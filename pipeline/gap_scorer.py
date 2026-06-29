"""Gap scoring engine: clusters limitation statements and ranks research gaps.

This is the discovery core. It pulls every Limitation node out of Neo4j, groups
semantically-similar limitations into clusters via Qdrant similarity, and scores
each cluster with the weighted formula from CLAUDE.md:

    score = 0.40*frequency + 0.35*recency + 0.25*solution_deficit
"""

import logging
import os
from collections import Counter

from pydantic import BaseModel

from graph.populate import get_neo4j_driver
from vectors.embed import (
    _COLLECTION_FUTURE_DIRECTIONS,
    _COLLECTION_LIMITATIONS,
    _embed_texts,
    get_qdrant_client,
    load_embedding_model,
)

logger = logging.getLogger(__name__)

# Similarity threshold for grouping two limitation statements into one cluster.
_CLUSTER_THRESHOLD = 0.82
# Similarity threshold for deciding a FutureDirection "addresses" a Limitation.
_SOLUTION_THRESHOLD = 0.75
# Upper bound on how many future-direction hits we inspect per cluster.
_MAX_FD_RESULTS = 100
# How much each paper's report counts toward frequency, by extraction tier.
# Explicitly-stated limitations are the strongest signal; inferred ones the weakest.
_TIER_WEIGHTS = {"explicit": 1.0, "conclusion": 0.75, "inferred": 0.5}
_DEFAULT_TIER_WEIGHT = 1.0


class GapResult(BaseModel):
    gap_description: str
    score: float
    frequency_score: float
    recency_score: float
    solution_deficit_score: float
    supporting_papers: list[str]
    proposed_solutions: list[str]


def get_all_limitations(domain: str = "computer_vision") -> list[dict]:
    """Query Neo4j for every Limitation node in a domain with its papers and years.

    Returns a list of dicts, one per Limitation node, with keys:
        text:      the limitation statement
        paper_ids: arxiv_ids of papers that report it
        years:     publication years, parallel to paper_ids
        tiers:     extraction tier per paper, parallel to paper_ids
    """
    driver = get_neo4j_driver()
    records: list[dict] = []
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run(
            """
            MATCH (p:Paper)-[r:REPORTS_LIMITATION]->(l:Limitation)
            WHERE p.domain = $domain
            RETURN l.text                AS text,
                   collect(p.arxiv_id)   AS paper_ids,
                   collect(p.year)       AS years,
                   collect(r.tier)       AS tiers
            """,
            domain=domain,
        )
        for record in result:
            records.append(
                {
                    "text": record["text"],
                    "paper_ids": list(record["paper_ids"]),
                    "years": list(record["years"]),
                    "tiers": list(record["tiers"]),
                }
            )
    driver.close()
    return records


def cluster_limitations(
    limitations: list[dict], min_cluster_size: int = 2
) -> list[list[dict]]:
    """Group semantically-similar limitations into clusters using Qdrant similarity.

    For each limitation we embed its text and search the 'limitations' collection,
    then union it with any other limitation in the input whose stored embedding
    scores at or above _CLUSTER_THRESHOLD (0.82). Limitations that match nothing
    else end up as singleton clusters.

    Returns a list of clusters; each cluster is a list of limitation dicts.
    """
    if not limitations:
        return []

    client = get_qdrant_client()
    model = load_embedding_model()

    # Map limitation text -> its index so we can resolve Qdrant hits back to inputs.
    text_to_idx: dict[str, int] = {}
    for i, lim in enumerate(limitations):
        text_to_idx.setdefault(lim["text"], i)

    # Union-find over limitation indices.
    parent = list(range(len(limitations)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, lim in enumerate(limitations):
        vector = _embed_texts(model, [lim["text"]])[0]
        results = client.query_points(
            collection_name=_COLLECTION_LIMITATIONS,
            query=vector,
            limit=len(limitations),
        )
        for hit in results.points:
            if hit.score < _CLUSTER_THRESHOLD:
                continue
            hit_text = hit.payload.get("limitation_text", "")
            j = text_to_idx.get(hit_text)
            if j is not None and j != i:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i in range(len(limitations)):
        groups.setdefault(find(i), []).append(limitations[i])

    return list(groups.values())


def compute_frequency_score(cluster: list[dict], total_papers: int) -> float:
    """Tier-weighted fraction of domain papers reporting any limitation in this cluster.

    Each unique paper contributes its tier weight (explicit=1.0, conclusion=0.75,
    inferred=0.5; default 1.0 when tier is missing). frequency_score = sum of those
    weights / total_papers_in_domain, capped at 1.0.
    """
    if total_papers <= 0:
        return 0.0

    # A paper's tier is consistent across its limitations; if it somehow appears
    # with several tiers, keep the strongest (highest weight).
    paper_weights: dict[str, float] = {}
    for lim in cluster:
        paper_ids = lim.get("paper_ids", [])
        tiers = lim.get("tiers", [])
        for i, paper_id in enumerate(paper_ids):
            tier = tiers[i] if i < len(tiers) else None
            weight = _TIER_WEIGHTS.get(tier, _DEFAULT_TIER_WEIGHT)
            if paper_id not in paper_weights or weight > paper_weights[paper_id]:
                paper_weights[paper_id] = weight

    weighted_sum = sum(paper_weights.values())
    return min(weighted_sum / total_papers, 1.0)


def compute_recency_score(cluster: list[dict], current_year: int = 2024) -> float:
    """Ratio of last-2-year papers vs all-time papers reporting this cluster.

    "Last 2 years" means year >= current_year - 1 (e.g. 2023 and 2024 for a 2024
    baseline). Returns 0.5 when no year data is available for any paper.
    """
    paper_year: dict[str, int] = {}
    for lim in cluster:
        paper_ids = lim.get("paper_ids", [])
        years = lim.get("years", [])
        for pid, yr in zip(paper_ids, years):
            if yr:
                paper_year[pid] = yr

    if not paper_year:
        return 0.5

    all_time = len(paper_year)
    recent = sum(1 for yr in paper_year.values() if yr >= current_year - 1)
    return recent / all_time


def compute_solution_deficit_score(cluster: list[dict]) -> float:
    """How unaddressed this cluster is by the corpus's future directions.

    solution_deficit = 1 - (future_directions_addressing / papers_reporting), where
    an addressing future direction is any FutureDirection whose embedding scores at
    or above _SOLUTION_THRESHOLD (0.75) against the cluster's centroid text. Capped
    to [0.0, 1.0]. A cluster nobody has proposed solutions for scores near 1.0.
    """
    if not cluster:
        return 1.0

    unique_papers: set[str] = set()
    for lim in cluster:
        unique_papers.update(lim.get("paper_ids", []))
    papers_reporting = len(unique_papers)
    if papers_reporting == 0:
        return 1.0

    centroid_text = _cluster_centroid_text(cluster)
    matches = len(_find_addressing_solutions(centroid_text))

    score = 1.0 - (matches / papers_reporting)
    return max(0.0, min(1.0, score))


def score_gaps(domain: str = "computer_vision", top_n: int = 20) -> list[GapResult]:
    """Discover, score, and rank research gaps for a domain.

    Pulls all limitations, clusters them, scores each cluster with the weighted
    formula, and returns the top_n GapResults sorted by score descending. The most
    frequent limitation text in a cluster becomes its gap_description.
    """
    limitations = get_all_limitations(domain)
    if not limitations:
        return []

    total_papers = _count_papers_in_domain(domain)
    clusters = cluster_limitations(limitations)

    results: list[GapResult] = []
    for cluster in clusters:
        frequency = compute_frequency_score(cluster, total_papers)
        recency = compute_recency_score(cluster)
        deficit = compute_solution_deficit_score(cluster)

        score = (0.40 * frequency) + (0.35 * recency) + (0.25 * deficit)

        centroid_text = _cluster_centroid_text(cluster)
        supporting_papers = sorted(
            {pid for lim in cluster for pid in lim.get("paper_ids", [])}
        )
        proposed_solutions = _find_addressing_solutions(centroid_text)

        results.append(
            GapResult(
                gap_description=centroid_text,
                score=round(score, 4),
                frequency_score=round(frequency, 4),
                recency_score=round(recency, 4),
                solution_deficit_score=round(deficit, 4),
                supporting_papers=supporting_papers,
                proposed_solutions=proposed_solutions,
            )
        )

    results.sort(key=lambda gap: gap.score, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cluster_centroid_text(cluster: list[dict]) -> str:
    """Return the most frequently occurring limitation text in the cluster."""
    if not cluster:
        return ""
    counter = Counter(lim["text"] for lim in cluster)
    return counter.most_common(1)[0][0]


def _find_addressing_solutions(centroid_text: str) -> list[str]:
    """Return future-direction texts that address a limitation above threshold.

    Searches the Qdrant 'future_directions' collection with the centroid embedding
    and returns the payload text of every hit scoring >= _SOLUTION_THRESHOLD.
    """
    if not centroid_text:
        return []

    client = get_qdrant_client()
    model = load_embedding_model()
    vector = _embed_texts(model, [centroid_text])[0]

    results = client.query_points(
        collection_name=_COLLECTION_FUTURE_DIRECTIONS,
        query=vector,
        limit=_MAX_FD_RESULTS,
    )

    solutions: list[str] = []
    for hit in results.points:
        if hit.score < _SOLUTION_THRESHOLD:
            continue
        # embed.py stores future-direction payloads under the 'limitation_text' key.
        text = hit.payload.get("limitation_text", "")
        if text:
            solutions.append(text)
    return solutions


def _count_papers_in_domain(domain: str) -> int:
    """Return the total number of Paper nodes in the given domain."""
    driver = get_neo4j_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run(
            "MATCH (p:Paper {domain: $domain}) RETURN count(p) AS n",
            domain=domain,
        )
        record = result.single()
        count = record["n"] if record else 0
    driver.close()
    return count
