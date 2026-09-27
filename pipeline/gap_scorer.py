"""Gap scoring engine: clusters limitation statements and ranks research gaps.

This is the discovery core. It pulls every Limitation node out of Neo4j, groups
semantically-similar limitations into clusters via Qdrant similarity, and scores
each cluster with the weighted formula from CLAUDE.md:

    score = 0.40*frequency + 0.35*recency + 0.25*solution_deficit

**What those weights do and do not mean.** They are the coefficients, not the
observed influence on the ranking. `recency` and `solution_deficit` both span the
full [0, 1] interval on real data; `frequency` spans a small fraction of it,
because at this corpus size a gap is reported by one to a handful of papers out of
a few dozen. So the ordering is dominated by recency and deficit, and frequency
acts as a weak corroboration signal rather than the largest term. The measured
figures are recorded in CLAUDE.md and refreshed by
scripts/measure_weight_influence.py.

This is deliberately documented rather than corrected by normalisation: rescaling
terms to make the coefficients match their influence (PLAN.md #4 option 4B) would
make every score relative to whatever else is in the result set, so adding one
paper would move every number with no change in evidence — destroying the
cross-run comparability _corpus_reference_year exists to guarantee.
"""

import logging
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue, QueryRequest

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
#
# Per-domain 95th percentile of the *null distribution* — the similarity that
# arbitrary same-domain (limitation, limitation) pairs reach by chance, measured
# over every such pair in the corpus using the stored Specter2 vectors. Derived by
# scripts/derive_thresholds.py; re-run it after significant ingestion.
#
# Advisor decision, 2026-09-28 (PLAN.md #3, option 3A). The previous single 0.86
# was never derived at all. Its justification was that it "sits well above the
# median" — but the median of a *noise* distribution is not a bar, and using it as
# one is the exact error this project already identified and corrected for the
# cross-domain threshold ("0.82 sat below the median of pure noise"). That
# correction was applied to _SOLUTION_THRESHOLDS and to find_cross_domain_matches
# and never to the threshold that decides what a gap *is*.
#
# Measured against 0.86 on the pre-curation corpus: 10.9% of arbitrary CV pairs
# and 28.3% of arbitrary MI pairs cleared it. At a ~28% per-pair noise rate a long
# generic seed absorbs a quarter of its domain by chance, which is how 52 of 104
# medical-imaging limitations ended up in one "gap" labelled with a sentence about
# convolution kernel sizes while containing unrelated statements about stain
# normalisation and FC-layer matrix sizes.
#
# NOTE these are not comparable to _SOLUTION_THRESHOLDS below. They measure
# different pair populations (limitation x limitation vs limitation x
# future-direction) and the numbers must never be reasoned about side by side —
# doing so is what made 0.85 "look conservative" next to 0.86.
_CLUSTER_THRESHOLDS = {
    "computer_vision": 0.8769,   # null p95 over n=1,128 pairs (mean 0.8228, p50 0.8238)
    "medical_imaging": 0.8954,   # null p95 over n=5,565 pairs (mean 0.8401, p50 0.8406)
}


def _cluster_threshold(domain: str) -> float:
    """The clustering threshold for a domain.

    An unmeasured domain falls back to the strictest known floor, matching
    _solution_threshold's reasoning: guessing low silently merges unrelated
    limitations in a domain whose null was never measured, which corrupts every
    downstream score. Guessing high only fragments clusters, which is visible in
    the output as more single-paper gaps and is recoverable.
    """
    return _CLUSTER_THRESHOLDS.get(domain, max(_CLUSTER_THRESHOLDS.values()))


# No cluster may hold more than this share of its domain's limitations
# (PLAN.md #3, the structural half of the decision — option 3C).
#
# A percentile threshold bounds the per-*pair* error rate; it does nothing about
# the per-*cluster* outcome, because a seed is compared against every candidate
# and the expected number of chance members therefore grows with the corpus. The
# cap is what actually prevents one cluster from becoming half a domain, and it
# fails loudly by splitting rather than quietly by truncating.
_MAX_CLUSTER_SHARE = 0.20
# Floor so a small domain is not forbidden from clustering at all: 20% of 8
# limitations is 2, and a cap of 1 would mean "no clusters exist". This is a
# degeneracy guard, not a tuning dial.
_MIN_CLUSTER_CAP = 2
# Similarity thresholds for deciding a FutureDirection "addresses" a Limitation.
#
# Per-domain 95th percentile of the *null distribution* — the similarity that
# random, unrelated same-domain (limitation, future-direction) pairs reach by
# chance, measured over every such pair in the corpus using the stored Specter2
# vectors. At these values a match has at most a 5% chance of being noise.
#
# Advisor decision A1, 2026-08-23. The previous single 0.85 was below both
# floors: 20.1% of random CV pairs and 44.8% of random MI pairs cleared it, so a
# large share of "this gap is addressed" verdicts were chance. 0.85 looked
# conservative beside the 0.86 cluster threshold, but the two measure different
# populations and are not comparable.
#
# Corpus-dependent: re-derive after significant ingestion. See
# PROJECT_HARDENING_PLAN.md item A1.
_SOLUTION_THRESHOLDS = {
    "computer_vision": 0.8773,  # n=2,176 random pairs
    "medical_imaging": 0.8987,  # n=250 random pairs
}


def _solution_threshold(domain: str) -> float:
    """The addressing threshold for a domain.

    An unmeasured domain falls back to the strictest known floor. Guessing low
    would silently admit noise in a domain whose null was never measured;
    guessing high only under-counts solutions, which shows up as gaps looking
    more open than they are — visible, and recoverable.
    """
    return _SOLUTION_THRESHOLDS.get(domain, max(_SOLUTION_THRESHOLDS.values()))


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


def _cluster_cap(total_limitations: int) -> int:
    """Largest cluster permitted in a domain of this size.

    See _MAX_CLUSTER_SHARE. Ceiling rather than floor so the cap is never *below*
    the share, and floored at _MIN_CLUSTER_CAP so a small domain can still form
    clusters at all.
    """
    if total_limitations <= 0:
        return _MIN_CLUSTER_CAP
    share_cap = -(-int(total_limitations) * int(_MAX_CLUSTER_SHARE * 100) // 100)
    return max(_MIN_CLUSTER_CAP, share_cap)


def cluster_limitations(
    limitations: list[dict], domain: str = "computer_vision"
) -> list[list[dict]]:
    """Group semantically-similar limitations via seed-anchored direct grouping.

    All limitation texts are embedded in one batched Specter2 call and their
    Qdrant neighbours fetched up front (one batched query when the client
    supports it), **filtered to `domain`**. Limitations are then visited
    longest-text-first (a proxy for specificity): each unassigned limitation
    seeds a new cluster and pulls in the still-unassigned limitations whose
    similarity *to the seed itself* is at or above the domain's derived
    threshold, taking the **most similar first** and stopping at the size cap.
    Assigned limitations never join a second cluster.

    Two guards, both added 2026-09-28 (PLAN.md #3):

    **The neighbour query is domain-filtered.** It previously was not — the only
    filtered query in the project that wasn't — while asking for
    ``limit=len(limitations)`` from a collection holding every domain. Measured on
    the pre-curation corpus: a mean 42 of every 64 CV hits were other-domain
    points that were fetched, discarded, and crowded genuine same-domain
    neighbours out of the window entirely. Three above-threshold CV pairs were
    lost that way, and the loss grew with the other domain's size.

    **Cluster size is capped, and an over-cap cluster splits rather than
    truncating.** Candidates are considered in descending similarity, so the
    tightest subgroup stays with the seed; everything past the cap is simply left
    unassigned and re-seeds its own cluster later in the same pass. Nothing is
    dropped — the clusters always partition the input exactly. Truncating instead
    would silently delete limitations from the corpus's own accounting.

    A note on why the threshold alone was not enough. Seed-anchoring does remove
    the transitive A→B→C chain mechanism, and the previous docstring claimed that
    therefore "prevents ... one giant cluster". It does not: at a per-pair noise
    rate of ~28% a long generic seed absorbs a quarter of its domain *directly*,
    which is how 52 of 104 medical-imaging limitations became one cluster. The
    percentile threshold bounds the error per pair; the cap bounds the outcome.

    Singleton clusters are valid output and are never discarded. There is no
    minimum-size filter: a `min_cluster_size` parameter used to be declared here
    but was never read, which implied a filtering step that did not exist.
    Introducing one would change rankings and needs an advisor decision.

    Returns a list of clusters; each cluster is a list of limitation dicts.
    """
    if not limitations:
        return []

    client = get_qdrant_client()
    model = load_embedding_model()

    # One batched encode call for every text — never one call per limitation.
    texts = [lim["text"] for lim in limitations]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )
    vectors = [np.asarray(vec, dtype=float).tolist() for vec in embeddings]

    # Map limitation text -> its index so we can resolve Qdrant hits back to inputs.
    text_to_idx: dict[str, int] = {}
    for i, lim in enumerate(limitations):
        text_to_idx.setdefault(lim["text"], i)

    neighbours = _query_neighbours(
        client, vectors, limit=len(limitations), domain=domain
    )

    threshold = _cluster_threshold(domain)
    cap = _cluster_cap(len(limitations))

    # Longest text first: more specific statements make better cluster seeds.
    seed_order = sorted(
        range(len(limitations)), key=lambda i: len(texts[i]), reverse=True
    )

    assigned = [False] * len(limitations)
    clusters: list[list[dict]] = []
    split_count = 0
    for seed in seed_order:
        if assigned[seed]:
            continue
        assigned[seed] = True
        member_idxs = [seed]

        # Gather every eligible candidate first, then admit the most similar up to
        # the cap. Taking them in hit order instead would make membership depend
        # on Qdrant's return order rather than on similarity.
        candidates: list[tuple[float, int]] = []
        for hit in neighbours[seed]:
            # hit.score is cosine similarity against the seed's own vector, so
            # this threshold anchors membership to the seed, not to other members.
            if hit.score < threshold:
                continue
            j = text_to_idx.get(hit.payload.get("limitation_text", ""))
            if j is None or j == seed or assigned[j]:
                continue
            candidates.append((float(hit.score), j))
        candidates.sort(key=lambda pair: pair[0], reverse=True)

        admitted = candidates[: max(0, cap - 1)]
        if len(candidates) > len(admitted):
            split_count += 1
            logger.info(
                "Cluster cap %d reached in %s: seed %r kept the %d most similar of "
                "%d eligible members; the remaining %d re-seed their own clusters.",
                cap,
                domain,
                texts[seed][:60],
                len(admitted),
                len(candidates),
                len(candidates) - len(admitted),
            )
        for _score, j in admitted:
            assigned[j] = True
            member_idxs.append(j)

        clusters.append([limitations[k] for k in member_idxs])

    if split_count:
        logger.info(
            "%s: %d cluster(s) hit the %d-member cap (%.0f%% of %d limitations) and split.",
            domain,
            split_count,
            cap,
            _MAX_CLUSTER_SHARE * 100,
            len(limitations),
        )
    return clusters


def _query_neighbours(
    client, vectors: list[list[float]], limit: int, domain: str | None = None
) -> list[list]:
    """Fetch each vector's neighbours from the 'limitations' collection.

    Uses Qdrant's batch query endpoint (a single round trip) when the client
    provides it; otherwise falls back to sequential query_points calls reusing
    the cached embeddings. Returns hit lists index-aligned with `vectors`.

    `domain` narrows the search server-side, matching every other filtered query
    in the project (vectors/search.py, _find_addressing_solutions,
    cross_domain.find_cross_domain_matches). Without it the `limit` window is
    consumed by other-domain points that are then discarded client-side, which
    silently evicts genuine same-domain neighbours — see cluster_limitations.
    """
    query_filter = None
    if domain is not None:
        query_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    batch_query = getattr(client, "query_batch_points", None)
    if callable(batch_query):
        requests = [
            QueryRequest(
                query=vector, limit=limit, with_payload=True, filter=query_filter
            )
            for vector in vectors
        ]
        responses = batch_query(
            collection_name=_COLLECTION_LIMITATIONS, requests=requests
        )
        return [response.points for response in responses]

    return [
        client.query_points(
            collection_name=_COLLECTION_LIMITATIONS,
            query=vector,
            limit=limit,
            query_filter=query_filter,
        ).points
        for vector in vectors
    ]


def compute_frequency_score(cluster: list[dict], total_papers: int) -> float:
    """Tier-weighted share of *contributing* papers reporting a limitation in this cluster.

    Each unique paper contributes its tier weight (explicit=1.0, conclusion=0.75,
    inferred=0.5; default 1.0 when tier is missing). frequency_score = sum of those
    weights / total_papers, capped at 1.0.

    `total_papers` is supplied by the caller and **must be the number of papers
    that could possibly appear in a numerator** — i.e. papers with at least one
    extracted limitation (`_count_contributing_papers`), not every Paper node in
    the domain. score_gaps passes the former. Using the latter, as this did until
    2026-09-28, silently multiplied the metric by the extraction success rate:
    43% of CV papers and 38% of MI papers extracted no limitations at all, so they
    could only ever dilute. See PLAN.md #4.

    **Read this term as a weak corroboration signal, not as a driver of the
    ranking.** Even after the denominator fix its dynamic range is far narrower
    than recency's or solution-deficit's, both of which span the full unit
    interval, so its measured influence on the ordering is nowhere near its
    nominal 0.40 weight. The honest reading of a frequency_score is "how many
    papers, out of those that said anything, said this" — at this corpus size that
    is a count of one to a handful, and the 95% confidence intervals of a
    one-paper and a three-paper gap overlap heavily. The composite score is
    dominated by recency and deficit; that is a property of the data, not a bug,
    and it is documented rather than normalised away (option 4B was rejected,
    because normalising would make scores relative to the result set and destroy
    the cross-run comparability _corpus_reference_year exists to protect).
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


def compute_recency_score(cluster: list[dict], current_year: int | None = None) -> float:
    """Ratio of last-2-year papers vs all-time papers reporting this cluster.

    "Last 2 years" means year >= current_year - 1 (e.g. 2024 and 2025 for a 2025
    baseline). Returns 0.5 when no year data is available for any paper.

    current_year is the baseline the window hangs off. score_gaps() always passes
    the corpus-derived reference year explicitly — see _corpus_reference_year for
    why the corpus, not the wall clock, defines "now". The wall-clock fallback
    here only applies to standalone calls that supply no baseline of their own.
    """
    if current_year is None:
        current_year = _current_year()

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


def compute_solution_deficit_score(
    cluster: list[dict], domain: str = "computer_vision"
) -> float:
    """How unaddressed this cluster is by the corpus's future directions.

    solution_deficit = 1 - (future_directions_addressing / papers_reporting), where
    an addressing future direction is any FutureDirection that scores at or above
    the domain's noise floor (_solution_threshold) against the cluster's centroid
    text, is in the same domain, and does not come from a paper that reports it
    (see _find_addressing_solutions). Clamped to [0.0, 1.0]. A cluster nobody has
    proposed solutions for scores near 1.0.
    """
    if not cluster:
        return 1.0

    unique_papers: set[str] = set()
    for lim in cluster:
        unique_papers.update(lim.get("paper_ids", []))
    papers_reporting = len(unique_papers)
    if papers_reporting == 0:
        return 1.0

    centroid_text = _cluster_representative_text(cluster)
    matches = len(
        _find_addressing_solutions(
            centroid_text, domain=domain, exclude_paper_ids=unique_papers
        )
    )

    # No min(matches, papers_reporting) here: it was dead code. The clamp below
    # already floors any ratio above 1.0 at zero, and capping was verified to
    # produce identical scores on every cluster in the corpus.
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

    # The frequency denominator is the papers that could appear in a numerator,
    # not every paper in the domain — see _count_contributing_papers (PLAN.md #4).
    total_papers = _count_contributing_papers(domain)
    clusters = cluster_limitations(limitations, domain=domain)

    # One corpus-wide baseline for every cluster. Deriving it per-cluster would be
    # wrong: a cluster's own newest paper would always fall inside its own window,
    # so an all-2019 cluster would score as recent as an all-2025 one.
    reference_year = _corpus_reference_year(limitations)

    # (gap, newest_year) pairs: the newest publication year in a cluster is a
    # tie-break key but is not part of GapResult, so it is carried alongside rather
    # than widening the response model for an internal ordering concern.
    scored: list[tuple[GapResult, int]] = []
    for cluster in clusters:
        frequency = compute_frequency_score(cluster, total_papers)
        recency = compute_recency_score(cluster, current_year=reference_year)
        deficit = compute_solution_deficit_score(cluster, domain=domain)

        score = (0.40 * frequency) + (0.35 * recency) + (0.25 * deficit)

        centroid_text = _cluster_representative_text(cluster)
        supporting_papers = sorted(
            {pid for lim in cluster for pid in lim.get("paper_ids", [])}
        )
        # Same filtering as the score above — the solutions shown to the user are
        # exactly the ones counted against the deficit, never a looser set.
        proposed_solutions = _find_addressing_solutions(
            centroid_text, domain=domain, exclude_paper_ids=set(supporting_papers)
        )

        newest_year = max(
            (yr for lim in cluster for yr in lim.get("years", []) if isinstance(yr, int)),
            default=0,
        )

        scored.append((
            GapResult(
                gap_description=centroid_text,
                score=round(score, 4),
                frequency_score=round(frequency, 4),
                recency_score=round(recency, 4),
                solution_deficit_score=round(deficit, 4),
                supporting_papers=supporting_papers,
                proposed_solutions=proposed_solutions,
            ),
            newest_year,
        ))

    scored.sort(key=_ranking_key)
    return [gap for gap, _year in scored][:top_n]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ranking_key(entry: tuple[GapResult, int]) -> tuple:
    """Total ordering for the gap list: score, then corroboration, then recency, then text.

    Sorting by score alone left ties to the stable sort, which preserved cluster
    creation order — and clusters are created longest-seed-text-first, so **rank
    within a tie block tracked description length**. That was not a subtle bias:
    on the pre-fix corpus, ranks 1-3 tied at 0.6065 had descriptions of 72, 70 and
    64 characters and ranks 4-7 tied at 0.6043 had 50, 44, 41 and 34, monotonically
    descending. A copyright-law paper held rank 1 because its sentence was the
    longest in its tie group.

    Ties remain common because recency and solution-deficit both saturate — a
    single recent paper with no addressing future direction scores 1.0 on both — so
    after fixing the denominator in #4 there are still 15 of 28 CV gaps and 27 of 54
    MI gaps sharing a score with another gap. The keys, in order:

    1. **score**, descending — the formula still decides.
    2. **supporting-paper count**, descending. Among equally-scored gaps the
       better-corroborated one ranks higher. This is the qualifier `SupportBadge`
       already foregrounds in the UI, whose own comment notes that a one-paper gap
       can out-rank a well-attested one.
    3. **newest supporting paper's year**, descending — a live concern before a
       dormant one.
    4. **description**, ascending lexicographic. Not meaningful, but it guarantees
       a *total* order, so the output is reproducible across runs and processes
       rather than depending on dict or set iteration order.

    Deliberately NOT a key: description length, in either direction.
    """
    gap, newest_year = entry
    return (-gap.score, -len(gap.supporting_papers), -newest_year, gap.gap_description)


def _current_year() -> int:
    """The wall-clock year. Isolated so tests can pin 'today' without patching time."""
    return datetime.now(timezone.utc).year


def _corpus_reference_year(limitations: list[dict]) -> int:
    """The baseline year recency is measured against: the corpus's leading edge.

    Recency is a *relative* discriminator carrying 0.35 of the composite score, so
    it is anchored to the newest publication year actually present in the corpus
    rather than to the wall clock. Two reasons:

    1. The term keeps discriminating. Ingestion always lags publication, and the
       gap only widens as a corpus sits. Measuring against the calendar means an
       ageing corpus eventually has *no* papers inside the window, every cluster
       scores 0.0, and 35% of the formula silently becomes dead weight that
       separates nothing. Anchoring to the data guarantees a non-empty numerator.
    2. Scoring stays deterministic. score_gaps() is a ranking function; the same
       corpus must rank the same way whenever it runs. A wall-clock baseline makes
       rankings drift with no data change and makes a stored GapResult
       incomparable to a freshly computed one.

    The one thing the corpus cannot self-correct is future-dated metadata: a
    single paper mislabelled 2030 would push the window past every real paper and
    zero out the term for everything. The result is therefore clamped to the
    current year. For any sane corpus the clamp never binds, so determinism holds.

    Falls back to the current year when no usable year data exists at all.
    """
    years = [
        yr
        for lim in limitations
        for yr in lim.get("years", [])
        if isinstance(yr, int) and yr > 0
    ]
    if not years:
        return _current_year()
    return min(max(years), _current_year())


# Memo for _cluster_representative_text, keyed by the cluster's texts. The
# representative is needed twice per cluster (once for the deficit score, once for
# the displayed description and solutions) and computing it embeds every member, so
# without this the embedding work doubles. Process-lifetime, like _model_cache.
_representative_cache: dict[tuple[str, ...], str] = {}


def _cluster_representative_text(cluster: list[dict]) -> str:
    """The cluster member closest to the cluster's own vector centroid.

    This text is what users see as `gap_description`, *and* what the deficit score
    and `proposed_solutions` are computed against — so a bad choice here is visible
    three ways at once.

    **Renamed from `_cluster_centroid_text`, which never computed a centroid.** It
    returned `Counter(lim["text"] for lim in cluster).most_common(1)[0][0]` and its
    docstring called that "the most frequently occurring limitation text". But
    `Limitation` is `UNIQUE` on `text` in Neo4j, so within a cluster every count is
    always exactly 1; `most_common` therefore always resolved a total tie by
    insertion order, which is the seed, which `cluster_limitations` picks as the
    **longest string**. Measured before the fix: the description equalled the
    longest member in 27 of 27 CV clusters. The "most frequent" code path could not
    fire, and the effect was that a six-member cluster was labelled with whichever
    member had the most characters — in one live case a sentence about imaged
    anatomy written by exactly one of its five papers, which then drove the top
    cross-domain match.

    The centroid is the mean of the L2-normalised member vectors, renormalised, so
    "nearest" is cosine similarity — the same metric clustering and thresholding
    use. A singleton short-circuits without embedding anything. Ties in similarity
    fall back to the lexicographically smallest text, so the result is
    deterministic rather than dependent on member order.

    **A two-member cluster's label is always decided by that tie-break, never by
    similarity.** Both members of a pair are exactly equidistant from their own
    centroid by symmetry, so the comparison is always a tie. That is not a defect
    to fix — there is no principled "more central" member of a pair — but it does
    mean the label of a two-member gap carries no semantic claim to being the
    better summary of the two, and a reader should not infer one. It also means a
    test asserting this function's output must mirror the lexicographic tie-break
    rather than use argmax, which would silently assert member ordering instead.
    """
    if not cluster:
        return ""
    texts = [lim["text"] for lim in cluster]
    if len(texts) == 1:
        return texts[0]

    key = tuple(texts)
    cached = _representative_cache.get(key)
    if cached is not None:
        return cached

    model = load_embedding_model()
    vectors = np.asarray(_embed_texts(model, texts), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector would divide by zero; it cannot be near anything, so leave it.
    norms[norms == 0] = 1.0
    unit = vectors / norms

    centroid = unit.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        # Mutually opposed members — no meaningful centre. Fall back to the
        # deterministic lexicographic choice rather than an arbitrary index.
        representative = min(texts)
    else:
        similarities = unit @ (centroid / centroid_norm)
        best = float(similarities.max())
        # Lexicographic tie-break keeps this independent of member ordering.
        representative = min(
            text for text, sim in zip(texts, similarities) if sim >= best - 1e-12
        )

    _representative_cache[key] = representative
    return representative


def _find_addressing_solutions(
    centroid_text: str,
    domain: str = "computer_vision",
    exclude_paper_ids: set[str] | None = None,
) -> list[str]:
    """Return future-direction texts that genuinely address a limitation.

    A future direction counts only if all three hold:

    1. It scores >= the domain's noise floor (_solution_threshold: 0.8773 for CV,
       0.8987 for medical imaging) against the centroid embedding. Below that a
       match is statistically indistinguishable from a random pairing.
    2. It belongs to `domain`. Without this filter a medical-imaging suggestion
       could mark a CV gap as solved. Note that pipeline/cross_domain.py treats
       exactly that pairing as a *discovery* — so counting it here as a solution
       would demote the very gaps the cross-domain feature exists to surface.
    3. It comes from a paper outside `exclude_paper_ids` — the papers reporting
       this limitation. A paper restating its own open problem as future work is
       the definition of an unsolved gap, not evidence that anyone solved it;
       counting it inverted the signal.

    The domain filter is applied server-side by Qdrant, mirroring the pattern in
    vectors/search.py's find_similar_future_directions(). The query is kept here
    rather than delegated so the threshold and self-exclusion happen in one pass
    over the hits, and so this module's Qdrant client stays a single test seam.
    """
    if not centroid_text:
        return []

    client = get_qdrant_client()
    model = load_embedding_model()
    vector = _embed_texts(model, [centroid_text])[0]

    results = client.query_points(
        collection_name=_COLLECTION_FUTURE_DIRECTIONS,
        query=vector,
        query_filter=Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        ),
        limit=_MAX_FD_RESULTS,
    )

    threshold = _solution_threshold(domain)
    excluded = exclude_paper_ids or set()
    solutions: list[str] = []
    for hit in results.points:
        if hit.score < threshold:
            continue
        # A future direction can be attached to several papers; if any of them
        # reports this limitation, it is self-referential and does not count.
        if excluded and set(hit.payload.get("paper_ids") or []) & excluded:
            continue
        # embed.py stores future-direction payloads under the 'limitation_text' key.
        text = hit.payload.get("limitation_text", "")
        if text:
            solutions.append(text)
    return solutions


def _count_contributing_papers(domain: str) -> int:
    """Papers in `domain` with at least one extracted limitation.

    This is the frequency denominator. The population a frequency is a fraction
    *of* has to be the population that could appear in its numerator; a paper that
    extracted no limitations cannot corroborate any gap, so counting it makes
    `frequency_score` the product of two unrelated things — how widely a
    limitation is reported, and how often extraction succeeded.

    Measured on the curated corpus: 31 CV papers of which 22 contribute, and 85 MI
    papers of which 54 contribute. Dividing by 31 and 85 understated every CV
    frequency by ~29% and every MI frequency by ~36%.

    Kept separate from _count_papers_in_domain, which /corpus still reports as the
    corpus size — a reader asking "how big is this corpus" wants every paper, and
    a reader asking "what divides these scores" wants this. Conflating them is why
    the banner previously described a number that was not the divisor.
    """
    driver = get_neo4j_driver()
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run(
            """
            MATCH (p:Paper {domain: $domain})-[:REPORTS_LIMITATION]->(:Limitation)
            RETURN count(DISTINCT p) AS n
            """,
            domain=domain,
        )
        record = result.single()
        count = record["n"] if record else 0
    driver.close()
    return count


def _count_papers_in_domain(domain: str) -> int:
    """Return the total number of Paper nodes in the given domain.

    This is the corpus *size* for a domain, reported by /corpus. It is deliberately
    **not** the frequency denominator — see _count_contributing_papers.
    """
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
