"""Cross-domain hypothesis matching: CV limitations ↔ Medical Imaging future directions.

The discovery thesis: when a solution direction proposed in one domain semantically
matches an unresolved limitation in another, that pairing is a cross-domain research
hypothesis worth surfacing. This module ingests papers under an explicit domain tag,
finds genuinely-unresolved gaps in a source domain, and matches them against the
target domain's future directions in Qdrant.
"""

import logging
import os
import time

from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue

from graph.populate import (
    _upsert_paper_counting,
    create_constraints,
    get_neo4j_driver,
)
from pipeline.batch import _get_db, _log_failure, _paper_to_row
from pipeline.extractor import extract_paper
from pipeline.gap_scorer import GapResult, score_gaps
from vectors.embed import (
    _COLLECTION_FUTURE_DIRECTIONS,
    _embed_texts,
    embed_future_directions,
    embed_limitations,
    get_qdrant_client,
    load_embedding_model,
)

logger = logging.getLogger(__name__)

# Gaps with a deficit at or below this floor are considered adequately addressed.
_UNRESOLVED_DEFICIT_FLOOR = 0.3
# Seconds between individual paper ingestions (Semantic Scholar free tier: 1 req/sec).
_FETCH_SLEEP_SECONDS = 2
# How many future-direction candidates to pull per gap before thresholding.
_MAX_FD_CANDIDATES = 20

_EXPLAIN_PROMPT = """\
You are a scientific research strategist evaluating a cross-domain research hypothesis.

An unresolved problem in {source_domain}:
"{source_gap}"

A solution direction proposed in {target_domain}:
"{target_solution}"

In 2-3 sentences, explain why applying this {target_domain} solution direction to the \
{source_domain} problem is scientifically interesting. Be specific about the shared \
structure between the problem and the solution. Return only the explanation text, \
no preamble."""


class CrossDomainMatch(BaseModel):
    source_gap: str              # unresolved limitation in source domain
    target_solution: str         # future_direction from target domain
    similarity_score: float      # cosine similarity; cross-domain default 0.82 (lower than clustering 0.86 — cross-domain vocab divergence compresses scores)
    source_papers: list[str]
    target_papers: list[str]
    source_domain: str
    target_domain: str


def ingest_domain_papers(arxiv_ids: list[str], domain: str) -> dict:
    """Ingest papers for a specific domain, overriding the extracted domain field.

    Runs the existing pipeline per paper — extract_paper → SQLite → Neo4j — with the
    domain field patched after extraction (extract_paper hardcodes computer_vision).
    Papers already present in SQLite are skipped, never re-tagged. After ingestion,
    both Qdrant collections are re-synced from Neo4j so the new domain's limitations
    and future_directions carry the correct domain payload.

    Failures are logged to data/failed_extractions.log and do not abort the batch.
    Returns {"ingested": n, "failed": n, "skipped": n}.
    """
    db = _get_db()
    table = db["papers"]
    existing: set[str] = set()
    if "papers" in db.table_names():
        existing = {
            row[0] for row in db.execute("SELECT arxiv_id FROM papers").fetchall()
        }

    driver = get_neo4j_driver()
    create_constraints(driver)

    total = len(arxiv_ids)
    ingested = 0
    failed = 0
    skipped = 0

    for i, arxiv_id in enumerate(arxiv_ids, start=1):
        print(f"[{i}/{total}] {arxiv_id} ({domain})")

        if arxiv_id in existing:
            logger.info("Skipping %s — already ingested", arxiv_id)
            skipped += 1
            continue

        try:
            paper = extract_paper(arxiv_id)
            # extract_paper hardcodes domain="computer_vision"; patch it here.
            paper = paper.model_copy(update={"domain": domain})
            table.insert(_paper_to_row(paper), pk="arxiv_id", replace=False, alter=True)

            with driver.session(
                database=os.getenv("NEO4J_DATABASE", "neo4j")
            ) as session:
                session.execute_write(
                    lambda tx, p=paper.model_dump(): _upsert_paper_counting(tx, p)
                )
            ingested += 1
        except Exception as exc:  # noqa: BLE001 — log and continue, never crash the batch
            logger.error("Failed to ingest %s: %s", arxiv_id, exc)
            _log_failure(arxiv_id, str(exc))
            failed += 1

        # Semantic Scholar free tier is 1 req/sec — pace individual fetches.
        time.sleep(_FETCH_SLEEP_SECONDS)

    driver.close()

    if ingested:
        # Re-sync Qdrant from Neo4j: payload domains come from the Paper nodes, so
        # the new domain's texts land in both collections correctly tagged.
        embed_limitations()
        embed_future_directions()

    return {"ingested": ingested, "failed": failed, "skipped": skipped}


def get_unresolved_gaps(domain: str, top_n: int = 20) -> list[GapResult]:
    """Ranked gaps for a domain that remain genuinely unresolved.

    Runs score_gaps and keeps only gaps whose solution_deficit_score exceeds
    _UNRESOLVED_DEFICIT_FLOOR (0.3) — those the corpus's own future directions
    have not meaningfully addressed.
    """
    gaps = score_gaps(domain=domain, top_n=top_n)
    return [gap for gap in gaps if gap.solution_deficit_score > _UNRESOLVED_DEFICIT_FLOOR]


def find_cross_domain_matches(
    source_domain: str = "computer_vision",
    target_domain: str = "medical_imaging",
    top_n: int = 10,
    # Cross-domain threshold is intentionally lower than the within-domain cluster
    # threshold (0.86): different field vocabularies (CV "tracking" vs MI
    # "registration") naturally compress Specter2 scores, so the best achievable
    # cross-domain pairs peak at ~0.83-0.84 even when semantically equivalent.
    # 0.82 is the right operating point here; 0.84 only passes within-domain pairs.
    similarity_threshold: float = 0.82,
) -> list[CrossDomainMatch]:
    """Match unresolved source-domain gaps to target-domain future directions.

    Embeds every unresolved gap description in one batched Specter2 call, then
    searches the Qdrant 'future_directions' collection filtered to the target
    domain. Pairs scoring at or above similarity_threshold become
    CrossDomainMatch objects, sorted by similarity_score descending, top_n kept.
    """
    gaps = get_unresolved_gaps(source_domain)
    if not gaps:
        return []

    client = get_qdrant_client()
    model = load_embedding_model()
    # One batched encode call for all gap descriptions.
    vectors = _embed_texts(model, [gap.gap_description for gap in gaps])

    domain_filter = Filter(
        must=[FieldCondition(key="domain", match=MatchValue(value=target_domain))]
    )

    matches: list[CrossDomainMatch] = []
    for gap, vector in zip(gaps, vectors):
        results = client.query_points(
            collection_name=_COLLECTION_FUTURE_DIRECTIONS,
            query=vector,
            query_filter=domain_filter,
            limit=_MAX_FD_CANDIDATES,
        )
        for hit in results.points:
            if hit.score < similarity_threshold:
                continue
            # embed.py stores future-direction payloads under the 'limitation_text' key.
            solution_text = hit.payload.get("limitation_text", "")
            if not solution_text:
                continue
            matches.append(
                CrossDomainMatch(
                    source_gap=gap.gap_description,
                    target_solution=solution_text,
                    similarity_score=round(float(hit.score), 4),
                    source_papers=gap.supporting_papers,
                    target_papers=list(hit.payload.get("paper_ids", [])),
                    source_domain=source_domain,
                    target_domain=target_domain,
                )
            )

    matches.sort(key=lambda match: match.similarity_score, reverse=True)
    return matches[:top_n]


def explain_match(match: CrossDomainMatch) -> str:
    """Generate a 2-3 sentence explanation of why a cross-domain match is interesting.

    Calls Ollama with a prompt containing both the source gap and the target
    solution and returns the model's free-text explanation.
    """
    prompt = _EXPLAIN_PROMPT.format(
        source_domain=match.source_domain.replace("_", " "),
        target_domain=match.target_domain.replace("_", " "),
        source_gap=match.source_gap,
        target_solution=match.target_solution,
    )
    return _call_ollama_text(prompt)


def _call_ollama_text(prompt: str) -> str:
    """Send a free-text prompt to Ollama and return the stripped response text."""
    import ollama as _ollama  # lazy import so the module loads without Ollama running

    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    client = _ollama.Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"].strip()
