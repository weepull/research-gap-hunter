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

import numpy as np
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue

from graph.populate import (
    _upsert_paper_counting,
    create_constraints,
    get_neo4j_driver,
)
from pipeline.batch import _get_db, _log_failure, _paper_to_row
from pipeline.domains import validate_domain
from pipeline.extractor import extract_paper
from pipeline.gap_scorer import (
    GapResult,
    _solution_threshold,
    score_gaps,
)
from vectors.embed import (
    _COLLECTION_FUTURE_DIRECTIONS,
    _COLLECTION_LIMITATIONS,
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

# 95th percentile of the *null distribution*: the similarity that random,
# unrelated cross-domain pairs reach by chance. Measured over all possible
# (limitation, future-direction) pairs in both directions using the stored
# Specter2 vectors. A match at or above this has at most a 5% chance of being
# noise.
#
# Advisor decision A2, 2026-08-23, replacing 0.82. That value was not merely
# loose, it sat *below the median of pure noise*: 61.6% of random pairs cleared
# it. The earlier rationale — that cross-domain vocabulary divergence compresses
# scores, so the threshold should sit below the within-domain one — had the logic
# backwards. Compression raises the noise floor as well as the signal, so a
# compressed space needs a *higher* bar, not a lower one.
#
# Hoisted from an inline default on find_cross_domain_matches on 2026-09-28 so
# that verify_pairing gates on exactly the number the matcher admits; two copies
# of the literal could drift. The value is unchanged.
#
# Corpus-dependent: re-derive with scripts/derive_thresholds.py after significant
# ingestion. Re-derived on the curated 116-paper corpus it comes to 0.8793 — a
# 0.0001 difference, left alone because changing it is an advisor decision. See
# PROJECT_HARDENING_PLAN.md item A2.
_CROSS_DOMAIN_THRESHOLD = 0.8792

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
    # Cosine similarity. The default cross-domain threshold is
    # _CROSS_DOMAIN_THRESHOLD, the 95th percentile of randomly-paired
    # cross-domain similarity — i.e. the score a meaningless pairing reaches by
    # chance. See find_cross_domain_matches and verify_pairing.
    similarity_score: float
    source_papers: list[str]
    target_papers: list[str]
    source_domain: str
    target_domain: str


def ingest_domain_papers(arxiv_ids: list[str], domain: str) -> dict:
    """Ingest papers for a specific domain, overriding the extracted domain field.

    Runs the existing pipeline per paper — extract_paper → SQLite → Neo4j — passing
    ``domain`` straight into extraction, which requires and validates it.
    Papers already present in SQLite are skipped, never re-tagged. After ingestion,
    both Qdrant collections are re-synced from Neo4j so the new domain's limitations
    and future_directions carry the correct domain payload.

    Failures are logged to data/failed_extractions.log and do not abort the batch.
    Returns {"ingested": n, "failed": n, "skipped": n}.
    """
    domain = validate_domain(domain)
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
            # domain is passed in, not patched afterwards: extract_paper now
            # requires and validates it (PLAN.md #1, option 1A).
            paper = extract_paper(arxiv_id, domain=domain)
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
    # See _CROSS_DOMAIN_THRESHOLD. Kept as an overridable parameter so a caller can
    # explore a different operating point, but the default is the derived constant
    # rather than a literal repeated here — verify_pairing must gate on exactly the
    # same number this matcher admits, and two literals could drift apart.
    similarity_threshold: float = _CROSS_DOMAIN_THRESHOLD,
) -> list[CrossDomainMatch]:
    """Match unresolved source-domain gaps to target-domain future directions.

    Embeds every unresolved gap description in one batched Specter2 call, then
    searches the Qdrant 'future_directions' collection filtered to the target
    domain. Pairs scoring at or above similarity_threshold become
    CrossDomainMatch objects, sorted by similarity_score descending, top_n kept.

    **An empty list is a valid, expected result.** On a corpus where no pair
    clears the noise floor, returning nothing is the honest answer — it means
    there is no defensible cross-domain hypothesis in the data, not that
    something failed. Callers must render the empty case rather than treat it as
    an error; the frontend already shows a "No connections found" state.
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


class UngroundedPairingError(ValueError):
    """A requested (gap, solution) pairing is not a defensible corpus match.

    Raised *before* any LLM call, by design. See verify_pairing.
    """


class VerifiedPairing(BaseModel):
    """A pairing confirmed to exist in the corpus and to clear its noise floor."""

    source_gap: str
    target_solution: str
    source_domain: str
    target_domain: str
    # Recomputed from the stored vectors, not re-embedded and never a placeholder.
    similarity_score: float
    threshold: float
    source_papers: list[str]
    target_papers: list[str]


def _find_stored_point(collection: str, domain: str, text: str):
    """Locate an exact text in a collection, returning its point (with vector).

    Scrolls the **domain-filtered** slice and matches the text client-side rather
    than filtering on the text field server-side. That is deliberate: `domain` is
    the only indexed payload field (vectors/embed.py:_INDEXED_PAYLOAD_FIELDS), and
    Qdrant Cloud *refuses* a filter on an unindexed field rather than falling back
    to a scan — so a server-side text filter would work locally and fail in
    production, which is the exact failure mode documented in ensure_payload_indexes.
    Adding a keyword index over a multi-sentence text field to avoid one scan of a
    few hundred points would be a poor trade.
    """
    client = get_qdrant_client()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
            ),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            if point.payload.get("limitation_text") == text:
                return point
        if offset is None:
            return None


def verify_pairing(
    source_gap: str,
    target_solution: str,
    source_domain: str = "computer_vision",
    target_domain: str = "medical_imaging",
) -> VerifiedPairing:
    """Confirm a pairing is real and above its noise floor, or raise.

    This is the deterministic, server-side gate that PLAN.md #5 (option 5A)
    requires, and it runs **before** any generation.

    What it fixes. `/explain` accepted `source_gap` and `target_solution` as two
    arbitrary client-supplied strings, built a CrossDomainMatch with
    `similarity_score=0.0` hardcoded, and called the LLM — whose prompt never saw
    the score and instructed it to "be specific about the shared structure". The
    model therefore could not decline. Fed an analytic-number-theory limitation and
    a histopathology future direction, llama3.1:8b produced a fluent account of how
    "both domains involve distributed data and computational resources", and echoed
    a false `computer_vision` label back as fact.

    Prompt wording alone was explicitly rejected as the fix (option 5B): asking an
    8B model to refuse a leading question is weak, and a probabilistic refusal
    cannot be asserted by a test. The gate is code.

    Two independent conditions, in order:

    1. **Both texts must exist in the corpus.** The gap must be a stored Limitation
       in `source_domain` and the solution a stored FutureDirection in
       `target_domain`. This closes the arbitrary-string hole outright — no amount
       of similarity makes a sentence that appears in no paper a finding about the
       literature.
    2. **The pair must clear the same noise floor the matcher uses.** Similarity is
       recomputed from the two **stored** vectors, so it is exactly the number
       `find_cross_domain_matches` would have produced, not a fresh embedding that
       could drift. The bar is the cross-domain threshold for a cross-domain pair,
       or `_solution_threshold(domain)` for a same-domain one — reusing the derived
       nulls rather than inventing a third constant.
    """
    if not source_gap.strip() or not target_solution.strip():
        raise UngroundedPairingError("source_gap and target_solution must be non-empty")

    gap_point = _find_stored_point(_COLLECTION_LIMITATIONS, source_domain, source_gap)
    if gap_point is None:
        raise UngroundedPairingError(
            f"not a corpus match: no limitation with this exact text is recorded in "
            f"{source_domain!r}. Explanations are only generated for pairings that "
            f"exist in the corpus."
        )

    solution_point = _find_stored_point(
        _COLLECTION_FUTURE_DIRECTIONS, target_domain, target_solution
    )
    if solution_point is None:
        raise UngroundedPairingError(
            f"not a corpus match: no future direction with this exact text is "
            f"recorded in {target_domain!r}."
        )

    left = np.asarray(gap_point.vector, dtype=float)
    right = np.asarray(solution_point.vector, dtype=float)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    similarity = float(left @ right / denominator) if denominator else 0.0

    threshold = (
        _CROSS_DOMAIN_THRESHOLD
        if source_domain != target_domain
        else _solution_threshold(source_domain)
    )
    if similarity < threshold:
        raise UngroundedPairingError(
            f"below the noise floor: this pairing scores {similarity:.4f} against a "
            f"threshold of {threshold:.4f}, the 95th percentile of randomly-paired "
            f"similarity. At or below that a match is statistically indistinguishable "
            f"from chance, so no explanation is generated."
        )

    return VerifiedPairing(
        source_gap=source_gap,
        target_solution=target_solution,
        source_domain=source_domain,
        target_domain=target_domain,
        similarity_score=round(similarity, 4),
        threshold=threshold,
        source_papers=list(gap_point.payload.get("paper_ids") or []),
        target_papers=list(solution_point.payload.get("paper_ids") or []),
    )


def explain_match(match: CrossDomainMatch) -> str:
    """Generate a 2-3 sentence explanation of why a cross-domain match is interesting.

    Always calls the local Ollama. There is deliberately no hosted-LLM path: a
    public deployment refuses /explain outright (see api/main.py) rather than
    paying per-request API costs for anonymous callers, so this function is only
    ever reached when Ollama is available.
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
