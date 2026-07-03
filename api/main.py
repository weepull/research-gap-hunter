"""FastAPI application — REST layer for Research Gap Hunter (port 8000).

All heavy backend objects (Specter2 model, Qdrant client, Neo4j driver) are
initialised once in the lifespan context manager and stored in app.state.
Endpoint handlers call module-level backend functions so tests can patch them
at the api.main namespace without touching the real services.
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level backend imports — patched by tests via api.main.<name>
# ---------------------------------------------------------------------------
from graph.populate import _upsert_paper_counting, get_neo4j_driver  # noqa: E402
from pipeline.batch import _get_db, _log_failure, _paper_to_row, get_paper  # noqa: E402
from pipeline.cross_domain import (  # noqa: E402
    CrossDomainMatch,
    explain_match,
    find_cross_domain_matches,
)
from pipeline.extractor import extract_paper  # noqa: E402
from pipeline.gap_scorer import GapResult, score_gaps  # noqa: E402
from vectors.embed import (  # noqa: E402
    embed_future_directions,
    embed_limitations,
    get_qdrant_client,
    load_embedding_model,
)
from vectors.search import find_similar_limitations  # noqa: E402

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    papers: int
    limitations: int
    future_directions: int


class LimitationResult(BaseModel):
    limitation_text: str
    score: float
    paper_ids: list[str]
    domain: str


class IngestRequest(BaseModel):
    arxiv_id: str
    domain: str = "computer_vision"


class IngestResponse(BaseModel):
    status: str
    arxiv_id: str
    limitations_found: int
    tier: str


class ErrorResponse(BaseModel):
    status: str
    message: str


class ExplainResponse(BaseModel):
    explanation: str


# ---------------------------------------------------------------------------
# Lifespan — warm model cache and verify service connectivity once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Research Gap Hunter API…")
    app.state.model = load_embedding_model()
    app.state.qdrant = get_qdrant_client()
    app.state.neo4j = get_neo4j_driver()
    logger.info("All backend services connected.")
    yield
    logger.info("Shutting down…")
    app.state.neo4j.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Gap Hunter",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness check — counts papers, limitations, future_directions."""
    db = _get_db()
    if "papers" in db.table_names():
        papers = db.execute("SELECT count(*) FROM papers").fetchone()[0]
    else:
        papers = 0

    client = get_qdrant_client()
    try:
        lim_count = client.get_collection("limitations").points_count or 0
    except Exception:
        lim_count = 0
    try:
        fd_count = client.get_collection("future_directions").points_count or 0
    except Exception:
        fd_count = 0

    return HealthResponse(
        status="ok",
        papers=papers,
        limitations=lim_count,
        future_directions=fd_count,
    )


@app.get("/gaps", response_model=list[GapResult])
def get_gaps(
    domain: str = Query(default="computer_vision"),
    top_n: int = Query(default=20, ge=1, le=100),
) -> list[GapResult]:
    """Return top-ranked research gaps for a domain, scored by frequency/recency/deficit."""
    return score_gaps(domain=domain, top_n=top_n)


@app.get("/search", response_model=list[LimitationResult])
def search_limitations(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(default=10, ge=1, le=50),
    domain: str = Query(default="computer_vision"),
) -> list[LimitationResult]:
    """Vector-search the limitations collection and return semantically similar statements."""
    results = find_similar_limitations(query_text=q, top_k=top_k, domain=domain)
    return [
        LimitationResult(
            limitation_text=r["limitation_text"],
            score=r["score"],
            paper_ids=r.get("paper_ids", []),
            domain=r.get("domain", domain),
        )
        for r in results
    ]


@app.get("/cross-domain", response_model=list[CrossDomainMatch])
def get_cross_domain_matches(
    source: str = Query(default="computer_vision"),
    target: str = Query(default="medical_imaging"),
    top_n: int = Query(default=10, ge=1, le=50),
) -> list[CrossDomainMatch]:
    """Return ranked cross-domain research hypotheses above the similarity threshold."""
    return find_cross_domain_matches(
        source_domain=source,
        target_domain=target,
        top_n=top_n,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_paper(body: IngestRequest) -> IngestResponse:
    """Extract a paper from arXiv, store it in SQLite + Neo4j, and re-sync Qdrant.

    Extraction runs Semantic Scholar metadata fetch + PDF download + Ollama LLM.
    Expect ~30–60 s per paper. The domain field overrides the extractor default.
    """
    arxiv_id = body.arxiv_id.strip()
    domain = body.domain.strip()

    try:
        paper = extract_paper(arxiv_id)
        paper = paper.model_copy(update={"domain": domain})

        # SQLite — alter=True handles missing columns from schema drift
        db = _get_db()
        db["papers"].insert(_paper_to_row(paper), pk="arxiv_id", replace=True, alter=True)

        # Neo4j — single-paper upsert using the same transaction helper as batch
        driver = get_neo4j_driver()
        try:
            with driver.session(
                database=os.getenv("NEO4J_DATABASE", "neo4j")
            ) as session:
                session.execute_write(
                    lambda tx, p=paper.model_dump(): _upsert_paper_counting(tx, p)
                )
        finally:
            driver.close()

        # Qdrant — re-sync both collections from Neo4j
        embed_limitations()
        embed_future_directions()

        return IngestResponse(
            status="ok",
            arxiv_id=arxiv_id,
            limitations_found=len(paper.limitations),
            tier=paper.extraction_tier,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Ingest failed for %s: %s", arxiv_id, exc)
        _log_failure(arxiv_id, str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/explain", response_model=ExplainResponse)
def explain_connection(
    source_gap: str = Query(..., min_length=1),
    target_solution: str = Query(..., min_length=1),
    source: str = Query(default="computer_vision"),
    target: str = Query(default="medical_imaging"),
) -> ExplainResponse:
    """Generate an Ollama explanation for a cross-domain gap↔solution pairing.

    Calls llama3.1:8b — expect a few seconds of latency per request.
    """
    match = CrossDomainMatch(
        source_gap=source_gap,
        target_solution=target_solution,
        similarity_score=0.0,  # not used by the explanation prompt
        source_papers=[],
        target_papers=[],
        source_domain=source,
        target_domain=target,
    )
    try:
        return ExplainResponse(explanation=explain_match(match))
    except Exception as exc:  # noqa: BLE001 — Ollama may be down
        logger.error("Explain failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/paper/{arxiv_id}")
def get_paper_by_id(arxiv_id: str) -> dict:
    """Return the full stored paper record or 404 if not found."""
    paper = get_paper(arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id!r} not found")
    return paper


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
