"""FastAPI application — REST layer for Research Gap Hunter (port 8000).

All heavy backend objects (Specter2 model, Qdrant client, Neo4j driver) are
initialised once in the lifespan context manager and stored in app.state.
Endpoint handlers call module-level backend functions so tests can patch them
at the api.main namespace without touching the real services.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

# This module is the application entrypoint, and per the convention recorded in
# pipeline/extractor.py it is the one place allowed to configure the root logger.
# Nothing did, which meant every logger under pipeline/, graph/ and vectors/ was
# emitting into a root logger with no handler — uvicorn only configures its own
# "uvicorn.*" loggers. In practice that made the startup self-heal silent: the
# WARNING that says a persistent volume did not work never reached the deploy
# logs, and that warning is the entire point of having the safety net.
#
# force=False so an operator who has already configured logging (a container
# runtime, a test harness) keeps their setup.
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").strip().upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level backend imports — patched by tests via api.main.<name>
# ---------------------------------------------------------------------------
from graph.populate import _upsert_paper_counting, get_neo4j_driver  # noqa: E402
from pipeline.batch import _get_db, _log_failure, _paper_to_row, get_paper  # noqa: E402
from pipeline.config import allowed_origins, is_demo_mode  # noqa: E402
from pipeline import corpus_rules  # noqa: E402
from pipeline.domains import RESEARCH_DOMAINS, validate_domain  # noqa: E402
from pipeline.cross_domain import (  # noqa: E402
    CrossDomainMatch,
    CrossDomainReport,
    UngroundedPairingError,
    cross_domain_report,
    explain_match,
    verify_pairing,
)
from pipeline.extractor import extract_paper, is_valid_arxiv_id  # noqa: E402
from pipeline.selfheal import (  # noqa: E402
    reconcile_sqlite_from_graph,
    selfheal_enabled,
)
from pipeline.gap_scorer import (  # noqa: E402
    GapResult,
    _count_contributing_papers,
    _count_papers_in_domain,
    score_gaps,
)
from vectors.embed import (  # noqa: E402
    embed_future_directions,
    embed_limitations,
    get_qdrant_client,
    load_embedding_model,
)
from vectors.search import find_similar_limitations  # noqa: E402

from api.rate_limit import rate_limit_middleware  # noqa: E402

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


# Per-dependency probe outcomes. The distinction that matters is ABSENT vs
# UNREACHABLE: a collection that does not exist yet is a normal cold-start state,
# while a store that cannot be reached is an outage. Collapsing both to "0" is
# what let a fully broken deployment report itself healthy.
_OK = "ok"
_ABSENT = "absent"        # store reachable, the thing asked for is not there yet
_UNREACHABLE = "unreachable"  # store cannot be reached at all


class HealthResponse(BaseModel):
    """Readiness, per dependency.

    `status` is DERIVED from `services`, never asserted. It was previously the
    literal string "ok" computed from nothing, while Qdrant errors were swallowed
    into zero counts and Neo4j was not contacted at all — so an instance with both
    stores down returned 200 {"status":"ok"}. A probe that cannot fail is not a
    probe.

    Counts are `None` when the store backing them is unreachable, so a reader can
    tell "nothing there" from "could not look".
    """

    status: str
    papers: int | None
    limitations: int | None
    future_directions: int | None
    services: dict[str, str]


class LimitationResult(BaseModel):
    limitation_text: str
    score: float
    paper_ids: list[str]
    domain: str


class IngestRequest(BaseModel):
    arxiv_id: str
    # Required, with no default (PLAN.md #1, option 1A). Defaulting to
    # "computer_vision" here carried exactly the hazard the extractor default
    # did: a caller who omits the field gets a plausible-looking but unverified
    # label on a paper that drives scoring. A 422 is the better outcome.
    domain: str

    @field_validator("arxiv_id")
    @classmethod
    def _check_arxiv_id(cls, value: str) -> str:
        # arxiv_id is interpolated into outbound arxiv.org / Semantic Scholar
        # URLs, so reject anything that is not a real id before it gets there.
        candidate = value.strip()
        if not is_valid_arxiv_id(candidate):
            raise ValueError(f"{value!r} is not a valid arXiv id")
        return candidate

    @field_validator("domain")
    @classmethod
    def _check_domain(cls, value: str) -> str:
        return validate_domain(value)


class IngestResponse(BaseModel):
    status: str
    arxiv_id: str
    limitations_found: int
    tier: str


class CorpusInfo(BaseModel):
    """What the numbers on a results page were actually computed over.

    Two paper counts, because they answer different questions and conflating them
    is what made the previous banner describe a number that was not the divisor:

    - `papers` — every Paper node in the domain. This is the corpus *size*, which
      is what a reader asking "how big is this?" wants.
    - `papers_reporting_limitations` — papers with at least one extracted
      limitation. **This is what divides `frequency_score`** (PLAN.md #4). A paper
      that extracted nothing cannot corroborate any gap, so including it in the
      denominator made frequency the product of two unrelated things: how widely a
      limitation is reported, and how often extraction succeeded.

    The gap between them is large and worth showing: on the curated corpus only 18
    of 31 CV papers and 51 of 85 MI papers reported any limitation at all.

    `limitations` / `future_directions` are `None`, and `vectors_available` is
    False, when Qdrant cannot be reached. Previously any exception became `0`,
    so an unreachable vector store rendered as a confident "0 limitations"
    underneath a working results list — indistinguishable from a genuinely empty
    corpus. A missing collection still reports 0 with `vectors_available` True,
    because that is a real cold-start state rather than an outage.
    """

    domain: str
    papers: int | None
    papers_reporting_limitations: int | None
    limitations: int | None
    future_directions: int | None
    last_updated: str | None
    graph_available: bool = True
    vectors_available: bool = True


class ErrorResponse(BaseModel):
    status: str
    message: str


class ExplainResponse(BaseModel):
    """An explanation, shipped with the evidence that justified generating it.

    PLAN.md #5 (option 5C). The endpoint used to return the prose alone, which gave
    a reader no way to tell a 0.90 pairing from one the system had never measured —
    and it had never measured any of them, because `similarity_score` was hardcoded
    to 0.0 and the prompt never saw it.

    `is_hypothesis` is always True and is meant to be rendered, not inspected: an
    LLM's account of why two papers might connect is a suggestion to evaluate, not
    a finding about the literature, however fluent it reads.
    """

    explanation: str
    # Recomputed from the stored vectors by verify_pairing — never a placeholder.
    similarity_score: float
    # The noise floor this pairing had to clear to be explained at all.
    threshold: float
    # Why this was considered groundable. Currently only "corpus_match": both texts
    # are in the corpus and the pair clears its domain's threshold.
    grounding: str
    # Always True. An explanation is a hypothesis, never a finding.
    is_hypothesis: bool = True
    source_papers: list[str] = []
    target_papers: list[str] = []


# ---------------------------------------------------------------------------
# Lifespan — warm model cache and verify service connectivity once at startup
# ---------------------------------------------------------------------------


def _warm(name: str, factory):
    """Initialise a backend without letting its absence prevent startup.

    Every backend used to be constructed unguarded here, and `get_neo4j_driver()`
    calls `verify_connectivity()`, so an unreachable store made the application
    fail to start. On a container platform that is a crash loop, which means
    /health is never served and the degraded state it now reports (PLAN.md #6)
    could never actually be observed in the situation it exists for. A process
    that starts and truthfully says "neo4j: unreachable" is strictly more
    diagnosable than one that dies before it can answer.

    The failure is logged at ERROR, because it is always worth investigating.
    """
    try:
        return factory()
    except Exception:  # noqa: BLE001 — report and degrade, never block startup
        logger.error(
            "Backend %r is unavailable at startup. The API will serve traffic and "
            "/health will report it as unreachable; endpoints needing it will fail.",
            name,
            exc_info=True,
        )
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Research Gap Hunter API…")
    app.state.model = _warm("embedding_model", load_embedding_model)
    app.state.qdrant = _warm("qdrant", get_qdrant_client)
    app.state.neo4j = _warm("neo4j", get_neo4j_driver)
    if all((app.state.model, app.state.qdrant, app.state.neo4j)):
        logger.info("All backend services connected.")
    else:
        logger.error("Started with one or more backends unavailable — see /health.")

    # Second layer of the G3 fix. The first layer is PAPERS_DB_PATH pointing at
    # a persistent volume; this catches the case where that did not work, by
    # rebuilding any paper rows SQLite is missing straight from the graph.
    #
    # Deliberately non-fatal: a failure here means /paper/{id} may 404 for some
    # papers, which is worse than it was but far better than an API that will
    # not start at all.
    app.state.selfheal = None
    if selfheal_enabled() and app.state.neo4j is None:
        logger.warning(
            "Skipping startup paper-store reconciliation: Neo4j is unavailable. "
            "/paper/{arxiv_id} may 404 for papers that only exist in the graph."
        )
    elif selfheal_enabled():
        try:
            app.state.selfheal = reconcile_sqlite_from_graph(app.state.neo4j)
        except Exception:  # noqa: BLE001 — never block startup on the safety net
            logger.exception(
                "Startup paper-store reconciliation failed. The API will serve "
                "traffic, but /paper/{arxiv_id} may 404 for papers that only "
                "exist in Neo4j."
            )
    else:
        logger.info("Startup paper-store reconciliation disabled by SELFHEAL_ON_STARTUP.")

    yield
    logger.info("Shutting down…")
    if app.state.neo4j is not None:
        app.state.neo4j.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Gap Hunter",
    version="0.1.0",
    lifespan=lifespan,
)

# Order matters: Starlette runs the *last* registered middleware outermost, so
# the rate limiter is registered first and CORS second. That puts CORS outside
# the limiter, which means a short-circuited 429 still carries CORS headers and
# the frontend can read the status instead of seeing an opaque network error.
app.middleware("http")(rate_limit_middleware)

# Open locally so a dev frontend on any port works; restricted to the configured
# frontend origins in demo mode, where the browser is untrusted. allowed_origins()
# returns [] if demo mode is on and ALLOWED_ORIGINS is unset — a misconfigured
# deploy should fail visibly in the browser rather than quietly serving everyone.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _known_domain(value: str, param: str = "domain") -> str:
    """The domain a read endpoint was asked for, or a 422 (P3).

    Read endpoints used to pass any string straight through. An unknown domain then
    matched nothing and came back as 200 with an empty result, which this project
    treats as meaningful output ("no gaps", "no connections") — so a typo produced a
    false finding. Validation reuses `validate_domain`; the message is written for a
    reader rather than for ingestion.
    """
    try:
        return validate_domain(value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"unknown {param} {value!r}; expected one of {list(RESEARCH_DOMAINS)}",
        )


def _internal_error(
    operation: str,
    exc: Exception,
    arxiv_id: str | None = None,
    log_failure: bool = False,
) -> HTTPException:
    """Log an exception in full server-side; return a 500 that reveals nothing.

    str(exc) on a driver or HTTP error routinely carries connection URIs,
    credentials and filesystem paths, and the frontend renders the response body
    verbatim. The caller gets an opaque error id that ties their report to the
    server log line holding the real detail.
    """
    error_id = uuid.uuid4().hex[:12]
    logger.error(
        "%s failed [error_id=%s]%s: %s",
        operation,
        error_id,
        f" arxiv_id={arxiv_id}" if arxiv_id else "",
        exc,
        exc_info=True,
    )
    if log_failure and arxiv_id:
        _log_failure(arxiv_id, str(exc))
    return HTTPException(
        status_code=500,
        detail=f"{operation} failed. Quote error id {error_id} when reporting this.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _probe_sqlite() -> tuple[str, int | None]:
    """(state, paper count). A missing table is absent, not an outage."""
    try:
        db = _get_db()
        if "papers" not in db.table_names():
            return _ABSENT, 0
        return _OK, db.execute("SELECT count(*) FROM papers").fetchone()[0]
    except Exception:  # noqa: BLE001 — classified, not swallowed; see _UNREACHABLE
        logger.warning("SQLite paper store unreachable during health probe", exc_info=True)
        return _UNREACHABLE, None


def _probe_neo4j() -> str:
    """Neo4j is the scoring source of truth, so readiness must actually ask it.

    `verify_connectivity()` alone would pass against a reachable server with the
    configured database missing, so the probe runs the same count query the
    scorer depends on.
    """
    driver = None
    try:
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            session.run("MATCH (p:Paper) RETURN count(p) AS n").single()
        return _OK
    except Exception:  # noqa: BLE001
        logger.warning("Neo4j unreachable during health probe", exc_info=True)
        return _UNREACHABLE
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:  # noqa: BLE001 — closing a broken driver must not mask the probe
                pass


def _probe_qdrant_collection(client, collection: str) -> tuple[str, int | None]:
    """(state, point count). UnexpectedResponse means the collection is absent."""
    try:
        return _OK, client.get_collection(collection).points_count or 0
    except UnexpectedResponse:
        # 404 from a reachable Qdrant: the collection has not been created yet.
        return _ABSENT, 0
    except Exception:  # noqa: BLE001
        logger.warning("Qdrant unreachable while probing %r", collection, exc_info=True)
        return _UNREACHABLE, None


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Readiness check, per dependency. Returns 503 when any store is unreachable.

    **Behaviour change (PLAN.md #6):** this used to return 200 with
    `status="ok"` unconditionally. It now returns **503** when a required store
    cannot be reached, so a platform health check can actually restart a broken
    container. See DEPLOYMENT.md.
    """
    sqlite_state, papers = _probe_sqlite()
    neo4j_state = _probe_neo4j()

    try:
        client = get_qdrant_client()
    except Exception:  # noqa: BLE001 — cannot even construct a client
        logger.warning("Could not create a Qdrant client during health probe", exc_info=True)
        qdrant_state, lim_count, fd_count = _UNREACHABLE, None, None
    else:
        lim_state, lim_count = _probe_qdrant_collection(client, "limitations")
        fd_state, fd_count = _probe_qdrant_collection(client, "future_directions")
        # The store is unreachable only if it could not be reached at all; one
        # absent collection does not condemn the other.
        if _UNREACHABLE in (lim_state, fd_state):
            qdrant_state = _UNREACHABLE
        elif _ABSENT in (lim_state, fd_state):
            qdrant_state = _ABSENT
        else:
            qdrant_state = _OK

    services = {"sqlite": sqlite_state, "neo4j": neo4j_state, "qdrant": qdrant_state}
    degraded = [name for name, state in services.items() if state == _UNREACHABLE]

    if degraded:
        logger.error("Health check degraded — unreachable: %s", ", ".join(sorted(degraded)))
        response.status_code = 503

    return HealthResponse(
        status="degraded" if degraded else "ok",
        papers=papers,
        limitations=lim_count,
        future_directions=fd_count,
        services=services,
    )


@app.get("/corpus", response_model=CorpusInfo)
def corpus_info(domain: str = Query(default="computer_vision")) -> CorpusInfo:
    """Corpus size and freshness for a domain, so results can state their basis.

    Every results page shows this. A ranked gap list means something very
    different over 31 papers than over 3,100, and a reader cannot judge the output
    without knowing which it is — nor without knowing that only 18 of those 31
    contributed any limitation at all, which is what `papers_reporting_limitations`
    is for.
    """
    domain = _known_domain(domain)
    graph_available = True
    try:
        papers = _count_papers_in_domain(domain)
        contributing = _count_contributing_papers(domain)
    except Exception:  # noqa: BLE001 — report the outage rather than a false zero
        logger.warning("Neo4j unreachable while reading /corpus", exc_info=True)
        papers, contributing, graph_available = None, None, False

    domain_filter = Filter(
        must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
    )

    vectors_available = True

    def _count(client, collection: str) -> int | None:
        """Domain-filtered point count, or None when the store cannot be reached.

        A missing collection legitimately counts as 0 — that is a cold start. A
        connection failure does not: returning 0 there put a confident
        "0 limitations" under a working results list.
        """
        nonlocal vectors_available
        try:
            return client.count(
                collection_name=collection, count_filter=domain_filter, exact=True
            ).count
        except UnexpectedResponse:
            return 0
        except Exception:  # noqa: BLE001
            logger.warning("Qdrant unreachable while counting %r", collection, exc_info=True)
            vectors_available = False
            return None

    try:
        client = get_qdrant_client()
    except Exception:  # noqa: BLE001
        logger.warning("Could not create a Qdrant client for /corpus", exc_info=True)
        vectors_available = False
        limitations = future_directions = None
    else:
        limitations = _count(client, "limitations")
        future_directions = _count(client, "future_directions")

    last_updated = None
    try:
        db = _get_db()
        if "papers" in db.table_names():
            row = db.execute(
                "SELECT max(ingested_at) FROM papers WHERE domain = ?", [domain]
            ).fetchone()
            if row:
                last_updated = row[0]
    except Exception:  # noqa: BLE001 — freshness is the least important field here
        logger.warning("SQLite unreachable while reading /corpus freshness", exc_info=True)

    return CorpusInfo(
        domain=domain,
        papers=papers,
        papers_reporting_limitations=contributing,
        limitations=limitations,
        future_directions=future_directions,
        last_updated=last_updated,
        graph_available=graph_available,
        vectors_available=vectors_available,
    )


@app.get("/gaps", response_model=list[GapResult])
def get_gaps(
    domain: str = Query(default="computer_vision"),
    top_n: int = Query(default=20, ge=1, le=100),
) -> list[GapResult]:
    """Return top-ranked research gaps for a domain, scored by frequency/recency/deficit."""
    return score_gaps(domain=_known_domain(domain), top_n=top_n)


@app.get("/search", response_model=list[LimitationResult])
def search_limitations(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(default=10, ge=1, le=50),
    domain: str = Query(default="computer_vision"),
) -> list[LimitationResult]:
    """Vector-search the limitations collection and return semantically similar statements."""
    domain = _known_domain(domain)
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


@app.get("/cross-domain", response_model=CrossDomainReport)
def get_cross_domain_matches(
    source: str = Query(default="computer_vision"),
    target: str = Query(default="medical_imaging"),
    top_n: int = Query(default=10, ge=1, le=50),
) -> CrossDomainReport:
    """Cross-domain hypotheses above the similarity threshold, with a stated status.

    **Behaviour change (T2, 2026-10-05):** this used to return a bare list, and an empty
    list read as "no such connections exist". It now returns a `CrossDomainReport` whose
    `status` says which kind of result it is (`matches_found`, `none_above_threshold`,
    `no_corroborated_gaps`, `no_data`), with a plain `message` and the counts behind it.
    """
    return cross_domain_report(
        source_domain=_known_domain(source, "source"),
        target_domain=_known_domain(target, "target"),
        top_n=top_n,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_paper(body: IngestRequest) -> IngestResponse:
    """Extract a paper from arXiv, store it in SQLite + Neo4j, and re-sync Qdrant.

    Extraction runs Semantic Scholar metadata fetch + PDF download + Ollama LLM.
    Expect ~30–60 s per paper.

    `domain` is required and validated — there is no default. A paper whose own
    text disagrees with the declared domain is still ingested, but the extractor
    logs a warning; see pipeline/domains.py for why the check does not override.
    """
    if is_demo_mode():
        # 403, not 501: the endpoint is implemented and works — this deployment
        # is not permitted to use it. Ingestion burns the Semantic Scholar quota,
        # runs a 30-60s LLM job and rewrites both Qdrant collections, so a public
        # instance must not expose it. See item E1 in PROJECT_HARDENING_PLAN.md.
        raise HTTPException(
            status_code=403,
            detail=(
                "Ingestion is disabled in the public demo. "
                "Run the project locally with DEMO_MODE=false to ingest papers."
            ),
        )

    arxiv_id = body.arxiv_id.strip()
    domain = body.domain.strip()

    # The corpus admission rule (P2), before any extraction or write. 422 for a paper
    # the rule rejects — the request is well-formed but the paper does not belong in
    # the declared domain — and 503 when arXiv cannot be asked, since an unverified
    # paper is not admitted.
    try:
        verdict = corpus_rules.admit(domain, [arxiv_id])[arxiv_id]
    except corpus_rules.AdmissionUnverifiable as exc:
        logger.error("Ingest of %s not attempted: %s", arxiv_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Cannot check this paper's arXiv primary category right now; try again later.",
        )
    if not verdict.accepted:
        raise HTTPException(status_code=422, detail=f"Not admitted: {verdict.reason}")

    try:
        # domain goes into extraction rather than being patched on afterwards:
        # extract_paper requires and validates it (PLAN.md #1, option 1A).
        paper = extract_paper(arxiv_id, domain=domain)

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
        raise _internal_error("Ingest", exc, arxiv_id=arxiv_id, log_failure=True)


@app.get("/explain", response_model=ExplainResponse)
def explain_connection(
    source_gap: str = Query(..., min_length=1),
    target_solution: str = Query(..., min_length=1),
    source: str = Query(default="computer_vision"),
    target: str = Query(default="medical_imaging"),
) -> ExplainResponse:
    """Generate an Ollama explanation for a cross-domain gap↔solution pairing.

    Calls llama3.1:8b — expect a few seconds of latency per request.

    **Refuses with 422 unless the pairing is grounded** (PLAN.md #5): both texts
    must exist in the corpus and the pair must clear the same measured noise floor
    `find_cross_domain_matches` uses. The check is deterministic and runs before
    any generation — see `verify_pairing` for why prompt wording was not accepted
    as the fix. The response carries the real similarity and an explicit
    `is_hypothesis` marker.
    """
    if is_demo_mode():
        # Same pattern as /ingest: the endpoint works, this deployment is not
        # permitted to run it. Explanations call an LLM per request, so exposing
        # them to anonymous callers is an open-ended cost. There is deliberately
        # no hosted-LLM fallback — see explain_match in pipeline/cross_domain.py.
        raise HTTPException(
            status_code=403,
            detail=(
                "Live explanations are disabled in the public demo to avoid API "
                "costs. Run this project locally with DEMO_MODE=false to generate "
                "live explanations — see the README for setup."
            ),
        )

    source = _known_domain(source, "source")
    target = _known_domain(target, "target")

    # Deterministic, server-side grounding BEFORE any generation (PLAN.md #5,
    # option 5A). Without this the endpoint would explain any two strings a client
    # sent: it previously hardcoded similarity_score=0.0, never showed the model a
    # score, and instructed it to be specific about the shared structure — so the
    # model could not decline. Asked to connect analytic number theory to
    # histopathology federated learning, it obliged.
    try:
        verified = verify_pairing(
            source_gap=source_gap,
            target_solution=target_solution,
            source_domain=source,
            target_domain=target,
        )
    except UngroundedPairingError as exc:
        # 422, not 500: the request is well-formed but asks for something the
        # corpus does not support, and the caller can act on the reason.
        raise HTTPException(status_code=422, detail=str(exc))

    match = CrossDomainMatch(
        source_gap=verified.source_gap,
        target_solution=verified.target_solution,
        similarity_score=verified.similarity_score,
        source_papers=verified.source_papers,
        target_papers=verified.target_papers,
        source_domain=verified.source_domain,
        target_domain=verified.target_domain,
    )
    try:
        explanation = explain_match(match)
    except Exception as exc:  # noqa: BLE001 — Ollama may be down
        raise _internal_error("Explain", exc)

    return ExplainResponse(
        explanation=explanation,
        similarity_score=verified.similarity_score,
        threshold=verified.threshold,
        grounding="corpus_match",
        is_hypothesis=True,
        source_papers=verified.source_papers,
        target_papers=verified.target_papers,
    )


@app.get("/paper/{arxiv_id}")
def get_paper_by_id(arxiv_id: str) -> dict:
    """Return the full stored paper record, 422 if malformed, 404 if not found."""
    if not is_valid_arxiv_id(arxiv_id):
        raise HTTPException(
            status_code=422, detail=f"{arxiv_id!r} is not a valid arXiv id"
        )
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
