"""Batch ingestion runner: search, extract, and persist arXiv papers to SQLite."""

import json
import logging
import os
import time
from pathlib import Path

import requests
import sqlite_utils
from dotenv import load_dotenv

from pipeline import corpus_rules
from pipeline.domains import validate_domain
from pipeline.extractor import (
    extract_paper,
    log_extraction_failure,
    rate_limit_wait_seconds,
)

load_dotenv()

_SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
# The SQLite path is configurable so a deployment can point it at a mounted
# volume. On Railway the container filesystem is ephemeral, so leaving this at
# the repo-relative default there would silently reset the corpus on every
# redeploy — see G3 in PROJECT_HARDENING_PLAN.md.
#
# Resolved once at import, after load_dotenv() above, so tests that patch this
# constant keep working exactly as before.
_DEFAULT_DB_PATH = Path("data/papers.db")
_DB_PATH = Path(os.getenv("PAPERS_DB_PATH", "").strip() or _DEFAULT_DB_PATH)
_LOG_PATH = Path("data/failed_extractions.log")
# Seconds between individual paper fetches (Semantic Scholar free tier: 1 req/sec).
# Mirrors cross_domain._FETCH_SLEEP_SECONDS.
_FETCH_SLEEP_SECONDS = 2
_LIST_FIELDS = (
    "objectives",
    "methods",
    "datasets",
    "evaluation_metrics",
    "limitations",
    "future_directions",
)

# No logging.basicConfig here — see pipeline/extractor.py.
logger = logging.getLogger(__name__)


def _get_db() -> sqlite_utils.Database:
    """Open (or create) the SQLite database at the configured path.

    Defaults to data/papers.db; PAPERS_DB_PATH overrides it. The parent
    directory is created if missing, so pointing this at a freshly mounted
    volume works without a provisioning step.
    """
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_utils.Database(_DB_PATH)


def _log_failure(arxiv_id: str, reason: str) -> None:
    """Record a batch-level failure, tagged so its origin is visible in the log.

    A thin wrapper over extractor.log_extraction_failure rather than a second
    implementation — the two used to be separate functions with different
    signatures and formats that could drift apart independently.

    _LOG_PATH is passed explicitly: tests redirect the log by patching this
    module's constant, and without it the delegate would resolve the extractor's
    own path and write to the real log during a test run.
    """
    log_extraction_failure(arxiv_id, reason, source="batch", log_path=_LOG_PATH)


def _paper_to_row(paper) -> dict:
    """Serialize a PaperExtract to a flat dict suitable for SQLite storage.

    list fields are stored as JSON strings.
    """
    row = paper.model_dump()
    for field in _LIST_FIELDS:
        row[field] = json.dumps(row[field])
    return row


def search_papers(query: str, limit: int = 100) -> list[dict]:
    """Search Semantic Scholar and return papers that have an arXiv ID.

    Each returned dict has keys: arxiv_id, title, year.
    Uses exponential backoff (max 3 attempts) on HTTP 429.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    url = f"{_SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,year,externalIds",
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    max_attempts = 3
    for attempt in range(max_attempts):
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 429:
            if attempt == max_attempts - 1:
                response.raise_for_status()
            wait = rate_limit_wait_seconds(attempt, response)
            logger.warning("Rate limited by Semantic Scholar, retrying in %ss", wait)
            time.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("data", []):
            external_ids = item.get("externalIds") or {}
            arxiv_id = external_ids.get("ArXiv")
            if not arxiv_id:
                continue
            results.append({
                "arxiv_id": arxiv_id,
                "title": item.get("title", ""),
                "year": item.get("year") or 0,
            })
        return results

    raise RuntimeError("search_papers exhausted retries")


def ingest_from_query(query: str, domain: str, limit: int = 100) -> dict:
    """Search Semantic Scholar, extract each paper, and persist to SQLite.

    ``domain`` is **required** and applies to every paper this call ingests
    (PLAN.md #1, option 1A). This function previously took no domain at all and
    relied on ``extract_paper``'s hardcoded "computer_vision", which is how
    analytic number theory, atomic physics, control theory, astronomy and several
    language-model papers entered the corpus labelled as computer vision: a
    keyword search returns whatever matches the words, and nothing here checked.

    Declaring the domain does not make the search results relevant — it only
    makes the label honest about what the caller intended. ``extract_paper``
    logs a warning when a paper's own text disagrees with the declaration, and
    that log is the signal to curate the query's results.

    Every new paper must pass the corpus admission rule (P2,
    ``corpus_rules.admit``) before extraction: its arXiv primary category is looked
    up and checked against ``domain``. A rejected paper is logged and counted, never
    extracted. If arXiv cannot be asked, this raises ``AdmissionUnverifiable`` before
    anything is extracted — an unverified paper is not admitted.

    Skips papers already present in the database.
    Logs extraction failures to data/failed_extractions.log and continues.
    Prints progress to stdout: "[{i}/{total}] {arxiv_id}".

    Returns {"ingested": n, "skipped": n, "failed": n, "rejected": n}.
    """
    domain = validate_domain(domain)
    papers = search_papers(query, limit=limit)
    db = _get_db()
    table = db["papers"]

    existing: set[str] = set()
    if "papers" in db.table_names():
        existing = {
            row[0]
            for row in db.execute("SELECT arxiv_id FROM papers").fetchall()
        }

    # One batched arXiv lookup for every paper not already stored, before any
    # extraction, so an unreachable arXiv fails the call cleanly.
    verdicts = corpus_rules.admit(
        domain, [p["arxiv_id"] for p in papers if p["arxiv_id"] not in existing]
    )

    total = len(papers)
    ingested = 0
    skipped = 0
    failed = 0
    rejected = 0

    for i, paper_stub in enumerate(papers, start=1):
        arxiv_id = paper_stub["arxiv_id"]
        print(f"[{i}/{total}] {arxiv_id}")

        if arxiv_id in existing:
            skipped += 1
            continue

        verdict = verdicts[arxiv_id]
        if not verdict.accepted:
            logger.warning("Not ingesting %s into %s: %s", arxiv_id, domain, verdict.reason)
            rejected += 1
            continue

        try:
            paper = extract_paper(arxiv_id, domain=domain)
            row = _paper_to_row(paper)
            # alter=True: new PaperExtract fields must be added to pre-existing
            # tables, since SQLite does not auto-migrate. Omitting it is how
            # extraction_tier broke a live database.
            table.insert(row, pk="arxiv_id", replace=False, alter=True)
            ingested += 1
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", arxiv_id, exc)
            _log_failure(arxiv_id, str(exc))
            failed += 1

        # Pace every attempt, including failures — a failed extraction still
        # spent Semantic Scholar quota. Matches ingest_domain_papers, which had
        # this from the start; its absence here caused the 429s in
        # data/failed_extractions.log.
        time.sleep(_FETCH_SLEEP_SECONDS)

    return {"ingested": ingested, "skipped": skipped, "failed": failed, "rejected": rejected}


def get_paper(arxiv_id: str) -> dict | None:
    """Retrieve one paper from SQLite by arxiv_id.

    Deserializes JSON string fields back to lists.
    Returns None if not found.
    """
    db = _get_db()
    if "papers" not in db.table_names():
        return None

    rows = list(db["papers"].rows_where("arxiv_id = ?", [arxiv_id]))
    if not rows:
        return None

    row = dict(rows[0])
    for field in _LIST_FIELDS:
        if isinstance(row.get(field), str):
            try:
                row[field] = json.loads(row[field])
            except (json.JSONDecodeError, TypeError):
                row[field] = []
    return row
