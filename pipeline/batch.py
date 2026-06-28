"""Batch ingestion runner: search, extract, and persist arXiv papers to SQLite."""

import json
import logging
import os
import time
from pathlib import Path

import requests
import sqlite_utils
from dotenv import load_dotenv

from pipeline.extractor import extract_paper

load_dotenv()

_SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
_DB_PATH = Path("data/papers.db")
_LOG_PATH = Path("data/failed_extractions.log")
_LIST_FIELDS = (
    "objectives",
    "methods",
    "datasets",
    "evaluation_metrics",
    "limitations",
    "future_directions",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_db() -> sqlite_utils.Database:
    """Open (or create) the SQLite database at data/papers.db."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_utils.Database(_DB_PATH)


def _log_failure(arxiv_id: str, reason: str) -> None:
    """Append a batch-level failure record to data/failed_extractions.log."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] batch arxiv_id={arxiv_id} reason={reason}\n")


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
            wait = 2 ** attempt
            logger.warning("Rate limited by Semantic Scholar, retrying in %ds", wait)
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


def ingest_from_query(query: str, limit: int = 100) -> dict:
    """Search Semantic Scholar, extract each paper, and persist to SQLite.

    Skips papers already present in the database.
    Logs extraction failures to data/failed_extractions.log and continues.
    Prints progress to stdout: "[{i}/{total}] {arxiv_id}".

    Returns {"ingested": n, "skipped": n, "failed": n}.
    """
    papers = search_papers(query, limit=limit)
    db = _get_db()
    table = db["papers"]

    existing: set[str] = set()
    if "papers" in db.table_names():
        existing = {
            row[0]
            for row in db.execute("SELECT arxiv_id FROM papers").fetchall()
        }

    total = len(papers)
    ingested = 0
    skipped = 0
    failed = 0

    for i, paper_stub in enumerate(papers, start=1):
        arxiv_id = paper_stub["arxiv_id"]
        print(f"[{i}/{total}] {arxiv_id}")

        if arxiv_id in existing:
            skipped += 1
            continue

        try:
            paper = extract_paper(arxiv_id)
            row = _paper_to_row(paper)
            table.insert(row, pk="arxiv_id", replace=False)
            ingested += 1
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", arxiv_id, exc)
            _log_failure(arxiv_id, str(exc))
            failed += 1

    return {"ingested": ingested, "skipped": skipped, "failed": failed}


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
            row[field] = json.loads(row[field])
    return row
