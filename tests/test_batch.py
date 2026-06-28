"""Tests for pipeline/batch.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import sqlite_utils

import pipeline.batch as batch_mod
from pipeline.batch import get_paper, ingest_from_query, search_papers
from pipeline.extractor import PaperExtract


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_SS_RESPONSE = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Object Detection with Transformers",
            "year": 2023,
            "externalIds": {"ArXiv": "2301.00234"},
        },
        {
            "paperId": "def456",
            "title": "Image Segmentation Survey",
            "year": 2023,
            "externalIds": {"ArXiv": "2303.05499"},
        },
        {
            "paperId": "no-arxiv",
            "title": "Paper with no arXiv ID",
            "year": 2022,
            "externalIds": {},
        },
    ]
}


def _make_paper_extract(arxiv_id: str = "2301.00234") -> PaperExtract:
    return PaperExtract(
        arxiv_id=arxiv_id,
        title="Object Detection with Transformers",
        year=2023,
        domain="computer_vision",
        objectives=["Detect objects efficiently"],
        methods=["DETR", "Transformer"],
        datasets=["COCO"],
        evaluation_metrics=["mAP"],
        limitations=["Slow convergence", "High memory use"],
        future_directions=["Apply to video", "Reduce params"],
        raw_json=json.dumps({"limitations": ["Slow convergence"]}),
        ingested_at="2026-06-29T00:00:00+00:00",
    )


def _make_ss_mock_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    if status != 200:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Redirect all DB and log paths to a temp directory for isolation."""
    db_path = tmp_path / "papers.db"
    log_path = tmp_path / "failed_extractions.log"
    monkeypatch.setattr(batch_mod, "_DB_PATH", db_path)
    monkeypatch.setattr(batch_mod, "_LOG_PATH", log_path)
    return tmp_path


# ---------------------------------------------------------------------------
# search_papers
# ---------------------------------------------------------------------------


def test_search_papers_returns_arxiv_papers():
    """search_papers should return only papers that have an ArXiv ID."""
    with patch("pipeline.batch.requests.get",
               return_value=_make_ss_mock_response(MOCK_SS_RESPONSE)):
        results = search_papers("object detection", limit=10)

    assert len(results) == 2  # paper with no arXiv ID is filtered out
    arxiv_ids = [r["arxiv_id"] for r in results]
    assert "2301.00234" in arxiv_ids
    assert "2303.05499" in arxiv_ids


def test_search_papers_result_shape():
    """Each result dict should have arxiv_id, title, and year keys."""
    with patch("pipeline.batch.requests.get",
               return_value=_make_ss_mock_response(MOCK_SS_RESPONSE)):
        results = search_papers("segmentation")

    for r in results:
        assert "arxiv_id" in r
        assert "title" in r
        assert "year" in r


def test_search_papers_sends_api_key(monkeypatch):
    """search_papers should include x-api-key header when env var is set."""
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key-123")

    with patch("pipeline.batch.requests.get",
               return_value=_make_ss_mock_response({"data": []})) as mock_get:
        search_papers("transformers")

    _, kwargs = mock_get.call_args
    assert kwargs["headers"].get("x-api-key") == "test-key-123"


def test_search_papers_retries_on_429():
    """search_papers should retry on HTTP 429 and succeed on the third attempt."""
    rate_limited = _make_ss_mock_response({}, status=429)
    rate_limited.raise_for_status.side_effect = None  # don't raise during retry

    success = _make_ss_mock_response({"data": []})
    responses = [rate_limited, rate_limited, success]

    with patch("pipeline.batch.requests.get", side_effect=responses), \
         patch("pipeline.batch.time.sleep"):
        results = search_papers("test")

    assert results == []


def test_search_papers_raises_after_max_retries():
    """search_papers should raise after 3 consecutive 429 responses."""
    rate_limited = _make_ss_mock_response({}, status=429)
    rate_limited.raise_for_status.side_effect = Exception("HTTP 429")

    with patch("pipeline.batch.requests.get", return_value=rate_limited), \
         patch("pipeline.batch.time.sleep"):
        with pytest.raises(Exception):
            search_papers("test")


def test_search_papers_skips_papers_without_arxiv_id():
    """Papers with empty or missing externalIds should not appear in results."""
    payload = {
        "data": [
            {"paperId": "x", "title": "No ID", "year": 2020, "externalIds": {}},
            {"paperId": "y", "title": "Null IDs", "year": 2020, "externalIds": None},
        ]
    }
    with patch("pipeline.batch.requests.get",
               return_value=_make_ss_mock_response(payload)):
        results = search_papers("test")

    assert results == []


# ---------------------------------------------------------------------------
# ingest_from_query
# ---------------------------------------------------------------------------


def test_ingest_from_query_returns_summary(monkeypatch, tmp_db):
    """ingest_from_query should return a dict with ingested, skipped, failed counts."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [{"arxiv_id": "2301.00234", "title": "T", "year": 2023}])
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: _make_paper_extract(arxiv_id))

    result = ingest_from_query("object detection")

    assert result == {"ingested": 1, "skipped": 0, "failed": 0}


def test_ingest_from_query_writes_to_sqlite(monkeypatch, tmp_db):
    """ingest_from_query should persist extracted papers to the papers table."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [{"arxiv_id": "2301.00234", "title": "T", "year": 2023}])
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: _make_paper_extract(arxiv_id))

    ingest_from_query("object detection")

    db = sqlite_utils.Database(tmp_db / "papers.db")
    assert "papers" in db.table_names()
    rows = list(db["papers"].rows_where("arxiv_id = ?", ["2301.00234"]))
    assert len(rows) == 1
    assert rows[0]["title"] == "Object Detection with Transformers"


def test_ingest_from_query_skips_existing(monkeypatch, tmp_db):
    """ingest_from_query should skip papers already present in SQLite."""
    # Pre-populate DB with the paper
    db = sqlite_utils.Database(tmp_db / "papers.db")
    db["papers"].insert({"arxiv_id": "2301.00234", "title": "existing"}, pk="arxiv_id")

    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [{"arxiv_id": "2301.00234", "title": "T", "year": 2023}])
    extract_called = []
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: extract_called.append(arxiv_id) or _make_paper_extract(arxiv_id))

    result = ingest_from_query("object detection")

    assert result == {"ingested": 0, "skipped": 1, "failed": 0}
    assert extract_called == [], "extract_paper should not be called for existing papers"


def test_ingest_from_query_handles_extraction_failure(monkeypatch, tmp_db):
    """ingest_from_query should log failures and continue — never crash."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [
                            {"arxiv_id": "good-id", "title": "Good", "year": 2023},
                            {"arxiv_id": "bad-id", "title": "Bad", "year": 2023},
                        ])

    def fake_extract(arxiv_id):
        if arxiv_id == "bad-id":
            raise RuntimeError("Ollama timed out")
        return _make_paper_extract(arxiv_id)

    monkeypatch.setattr(batch_mod, "extract_paper", fake_extract)

    result = ingest_from_query("mixed batch")

    assert result["ingested"] == 1
    assert result["failed"] == 1
    assert result["skipped"] == 0


def test_ingest_from_query_logs_failure_to_file(monkeypatch, tmp_db):
    """Extraction failures should be appended to data/failed_extractions.log."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [{"arxiv_id": "bad-id", "title": "Bad", "year": 2023}])
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: (_ for _ in ()).throw(RuntimeError("boom")))

    ingest_from_query("failing query")

    log_file = tmp_db / "failed_extractions.log"
    assert log_file.exists()
    assert "bad-id" in log_file.read_text()


def test_ingest_from_query_prints_progress(monkeypatch, tmp_db, capsys):
    """ingest_from_query should print '[i/total] arxiv_id' for each paper."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [
                            {"arxiv_id": "2301.00234", "title": "T1", "year": 2023},
                            {"arxiv_id": "2303.05499", "title": "T2", "year": 2023},
                        ])
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: _make_paper_extract(arxiv_id))

    ingest_from_query("progress test")

    captured = capsys.readouterr().out
    assert "[1/2] 2301.00234" in captured
    assert "[2/2] 2303.05499" in captured


def test_ingest_stores_list_fields_as_json_strings(monkeypatch, tmp_db):
    """List fields in SQLite rows should be stored as JSON strings."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [{"arxiv_id": "2301.00234", "title": "T", "year": 2023}])
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: _make_paper_extract(arxiv_id))

    ingest_from_query("object detection")

    db = sqlite_utils.Database(tmp_db / "papers.db")
    row = list(db["papers"].rows_where("arxiv_id = ?", ["2301.00234"]))[0]
    # Should be a JSON string, not a Python list
    assert isinstance(row["limitations"], str)
    parsed = json.loads(row["limitations"])
    assert isinstance(parsed, list)
    assert "Slow convergence" in parsed


# ---------------------------------------------------------------------------
# get_paper
# ---------------------------------------------------------------------------


def test_get_paper_returns_dict(monkeypatch, tmp_db):
    """get_paper should return a dict with list fields deserialized."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [{"arxiv_id": "2301.00234", "title": "T", "year": 2023}])
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: _make_paper_extract(arxiv_id))
    ingest_from_query("seed")

    result = get_paper("2301.00234")

    assert result is not None
    assert result["arxiv_id"] == "2301.00234"
    assert isinstance(result["limitations"], list)
    assert "Slow convergence" in result["limitations"]


def test_get_paper_returns_none_when_not_found(tmp_db):
    """get_paper should return None for an arxiv_id not in the database."""
    result = get_paper("9999.99999")
    assert result is None


def test_get_paper_returns_none_when_table_missing(tmp_db):
    """get_paper should return None gracefully if the papers table doesn't exist yet."""
    result = get_paper("2301.00234")
    assert result is None


def test_get_paper_deserializes_all_list_fields(monkeypatch, tmp_db):
    """get_paper should deserialize all six list fields back to Python lists."""
    monkeypatch.setattr(batch_mod, "search_papers",
                        lambda q, limit: [{"arxiv_id": "2301.00234", "title": "T", "year": 2023}])
    monkeypatch.setattr(batch_mod, "extract_paper",
                        lambda arxiv_id: _make_paper_extract(arxiv_id))
    ingest_from_query("seed")

    result = get_paper("2301.00234")

    for field in ("objectives", "methods", "datasets", "evaluation_metrics",
                  "limitations", "future_directions"):
        assert isinstance(result[field], list), f"{field} should be a list"
