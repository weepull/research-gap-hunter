"""Tests for pipeline/selfheal.py — startup Neo4j -> SQLite reconciliation.

The behaviour under test is the safety net for G3: when SQLite is missing paper
rows that Neo4j has, they are rebuilt from graph relationships alone. These
tests pin both directions — that it repairs a real mismatch correctly, and that
it does nothing at all when the two stores already agree.
"""

import json
from unittest.mock import MagicMock

import pytest
import sqlite_utils

import pipeline.selfheal as selfheal_mod
from pipeline.selfheal import (
    RECONSTRUCTION_SOURCE,
    fetch_papers_from_graph,
    graph_paper_ids,
    reconcile_sqlite_from_graph,
    selfheal_enabled,
    sqlite_paper_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GRAPH_RECORDS = {
    "2301.00234": {
        "arxiv_id": "2301.00234",
        "title": "Object Detection with Transformers",
        "year": 2023,
        "domain": "computer_vision",
        "limitations": ["Slow convergence", "High memory use"],
        "future_directions": ["Extend to video"],
        "methods": ["DETR", "Transformer"],
        "datasets": ["COCO"],
        "tiers": ["conclusion", "conclusion"],
    },
    "2404.01197": {
        "arxiv_id": "2404.01197",
        "title": "Segmentation Under Domain Shift",
        "year": 2024,
        "domain": "medical_imaging",
        "limitations": ["Fails on unseen scanners"],
        "future_directions": [],
        "methods": ["U-Net"],
        "datasets": ["BraTS"],
        "tiers": ["explicit"],
    },
}


def _make_driver(graph_ids, records=None):
    """A mock driver whose session answers the two queries selfheal issues."""
    records = records if records is not None else GRAPH_RECORDS

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    def run(query, **kwargs):
        if "RETURN p.arxiv_id AS arxiv_id" in query and "REPORTS_LIMITATION" not in query:
            return [{"arxiv_id": pid} for pid in graph_ids]
        # The reconstruction query, filtered to the requested ids.
        return [records[pid] for pid in kwargs.get("ids", []) if pid in records]

    session.run.side_effect = run
    driver = MagicMock()
    driver.session.return_value = session
    return driver


def _db(tmp_path, rows=()):
    db = sqlite_utils.Database(tmp_path / "papers.db")
    for row in rows:
        db["papers"].insert(row, pk="arxiv_id", replace=True, alter=True)
    return db


def _existing_row(arxiv_id, title="Already here"):
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "year": 2022,
        "domain": "computer_vision",
        "objectives": json.dumps(["Original objective"]),
        "methods": json.dumps(["Original method"]),
        "datasets": json.dumps([]),
        "evaluation_metrics": json.dumps(["mAP"]),
        "limitations": json.dumps(["Original limitation"]),
        "future_directions": json.dumps([]),
        "raw_json": '{"source": "extracted"}',
        "ingested_at": "2026-01-01T00:00:00+00:00",
        "extraction_tier": "explicit",
    }


# ---------------------------------------------------------------------------
# The no-op case: stores already agree
# ---------------------------------------------------------------------------


def test_does_nothing_when_counts_match(tmp_path):
    """No writes, no rebuilds, when SQLite already holds every graph paper."""
    ids = ["2301.00234", "2404.01197"]
    db = _db(tmp_path, [_existing_row(i) for i in ids])
    driver = _make_driver(ids)

    before = {r["arxiv_id"]: dict(r) for r in db["papers"].rows}
    summary = reconcile_sqlite_from_graph(driver, db=db)

    assert summary["missing_from_sqlite"] == 0
    assert summary["rebuilt"] == 0
    assert summary["failed"] == 0
    assert summary["graph_papers"] == 2
    assert summary["sqlite_papers"] == 2
    # Byte-identical rows: reconciliation must not rewrite what it did not add.
    assert {r["arxiv_id"]: dict(r) for r in db["papers"].rows} == before


def test_does_not_rebuild_when_sqlite_has_extra_rows(tmp_path):
    """Rows SQLite has and the graph does not are reported, never deleted."""
    db = _db(tmp_path, [_existing_row("2301.00234"), _existing_row("9999.11111")])
    driver = _make_driver(["2301.00234"])

    summary = reconcile_sqlite_from_graph(driver, db=db)

    assert summary["rebuilt"] == 0
    assert summary["sqlite_only"] == 1
    # The extra row survives — deleting it would destroy data and change nothing.
    assert db["papers"].count == 2


# ---------------------------------------------------------------------------
# The repair case
# ---------------------------------------------------------------------------


def test_rebuilds_missing_rows_from_graph(tmp_path):
    """A row Neo4j has and SQLite lost is reconstructed from relationships."""
    db = _db(tmp_path, [_existing_row("2301.00234")])
    driver = _make_driver(["2301.00234", "2404.01197"])

    summary = reconcile_sqlite_from_graph(driver, db=db)

    assert summary["missing_from_sqlite"] == 1
    assert summary["rebuilt"] == 1
    assert summary["failed"] == 0
    assert db["papers"].count == 2

    rebuilt = next(db["papers"].rows_where("arxiv_id = ?", ["2404.01197"]))
    assert rebuilt["title"] == "Segmentation Under Domain Shift"
    assert rebuilt["year"] == 2024
    assert rebuilt["domain"] == "medical_imaging"
    assert rebuilt["extraction_tier"] == "explicit"


def test_rebuilt_list_fields_are_json_not_python_repr(tmp_path):
    """List fields must be json.dumps'd, per the SQLite rule in CLAUDE.md.

    str(list) produces Python repr, which json.loads cannot parse — the exact
    corruption this project has hit before.
    """
    db = _db(tmp_path)
    driver = _make_driver(["2301.00234"])

    reconcile_sqlite_from_graph(driver, db=db)

    row = next(db["papers"].rows_where("arxiv_id = ?", ["2301.00234"]))
    for field in ("limitations", "future_directions", "methods", "datasets",
                  "objectives", "evaluation_metrics"):
        assert isinstance(row[field], str), f"{field} should be stored as a string"
        json.loads(row[field])  # must not raise

    assert json.loads(row["limitations"]) == ["Slow convergence", "High memory use"]
    assert json.loads(row["methods"]) == ["DETR", "Transformer"]
    assert json.loads(row["datasets"]) == ["COCO"]


def test_rebuilt_rows_record_their_provenance(tmp_path):
    """A rebuilt row is visibly reconstructed, not mistakable for an extraction."""
    db = _db(tmp_path)
    driver = _make_driver(["2404.01197"])

    reconcile_sqlite_from_graph(driver, db=db)

    row = next(db["papers"].rows_where("arxiv_id = ?", ["2404.01197"]))
    provenance = json.loads(row["raw_json"])
    assert provenance["source"] == RECONSTRUCTION_SOURCE
    assert "reconstructed_at" in provenance
    assert "objectives" in provenance["note"]


def test_fields_absent_from_the_graph_come_back_empty(tmp_path):
    """objectives / evaluation_metrics are not in Neo4j and must not be invented."""
    db = _db(tmp_path)
    driver = _make_driver(["2301.00234"])

    reconcile_sqlite_from_graph(driver, db=db)

    row = next(db["papers"].rows_where("arxiv_id = ?", ["2301.00234"]))
    assert json.loads(row["objectives"]) == []
    assert json.loads(row["evaluation_metrics"]) == []


def test_tier_is_recovered_from_the_relationship(tmp_path):
    """extraction_tier lives on REPORTS_LIMITATION, not on the Paper node."""
    db = _db(tmp_path)
    driver = _make_driver(["2301.00234"])

    reconcile_sqlite_from_graph(driver, db=db)

    row = next(db["papers"].rows_where("arxiv_id = ?", ["2301.00234"]))
    assert row["extraction_tier"] == "conclusion"


def test_tier_defaults_when_paper_has_no_limitations(tmp_path):
    """A paper with no limitations carries no tier; the model default applies."""
    records = {
        "2500.00001": {
            "arxiv_id": "2500.00001", "title": "No limitations stated", "year": 2025,
            "domain": "computer_vision", "limitations": [], "future_directions": [],
            "methods": [], "datasets": [], "tiers": [],
        }
    }
    db = _db(tmp_path)
    driver = _make_driver(["2500.00001"], records=records)

    reconcile_sqlite_from_graph(driver, db=db)

    row = next(db["papers"].rows_where("arxiv_id = ?", ["2500.00001"]))
    assert row["extraction_tier"] == "explicit"


def test_existing_rows_are_never_overwritten(tmp_path):
    """An existing row is richer than anything the graph can rebuild."""
    db = _db(tmp_path, [_existing_row("2301.00234", title="Original title")])
    driver = _make_driver(["2301.00234", "2404.01197"])

    reconcile_sqlite_from_graph(driver, db=db)

    kept = next(db["papers"].rows_where("arxiv_id = ?", ["2301.00234"]))
    assert kept["title"] == "Original title"
    assert json.loads(kept["objectives"]) == ["Original objective"]
    assert json.loads(kept["raw_json"])["source"] == "extracted"


def test_rebuild_continues_after_one_bad_row(tmp_path, monkeypatch):
    """One unwritable row must not abandon the rest of the repair."""
    db = _db(tmp_path)
    driver = _make_driver(["2301.00234", "2404.01197"])

    real_to_row = selfheal_mod._paper_to_row

    def flaky(paper):
        if paper.arxiv_id == "2301.00234":
            raise ValueError("simulated serialisation failure")
        return real_to_row(paper)

    monkeypatch.setattr(selfheal_mod, "_paper_to_row", flaky)

    summary = reconcile_sqlite_from_graph(driver, db=db)

    assert summary["failed"] == 1
    assert summary["rebuilt"] == 1
    assert db["papers"].count == 1


def test_missing_papers_table_is_treated_as_empty(tmp_path):
    """A brand-new volume has no table at all; every paper is missing."""
    db = sqlite_utils.Database(tmp_path / "papers.db")
    assert "papers" not in db.table_names()
    driver = _make_driver(["2301.00234", "2404.01197"])

    summary = reconcile_sqlite_from_graph(driver, db=db)

    assert summary["sqlite_papers"] == 0
    assert summary["rebuilt"] == 2
    assert db["papers"].count == 2


# ---------------------------------------------------------------------------
# Store readers and the kill switch
# ---------------------------------------------------------------------------


def test_graph_paper_ids_skips_null_ids():
    driver = _make_driver(["2301.00234"])
    driver.session.return_value.run.side_effect = lambda q, **kw: [
        {"arxiv_id": "2301.00234"}, {"arxiv_id": None},
    ]
    assert graph_paper_ids(driver) == {"2301.00234"}


def test_sqlite_paper_ids_empty_without_table(tmp_path):
    db = sqlite_utils.Database(tmp_path / "papers.db")
    assert sqlite_paper_ids(db) == set()


def test_fetch_papers_from_graph_returns_nothing_for_empty_input():
    driver = _make_driver([])
    assert fetch_papers_from_graph(driver, []) == []
    driver.session.assert_not_called()


@pytest.mark.parametrize(
    "value,expected",
    [(None, True), ("true", True), ("1", True), ("", True),
     ("false", False), ("0", False), ("no", False), ("OFF", False)],
)
def test_selfheal_kill_switch(monkeypatch, value, expected):
    """Default on; explicit falsey values turn it off."""
    if value is None:
        monkeypatch.delenv("SELFHEAL_ON_STARTUP", raising=False)
    else:
        monkeypatch.setenv("SELFHEAL_ON_STARTUP", value)
    assert selfheal_enabled() is expected
