"""Tests for graph/populate.py."""

from unittest.mock import MagicMock, call, patch

import pytest

import graph.populate as graph_mod
from graph.populate import (
    _CONSTRAINTS,
    _upsert_paper_counting,
    create_constraints,
    get_neo4j_driver,
    populate_graph,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_PAPER = {
    "arxiv_id": "2301.00234",
    "title": "Object Detection with Transformers",
    "year": 2023,
    "domain": "computer_vision",
    "objectives": ["Detect objects"],
    "methods": ["DETR", "Transformer"],
    "datasets": ["COCO", "LVIS"],
    "evaluation_metrics": ["mAP"],
    "limitations": ["Slow convergence", "High memory"],
    "future_directions": ["Extend to video"],
    "raw_json": "{}",
    "ingested_at": "2026-06-29T00:00:00+00:00",
}


def _make_tx() -> MagicMock:
    """Return a mock Neo4j transaction whose run() returns a result with consume()."""
    tx = MagicMock()
    result = MagicMock()
    counters = MagicMock()
    counters.nodes_created = 0
    counters.relationships_created = 0
    result.consume.return_value = MagicMock(counters=counters)
    tx.run.return_value = result
    return tx


def _make_driver(nodes_created: int = 0, rels_created: int = 0) -> MagicMock:
    """Return a mock Neo4j driver whose session().execute_write() returns counts."""
    driver = MagicMock()

    counters = MagicMock()
    counters.nodes_created = nodes_created
    counters.relationships_created = rels_created

    result = MagicMock()
    result.consume.return_value = MagicMock(counters=counters)

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.run.return_value = result
    # execute_write calls the supplied function and returns whatever it returns
    session.execute_write.side_effect = lambda fn, **kw: fn(_make_tx())

    driver.session.return_value = session
    return driver


# ---------------------------------------------------------------------------
# get_neo4j_driver
# ---------------------------------------------------------------------------


def test_get_neo4j_driver_calls_verify_connectivity(monkeypatch):
    """get_neo4j_driver should call verify_connectivity() on the driver."""
    mock_driver = MagicMock()
    mock_gd = MagicMock(return_value=mock_driver)

    with patch("graph.populate.GraphDatabase.driver", mock_gd):
        driver = get_neo4j_driver()

    mock_driver.verify_connectivity.assert_called_once()
    assert driver is mock_driver


def test_get_neo4j_driver_uses_env_vars(monkeypatch):
    """get_neo4j_driver should read URI, user, and password from environment."""
    monkeypatch.setenv("NEO4J_URI", "bolt://testhost:7687")
    monkeypatch.setenv("NEO4J_USER", "testuser")
    monkeypatch.setenv("NEO4J_PASSWORD", "testpass")

    mock_driver = MagicMock()
    with patch("graph.populate.GraphDatabase.driver", return_value=mock_driver) as mock_gd:
        get_neo4j_driver()

    mock_gd.assert_called_once_with(
        "bolt://testhost:7687", auth=("testuser", "testpass")
    )


# ---------------------------------------------------------------------------
# create_constraints
# ---------------------------------------------------------------------------


def test_create_constraints_runs_all_five(monkeypatch):
    """create_constraints should execute all five CONSTRAINT Cypher statements."""
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = session

    create_constraints(driver)

    assert session.run.call_count == len(_CONSTRAINTS)


def test_create_constraints_covers_all_node_labels(monkeypatch):
    """Each uniqueness constraint should reference a distinct node label."""
    labels_covered = set()
    for cypher in _CONSTRAINTS:
        for label in ("Paper", "Limitation", "FutureDirection", "Method", "Dataset"):
            if label in cypher:
                labels_covered.add(label)

    assert labels_covered == {"Paper", "Limitation", "FutureDirection", "Method", "Dataset"}


# ---------------------------------------------------------------------------
# _upsert_paper_counting — the single production write path
#
# These were previously written against the upsert_* helper functions, which
# nothing in production ever called. They are ported here so the code that
# actually writes the graph is the code under test.
# ---------------------------------------------------------------------------


def _calls_containing(tx: MagicMock, needle: str) -> list:
    """Every tx.run call whose Cypher mentions `needle`."""
    return [c for c in tx.run.call_args_list if needle in c[0][0]]


def test_upsert_paper_counting_merges_paper_on_arxiv_id():
    """The Paper node is MERGEd on arxiv_id with title, year and domain set."""
    tx = _make_tx()
    _upsert_paper_counting(tx, SAMPLE_PAPER)

    paper_calls = [c for c in tx.run.call_args_list if "MERGE (p:Paper" in c[0][0]]
    assert len(paper_calls) == 1
    cypher, kwargs = paper_calls[0][0][0], paper_calls[0][1]
    assert "arxiv_id" in cypher
    assert kwargs["arxiv_id"] == "2301.00234"
    assert kwargs["title"] == "Object Detection with Transformers"
    assert kwargs["year"] == 2023
    assert kwargs["domain"] == "computer_vision"


def test_upsert_paper_counting_is_idempotent():
    """Two runs issue the same MERGEs — Neo4j MERGE guarantees no duplicate nodes."""
    tx = _make_tx()
    _upsert_paper_counting(tx, SAMPLE_PAPER)
    first_pass = tx.run.call_count
    _upsert_paper_counting(tx, SAMPLE_PAPER)

    assert tx.run.call_count == first_pass * 2
    for c in _calls_containing(tx, "MERGE (p:Paper"):
        assert c[1]["arxiv_id"] == "2301.00234"


def test_upsert_paper_counting_creates_limitations():
    """Each limitation MERGEs a Limitation node and a REPORTS_LIMITATION relationship."""
    tx = _make_tx()
    _upsert_paper_counting(tx, SAMPLE_PAPER)

    calls = _calls_containing(tx, "REPORTS_LIMITATION")
    assert len(calls) == len(SAMPLE_PAPER["limitations"])
    assert "Limitation" in calls[0][0][0]
    assert {c[1]["text"] for c in calls} == set(SAMPLE_PAPER["limitations"])
    for c in calls:
        assert c[1]["arxiv_id"] == "2301.00234"


def test_upsert_paper_counting_defaults_to_explicit_tier():
    """A paper with no extraction_tier records the explicit tier on the relationship."""
    tx = _make_tx()
    _upsert_paper_counting(tx, SAMPLE_PAPER)  # SAMPLE_PAPER has no extraction_tier

    calls = _calls_containing(tx, "REPORTS_LIMITATION")
    for c in calls:
        assert "r.tier" in c[0][0]
        assert c[1]["tier"] == "explicit"


def test_upsert_paper_counting_sets_limitation_tier():
    """_upsert_paper_counting should tag REPORTS_LIMITATION with the paper's tier."""
    tx = _make_tx()
    paper = {**SAMPLE_PAPER, "extraction_tier": "conclusion"}

    _upsert_paper_counting(tx, paper)

    limitation_calls = _calls_containing(tx, "REPORTS_LIMITATION")
    assert limitation_calls, "expected at least one REPORTS_LIMITATION query"
    for c in limitation_calls:
        assert "r.tier" in c[0][0]
        assert c[1]["tier"] == "conclusion"


def test_upsert_paper_counting_creates_future_directions():
    """Each future direction MERGEs a FutureDirection node and SUGGESTS_FUTURE edge."""
    tx = _make_tx()
    _upsert_paper_counting(tx, SAMPLE_PAPER)

    calls = _calls_containing(tx, "SUGGESTS_FUTURE")
    assert len(calls) == len(SAMPLE_PAPER["future_directions"])
    assert "FutureDirection" in calls[0][0][0]
    assert {c[1]["text"] for c in calls} == set(SAMPLE_PAPER["future_directions"])


def test_upsert_paper_counting_creates_methods():
    """Each method MERGEs a Method node and a USES_METHOD relationship."""
    tx = _make_tx()
    _upsert_paper_counting(tx, SAMPLE_PAPER)

    calls = _calls_containing(tx, "USES_METHOD")
    assert len(calls) == len(SAMPLE_PAPER["methods"])
    assert "Method" in calls[0][0][0]
    assert {c[1]["name"] for c in calls} == set(SAMPLE_PAPER["methods"])


def test_upsert_paper_counting_creates_datasets():
    """Each dataset MERGEs a Dataset node and a USES_DATASET relationship."""
    tx = _make_tx()
    _upsert_paper_counting(tx, SAMPLE_PAPER)

    calls = _calls_containing(tx, "USES_DATASET")
    assert len(calls) == len(SAMPLE_PAPER["datasets"])
    assert "Dataset" in calls[0][0][0]
    assert {c[1]["name"] for c in calls} == set(SAMPLE_PAPER["datasets"])


def test_upsert_paper_counting_skips_empty_strings():
    """Empty list entries are skipped rather than creating blank nodes.

    Extraction occasionally emits "" inside a list; a blank Limitation node would
    be MERGEd corpus-wide and match many things, so it must never be written.
    """
    tx = _make_tx()
    paper = {
        **SAMPLE_PAPER,
        "limitations": ["real limitation", ""],
        "future_directions": ["", ""],
        "methods": ["DETR", ""],
        "datasets": [""],
    }

    _upsert_paper_counting(tx, paper)

    assert len(_calls_containing(tx, "REPORTS_LIMITATION")) == 1
    assert len(_calls_containing(tx, "SUGGESTS_FUTURE")) == 0
    assert len(_calls_containing(tx, "USES_METHOD")) == 1
    assert len(_calls_containing(tx, "USES_DATASET")) == 0


def test_upsert_paper_counting_handles_missing_lists():
    """A paper dict with no list fields at all writes only the Paper node."""
    tx = _make_tx()

    _upsert_paper_counting(tx, {"arxiv_id": "2301.00234"})

    assert len(_calls_containing(tx, "MERGE (p:Paper")) == 1
    assert _calls_containing(tx, "REPORTS_LIMITATION") == []
    assert _calls_containing(tx, "USES_METHOD") == []


# ---------------------------------------------------------------------------
# populate_graph
# ---------------------------------------------------------------------------


def test_populate_graph_returns_zero_when_no_table(monkeypatch, tmp_path):
    """populate_graph should return zeros when papers table does not exist."""
    import pipeline.batch as batch_mod
    monkeypatch.setattr(graph_mod, "_get_db",
                        lambda: _empty_db(tmp_path))

    result = populate_graph()

    assert result == {"nodes_created": 0, "relationships_created": 0}


def _empty_db(tmp_path):
    import sqlite_utils
    return sqlite_utils.Database(tmp_path / "empty.db")


def test_populate_graph_prints_progress(monkeypatch, tmp_path, capsys):
    """populate_graph should print '[i/total] arxiv_id → graph' for each paper."""
    _setup_db_and_driver(monkeypatch, tmp_path, [SAMPLE_PAPER])

    populate_graph()

    out = capsys.readouterr().out
    assert "[1/1] 2301.00234 → graph" in out


def test_populate_graph_returns_summary_dict(monkeypatch, tmp_path):
    """populate_graph should return a dict with nodes_created and relationships_created."""
    _setup_db_and_driver(monkeypatch, tmp_path, [SAMPLE_PAPER])

    result = populate_graph()

    assert "nodes_created" in result
    assert "relationships_created" in result
    assert isinstance(result["nodes_created"], int)
    assert isinstance(result["relationships_created"], int)


def test_populate_graph_handles_empty_limitations(monkeypatch, tmp_path):
    """populate_graph should not crash when limitations or future_directions are empty."""
    paper_no_limits = {**SAMPLE_PAPER, "limitations": [], "future_directions": []}
    _setup_db_and_driver(monkeypatch, tmp_path, [paper_no_limits])

    result = populate_graph()  # must not raise

    assert "nodes_created" in result


def test_populate_graph_handles_multiple_papers(monkeypatch, tmp_path, capsys):
    """populate_graph should process every paper returned from SQLite."""
    paper2 = {**SAMPLE_PAPER, "arxiv_id": "2303.05499", "title": "Segmentation Paper"}
    _setup_db_and_driver(monkeypatch, tmp_path, [SAMPLE_PAPER, paper2])

    populate_graph()

    out = capsys.readouterr().out
    assert "[1/2]" in out
    assert "[2/2]" in out
    assert "2301.00234" in out
    assert "2303.05499" in out


def test_populate_graph_creates_constraints(monkeypatch, tmp_path):
    """populate_graph should call create_constraints before upserting any paper."""
    mock_driver = _setup_db_and_driver(monkeypatch, tmp_path, [SAMPLE_PAPER])
    create_called = []

    original_cc = graph_mod.create_constraints

    def tracking_cc(driver):
        create_called.append(True)

    monkeypatch.setattr(graph_mod, "create_constraints", tracking_cc)

    populate_graph()

    assert create_called, "create_constraints should have been called"


# ---------------------------------------------------------------------------
# Helpers for populate_graph tests
# ---------------------------------------------------------------------------


def _setup_db_and_driver(monkeypatch, tmp_path, papers: list[dict]) -> MagicMock:
    """Seed a temp SQLite DB with papers and patch driver + get_paper in graph module."""
    import json
    import sqlite_utils
    import pipeline.batch as batch_mod

    db_path = tmp_path / "papers.db"
    db = sqlite_utils.Database(db_path)

    list_fields = ("objectives", "methods", "datasets", "evaluation_metrics",
                   "limitations", "future_directions")
    rows = []
    for p in papers:
        row = dict(p)
        for f in list_fields:
            row[f] = json.dumps(row.get(f, []))
        rows.append(row)

    db["papers"].insert_all(rows, pk="arxiv_id")

    monkeypatch.setattr(graph_mod, "_get_db", lambda: db)

    # get_paper reads from same db — patch it to deserialize from our temp db
    def fake_get_paper(arxiv_id):
        hits = list(db["papers"].rows_where("arxiv_id = ?", [arxiv_id]))
        if not hits:
            return None
        row = dict(hits[0])
        for f in list_fields:
            if isinstance(row.get(f), str):
                row[f] = json.loads(row[f])
        return row

    monkeypatch.setattr(graph_mod, "get_paper", fake_get_paper)

    mock_driver = _make_driver(nodes_created=1, rels_created=1)
    monkeypatch.setattr(graph_mod, "get_neo4j_driver", lambda: mock_driver)

    return mock_driver
