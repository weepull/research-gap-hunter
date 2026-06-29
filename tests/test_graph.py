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
    upsert_dataset,
    upsert_future_direction,
    upsert_limitation,
    upsert_method,
    upsert_paper,
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
# upsert_paper
# ---------------------------------------------------------------------------


def test_upsert_paper_merges_on_arxiv_id():
    """upsert_paper should issue a MERGE on arxiv_id with all properties."""
    tx = _make_tx()
    upsert_paper(tx, SAMPLE_PAPER)

    tx.run.assert_called_once()
    cypher, kwargs = tx.run.call_args[0][0], tx.run.call_args[1]
    assert "MERGE" in cypher
    assert "arxiv_id" in cypher
    assert kwargs["arxiv_id"] == "2301.00234"
    assert kwargs["title"] == "Object Detection with Transformers"
    assert kwargs["year"] == 2023
    assert kwargs["domain"] == "computer_vision"


def test_upsert_paper_is_idempotent():
    """Calling upsert_paper twice should issue two identical MERGE calls — no duplicate data."""
    tx = _make_tx()
    upsert_paper(tx, SAMPLE_PAPER)
    upsert_paper(tx, SAMPLE_PAPER)
    assert tx.run.call_count == 2
    # Both calls should use the same arxiv_id — Neo4j MERGE guarantees idempotency
    for c in tx.run.call_args_list:
        assert c[1]["arxiv_id"] == "2301.00234"


# ---------------------------------------------------------------------------
# upsert_limitation
# ---------------------------------------------------------------------------


def test_upsert_limitation_creates_node_and_relationship():
    """upsert_limitation should MERGE Limitation and MERGE REPORTS_LIMITATION."""
    tx = _make_tx()
    upsert_limitation(tx, "2301.00234", "Slow convergence")

    tx.run.assert_called_once()
    cypher = tx.run.call_args[0][0]
    assert "Limitation" in cypher
    assert "REPORTS_LIMITATION" in cypher
    assert tx.run.call_args[1]["text"] == "Slow convergence"
    assert tx.run.call_args[1]["arxiv_id"] == "2301.00234"
    # Defaults to the explicit tier and writes it onto the relationship.
    assert "r.tier" in cypher
    assert tx.run.call_args[1]["tier"] == "explicit"


def test_upsert_limitation_records_given_tier():
    """upsert_limitation should store the supplied extraction tier on the relationship."""
    tx = _make_tx()
    upsert_limitation(tx, "2301.00234", "Inferred limitation", tier="inferred")

    assert tx.run.call_args[1]["tier"] == "inferred"


def test_upsert_paper_counting_sets_limitation_tier():
    """_upsert_paper_counting should tag REPORTS_LIMITATION with the paper's tier."""
    tx = _make_tx()
    paper = {**SAMPLE_PAPER, "extraction_tier": "conclusion"}

    _upsert_paper_counting(tx, paper)

    limitation_calls = [
        c for c in tx.run.call_args_list if "REPORTS_LIMITATION" in c[0][0]
    ]
    assert limitation_calls, "expected at least one REPORTS_LIMITATION query"
    for c in limitation_calls:
        assert "r.tier" in c[0][0]
        assert c[1]["tier"] == "conclusion"


# ---------------------------------------------------------------------------
# upsert_future_direction
# ---------------------------------------------------------------------------


def test_upsert_future_direction_creates_node_and_relationship():
    """upsert_future_direction should MERGE FutureDirection and MERGE SUGGESTS_FUTURE."""
    tx = _make_tx()
    upsert_future_direction(tx, "2301.00234", "Apply to video understanding")

    tx.run.assert_called_once()
    cypher = tx.run.call_args[0][0]
    assert "FutureDirection" in cypher
    assert "SUGGESTS_FUTURE" in cypher
    assert tx.run.call_args[1]["text"] == "Apply to video understanding"


# ---------------------------------------------------------------------------
# upsert_method
# ---------------------------------------------------------------------------


def test_upsert_method_creates_node_and_relationship():
    """upsert_method should MERGE Method node and MERGE USES_METHOD relationship."""
    tx = _make_tx()
    upsert_method(tx, "2301.00234", "DETR")

    tx.run.assert_called_once()
    cypher = tx.run.call_args[0][0]
    assert "Method" in cypher
    assert "USES_METHOD" in cypher
    assert tx.run.call_args[1]["name"] == "DETR"


# ---------------------------------------------------------------------------
# upsert_dataset
# ---------------------------------------------------------------------------


def test_upsert_dataset_creates_node_and_relationship():
    """upsert_dataset should MERGE Dataset node and MERGE USES_DATASET relationship."""
    tx = _make_tx()
    upsert_dataset(tx, "2301.00234", "COCO")

    tx.run.assert_called_once()
    cypher = tx.run.call_args[0][0]
    assert "Dataset" in cypher
    assert "USES_DATASET" in cypher
    assert tx.run.call_args[1]["name"] == "COCO"


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
