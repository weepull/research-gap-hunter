"""Tests for api/main.py — FastAPI REST layer."""

import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pipeline.cross_domain import CrossDomainMatch
from pipeline.extractor import PaperExtract
from pipeline.gap_scorer import GapResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_gap(desc: str = "slow convergence on large datasets", score: float = 0.72) -> GapResult:
    return GapResult(
        gap_description=desc,
        score=score,
        frequency_score=0.5,
        recency_score=0.6,
        solution_deficit_score=0.8,
        supporting_papers=["2301.00234", "2303.05499"],
        proposed_solutions=["use adaptive learning rates"],
    )


def _make_match() -> CrossDomainMatch:
    return CrossDomainMatch(
        source_gap="confidence calibration under distribution shift",
        target_solution="uncertainty quantification for medical segmentation",
        similarity_score=0.85,
        source_papers=["2301.00234"],
        target_papers=["2306.13528"],
        source_domain="computer_vision",
        target_domain="medical_imaging",
    )


def _make_paper_extract(arxiv_id: str = "2301.00234") -> PaperExtract:
    return PaperExtract(
        arxiv_id=arxiv_id,
        title="Object Detection with Transformers",
        year=2023,
        domain="computer_vision",
        objectives=["detect objects"],
        methods=["transformer"],
        datasets=["COCO"],
        evaluation_metrics=["mAP"],
        limitations=["struggles with small objects", "high compute cost"],
        future_directions=["explore efficient architectures"],
        raw_json="{}",
        ingested_at="2026-07-03T00:00:00+00:00",
        extraction_tier="explicit",
    )


def _make_qdrant_collection(points_count: int) -> MagicMock:
    info = MagicMock()
    info.points_count = points_count
    return info


@pytest.fixture()
def client(monkeypatch):
    """TestClient with all external backends patched before lifespan runs."""
    mock_model = MagicMock()
    mock_qdrant = MagicMock()
    mock_neo4j = MagicMock()

    monkeypatch.setattr("api.main.load_embedding_model", lambda: mock_model)
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main.get_neo4j_driver", lambda: mock_neo4j)

    from api.main import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def _make_mock_db(paper_count: int = 2) -> MagicMock:
    """Return a _get_db() mock that reports paper_count without real SQLite threads."""
    db = MagicMock()
    db.table_names.return_value = ["papers"]
    db.execute.return_value.fetchone.return_value = (paper_count,)
    return db


def test_health_returns_ok(monkeypatch):
    """Health endpoint returns status=ok with counts from SQLite and Qdrant."""
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = lambda name: _make_qdrant_collection(
        79 if name == "limitations" else 44
    )

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main._get_db", lambda: _make_mock_db(2))

    from api.main import app
    with TestClient(app) as c:
        r = c.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["papers"] == 2
    assert body["limitations"] == 79
    assert body["future_directions"] == 44


def test_health_missing_collection_returns_zero(monkeypatch):
    """If a Qdrant collection doesn't exist yet, counts default to 0."""
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = Exception("collection not found")

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main._get_db", lambda: _make_mock_db(0))

    from api.main import app
    with TestClient(app) as c:
        r = c.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["limitations"] == 0
    assert body["future_directions"] == 0


# ---------------------------------------------------------------------------
# /gaps
# ---------------------------------------------------------------------------


def test_gaps_returns_gap_list(client, monkeypatch):
    """GET /gaps returns a list of GapResult objects serialised as JSON."""
    gaps = [_make_gap("slow convergence", 0.72), _make_gap("poor generalisation", 0.61)]
    monkeypatch.setattr("api.main.score_gaps", lambda domain, top_n: gaps)

    r = client.get("/gaps")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["gap_description"] == "slow convergence"
    assert body[0]["score"] == 0.72
    assert "supporting_papers" in body[0]


def test_gaps_passes_domain_and_top_n(client, monkeypatch):
    """Query parameters domain and top_n are forwarded to score_gaps."""
    captured = {}

    def fake_score_gaps(domain, top_n):
        captured["domain"] = domain
        captured["top_n"] = top_n
        return []

    monkeypatch.setattr("api.main.score_gaps", fake_score_gaps)

    r = client.get("/gaps?domain=medical_imaging&top_n=5")

    assert r.status_code == 200
    assert captured == {"domain": "medical_imaging", "top_n": 5}


def test_gaps_empty_domain_returns_empty_list(client, monkeypatch):
    """An empty domain (no papers ingested) returns an empty list, not an error."""
    monkeypatch.setattr("api.main.score_gaps", lambda domain, top_n: [])

    r = client.get("/gaps?domain=unknown_domain")

    assert r.status_code == 200
    assert r.json() == []


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------


def test_search_returns_limitation_results(client, monkeypatch):
    """GET /search returns LimitationResult objects from find_similar_limitations."""
    results = [
        {"limitation_text": "fails on small objects", "score": 0.91,
         "paper_ids": ["2301.00234"], "domain": "computer_vision"},
    ]
    monkeypatch.setattr("api.main.find_similar_limitations",
                        lambda query_text, top_k, domain: results)

    r = client.get("/search?q=object+detection+failure")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["limitation_text"] == "fails on small objects"
    assert body[0]["score"] == 0.91


def test_search_passes_all_params(client, monkeypatch):
    """Query params q, top_k, and domain are forwarded to find_similar_limitations."""
    captured = {}
    monkeypatch.setattr(
        "api.main.find_similar_limitations",
        lambda query_text, top_k, domain: captured.update(
            {"q": query_text, "top_k": top_k, "domain": domain}
        ) or [],
    )

    # + in a query string decodes to a space; use %20 for a literal plus or just use space
    client.get("/search?q=attention+mechanism&top_k=5&domain=medical_imaging")

    assert captured == {"q": "attention mechanism", "top_k": 5, "domain": "medical_imaging"}


def test_search_missing_q_returns_422(client):
    """Omitting the required q parameter yields a 422 Unprocessable Entity."""
    r = client.get("/search")
    assert r.status_code == 422


def test_search_empty_q_returns_422(client):
    """An empty q string (min_length=1) yields 422."""
    r = client.get("/search?q=")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /cross-domain
# ---------------------------------------------------------------------------


def test_cross_domain_returns_matches(client, monkeypatch):
    """GET /cross-domain returns CrossDomainMatch objects as JSON."""
    monkeypatch.setattr(
        "api.main.find_cross_domain_matches",
        lambda source_domain, target_domain, top_n: [_make_match()],
    )

    r = client.get("/cross-domain")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    m = body[0]
    assert m["source_domain"] == "computer_vision"
    assert m["target_domain"] == "medical_imaging"
    assert m["similarity_score"] == 0.85
    assert "source_papers" in m
    assert "target_papers" in m


def test_cross_domain_passes_params(client, monkeypatch):
    """source, target, and top_n query params are forwarded to find_cross_domain_matches."""
    captured = {}

    def fake(source_domain, target_domain, top_n):
        captured.update({"source": source_domain, "target": target_domain, "top_n": top_n})
        return []

    monkeypatch.setattr("api.main.find_cross_domain_matches", fake)

    client.get("/cross-domain?source=medical_imaging&target=computer_vision&top_n=3")

    assert captured == {"source": "medical_imaging", "target": "computer_vision", "top_n": 3}


def test_cross_domain_empty_result(client, monkeypatch):
    """No matches returns an empty list, not an error."""
    monkeypatch.setattr(
        "api.main.find_cross_domain_matches",
        lambda source_domain, target_domain, top_n: [],
    )

    r = client.get("/cross-domain")
    assert r.status_code == 200
    assert r.json() == []


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


def test_ingest_happy_path(monkeypatch):
    """POST /ingest extracts, stores, and returns limitations_found and tier."""
    paper = _make_paper_extract()
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=MagicMock())  # db["papers"]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", lambda: mock_driver)
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id: paper)
    monkeypatch.setattr("api.main._get_db", lambda: mock_db)
    monkeypatch.setattr("api.main._paper_to_row", lambda p: {"arxiv_id": p.arxiv_id})
    monkeypatch.setattr("api.main._upsert_paper_counting", MagicMock())
    monkeypatch.setattr("api.main.embed_limitations", MagicMock())
    monkeypatch.setattr("api.main.embed_future_directions", MagicMock())

    from api.main import app
    with TestClient(app) as c:
        r = c.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "computer_vision"})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["arxiv_id"] == "2301.00234"
    assert body["limitations_found"] == 2
    assert body["tier"] == "explicit"


def test_ingest_patches_domain(monkeypatch):
    """The domain from the request body overrides the extractor default."""
    paper = _make_paper_extract()  # domain defaults to computer_vision
    stored = {}

    def fake_get_db():
        db = MagicMock()
        def fake_insert(row, **kw):
            stored["domain"] = row.get("domain")
        db.__getitem__ = MagicMock(return_value=MagicMock(insert=fake_insert))
        return db

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", lambda: mock_driver)
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id: paper)
    monkeypatch.setattr("api.main._get_db", fake_get_db)
    monkeypatch.setattr("api.main._paper_to_row",
                        lambda p: {"arxiv_id": p.arxiv_id, "domain": p.domain})
    monkeypatch.setattr("api.main._upsert_paper_counting", MagicMock())
    monkeypatch.setattr("api.main.embed_limitations", MagicMock())
    monkeypatch.setattr("api.main.embed_future_directions", MagicMock())

    from api.main import app
    with TestClient(app) as c:
        c.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "medical_imaging"})

    assert stored["domain"] == "medical_imaging"


def test_ingest_extraction_failure_returns_500(monkeypatch):
    """If extract_paper raises, the endpoint returns HTTP 500."""
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr("api.main.extract_paper",
                        lambda arxiv_id: (_ for _ in ()).throw(RuntimeError("PDF not found")))
    monkeypatch.setattr("api.main._log_failure", lambda aid, reason: None)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/ingest", json={"arxiv_id": "0000.99999", "domain": "computer_vision"})

    assert r.status_code == 500


def test_ingest_missing_body_returns_422(client):
    """POST /ingest with no body returns 422."""
    r = client.post("/ingest", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /paper/{arxiv_id}
# ---------------------------------------------------------------------------


def test_get_paper_returns_full_record(client, monkeypatch):
    """GET /paper/{arxiv_id} returns the full paper dict from SQLite."""
    paper_dict = {
        "arxiv_id": "2301.00234", "title": "Object Detection", "year": 2023,
        "domain": "computer_vision", "limitations": ["a", "b"],
    }
    monkeypatch.setattr("api.main.get_paper", lambda arxiv_id: paper_dict)

    r = client.get("/paper/2301.00234")

    assert r.status_code == 200
    body = r.json()
    assert body["arxiv_id"] == "2301.00234"
    assert body["title"] == "Object Detection"


def test_get_paper_not_found_returns_404(client, monkeypatch):
    """GET /paper/{arxiv_id} returns 404 when the paper is not in SQLite."""
    monkeypatch.setattr("api.main.get_paper", lambda arxiv_id: None)

    r = client.get("/paper/9999.00000")

    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Lifespan and CORS
# ---------------------------------------------------------------------------


def test_lifespan_initialises_app_state(monkeypatch):
    """Startup loads model, qdrant client, and neo4j driver into app.state."""
    mock_model = MagicMock(name="model")
    mock_qdrant = MagicMock(name="qdrant")
    mock_neo4j = MagicMock(name="neo4j")

    monkeypatch.setattr("api.main.load_embedding_model", lambda: mock_model)
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main.get_neo4j_driver", lambda: mock_neo4j)

    from api.main import app
    with TestClient(app) as c:
        assert app.state.model is mock_model
        assert app.state.qdrant is mock_qdrant
        assert app.state.neo4j is mock_neo4j


def test_cors_headers_present(monkeypatch):
    """Responses include CORS headers allowing the Next.js dev origin."""
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr("api.main.score_gaps", lambda domain, top_n: [])

    from api.main import app
    with TestClient(app) as c:
        r = c.get("/gaps", headers={"Origin": "http://localhost:3000"})

    assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
