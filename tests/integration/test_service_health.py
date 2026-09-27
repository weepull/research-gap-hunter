"""Integration proof for PLAN.md item #6 — /health and /corpus honesty.

Fails against pre-fix code: /health returns status="ok" as a literal, swallows
Qdrant failures into 0, and never contacts Neo4j at all — so a deployment with
both stores down returns 200 {"status":"ok",...}.
"""

import pytest

pytestmark = pytest.mark.integration


def test_health_reports_every_service_when_all_are_up(loaded_corpus):
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["services"]) >= {"neo4j", "qdrant"}
    assert all(state == "ok" for state in body["services"].values()), body["services"]


def test_health_is_degraded_and_503_when_neo4j_is_unreachable(loaded_corpus, monkeypatch):
    """Neo4j is the scoring source of truth; a probe that ignores it is not a probe."""
    from fastapi.testclient import TestClient

    import api.main

    def unreachable():
        raise OSError("simulated: Neo4j connection refused")

    monkeypatch.setattr(api.main, "get_neo4j_driver", unreachable)

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"]["neo4j"] == "unreachable"


def test_corpus_distinguishes_unavailable_from_empty(loaded_corpus, monkeypatch):
    """An unreachable Qdrant must not render as a confident '0 limitations'."""
    from fastapi.testclient import TestClient

    import api.main

    class DeadClient:
        def count(self, *a, **k):
            raise OSError("simulated: Qdrant connection refused")

        def get_collection(self, *a, **k):
            raise OSError("simulated: Qdrant connection refused")

        def get_collections(self, *a, **k):
            raise OSError("simulated: Qdrant connection refused")

    monkeypatch.setattr(api.main, "get_qdrant_client", lambda: DeadClient())

    from api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/corpus", params={"domain": "computer_vision"})

    body = response.json()
    assert body.get("limitations") != 0, (
        "an unreachable Qdrant reported 0 limitations, which is indistinguishable "
        "from a genuinely empty corpus"
    )
    assert body.get("vectors_available") is False
