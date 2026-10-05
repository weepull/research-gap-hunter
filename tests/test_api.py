"""Tests for api/main.py — FastAPI REST layer."""

import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import rate_limit
from pipeline.cross_domain import CrossDomainMatch
from pipeline.extractor import PaperExtract
from pipeline.gap_scorer import GapResult
from tests.conftest import arxiv_reports

# Every /ingest test reaches the P2 admission check; arXiv is answered offline.
pytestmark = pytest.mark.usefixtures("arxiv_admits")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_rate_limit_buckets():
    """Token buckets live on the module, and every test shares one `app`.

    Without this, quota consumed by one test would leak into the next and make
    failures depend on test ordering.
    """
    rate_limit.reset()
    yield
    rate_limit.reset()


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


def _unexpected_response(status: int = 404) -> Exception:
    """A real Qdrant 'collection not found', which is ABSENT rather than an outage.

    The endpoints now narrow on this type: a reachable Qdrant answering 404 means
    the collection has not been created yet (a normal cold start, count 0), while
    a connection failure means the store is unreachable (count None). A bare
    Exception cannot express that difference, which is why these tests no longer
    raise one.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse

    return UnexpectedResponse(
        status_code=status, reason_phrase="Not Found", content=b"", headers=None
    )


def _patch_grounding(monkeypatch, similarity=0.9039, threshold=0.8792,
                     source_papers=None, target_papers=None):
    """Stub the /explain grounding gate with a successful verification.

    /explain now refuses anything verify_pairing cannot ground (PLAN.md #5), and
    verify_pairing reads real Qdrant vectors. Tests that are about the endpoint's
    other behaviour have to say explicitly that the pairing is grounded, rather
    than relying on an endpoint that explained whatever it was handed.
    """
    from pipeline.cross_domain import VerifiedPairing

    def fake_verify(source_gap, target_solution, source_domain, target_domain):
        return VerifiedPairing(
            source_gap=source_gap,
            target_solution=target_solution,
            source_domain=source_domain,
            target_domain=target_domain,
            similarity_score=similarity,
            threshold=threshold,
            source_papers=source_papers or ["2304.02643"],
            target_papers=target_papers or ["2403.16502"],
        )

    monkeypatch.setattr("api.main.verify_pairing", fake_verify)


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
    # status is derived from per-service probes, never asserted as a literal.
    assert body["services"] == {"sqlite": "ok", "neo4j": "ok", "qdrant": "ok"}


def test_health_missing_collection_returns_zero(monkeypatch):
    """A collection that does not exist yet is ABSENT: still 200, counts 0.

    This is the cold-start case and must stay non-fatal — the intention the
    original test encoded. What changed is that it now has to be expressed as a
    real Qdrant 404 rather than a bare Exception, because a bare Exception is
    now classified as an outage instead.
    """
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = _unexpected_response(404)

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main._get_db", lambda: _make_mock_db(0))

    from api.main import app
    with TestClient(app) as c:
        r = c.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["limitations"] == 0
    assert body["future_directions"] == 0
    assert body["services"]["qdrant"] == "absent"


def test_health_unreachable_qdrant_is_degraded_with_503(monkeypatch):
    """An unreachable Qdrant must fail visibly, not report zero counts.

    Fails against pre-fix code, which returned 200 status="ok" with both counts
    swallowed to 0 — indistinguishable from an empty corpus.
    """
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = OSError("connection refused")

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main._get_db", lambda: _make_mock_db(2))

    from api.main import app
    with TestClient(app) as c:
        r = c.get("/health")

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["services"]["qdrant"] == "unreachable"
    assert body["limitations"] is None
    assert body["future_directions"] is None


def test_health_probes_neo4j_at_all(monkeypatch):
    """Neo4j is the scoring source of truth, so readiness must actually ask it.

    Fails against pre-fix code: /health never contacted Neo4j, so an instance
    whose graph was completely unreachable still reported status="ok".
    """
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = lambda name: _make_qdrant_collection(5)

    def unreachable():
        raise OSError("connection refused")

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main._get_db", lambda: _make_mock_db(2))
    monkeypatch.setattr("api.main.get_neo4j_driver", unreachable)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/health")

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["services"]["neo4j"] == "unreachable"


def test_health_verifies_connectivity_rather_than_just_constructing_a_driver(monkeypatch):
    """A driver object is not a working connection — verify_connectivity must run."""
    mock_qdrant = MagicMock()
    mock_qdrant.get_collection.side_effect = lambda name: _make_qdrant_collection(5)
    driver = MagicMock()
    driver.verify_connectivity.side_effect = OSError("handshake failed")

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: mock_qdrant)
    monkeypatch.setattr("api.main._get_db", lambda: _make_mock_db(2))
    monkeypatch.setattr("api.main.get_neo4j_driver", lambda: driver)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/health")

    assert r.status_code == 503
    assert r.json()["services"]["neo4j"] == "unreachable"
    driver.verify_connectivity.assert_called()
    driver.close.assert_called()


# ---------------------------------------------------------------------------
# /gaps
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /corpus — H1 corpus transparency
# ---------------------------------------------------------------------------


def _patch_corpus_backends(
    monkeypatch, papers=31, contributing=18, lim=48, fd=24,
    last="2026-08-23T01:00:00+00:00",
):
    monkeypatch.setattr("api.main._count_papers_in_domain", lambda domain: papers)
    monkeypatch.setattr("api.main._count_contributing_papers", lambda domain: contributing)
    qdrant = MagicMock()
    qdrant.count.side_effect = lambda collection_name, **kw: types.SimpleNamespace(
        count=lim if collection_name == "limitations" else fd
    )
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: qdrant)
    db = MagicMock()
    db.table_names.return_value = ["papers"]
    db.execute.return_value.fetchone.return_value = (last,)
    monkeypatch.setattr("api.main._get_db", lambda: db)
    return qdrant


def test_corpus_reports_domain_size_and_freshness(client, monkeypatch):
    """/corpus states what the scores were computed over."""
    _patch_corpus_backends(monkeypatch)

    r = client.get("/corpus?domain=computer_vision")

    assert r.status_code == 200
    body = r.json()
    assert body == {
        "domain": "computer_vision",
        "papers": 31,
        "papers_reporting_limitations": 18,
        "limitations": 48,
        "future_directions": 24,
        "last_updated": "2026-08-23T01:00:00+00:00",
        "graph_available": True,
        "vectors_available": True,
    }


def test_corpus_reports_the_real_scoring_denominator(client, monkeypatch):
    """`papers_reporting_limitations` must be the value frequency_score divides by.

    Rewritten from test_corpus_paper_count_matches_the_scoring_denominator, which
    asserted that `papers` was the denominator. That was true and is no longer:
    the divisor is now the papers that extracted at least one limitation (PLAN.md
    #4), because a paper that extracted nothing cannot corroborate a gap and was
    only ever diluting the metric. Both numbers are reported, and this test pins
    which one is which — the banner's promise is that the divisor is visible, not
    that it is the corpus size.
    """
    _patch_corpus_backends(monkeypatch)
    size_seen, denominator_seen = [], []
    monkeypatch.setattr(
        "api.main._count_papers_in_domain", lambda domain: size_seen.append(domain) or 99
    )
    monkeypatch.setattr(
        "api.main._count_contributing_papers",
        lambda domain: denominator_seen.append(domain) or 70,
    )

    r = client.get("/corpus?domain=medical_imaging")

    body = r.json()
    assert body["papers"] == 99, "corpus size is every Paper node in the domain"
    assert body["papers_reporting_limitations"] == 70, "the divisor is the contributing count"
    assert size_seen == ["medical_imaging"]
    assert denominator_seen == ["medical_imaging"]


def test_score_gaps_divides_by_contributing_papers_not_domain_size(monkeypatch):
    """score_gaps must call _count_contributing_papers for its denominator.

    Fails against pre-fix code, which called _count_papers_in_domain — so every
    frequency was multiplied by the extraction success rate (43% of CV papers and
    38% of MI papers extracted no limitations).
    """
    import pipeline.gap_scorer as gs

    monkeypatch.setattr(
        gs, "get_all_limitations",
        lambda domain: [
            {"text": "a limitation", "paper_ids": ["p1"], "years": [2025], "tiers": ["explicit"]}
        ],
    )
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: [lims])
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda *a, **k: [])
    # The continuous deficit (A9 Option F) embeds the representative text, so the
    # real Specter2 weights load unless _addressing_hits is stubbed too. Without
    # this the hermetic unit suite jumps from 0.7s to 6s.
    monkeypatch.setattr(gs, "_addressing_hits", lambda *a, **k: [])

    def must_not_be_the_denominator(domain):
        raise AssertionError("score_gaps must not divide by the domain's total paper count")

    monkeypatch.setattr(gs, "_count_papers_in_domain", must_not_be_the_denominator)
    monkeypatch.setattr(gs, "_count_contributing_papers", lambda domain: 4)

    gaps = gs.score_gaps(domain="computer_vision", top_n=5)

    # one explicit-tier paper over 4 contributing papers
    assert gaps[0].frequency_score == pytest.approx(0.25)


def test_corpus_counts_are_domain_filtered(client, monkeypatch):
    """Limitation/future-direction counts are per-domain, not corpus-wide."""
    qdrant = _patch_corpus_backends(monkeypatch)

    client.get("/corpus?domain=medical_imaging")

    for call in qdrant.count.call_args_list:
        condition = call.kwargs["count_filter"].must[0]
        assert condition.key == "domain"
        assert condition.match.value == "medical_imaging"


def test_corpus_survives_missing_collections(client, monkeypatch):
    """A missing Qdrant collection reports zero rather than failing the page.

    The original intention holds — a cold start must not break the results page.
    It now has to be expressed as a real Qdrant 404, because an arbitrary
    exception is classified as an outage and reports None instead of 0.
    """
    monkeypatch.setattr("api.main._count_papers_in_domain", lambda domain: 5)
    monkeypatch.setattr("api.main._count_contributing_papers", lambda domain: 3)
    qdrant = MagicMock()
    qdrant.count.side_effect = _unexpected_response(404)
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: qdrant)
    db = MagicMock()
    db.table_names.return_value = []
    monkeypatch.setattr("api.main._get_db", lambda: db)

    r = client.get("/corpus")

    assert r.status_code == 200
    assert r.json()["limitations"] == 0
    assert r.json()["future_directions"] == 0
    assert r.json()["last_updated"] is None
    assert r.json()["vectors_available"] is True
    # The premise is that only Qdrant is degraded. Without this, a graph read that
    # failed and was swallowed by /corpus passed unnoticed (it did, through P1).
    assert r.json()["graph_available"] is True


def test_corpus_reports_unreachable_vectors_as_unknown_not_zero(client, monkeypatch):
    """An unreachable Qdrant must not render as a confident "0 limitations".

    Fails against pre-fix code, where every exception became 0 — so the banner
    asserted an empty corpus underneath a working gap list.
    """
    monkeypatch.setattr("api.main._count_papers_in_domain", lambda domain: 5)
    monkeypatch.setattr("api.main._count_contributing_papers", lambda domain: 3)
    qdrant = MagicMock()
    qdrant.count.side_effect = OSError("connection refused")
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: qdrant)
    db = MagicMock()
    db.table_names.return_value = []
    monkeypatch.setattr("api.main._get_db", lambda: db)

    r = client.get("/corpus")

    assert r.status_code == 200
    assert r.json()["limitations"] is None
    assert r.json()["future_directions"] is None
    assert r.json()["vectors_available"] is False
    # The premise is that only Qdrant is degraded. Without this, a graph read that
    # failed and was swallowed by /corpus passed unnoticed (it did, through P1).
    assert r.json()["graph_available"] is True


def test_corpus_reports_unreachable_graph_as_unknown_not_zero(client, monkeypatch):
    """An unreachable Neo4j must not render the denominator as 0."""
    def unreachable(domain):
        raise OSError("connection refused")

    monkeypatch.setattr("api.main._count_papers_in_domain", unreachable)
    qdrant = MagicMock()
    qdrant.count.return_value = MagicMock(count=7)
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: qdrant)
    db = MagicMock()
    db.table_names.return_value = []
    monkeypatch.setattr("api.main._get_db", lambda: db)

    r = client.get("/corpus")

    assert r.status_code == 200
    assert r.json()["papers"] is None
    assert r.json()["graph_available"] is False


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
    """An empty domain (no papers ingested) returns an empty list, not an error.

    Uses a known domain. This test previously sent `domain=unknown_domain` and asserted
    200 — standing in an unknown domain for an empty one, which is the conflation P3
    removed: an unknown domain is now a 422 (see the P3 tests below).
    """
    monkeypatch.setattr("api.main.score_gaps", lambda domain, top_n: [])

    r = client.get("/gaps?domain=medical_imaging")

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


def _report(status="matches_found", matches=None, **kw):
    """A CrossDomainReport as the pipeline would return it (imported lazily so a missing
    class fails the test that needs it, not the whole module)."""
    from pipeline.cross_domain import CrossDomainReport

    matches = [] if matches is None else matches
    base = dict(
        source_domain="computer_vision", target_domain="medical_imaging", status=status,
        message="m", matches=matches, total_matches=len(matches),
        similarity_threshold=0.8764, min_seed_papers=2, source_gaps=54, seed_gaps=7,
        target_future_directions=68,
    )
    base.update(kw)
    return CrossDomainReport(**base)


def test_cross_domain_returns_a_report_not_a_bare_list(client, monkeypatch):
    """T2: the response states its own status and the evidence behind it.

    Replaces the test that asserted a bare list of matches. Fails against pre-T2 code,
    which returned `[]` or `[match, ...]` with nothing saying why.
    """
    monkeypatch.setattr(
        "api.main.cross_domain_report",
        lambda source_domain, target_domain, top_n: _report(matches=[_make_match()]),
    )

    r = client.get("/cross-domain")

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert body["status"] == "matches_found"
    assert body["similarity_threshold"] == 0.8764 and body["min_seed_papers"] == 2
    m = body["matches"][0]
    assert m["source_domain"] == "computer_vision" and m["similarity_score"] == 0.85


def test_cross_domain_passes_params(client, monkeypatch):
    """source, target, and top_n query params are forwarded to cross_domain_report."""
    captured = {}

    def fake(source_domain, target_domain, top_n):
        captured.update({"source": source_domain, "target": target_domain, "top_n": top_n})
        return _report(source_domain=source_domain, target_domain=target_domain,
                       status="none_above_threshold")

    monkeypatch.setattr("api.main.cross_domain_report", fake)

    client.get("/cross-domain?source=medical_imaging&target=computer_vision&top_n=3")

    assert captured == {"source": "medical_imaging", "target": "computer_vision", "top_n": 3}


@pytest.mark.parametrize("status", ["none_above_threshold", "no_corroborated_gaps", "no_data"])
def test_cross_domain_states_why_a_result_is_empty(client, monkeypatch, status):
    """T2: an empty result says which kind of empty it is, with a message.

    Replaces `test_cross_domain_empty_result`, which asserted a bare `[]`: an empty list
    reads as "no such connections exist", when the honest statement is "none above the
    noise floor at this corpus size", or "no gap met the evidence gate", or "no data".
    """
    monkeypatch.setattr(
        "api.main.cross_domain_report",
        lambda source_domain, target_domain, top_n: _report(status=status, message=f"why: {status}"),
    )

    r = client.get("/cross-domain")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == status and body["matches"] == []
    assert body["message"] == f"why: {status}"


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
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id, domain: paper)
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


def _patch_ingest_backends(monkeypatch, paper):
    """Patch everything /ingest touches after admission; return the extract mock."""
    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=MagicMock())
    mock_driver = MagicMock()
    extract = MagicMock(return_value=paper)
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", lambda: mock_driver)
    monkeypatch.setattr("api.main.extract_paper", extract)
    monkeypatch.setattr("api.main._get_db", lambda: mock_db)
    monkeypatch.setattr("api.main._paper_to_row", lambda p: {"arxiv_id": p.arxiv_id})
    monkeypatch.setattr("api.main._upsert_paper_counting", MagicMock())
    monkeypatch.setattr("api.main.embed_limitations", MagicMock())
    monkeypatch.setattr("api.main.embed_future_directions", MagicMock())
    return extract


def test_ingest_refuses_a_paper_its_domain_rule_rejects(monkeypatch):
    """/ingest applies the same arXiv primary-category rule as the tranche path (P2).

    Fails against pre-fix code, where /ingest extracted any id it was handed: a paper
    with Kosmos-2's profile (primary cs.CL, cross-listed cs.CV) declared as computer
    vision went straight to extraction and into the corpus.
    """
    arxiv_reports(monkeypatch, "cs.CL", title="Grounding multimodal large language models")
    extract = _patch_ingest_backends(monkeypatch, _make_paper_extract())

    from api.main import app
    with TestClient(app) as c:
        r = c.post("/ingest", json={"arxiv_id": "2306.14824", "domain": "computer_vision"})

    assert extract.call_count == 0
    assert r.status_code == 422
    assert "cs.CL" in r.json()["detail"]


def test_ingest_fails_closed_when_arxiv_cannot_be_asked(monkeypatch):
    """An unverifiable paper is not admitted: 503, and nothing is extracted (P2)."""
    from pipeline import arxiv_source

    def unreachable(arxiv_ids):
        raise RuntimeError("arXiv request failed after 4 attempts: timed out")

    monkeypatch.setattr(arxiv_source, "fetch_by_ids", unreachable)
    extract = _patch_ingest_backends(monkeypatch, _make_paper_extract())

    from api.main import app
    with TestClient(app) as c:
        r = c.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "computer_vision"})

    assert extract.call_count == 0
    assert r.status_code == 503


def test_ingest_passes_the_requested_domain_into_extraction(monkeypatch):
    """The request body's domain is passed INTO extract_paper, not patched after it.

    Replaces test_ingest_patches_domain, whose premise the fix removes: there is
    no extractor default left to override (PLAN.md #1, option 1A). The old flow
    extracted with a hardcoded computer_vision and then rewrote the field, which
    meant any call site that forgot the rewrite stored the wrong label.
    """
    paper = _make_paper_extract()
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
    seen: dict = {}

    def fake_extract(arxiv_id, domain):
        seen["arxiv_id"] = arxiv_id
        seen["domain"] = domain
        return paper.model_copy(update={"domain": domain})

    monkeypatch.setattr("api.main.extract_paper", fake_extract)
    monkeypatch.setattr("api.main._get_db", fake_get_db)
    monkeypatch.setattr("api.main._paper_to_row",
                        lambda p: {"arxiv_id": p.arxiv_id, "domain": p.domain})
    monkeypatch.setattr("api.main._upsert_paper_counting", MagicMock())
    monkeypatch.setattr("api.main.embed_limitations", MagicMock())
    monkeypatch.setattr("api.main.embed_future_directions", MagicMock())

    from api.main import app
    with TestClient(app) as c:
        c.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "medical_imaging"})

    assert seen == {"arxiv_id": "2301.00234", "domain": "medical_imaging"}
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


# ---------------------------------------------------------------------------
# DEMO_MODE — /ingest disabled on public deployments (G1 / E1)
# ---------------------------------------------------------------------------


def test_ingest_refused_in_demo_mode(client, monkeypatch):
    """A public demo must not accept ingestion requests."""
    monkeypatch.setattr("api.main.is_demo_mode", lambda: True)
    called = []
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id, domain: called.append(arxiv_id))

    r = client.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "computer_vision"})

    assert r.status_code == 403
    assert called == [], "demo mode still reached the extraction path"


def test_ingest_demo_refusal_explains_itself(client, monkeypatch):
    """The refusal says what happened and how to ingest, not just 'forbidden'.

    A bare 403 reads like a bug or a missing credential; this is a deliberate
    deployment policy and the message has to say so.
    """
    monkeypatch.setattr("api.main.is_demo_mode", lambda: True)
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id, domain: None)

    body = client.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "computer_vision"}).json()

    assert "disabled in the public demo" in body["detail"]
    assert "DEMO_MODE=false" in body["detail"]


def test_ingest_demo_refusal_precedes_validation(client, monkeypatch):
    """Demo mode refuses before doing any work, including on a malformed id.

    Ordering matters: a 422 would tell a prober that ingestion is reachable and
    only the payload was wrong.
    """
    monkeypatch.setattr("api.main.is_demo_mode", lambda: True)
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id, domain: None)

    # A valid id is refused; an invalid one is still rejected by the validator,
    # which runs before the handler — both refuse, neither ingests.
    assert client.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "computer_vision"}).status_code == 403
    assert client.post("/ingest", json={"arxiv_id": "../etc", "domain": "computer_vision"}).status_code == 422


def test_ingest_works_when_not_in_demo_mode(monkeypatch):
    """Local development is unaffected — the endpoint stays fully functional."""
    monkeypatch.setattr("api.main.is_demo_mode", lambda: False)
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id, domain: _make_paper_extract())
    monkeypatch.setattr("api.main._paper_to_row", lambda p: {"arxiv_id": p.arxiv_id})
    monkeypatch.setattr("api.main._get_db", lambda: MagicMock())
    monkeypatch.setattr("api.main._upsert_paper_counting", MagicMock())
    monkeypatch.setattr("api.main.embed_limitations", MagicMock())
    monkeypatch.setattr("api.main.embed_future_directions", MagicMock())

    from api.main import app

    with TestClient(app) as c:
        r = c.post("/ingest", json={"arxiv_id": "2301.00234", "domain": "computer_vision"})

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_explain_refused_in_demo_mode(client, monkeypatch):
    """Explanations call an LLM per request, so a public demo refuses them."""
    monkeypatch.setattr("api.main.is_demo_mode", lambda: True)
    called = []
    monkeypatch.setattr("api.main.explain_match", lambda match: called.append(match))

    r = client.get("/explain", params={"source_gap": "a", "target_solution": "b"})

    assert r.status_code == 403
    assert called == [], "demo mode still reached the LLM call"


def test_explain_demo_refusal_explains_itself(client, monkeypatch):
    """The message states the reason and the way to get a live explanation."""
    monkeypatch.setattr("api.main.is_demo_mode", lambda: True)
    monkeypatch.setattr("api.main.explain_match", lambda match: "should not run")

    detail = client.get(
        "/explain", params={"source_gap": "a", "target_solution": "b"}
    ).json()["detail"]

    assert "disabled in the public demo" in detail
    assert "API costs" in detail
    assert "DEMO_MODE=false" in detail


def test_explain_works_when_not_in_demo_mode(client, monkeypatch):
    """Local development keeps live explanations."""
    monkeypatch.setattr("api.main.is_demo_mode", lambda: False)
    _patch_grounding(monkeypatch)
    monkeypatch.setattr("api.main.explain_match", lambda match: "because X relates to Y")

    r = client.get("/explain", params={"source_gap": "a", "target_solution": "b"})

    assert r.status_code == 200
    assert r.json()["explanation"] == "because X relates to Y"


def test_read_endpoints_unaffected_by_demo_mode(client, monkeypatch):
    """Demo mode must not disable the read paths the demo exists to show."""
    monkeypatch.setattr("api.main.is_demo_mode", lambda: True)
    monkeypatch.setattr("api.main.score_gaps", lambda domain, top_n: [])
    monkeypatch.setattr(
        "api.main.cross_domain_report",
        lambda source_domain, target_domain, top_n: _report(status="none_above_threshold"),
    )

    assert client.get("/gaps").status_code == 200
    assert client.get("/cross-domain").status_code == 200


def test_ingest_error_does_not_leak_internals(monkeypatch):
    """A 500 must not echo the raw exception text back to the caller.

    str(exc) on a driver or HTTP error routinely carries connection URIs, file
    paths and library internals, and the frontend renders the body verbatim.
    """
    secret = "bolt://neo4j:hunter2@10.0.0.5:7687 /Users/someone/secret/path.py"
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr(
        "api.main.extract_paper",
        lambda arxiv_id: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr("api.main._log_failure", lambda aid, reason: None)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/ingest", json={"arxiv_id": "0000.99999", "domain": "computer_vision"})

    assert r.status_code == 500
    body = r.text
    assert "hunter2" not in body
    assert "bolt://" not in body
    assert "/Users/" not in body


def test_explain_error_does_not_leak_internals(monkeypatch):
    """/explain must not echo raw exception text either."""
    secret = "http://localhost:11434 connection refused /private/tmp/x.py"
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    _patch_grounding(monkeypatch)
    monkeypatch.setattr(
        "api.main.explain_match",
        lambda match: (_ for _ in ()).throw(ConnectionError(secret)),
    )

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/explain", params={"source_gap": "a", "target_solution": "b"})

    assert r.status_code == 500
    assert "11434" not in r.text
    assert "/private/tmp" not in r.text


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../../etc/passwd",
        "2301.00234/../../admin",
        "not-an-id",
        "2301.00234?fields=all",
        "",
    ],
)
def test_ingest_rejects_malformed_arxiv_id(client, monkeypatch, bad_id):
    """Malformed ids are rejected before reaching an outbound URL.

    arxiv_id is interpolated into arxiv.org and Semantic Scholar URLs, so an
    unvalidated value can redirect the fetch to another path on those hosts.

    extract_paper is stubbed so this can never make a real network call even if
    validation regresses — without it, these cases reach the live handler.
    """
    called = []
    monkeypatch.setattr("api.main.extract_paper", lambda arxiv_id, domain: called.append(arxiv_id))
    monkeypatch.setattr("api.main._log_failure", lambda aid, reason: None)

    r = client.post("/ingest", json={"arxiv_id": bad_id, "domain": "computer_vision"})

    assert r.status_code == 422
    assert called == [], "malformed id reached the extraction path"


@pytest.mark.parametrize(
    "good_id",
    ["2301.00234", "2301.00234v2", "1234.5678", "math/0309136", "cs.CV/0309136"],
)
def test_ingest_accepts_valid_arxiv_id_forms(monkeypatch, good_id):
    """Both the modern and the pre-2007 arXiv id forms are accepted."""
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    monkeypatch.setattr(
        "api.main.extract_paper",
        lambda arxiv_id: (_ for _ in ()).throw(RuntimeError("stop here")),
    )
    monkeypatch.setattr("api.main._log_failure", lambda aid, reason: None)

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/ingest", json={"arxiv_id": good_id, "domain": "computer_vision"})

    # Reaches the handler (which then fails on the stub) rather than 422ing.
    assert r.status_code != 422


@pytest.mark.parametrize("bad_id", ["not-an-id", "abc", "2301", "v2"])
def test_get_paper_rejects_malformed_arxiv_id(client, monkeypatch, bad_id):
    """The path param is validated too, not just the request body.

    A query string is not tested here — HTTP splits it off before routing, so it
    never reaches the path param. It is covered on the request body instead.
    """
    called = []
    monkeypatch.setattr("api.main.get_paper", lambda aid: called.append(aid))

    r = client.get(f"/paper/{bad_id}")

    assert r.status_code == 422
    assert called == [], "malformed id reached the SQLite lookup"


def test_get_paper_path_traversal_never_reaches_handler(client, monkeypatch):
    """An encoded traversal attempt is rejected by routing before the handler.

    Decoded, "..%2f..%2fetc" contains slashes, so it does not match the
    /paper/{arxiv_id} route at all and returns 404 rather than 422. Either way
    it must not reach the lookup — that is what this asserts.
    """
    called = []
    monkeypatch.setattr("api.main.get_paper", lambda aid: called.append(aid))

    r = client.get("/paper/..%2f..%2fetc")

    assert r.status_code in (404, 422)
    assert called == []


def test_ingest_missing_body_returns_422(client):
    """POST /ingest with no body returns 422."""
    r = client.post("/ingest", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /explain
# ---------------------------------------------------------------------------


def test_explain_returns_explanation(client, monkeypatch):
    """GET /explain builds a CrossDomainMatch and returns the Ollama explanation."""
    captured = {}

    def fake_explain(match):
        captured["source_gap"] = match.source_gap
        captured["target_solution"] = match.target_solution
        captured["source_domain"] = match.source_domain
        captured["target_domain"] = match.target_domain
        return "Both domains share a registration problem."

    _patch_grounding(monkeypatch)
    monkeypatch.setattr("api.main.explain_match", fake_explain)

    r = client.get(
        "/explain",
        params={
            "source_gap": "loses track of objects",
            "target_solution": "robust registration techniques",
            "source": "computer_vision",
            "target": "medical_imaging",
        },
    )

    assert r.status_code == 200
    assert r.json() == {
        "explanation": "Both domains share a registration problem.",
        # PLAN.md #5 (option 5C): the evidence ships with the prose. similarity_score
        # is the real recomputed value, replacing the hardcoded 0.0 the endpoint used
        # to invent and never show the model.
        "similarity_score": 0.9039,
        "threshold": 0.8792,
        "grounding": "corpus_match",
        "is_hypothesis": True,
        "source_papers": ["2304.02643"],
        "target_papers": ["2403.16502"],
    }
    assert captured == {
        "source_gap": "loses track of objects",
        "target_solution": "robust registration techniques",
        "source_domain": "computer_vision",
        "target_domain": "medical_imaging",
    }


def test_explain_missing_params_returns_422(client):
    """Omitting source_gap or target_solution yields 422."""
    assert client.get("/explain").status_code == 422
    assert client.get("/explain", params={"source_gap": "x"}).status_code == 422


def test_explain_ollama_failure_returns_500(monkeypatch):
    """If Ollama is unreachable, /explain returns HTTP 500."""
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", MagicMock())
    _patch_grounding(monkeypatch)
    monkeypatch.setattr(
        "api.main.explain_match",
        lambda match: (_ for _ in ()).throw(ConnectionError("ollama down")),
    )

    from api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/explain", params={"source_gap": "a", "target_solution": "b"})

    assert r.status_code == 500


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


# ---------------------------------------------------------------------------
# Startup resilience — PLAN.md #6
# ---------------------------------------------------------------------------


def test_startup_survives_an_unavailable_backend(monkeypatch):
    """The app must start with a store down so /health can report it.

    Every backend used to be constructed unguarded in the lifespan, and
    get_neo4j_driver() calls verify_connectivity(), so an unreachable store made
    startup raise. On a container platform that is a crash loop — /health is
    never served, and the degraded state it now reports could never be observed
    in the one situation it exists for.
    """
    def unreachable():
        raise OSError("connection refused")

    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", unreachable)

    from api.main import app
    with TestClient(app) as c:
        assert app.state.neo4j is None
        # and the app is genuinely serving
        assert c.get("/paper/not-an-arxiv-id").status_code == 422


def test_startup_skips_selfheal_when_the_graph_is_unavailable(monkeypatch):
    """Reconciliation reads from Neo4j, so it must be skipped rather than attempted."""
    def unreachable():
        raise OSError("connection refused")

    called = []
    monkeypatch.setattr("api.main.load_embedding_model", MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", unreachable)
    monkeypatch.setattr("api.main.selfheal_enabled", lambda: True)
    monkeypatch.setattr(
        "api.main.reconcile_sqlite_from_graph", lambda driver: called.append(driver)
    )

    from api.main import app
    with TestClient(app):
        pass

    assert called == [], "reconciliation must not run against a missing driver"


# ---------------------------------------------------------------------------
# P3 — read endpoints reject unknown domains
# ---------------------------------------------------------------------------

_READ_ENDPOINT_DOMAIN_CASES = [
    # (path, query params with one misspelt domain, backend that must not be called)
    ("/gaps", {"domain": "computer_visoin"}, "score_gaps"),
    ("/search", {"q": "occlusion", "domain": "computer_visoin"}, "find_similar_limitations"),
    ("/cross-domain", {"source": "computer_visoin"}, "cross_domain_report"),
    ("/cross-domain", {"target": "medical_imagng"}, "cross_domain_report"),
    ("/corpus", {"domain": "computer_visoin"}, "_count_papers_in_domain"),
    ("/explain", {"source_gap": "g", "target_solution": "s", "source": "computer_visoin"},
     "verify_pairing"),
    ("/explain", {"source_gap": "g", "target_solution": "s", "target": "medical_imagng"},
     "verify_pairing"),
]


@pytest.mark.parametrize("path, params, backend", _READ_ENDPOINT_DOMAIN_CASES)
def test_read_endpoints_reject_an_unknown_domain_with_422(client, monkeypatch, path, params, backend):
    """A misspelt domain is a bad request, not an empty corpus (P3).

    Fails against pre-fix code, where these endpoints took `domain` as a plain string
    and passed it straight through: `/gaps?domain=computer_visoin` returned 200 with an
    empty list, which by this project's own doctrine reads as "no gaps found" — a false
    finding produced by a typo.
    """
    if backend == "verify_pairing":
        # /explain has no list to return empty; reaching the gate at all is the defect.
        called = MagicMock(side_effect=AssertionError(f"{backend} reached with an unknown domain"))
    else:
        called = MagicMock(return_value=[])
    monkeypatch.setattr(f"api.main.{backend}", called)
    monkeypatch.setattr("api.main.explain_match", MagicMock(return_value="x"))

    r = client.get(path, params=params)

    assert r.status_code == 422
    assert called.call_count == 0
    assert "computer_vision" in r.json()["detail"] and "medical_imaging" in r.json()["detail"]
