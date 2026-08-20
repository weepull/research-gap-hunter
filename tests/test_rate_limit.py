"""Tests for api/rate_limit.py — token-bucket limiting on the FastAPI layer."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api import rate_limit
from api.rate_limit import check, client_key, is_enabled, tier_for_path


@pytest.fixture(autouse=True)
def _clean_buckets():
    rate_limit.reset()
    yield
    rate_limit.reset()


@pytest.fixture()
def fake_clock(monkeypatch):
    """Controllable monotonic clock so refill is tested without sleeping."""

    class Clock:
        def __init__(self) -> None:
            self.t = 1000.0

        def advance(self, seconds: float) -> None:
            self.t += seconds

    clock = Clock()
    monkeypatch.setattr(rate_limit, "_now", lambda: clock.t)
    return clock


# ---------------------------------------------------------------------------
# tier_for_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/ingest", "ingest"),
        ("/explain", "llm"),
        ("/gaps", "heavy"),
        ("/cross-domain", "heavy"),
        ("/search", "search"),
        ("/health", "light"),
        ("/paper/2301.00234", "light"),
    ],
)
def test_tier_for_path_maps_each_endpoint(path, expected):
    assert tier_for_path(path) == expected


def test_tier_for_path_unknown_path_defaults_to_light():
    """An unmapped route must not be unlimited — it falls back to the cheap tier."""
    assert tier_for_path("/does-not-exist") == "light"
    assert tier_for_path("/") == "light"


def test_tier_for_path_respects_query_free_prefixes():
    """Path params still resolve via prefix match."""
    assert tier_for_path("/paper/anything/at/all") == "light"


# ---------------------------------------------------------------------------
# check — bucket mechanics
# ---------------------------------------------------------------------------


def test_check_allows_up_to_capacity_then_blocks(fake_clock, monkeypatch):
    """Capacity is the burst allowance; the next call is refused."""
    monkeypatch.setitem(rate_limit._TIERS, "test", (3, 1 / 60))

    assert check("1.2.3.4", "test")[0] is True
    assert check("1.2.3.4", "test")[0] is True
    assert check("1.2.3.4", "test")[0] is True

    allowed, retry_after = check("1.2.3.4", "test")
    assert allowed is False
    assert retry_after > 0


def test_check_refills_over_time(fake_clock, monkeypatch):
    """Tokens come back at the refill rate — 1 per 60s here."""
    monkeypatch.setitem(rate_limit._TIERS, "test", (2, 1 / 60))

    assert check("1.2.3.4", "test")[0] is True
    assert check("1.2.3.4", "test")[0] is True
    assert check("1.2.3.4", "test")[0] is False

    fake_clock.advance(59)
    assert check("1.2.3.4", "test")[0] is False, "not yet a full token"

    fake_clock.advance(2)
    assert check("1.2.3.4", "test")[0] is True


def test_check_refill_never_exceeds_capacity(fake_clock, monkeypatch):
    """A long idle period cannot bank more than the burst allowance."""
    monkeypatch.setitem(rate_limit._TIERS, "test", (2, 1 / 60))

    assert check("k", "test")[0] is True
    fake_clock.advance(10_000)  # idle far longer than it takes to refill

    assert check("k", "test")[0] is True
    assert check("k", "test")[0] is True
    assert check("k", "test")[0] is False, "capacity is still the ceiling"


def test_check_retry_after_reflects_refill_rate(fake_clock, monkeypatch):
    """retry_after is the wait for one whole token, not an arbitrary constant."""
    monkeypatch.setitem(rate_limit._TIERS, "test", (1, 1 / 10))  # 1 token per 10s

    assert check("k", "test")[0] is True
    allowed, retry_after = check("k", "test")

    assert allowed is False
    assert retry_after == pytest.approx(10.0, abs=0.01)


def test_check_keys_are_independent(fake_clock, monkeypatch):
    """One client exhausting its quota must not affect another."""
    monkeypatch.setitem(rate_limit._TIERS, "test", (1, 1 / 60))

    assert check("client-a", "test")[0] is True
    assert check("client-a", "test")[0] is False
    assert check("client-b", "test")[0] is True


def test_check_tiers_are_independent(fake_clock, monkeypatch):
    """Exhausting an expensive tier must not lock out cheap endpoints."""
    monkeypatch.setitem(rate_limit._TIERS, "cheap", (5, 1.0))
    monkeypatch.setitem(rate_limit._TIERS, "pricey", (1, 1 / 60))

    assert check("k", "pricey")[0] is True
    assert check("k", "pricey")[0] is False
    assert check("k", "cheap")[0] is True


def test_check_unknown_tier_falls_back_to_default(fake_clock):
    """An unregistered tier name uses the default tier rather than crashing."""
    allowed, _ = check("k", "no-such-tier")
    assert allowed is True


def test_reset_clears_state(fake_clock, monkeypatch):
    monkeypatch.setitem(rate_limit._TIERS, "test", (1, 1 / 60))

    assert check("k", "test")[0] is True
    assert check("k", "test")[0] is False

    rate_limit.reset()
    assert check("k", "test")[0] is True


# ---------------------------------------------------------------------------
# is_enabled / client_key
# ---------------------------------------------------------------------------


def test_is_enabled_defaults_true(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
    assert is_enabled() is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", " false "])
def test_is_enabled_false_values(monkeypatch, value):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", value)
    assert is_enabled() is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "anything-else"])
def test_is_enabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", value)
    assert is_enabled() is True


def test_client_key_uses_peer_address():
    request = MagicMock()
    request.client.host = "10.0.0.7"
    assert client_key(request) == "10.0.0.7"


def test_client_key_handles_missing_client():
    """A request with no peer info still yields a usable bucket key."""
    request = MagicMock()
    request.client = None
    assert client_key(request) == "unknown"


def test_client_key_ignores_forwarded_for_header():
    """X-Forwarded-For is spoofable, so it must not become the bucket key."""
    request = MagicMock()
    request.client.host = "10.0.0.7"
    request.headers = {"X-Forwarded-For": "1.1.1.1"}
    assert client_key(request) == "10.0.0.7"


# ---------------------------------------------------------------------------
# Middleware behaviour through the real app
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(monkeypatch):
    """TestClient with backends stubbed, for exercising the middleware."""
    monkeypatch.setattr("api.main.load_embedding_model", lambda: MagicMock())
    monkeypatch.setattr("api.main.get_qdrant_client", lambda: MagicMock())
    monkeypatch.setattr("api.main.get_neo4j_driver", lambda: MagicMock())

    db = MagicMock()
    db.table_names.return_value = ["papers"]
    db.execute.return_value.fetchone.return_value = (2,)
    monkeypatch.setattr("api.main._get_db", lambda: db)

    from api.main import app

    with TestClient(app) as c:
        yield c


def test_middleware_returns_429_with_retry_after(api_client, monkeypatch):
    """Over-quota requests get 429 plus a Retry-After header."""
    monkeypatch.setitem(rate_limit._TIERS, "light", (2, 1 / 60))

    assert api_client.get("/health").status_code == 200
    assert api_client.get("/health").status_code == 200

    response = api_client.get("/health")
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    assert response.json()["status"] == "rate_limited"


def test_middleware_retry_after_is_never_zero(api_client, monkeypatch):
    """Retry-After must be a positive integer — 0 would invite an instant retry."""
    monkeypatch.setitem(rate_limit._TIERS, "light", (1, 10.0))  # refills fast

    api_client.get("/health")
    response = api_client.get("/health")

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


def test_middleware_allows_traffic_under_the_limit(api_client):
    """Normal interactive use is untouched by the limiter."""
    for _ in range(10):
        assert api_client.get("/health").status_code == 200


def test_middleware_isolates_expensive_endpoints_from_cheap_ones(
    api_client, monkeypatch
):
    """Exhausting the ingest tier must leave /health usable."""
    monkeypatch.setitem(rate_limit._TIERS, "ingest", (1, 1 / 60))
    monkeypatch.setattr("api.main.extract_paper", MagicMock(side_effect=RuntimeError("x")))
    monkeypatch.setattr("api.main._log_failure", lambda *a, **k: None)

    # First call is allowed through (the handler itself then 500s — irrelevant here).
    first = api_client.post("/ingest", json={"arxiv_id": "1234.5678"})
    assert first.status_code != 429

    second = api_client.post("/ingest", json={"arxiv_id": "1234.5678"})
    assert second.status_code == 429

    assert api_client.get("/health").status_code == 200


def test_middleware_can_be_disabled_by_env(api_client, monkeypatch):
    """RATE_LIMIT_ENABLED=false bypasses limiting entirely."""
    monkeypatch.setitem(rate_limit._TIERS, "light", (1, 1 / 60))
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")

    for _ in range(5):
        assert api_client.get("/health").status_code == 200


def test_middleware_429_still_carries_cors_headers(api_client, monkeypatch):
    """A browser must be able to read the 429 rather than see an opaque failure."""
    monkeypatch.setitem(rate_limit._TIERS, "light", (1, 1 / 60))

    api_client.get("/health", headers={"Origin": "http://localhost:3000"})
    response = api_client.get("/health", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 429
    assert "access-control-allow-origin" in {k.lower() for k in response.headers}
