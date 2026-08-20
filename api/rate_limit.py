"""In-process token-bucket rate limiting for the FastAPI layer.

Why a hand-rolled limiter rather than slowapi (the usual FastAPI choice):

* CLAUDE.md rule 8 discourages new dependencies, and slowapi would pull in
  limits/deprecated transitively for ~80 lines of logic we can own outright.
* slowapi's decorator requires every limited handler to accept a `request:
  Request` parameter. That means touching all seven endpoint signatures, which
  is a wider blast radius than the limiting itself warrants.
* Tests need a deterministic clock and a resettable bucket store. Owning the
  implementation gives both for free; with slowapi we would be monkeypatching
  its internals anyway.

The threat model here is *not* adversarial DDoS — this service has no auth and
is not deployed. It is protecting scarce local resources from runaway clients:
a frontend retry loop or a stuck browser tab can otherwise pin Ollama, exhaust
the Semantic Scholar quota attached to the API key, and re-embed both Qdrant
collections repeatedly. Limits are therefore per-endpoint-cost, not uniform.

A token bucket (rather than a fixed window) is used so that normal interactive
bursts pass untouched while the sustained rate stays bounded, and so there is no
window-boundary spike where 2x the limit slips through in a moment.

State is per-process and in-memory: it resets on restart and is not shared
across workers. That is the correct scope for a single-process local service;
a multi-worker or deployed setup would need a shared store (Redis) instead.
"""

import os
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse

# Bucket tiers: name -> (capacity, refill_tokens_per_second).
# Capacity is the burst allowance; the refill rate is the sustained ceiling.
_TIERS: dict[str, tuple[int, float]] = {
    # Semantic Scholar fetch + PDF download + Ollama extraction + full re-embed
    # of both Qdrant collections. 30-60s of work and outbound quota per call.
    "ingest": (3, 3 / 60),
    # Synchronous llama3.1:8b generation — seconds of CPU per call.
    "llm": (10, 10 / 60),
    # Clustering / cross-domain matching: Neo4j reads plus batched Specter2
    # encoding and many Qdrant queries.
    "heavy": (20, 20 / 60),
    # Single query embed + one Qdrant search.
    "search": (30, 30 / 60),
    # Counts and single-row lookups.
    "light": (120, 120 / 60),
}

_DEFAULT_TIER = "light"

# Longest-prefix match wins, so "/paper/{id}" is covered by the "/paper" entry.
_PATH_TIERS: tuple[tuple[str, str], ...] = (
    ("/ingest", "ingest"),
    ("/explain", "llm"),
    ("/gaps", "heavy"),
    ("/cross-domain", "heavy"),
    ("/search", "search"),
    ("/health", "light"),
    ("/paper", "light"),
)

_buckets: dict[tuple[str, str], tuple[float, float]] = {}
_lock = threading.Lock()


def _now() -> float:
    """Monotonic clock, isolated so tests can advance time without sleeping."""
    return time.monotonic()


def is_enabled() -> bool:
    """Rate limiting is on unless RATE_LIMIT_ENABLED is explicitly falsey.

    Read per-request rather than cached at import so tests and local debugging
    can toggle it without reimporting the module.
    """
    return os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() not in {
        "false",
        "0",
        "no",
    }


def reset() -> None:
    """Drop all bucket state. Used by tests to keep cases independent."""
    with _lock:
        _buckets.clear()


def tier_for_path(path: str) -> str:
    """Map a request path to its cost tier, longest prefix first."""
    best_tier = _DEFAULT_TIER
    best_len = -1
    for prefix, tier in _PATH_TIERS:
        if path.startswith(prefix) and len(prefix) > best_len:
            best_tier, best_len = tier, len(prefix)
    return best_tier


def client_key(request: Request) -> str:
    """Identify the caller.

    Deliberately uses the peer address only. X-Forwarded-For is trivially
    spoofable and would let any caller mint unlimited buckets; it should only be
    honoured behind a proxy that overwrites it, which this service does not have.
    """
    client = request.client
    return client.host if client else "unknown"


def check(key: str, tier: str) -> tuple[bool, float]:
    """Consume one token from (key, tier).

    Returns (allowed, retry_after_seconds). retry_after is 0.0 when allowed.
    """
    capacity, refill_rate = _TIERS.get(tier, _TIERS[_DEFAULT_TIER])
    now = _now()

    with _lock:
        tokens, last_seen = _buckets.get((key, tier), (float(capacity), now))

        # Refill for elapsed time, never above capacity.
        elapsed = max(0.0, now - last_seen)
        tokens = min(float(capacity), tokens + elapsed * refill_rate)

        if tokens >= 1.0:
            _buckets[(key, tier)] = (tokens - 1.0, now)
            return True, 0.0

        # Time until one whole token is available again.
        retry_after = (1.0 - tokens) / refill_rate if refill_rate > 0 else 60.0
        _buckets[(key, tier)] = (tokens, now)
        return False, retry_after


async def rate_limit_middleware(request: Request, call_next):
    """Reject over-quota requests with 429 before the handler runs."""
    if not is_enabled():
        return await call_next(request)

    tier = tier_for_path(request.url.path)
    allowed, retry_after = check(client_key(request), tier)

    if not allowed:
        seconds = max(1, int(retry_after + 0.999))
        return JSONResponse(
            status_code=429,
            content={
                "status": "rate_limited",
                "message": (
                    f"Rate limit exceeded for {request.url.path!r} "
                    f"({tier} tier). Retry in {seconds}s."
                ),
            },
            headers={"Retry-After": str(seconds)},
        )

    return await call_next(request)
