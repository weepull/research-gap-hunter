"""Deployment-mode flags shared by the API and pipeline layers.

Lives in `pipeline/` rather than `api/` because both need it and the dependency
runs api -> pipeline; putting it under `api/` would invert that.

The single switch is DEMO_MODE. It marks a deployment as publicly reachable and
therefore not allowed to ingest, and not able to reach a local Ollama:

    DEMO_MODE=false (default)  local development — /ingest enabled, Ollama used,
                               CORS open so a dev frontend on any port works.
    DEMO_MODE=true             public deployment — /ingest refused, explanations
                               served by a hosted LLM, CORS restricted to the
                               configured frontend origins.
"""

import os

# Origins allowed to call the API when DEMO_MODE is on. Comma-separated.
_ALLOWED_ORIGINS_VAR = "ALLOWED_ORIGINS"


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_demo_mode() -> bool:
    """True when running as a public demo rather than local development.

    Read per call rather than cached at import so tests and local debugging can
    flip it without reimporting the module.
    """
    return _is_truthy(os.getenv("DEMO_MODE", "false"))


def allowed_origins() -> list[str]:
    """Origins permitted by CORS.

    Wide open outside demo mode — a developer runs the frontend on whatever port
    is free and should not have to configure anything. In demo mode the browser
    is untrusted, so only the configured frontend origins are allowed.

    Returns [] when demo mode is on and ALLOWED_ORIGINS is unset. That is
    deliberately a closed door rather than a silent fallback to "*": a
    misconfigured deploy should fail visibly in the browser, not quietly serve
    every origin on the internet.
    """
    if not is_demo_mode():
        return ["*"]

    raw = os.getenv(_ALLOWED_ORIGINS_VAR, "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
