"""FastAPI application — REST layer for Research Gap Hunter (port 8000)."""

from fastapi import FastAPI

app = FastAPI(title="Research Gap Hunter", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Liveness check."""
    pass


@app.get("/gaps")
async def get_gaps(domain: str = "computer_vision", limit: int = 20) -> list:
    """Return top-ranked research gaps for the given domain."""
    pass


@app.get("/search")
async def search_limitations(q: str, top_k: int = 10) -> list:
    """Vector-search limitations collection and return similar limitation statements."""
    pass


@app.get("/cross-domain")
async def get_cross_domain_matches(
    source: str = "computer_vision",
    target: str = "medical_imaging",
    limit: int = 20,
) -> list:
    """Return cross-domain hypothesis matches above the similarity threshold."""
    pass
