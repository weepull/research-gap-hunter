"""Batch ingestion runner: fetch, extract, and persist a list of arXiv papers."""


def run_batch(arxiv_ids: list[str]) -> dict:
    """Run extract_paper() for each ID, store results in SQLite, return summary stats."""
    pass


def ingest_from_semantic_scholar(query: str, limit: int = 50) -> list[str]:
    """Search Semantic Scholar for CV papers and return a list of arXiv IDs."""
    pass
