"""Vector similarity search over the Qdrant 'limitations' collection."""


def find_similar_limitations(query: str, top_k: int = 10, domain: str | None = None) -> list[dict]:
    """Embed query with Specter2, search Qdrant 'limitations', return top_k results.

    Each result dict contains: limitation_text, paper_id, year, domain, cluster_id, score.
    """
    pass
