"""Vector similarity search over Qdrant collections."""

from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from vectors.embed import (
    _COLLECTION_LIMITATIONS,
    _COLLECTION_FUTURE_DIRECTIONS,
    get_qdrant_client,
    load_embedding_model,
    _embed_texts,
)


def find_similar_limitations(
    query_text: str,
    top_k: int = 10,
    domain: str = "computer_vision",
) -> list[dict]:
    """Embed query_text, search Qdrant 'limitations', return top_k results.

    Each result dict contains: limitation_text, paper_ids, score, domain.
    Domain filter is always applied.
    """
    client = get_qdrant_client()
    model = load_embedding_model()
    vector = _embed_texts(model, [query_text])[0]

    query_filter = Filter(
        must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
    )

    results = client.query_points(
        collection_name=_COLLECTION_LIMITATIONS,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
    )

    return [
        {
            "limitation_text": hit.payload.get("limitation_text", ""),
            "paper_ids": hit.payload.get("paper_ids", []),
            "domain": hit.payload.get("domain", ""),
            "score": hit.score,
        }
        for hit in results.points
    ]


def find_similar_future_directions(
    query_text: str,
    top_k: int = 10,
    domain: Optional[str] = None,
) -> list[dict]:
    """Embed query_text, search Qdrant 'future_directions', return top_k results.

    Each result dict contains: limitation_text, paper_ids, score, domain.
    Domain filter is optional — pass None to search across all domains.
    """
    client = get_qdrant_client()
    model = load_embedding_model()
    vector = _embed_texts(model, [query_text])[0]

    query_filter = None
    if domain is not None:
        query_filter = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )

    results = client.query_points(
        collection_name=_COLLECTION_FUTURE_DIRECTIONS,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
    )

    return [
        {
            "limitation_text": hit.payload.get("limitation_text", ""),
            "paper_ids": hit.payload.get("paper_ids", []),
            "domain": hit.payload.get("domain", ""),
            "score": hit.score,
        }
        for hit in results.points
    ]
