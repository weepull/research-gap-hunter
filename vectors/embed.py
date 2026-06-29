"""Generates Specter2 embeddings and upserts them into Qdrant collections."""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv()

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 768
_MODEL_NAME = "allenai/specter2_base"
_COLLECTION_LIMITATIONS = "limitations"
_COLLECTION_FUTURE_DIRECTIONS = "future_directions"

# Module-level cache so the model is loaded only once per process
_model_cache: dict = {}


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient using QDRANT_HOST and QDRANT_PORT from .env."""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def load_embedding_model():
    """Load allenai/specter2_base via sentence-transformers, cached for the process lifetime.

    Points the model, tokenizer, and config loaders at the local HuggingFace hub
    cache so repeated loads read from disk instead of hitting the HuggingFace API.
    """
    if "model" not in _model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s", _MODEL_NAME)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        _model_cache["model"] = SentenceTransformer(
            _MODEL_NAME,
            model_kwargs={"cache_dir": cache_dir},
            tokenizer_kwargs={"cache_dir": cache_dir},
            config_kwargs={"cache_dir": cache_dir},
        )
    return _model_cache["model"]


def ensure_collection(client: QdrantClient, collection_name: str = _COLLECTION_LIMITATIONS) -> None:
    """Create a Qdrant collection if it does not already exist.

    Idempotent — safe to call multiple times. Vector size: 768, distance: Cosine.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection '%s'", collection_name)


def _embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Return Specter2 embeddings (768-dim) for a list of strings."""
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [vec.tolist() for vec in embeddings]


def _query_neo4j_limitations() -> list[dict]:
    """Fetch all Limitation nodes and their connected Paper arxiv_ids and years from Neo4j."""
    from graph.populate import get_neo4j_driver
    driver = get_neo4j_driver()
    records = []
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run(
            """
            MATCH (p:Paper)-[:REPORTS_LIMITATION]->(l:Limitation)
            RETURN l.text AS text,
                   collect(p.arxiv_id) AS paper_ids,
                   collect(p.year)     AS years,
                   p.domain            AS domain
            """
        )
        for record in result:
            years = [y for y in record["years"] if y]
            records.append({
                "text": record["text"],
                "paper_ids": list(record["paper_ids"]),
                "year": max(years) if years else 0,
                "domain": record["domain"] or "computer_vision",
            })
    driver.close()
    return records


def _query_neo4j_future_directions() -> list[dict]:
    """Fetch all FutureDirection nodes and their connected Paper arxiv_ids and years from Neo4j."""
    from graph.populate import get_neo4j_driver
    driver = get_neo4j_driver()
    records = []
    with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run(
            """
            MATCH (p:Paper)-[:SUGGESTS_FUTURE]->(f:FutureDirection)
            RETURN f.text AS text,
                   collect(p.arxiv_id) AS paper_ids,
                   collect(p.year)     AS years,
                   p.domain            AS domain
            """
        )
        for record in result:
            years = [y for y in record["years"] if y]
            records.append({
                "text": record["text"],
                "paper_ids": list(record["paper_ids"]),
                "year": max(years) if years else 0,
                "domain": record["domain"] or "computer_vision",
            })
    driver.close()
    return records


def _upsert_records(
    records: list[dict],
    client: QdrantClient,
    model,
    collection_name: str,
) -> int:
    """Embed texts and upsert all records into the given Qdrant collection. Returns count."""
    if not records:
        return 0

    ensure_collection(client, collection_name)
    texts = [r["text"] for r in records]
    vectors = _embed_texts(model, texts)

    points = []
    for i, (record, vector) in enumerate(zip(records, vectors)):
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "limitation_text": record["text"],
                    "paper_ids": record["paper_ids"],
                    "domain": record["domain"],
                    "year": record["year"],
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Upserted %d points into '%s'", len(points), collection_name)
    return len(points)


def embed_limitations(client: Optional[QdrantClient] = None, model=None) -> dict:
    """Query Neo4j for all Limitation nodes, embed with Specter2, upsert into Qdrant.

    Returns {"embedded": n, "collection": "limitations"}.
    """
    if client is None:
        client = get_qdrant_client()
    if model is None:
        model = load_embedding_model()

    records = _query_neo4j_limitations()
    n = _upsert_records(records, client, model, _COLLECTION_LIMITATIONS)
    return {"embedded": n, "collection": _COLLECTION_LIMITATIONS}


def embed_future_directions(client: Optional[QdrantClient] = None, model=None) -> dict:
    """Query Neo4j for all FutureDirection nodes, embed with Specter2, upsert into Qdrant.

    Returns {"embedded": n, "collection": "future_directions"}.
    """
    if client is None:
        client = get_qdrant_client()
    if model is None:
        model = load_embedding_model()

    records = _query_neo4j_future_directions()
    n = _upsert_records(records, client, model, _COLLECTION_FUTURE_DIRECTIONS)
    return {"embedded": n, "collection": _COLLECTION_FUTURE_DIRECTIONS}
