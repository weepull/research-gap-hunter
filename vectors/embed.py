"""Generates Specter2 embeddings and upserts them into Qdrant collections."""

import logging
import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
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
    """Return a QdrantClient for either a local or a managed Qdrant.

    QDRANT_URL (plus optional QDRANT_API_KEY) takes precedence and is what a
    managed instance such as Qdrant Cloud requires — an HTTPS URL and a key.
    Host/port alone cannot reach one, so a deployment configured that way would
    have failed at connect time.

    Falls back to QDRANT_HOST / QDRANT_PORT for the local Docker setup, which
    stays the default when nothing is configured.
    """
    url = os.getenv("QDRANT_URL", "").strip()
    if url:
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        return QdrantClient(url=url, api_key=api_key)

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def hf_cache_dir() -> str:
    """The HuggingFace hub directory the model is loaded from.

    Defaults to the usual per-user location, so local behaviour is unchanged.
    HF_CACHE_DIR overrides it, which is what lets a container image bake the
    weights somewhere the app will actually look.

    This used to be hardcoded to ~/.cache/huggingface/hub, and that silently
    defeated the point of pre-caching weights into a container image: the image
    baked them into /opt/hf-cache, the app looked in /root/.cache, found
    nothing, and downloaded 440MB again on every cold start. Anything that bakes
    weights must set this to the same directory it baked into.
    """
    configured = os.getenv("HF_CACHE_DIR", "").strip()
    return configured or os.path.expanduser("~/.cache/huggingface/hub")


def load_embedding_model():
    """Load allenai/specter2_base via sentence-transformers, cached for the process lifetime.

    Points the model, tokenizer, and config loaders at the local HuggingFace hub
    cache so repeated loads read from disk instead of hitting the HuggingFace API.
    """
    if "model" not in _model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s", _MODEL_NAME)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        cache_dir = hf_cache_dir()
        logger.info("HuggingFace cache directory: %s", cache_dir)
        _model_cache["model"] = SentenceTransformer(
            _MODEL_NAME,
            model_kwargs={"cache_dir": cache_dir},
            tokenizer_kwargs={"cache_dir": cache_dir},
            config_kwargs={"cache_dir": cache_dir},
        )
    return _model_cache["model"]


# Payload fields that queries filter on and therefore need an index. Every
# filtered query in the project narrows by `domain` — vectors/search.py,
# pipeline/gap_scorer.py, pipeline/cross_domain.py and the /corpus counts in
# api/main.py all do it.
_INDEXED_PAYLOAD_FIELDS = ("domain",)


def ensure_payload_indexes(
    client: QdrantClient, collection_name: str = _COLLECTION_LIMITATIONS
) -> list[str]:
    """Create the payload indexes filtered queries require. Idempotent.

    A local Qdrant will happily filter on an unindexed payload field. Qdrant
    Cloud refuses:

        400 Bad request: Index required but not found for "domain" of one of
        the following types: [keyword]

    So a corpus loaded into a managed cluster without this looks perfectly
    healthy — right point counts, right dimensions, GREEN status — and then
    every filtered query fails, which is /search, /gaps, /cross-domain and the
    /corpus counts. That is to say: everything except /health.

    Returns the fields it actually created an index for on this call; a field
    that was already indexed is not listed.

    Existing indexes are detected by reading the collection's payload schema
    rather than by catching an error from a repeat create. Qdrant answers a
    duplicate create_payload_index with 200, not a conflict, so an
    exception-based check would report every run as having created everything.
    """
    existing = set(client.get_collection(collection_name).payload_schema)
    created = []
    for field in _INDEXED_PAYLOAD_FIELDS:
        if field in existing:
            logger.debug(
                "Payload index on '%s.%s' already present", collection_name, field
            )
            continue
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        created.append(field)
        logger.info("Created payload index on '%s.%s'", collection_name, field)
    return created


def ensure_collection(client: QdrantClient, collection_name: str = _COLLECTION_LIMITATIONS) -> None:
    """Create a Qdrant collection and its payload indexes if missing.

    Idempotent — safe to call multiple times. Vector size: 768, distance: Cosine.

    The index step runs even when the collection already exists, so a cluster
    populated before indexes were created here gets repaired on the next load
    rather than needing a manual fix.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", collection_name)

    ensure_payload_indexes(client, collection_name)


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


# Namespace for deterministic point ids. Fixed forever: changing it re-ids every point
# in both collections, which would orphan everything already stored.
_POINT_ID_NAMESPACE = uuid.UUID("6f1c3b6e-8a2d-5c41-9f3a-1d2e4b5a6c7d")


def point_id(domain: str, text: str) -> str:
    """A stable id for one (domain, text) pair.

    Point ids used to be the enumeration index of an unordered Cypher result
    (``id=i``), which had two consequences. Ids were not stable across runs, so the
    same limitation could land on a different id each re-embed; and because
    `_upsert_records` only ever upserted and never deleted, a corpus that *shrank*
    left orphaned points behind — holding stale text and a stale `domain` payload, and
    still matching queries. That is why every curation pass so far has had to drop and
    rebuild both collections wholesale.

    Deriving the id from the content makes a re-embed idempotent: the same limitation
    always occupies the same point, and points whose content no longer exists can be
    identified and removed (see `_delete_stale_points`).

    `domain` is part of the key because the same limitation text can legitimately be
    reported by papers in both domains, and those are two separate points with
    different payloads.
    """
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{domain}\u0000{text}"))


def _existing_point_ids(client: QdrantClient, collection_name: str) -> set[str]:
    """Every point id currently in a collection."""
    ids: set[str] = set()
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(p.id) for p in batch)
        if offset is None:
            return ids


def _delete_stale_points(
    client: QdrantClient, collection_name: str, live_ids: set[str]
) -> int:
    """Remove points whose content is no longer in the graph. Returns the count.

    This is the half that was missing. Without it a re-embed after any removal leaves
    points that still answer queries with text no paper reports any more.
    """
    try:
        existing = _existing_point_ids(client, collection_name)
    except UnexpectedResponse:
        return 0
    stale = existing - live_ids
    if not stale:
        return 0
    client.delete(collection_name=collection_name, points_selector=list(stale), wait=True)
    logger.warning(
        "Deleted %d stale point(s) from '%s' — content no longer present in the graph",
        len(stale), collection_name,
    )
    return len(stale)


def _upsert_records(
    records: list[dict],
    client: QdrantClient,
    model,
    collection_name: str,
    prune: bool = True,
) -> int:
    """Embed texts and upsert records into a Qdrant collection. Returns the count.

    Ids are deterministic (see `point_id`), so this is idempotent: re-running over an
    unchanged corpus rewrites the same points rather than accumulating new ones. With
    `prune=True` (the default) any point not in `records` is deleted afterwards, so the
    collection ends up an exact mirror of the graph.

    `prune=False` is for incremental tranche ingestion, where `records` is deliberately
    only the new slice and pruning would delete everything else.
    """
    if not records:
        return 0

    ensure_collection(client, collection_name)
    texts = [r["text"] for r in records]
    vectors = _embed_texts(model, texts)

    points = []
    live_ids: set[str] = set()
    for record, vector in zip(records, vectors):
        pid = point_id(record["domain"], record["text"])
        live_ids.add(pid)
        points.append(
            PointStruct(
                id=pid,
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
    if prune:
        _delete_stale_points(client, collection_name, live_ids)
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
