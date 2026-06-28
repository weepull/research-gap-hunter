"""Generates Specter2 embeddings and upserts them into the Qdrant 'limitations' collection."""


def get_qdrant_client():
    """Return a QdrantClient configured from env vars."""
    pass


def load_specter2():
    """Load and return the allenai/specter2_base SentenceTransformer model."""
    pass


def ensure_collection() -> None:
    """Create the 'limitations' Qdrant collection (vector size 768, cosine) if absent."""
    pass


def embed_text(model, texts: list[str]) -> list[list[float]]:
    """Return Specter2 embeddings for a list of strings."""
    pass


def embed_limitations(domain: str = "computer_vision") -> int:
    """Embed all limitations from SQLite and upsert into Qdrant. Return count upserted."""
    pass
