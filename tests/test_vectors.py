"""Tests for vectors/embed.py and vectors/search.py."""

import pytest
from vectors.embed import get_qdrant_client, load_specter2, ensure_collection, embed_text, embed_limitations
from vectors.search import find_similar_limitations


def test_load_specter2_returns_model():
    """load_specter2 should return a SentenceTransformer with output dim 768."""
    pass


def test_embed_text_shape():
    """embed_text should return one 768-dim vector per input string."""
    pass


def test_ensure_collection_is_idempotent():
    """Calling ensure_collection twice should not raise and collection should exist."""
    pass


def test_embed_limitations_returns_count():
    """embed_limitations should return a positive int equal to rows upserted."""
    pass


def test_find_similar_limitations_returns_list():
    """find_similar_limitations should return a list of dicts with required keys."""
    pass


def test_find_similar_limitations_respects_top_k():
    """find_similar_limitations(top_k=3) should return at most 3 results."""
    pass


def test_find_similar_limitations_domain_filter():
    """find_similar_limitations with domain='computer_vision' should only return CV results."""
    pass
