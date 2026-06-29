"""Tests for vectors/embed.py and vectors/search.py."""

import os
import types
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

import vectors.embed as embed_mod
import vectors.search as search_mod
from vectors.embed import (
    _COLLECTION_FUTURE_DIRECTIONS,
    _COLLECTION_LIMITATIONS,
    _embed_texts,
    _upsert_records,
    embed_future_directions,
    embed_limitations,
    ensure_collection,
    get_qdrant_client,
    load_embedding_model,
)
from vectors.search import find_similar_future_directions, find_similar_limitations


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "text": "Slow convergence during training",
        "paper_ids": ["2301.00234"],
        "year": 2023,
        "domain": "computer_vision",
    },
    {
        "text": "High memory requirements for large inputs",
        "paper_ids": ["2303.05499"],
        "year": 2023,
        "domain": "computer_vision",
    },
]


def _fake_vector(dim: int = 768) -> list[float]:
    return [0.01] * dim


def _make_model_mock(dim: int = 768) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() returns proper numpy arrays."""
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.array(
        [_fake_vector(dim) for _ in texts]
    )
    return model


def _make_qdrant_mock(existing_collections: list[str] | None = None) -> MagicMock:
    """Return a mock QdrantClient."""
    client = MagicMock()
    names = existing_collections or []
    # MagicMock treats 'name' as a reserved constructor kwarg (sets repr), not an attribute.
    # SimpleNamespace gives a plain .name attribute that ensure_collection can read.
    coll_objects = [types.SimpleNamespace(name=n) for n in names]
    client.get_collections.return_value = MagicMock(collections=coll_objects)
    return client


def _make_search_hit(text: str, paper_ids: list[str], domain: str, score: float) -> MagicMock:
    hit = MagicMock()
    hit.score = score
    hit.payload = {"limitation_text": text, "paper_ids": paper_ids, "domain": domain}
    return hit


# ---------------------------------------------------------------------------
# get_qdrant_client
# ---------------------------------------------------------------------------


def test_get_qdrant_client_uses_env_vars(monkeypatch):
    """get_qdrant_client should read QDRANT_HOST and QDRANT_PORT from environment."""
    monkeypatch.setenv("QDRANT_HOST", "myhost")
    monkeypatch.setenv("QDRANT_PORT", "9999")

    with patch("vectors.embed.QdrantClient") as mock_cls:
        get_qdrant_client()

    mock_cls.assert_called_once_with(host="myhost", port=9999)


# ---------------------------------------------------------------------------
# load_embedding_model
# ---------------------------------------------------------------------------


def test_load_embedding_model_caches_across_calls(monkeypatch):
    """load_embedding_model should return the same object on repeated calls."""
    embed_mod._model_cache.clear()

    mock_model = MagicMock()
    mock_st = MagicMock(return_value=mock_model)

    with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}):
        first = load_embedding_model()
        second = load_embedding_model()

    assert first is second
    assert mock_st.call_count == 1, "SentenceTransformer constructor called more than once"

    embed_mod._model_cache.clear()


def test_load_embedding_model_loads_correct_model(monkeypatch):
    """load_embedding_model should load allenai/specter2_base."""
    embed_mod._model_cache.clear()

    mock_model = MagicMock()
    mock_st = MagicMock(return_value=mock_model)

    with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}):
        load_embedding_model()

    # Loaded by name, with all loaders pointed at the local HuggingFace hub cache.
    mock_st.assert_called_once()
    args, kwargs = mock_st.call_args
    assert args == ("allenai/specter2_base",)
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    assert kwargs["model_kwargs"] == {"cache_dir": cache_dir}
    assert kwargs["tokenizer_kwargs"] == {"cache_dir": cache_dir}
    assert kwargs["config_kwargs"] == {"cache_dir": cache_dir}
    embed_mod._model_cache.clear()


# ---------------------------------------------------------------------------
# _embed_texts
# ---------------------------------------------------------------------------


def test_embed_texts_returns_one_vector_per_input():
    """_embed_texts should return exactly one vector for each input string."""
    model = _make_model_mock()
    texts = ["first sentence", "second sentence", "third sentence"]
    result = _embed_texts(model, texts)
    assert len(result) == 3


def test_embed_texts_vector_dimension():
    """Each vector returned by _embed_texts should be 768-dimensional."""
    model = _make_model_mock(dim=768)
    result = _embed_texts(model, ["some text"])
    assert len(result[0]) == 768


def test_embed_texts_returns_lists_of_floats():
    """_embed_texts should return Python lists, not numpy arrays."""
    model = _make_model_mock()
    result = _embed_texts(model, ["text"])
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], float)


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


def test_ensure_collection_creates_when_absent():
    """ensure_collection should call create_collection when name is not present."""
    client = _make_qdrant_mock(existing_collections=[])
    ensure_collection(client, "limitations")
    client.create_collection.assert_called_once()
    call_kwargs = client.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == "limitations"


def test_ensure_collection_skips_when_present():
    """ensure_collection should NOT call create_collection if collection already exists."""
    client = _make_qdrant_mock(existing_collections=["limitations"])
    ensure_collection(client, "limitations")
    client.create_collection.assert_not_called()


def test_ensure_collection_is_idempotent():
    """Calling ensure_collection twice for an absent collection should only create it once."""
    client = _make_qdrant_mock(existing_collections=[])

    # After first call the collection "exists" — simulate by updating mock
    def create_side_effect(**kwargs):
        name = kwargs["collection_name"]
        existing = client.get_collections.return_value.collections
        existing.append(types.SimpleNamespace(name=name))

    client.create_collection.side_effect = create_side_effect
    ensure_collection(client, "limitations")
    ensure_collection(client, "limitations")

    assert client.create_collection.call_count == 1


def test_ensure_collection_uses_cosine_768():
    """Created collection should use 768-dim vectors with Cosine distance."""
    from qdrant_client.models import Distance

    client = _make_qdrant_mock(existing_collections=[])
    ensure_collection(client, "limitations")

    _, kwargs = client.create_collection.call_args
    vc = kwargs["vectors_config"]
    assert vc.size == 768
    assert vc.distance == Distance.COSINE


# ---------------------------------------------------------------------------
# _upsert_records
# ---------------------------------------------------------------------------


def test_upsert_records_returns_count():
    """_upsert_records should return the number of points upserted."""
    client = _make_qdrant_mock(existing_collections=["limitations"])
    model = _make_model_mock()
    n = _upsert_records(SAMPLE_RECORDS, client, model, "limitations")
    assert n == len(SAMPLE_RECORDS)


def test_upsert_records_payload_structure():
    """Each upserted PointStruct should carry the correct payload keys."""
    client = _make_qdrant_mock(existing_collections=["limitations"])
    model = _make_model_mock()
    _upsert_records(SAMPLE_RECORDS, client, model, "limitations")

    client.upsert.assert_called_once()
    points = client.upsert.call_args[1]["points"]
    for point in points:
        assert "limitation_text" in point.payload
        assert "paper_ids" in point.payload
        assert "domain" in point.payload
        assert "year" in point.payload


def test_upsert_records_returns_zero_for_empty_input():
    """_upsert_records should return 0 without calling upsert when records list is empty."""
    client = _make_qdrant_mock()
    model = _make_model_mock()
    n = _upsert_records([], client, model, "limitations")
    assert n == 0
    client.upsert.assert_not_called()


def test_upsert_records_vector_per_record():
    """Each point should carry a 768-dim vector."""
    client = _make_qdrant_mock(existing_collections=["limitations"])
    model = _make_model_mock()
    _upsert_records(SAMPLE_RECORDS, client, model, "limitations")

    points = client.upsert.call_args[1]["points"]
    for point in points:
        assert len(point.vector) == 768


# ---------------------------------------------------------------------------
# embed_limitations
# ---------------------------------------------------------------------------


def test_embed_limitations_returns_dict(monkeypatch):
    """embed_limitations should return {'embedded': n, 'collection': 'limitations'}."""
    client = _make_qdrant_mock(existing_collections=["limitations"])
    model = _make_model_mock()

    monkeypatch.setattr(embed_mod, "_query_neo4j_limitations", lambda: SAMPLE_RECORDS)

    result = embed_limitations(client=client, model=model)

    assert result["collection"] == "limitations"
    assert result["embedded"] == len(SAMPLE_RECORDS)


def test_embed_limitations_handles_empty_neo4j(monkeypatch):
    """embed_limitations should return embedded=0 when Neo4j has no Limitation nodes."""
    client = _make_qdrant_mock()
    model = _make_model_mock()

    monkeypatch.setattr(embed_mod, "_query_neo4j_limitations", lambda: [])

    result = embed_limitations(client=client, model=model)

    assert result["embedded"] == 0
    client.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# embed_future_directions
# ---------------------------------------------------------------------------


def test_embed_future_directions_returns_dict(monkeypatch):
    """embed_future_directions should return {'embedded': n, 'collection': 'future_directions'}."""
    client = _make_qdrant_mock(existing_collections=["future_directions"])
    model = _make_model_mock()

    monkeypatch.setattr(embed_mod, "_query_neo4j_future_directions", lambda: SAMPLE_RECORDS)

    result = embed_future_directions(client=client, model=model)

    assert result["collection"] == "future_directions"
    assert result["embedded"] == len(SAMPLE_RECORDS)


def test_embed_future_directions_handles_empty_neo4j(monkeypatch):
    """embed_future_directions should return embedded=0 with no FutureDirection nodes."""
    client = _make_qdrant_mock()
    model = _make_model_mock()

    monkeypatch.setattr(embed_mod, "_query_neo4j_future_directions", lambda: [])

    result = embed_future_directions(client=client, model=model)

    assert result["embedded"] == 0


# ---------------------------------------------------------------------------
# find_similar_limitations
# ---------------------------------------------------------------------------


def test_find_similar_limitations_returns_list(monkeypatch):
    """find_similar_limitations should return a list of dicts with required keys."""
    hits = [_make_search_hit("slow training", ["2301.00234"], "computer_vision", 0.92)]
    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=hits)
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    result = find_similar_limitations("convergence issues")

    assert isinstance(result, list)
    assert len(result) == 1
    for key in ("limitation_text", "paper_ids", "domain", "score"):
        assert key in result[0]


def test_find_similar_limitations_respects_top_k(monkeypatch):
    """find_similar_limitations should pass top_k as limit to Qdrant."""
    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=[])
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    find_similar_limitations("query", top_k=3)

    _, kwargs = client.query_points.call_args
    assert kwargs["limit"] == 3


def test_find_similar_limitations_applies_domain_filter(monkeypatch):
    """find_similar_limitations should pass a domain filter to Qdrant."""
    from qdrant_client.models import Filter

    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=[])
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    find_similar_limitations("query", domain="computer_vision")

    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query_filter"], Filter)


def test_find_similar_limitations_returns_empty_list_gracefully(monkeypatch):
    """find_similar_limitations should return [] when Qdrant returns no hits."""
    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=[])
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    result = find_similar_limitations("obscure query no results")

    assert result == []


def test_find_similar_limitations_score_is_float(monkeypatch):
    """Score field in each result should be a float."""
    hits = [_make_search_hit("memory issue", ["2301.00234"], "computer_vision", 0.87)]
    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=hits)
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    result = find_similar_limitations("memory")

    assert isinstance(result[0]["score"], float)


# ---------------------------------------------------------------------------
# find_similar_future_directions
# ---------------------------------------------------------------------------


def test_find_similar_future_directions_returns_list(monkeypatch):
    """find_similar_future_directions should return a list of result dicts."""
    hits = [_make_search_hit("extend to video", ["2303.05499"], "computer_vision", 0.88)]
    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=hits)
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    result = find_similar_future_directions("video understanding")

    assert isinstance(result, list)
    assert len(result) == 1
    assert "limitation_text" in result[0]


def test_find_similar_future_directions_no_domain_filter(monkeypatch):
    """find_similar_future_directions with domain=None should pass no filter to Qdrant."""
    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=[])
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    find_similar_future_directions("some query", domain=None)

    _, kwargs = client.query_points.call_args
    assert kwargs["query_filter"] is None


def test_find_similar_future_directions_with_domain_filter(monkeypatch):
    """find_similar_future_directions with a domain should apply a Filter."""
    from qdrant_client.models import Filter

    client = _make_qdrant_mock()
    client.query_points.return_value = types.SimpleNamespace(points=[])
    model = _make_model_mock()

    monkeypatch.setattr(search_mod, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(search_mod, "load_embedding_model", lambda: model)

    find_similar_future_directions("query", domain="medical_imaging")

    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query_filter"], Filter)
