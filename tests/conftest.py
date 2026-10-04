"""Suite-wide guard: the unit tier must never reach a real service.

The unit tier is meant to be hermetic, but nothing enforced it. When commit `ad34741`
made `score_gaps` divide by `_count_contributing_papers`, eight tests that still
stubbed the old `_count_papers_in_domain` began opening a real Neo4j driver. They
passed while the production database was running — reading whatever the live corpus
held — and failed when it was not, so the reported pass count depended on which
services happened to be up.

This autouse fixture replaces the three construction points for live resources with
a function that raises `LiveServiceInUnitTest`, for every test *not* marked
`integration`. Each guard sits on the exact binding the unit tests already patch, so
a test that stubs the resource itself (``patch("graph.populate.GraphDatabase.driver")``,
``patch("vectors.embed.QdrantClient")``, ``patch.dict("sys.modules",
{"sentence_transformers": ...})``) overrides the guard for its own duration:

- Neo4j:  ``graph.populate.GraphDatabase.driver`` — the only place a driver is built.
- Qdrant: ``vectors.embed.QdrantClient`` — the only place a client is built.
- Model:  the ``sentence_transformers`` module, which ``load_embedding_model`` imports
  lazily. Guarding the module rather than the function covers every by-value binding
  of ``load_embedding_model`` (``pipeline.gap_scorer``, ``vectors.search``, …) without
  breaking the unit tests of ``load_embedding_model`` itself.

The integration tier (``-m integration``) is untouched: it exists to talk to real
services, and its own conftest isolates them.
"""

import sys
import types

import pytest


class LiveServiceInUnitTest(RuntimeError):
    """A unit test tried to reach a real Neo4j, Qdrant or embedding model."""


def _refuse(service: str, binding: str):
    def guard(*args, **kwargs):
        raise LiveServiceInUnitTest(
            f"unit test touched a real service: {service} via {binding}. "
            "Stub it in the test, or mark the test `integration`."
        )

    return guard


@pytest.fixture(autouse=True)
def _forbid_live_services(request, monkeypatch):
    if request.node.get_closest_marker("integration"):
        yield
        return

    import graph.populate
    import vectors.embed

    monkeypatch.setattr(
        graph.populate.GraphDatabase,
        "driver",
        _refuse("Neo4j", "graph.populate.GraphDatabase.driver"),
    )
    monkeypatch.setattr(
        vectors.embed,
        "QdrantClient",
        _refuse("Qdrant", "vectors.embed.QdrantClient"),
    )

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _refuse(
        "embedding model", "sentence_transformers.SentenceTransformer"
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    # A model cached by an earlier test would bypass the import entirely.
    monkeypatch.setattr(vectors.embed, "_model_cache", {})

    yield
