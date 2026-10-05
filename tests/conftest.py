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

- SQLite: ``sqlite3.connect`` to any file outside pytest's temp root. Every unit test
  also gets ``SELFHEAL_ON_STARTUP=false`` and ``pipeline.batch._DB_PATH`` under
  ``tmp_path``, so the API lifespan never opens the real ``data/papers.db``.

Refusals are recorded as well as raised, and the test fails at teardown if any occurred,
because the API swallows backend errors in several places and a raise alone was invisible.

The integration tier (``-m integration``) is untouched: it exists to talk to real
services, and its own conftest isolates them.
"""

import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest


class LiveServiceInUnitTest(RuntimeError):
    """A unit test tried to reach a real Neo4j, Qdrant, embedding model or SQLite file."""


def _refuse(service: str, binding: str, violations: list[str]):
    def guard(*args, **kwargs):
        message = (
            f"unit test touched a real service: {service} via {binding}. "
            "Stub it in the test, or mark the test `integration`."
        )
        violations.append(message)
        raise LiveServiceInUnitTest(message)

    return guard


@pytest.fixture(autouse=True)
def _forbid_live_services(request, monkeypatch, tmp_path, tmp_path_factory):
    if request.node.get_closest_marker("integration"):
        yield
        return

    import graph.populate
    import vectors.embed

    # Every refusal is recorded as well as raised, and the test fails at teardown if
    # any were recorded. Raising alone is not enough: the API lifespan deliberately
    # swallows backend failures (`_warm`, and the self-heal `except Exception`), so a
    # refused connection there was logged and the test still passed.
    violations: list[str] = []

    monkeypatch.setattr(
        graph.populate.GraphDatabase,
        "driver",
        _refuse("Neo4j", "graph.populate.GraphDatabase.driver", violations),
    )
    monkeypatch.setattr(
        vectors.embed,
        "QdrantClient",
        _refuse("Qdrant", "vectors.embed.QdrantClient", violations),
    )

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _refuse(
        "embedding model", "sentence_transformers.SentenceTransformer", violations
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    # A model cached by an earlier test would bypass the import entirely.
    monkeypatch.setattr(vectors.embed, "_model_cache", {})

    # SQLite: only in-memory databases and files under pytest's temp root. The root
    # (not this test's own tmp_path) is the boundary because every tmp_path lives
    # under it, and some tests build their database in a fixture's directory.
    allowed_root = Path(tmp_path_factory.getbasetemp()).resolve()
    real_connect = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        name = os.fsdecode(database) if isinstance(database, (bytes, os.PathLike)) else database
        in_memory = name in ("", ":memory:") or name.startswith("file::memory:") or (
            name.startswith("file:") and "mode=memory" in name
        )
        if not in_memory:
            path = Path(name.removeprefix("file:").split("?", 1)[0]).resolve()
            if not path.is_relative_to(allowed_root):
                _refuse(f"SQLite at {path}", "sqlite3.connect", violations)()
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)

    # Keep the unit tier off the real paper store, by two independent means. The
    # startup self-heal is a write path into SQLite and has no business running
    # against a mocked graph; and any code that does open the store gets a throwaway
    # file. Either alone leaves a hole: disabling self-heal does not cover /paper or
    # /corpus, and redirecting the path still lets self-heal write into it.
    import pipeline.batch

    monkeypatch.setenv("SELFHEAL_ON_STARTUP", "false")
    monkeypatch.setattr(pipeline.batch, "_DB_PATH", tmp_path / "papers.db")

    yield

    if violations:
        pytest.fail("\n".join(dict.fromkeys(violations)), pytrace=False)


# ---------------------------------------------------------------------------
# arXiv stub for ingestion tests (P2)
# ---------------------------------------------------------------------------

# Accepted by BOTH domain rules, so one stub serves every ingestion test: `cs.CV` is
# accepted outright for computer_vision, and the clinical title makes the medical
# keyword verifier fire, which medical_imaging requires of a `cs.CV` primary.
ADMISSIBLE_TITLE = "Deep learning segmentation of tumours on chest CT and MRI scans for radiology"


def arxiv_reports(monkeypatch, primary_category: str, title: str = ADMISSIBLE_TITLE):
    """Make `pipeline.arxiv_source.fetch_by_ids` report every requested id with this
    primary category, offline. Returns the list of id-lists it was asked for."""
    import re

    from pipeline import arxiv_source

    calls: list[list[str]] = []

    def fake_fetch_by_ids(arxiv_ids):
        calls.append(list(arxiv_ids))
        return {
            re.sub(r"v\d+$", "", i): arxiv_source.ArxivCandidate(
                arxiv_id=re.sub(r"v\d+$", "", i),
                title=title,
                abstract="",
                primary_category=primary_category,
                categories=(primary_category,),
                published="2025-01-01T00:00:00Z",
            )
            for i in arxiv_ids
        }

    monkeypatch.setattr(arxiv_source, "fetch_by_ids", fake_fetch_by_ids)
    return calls


@pytest.fixture()
def arxiv_admits(monkeypatch):
    """arXiv reports every id as admissible to either domain. For the existing
    ingestion tests, which are about what happens *after* admission."""
    return arxiv_reports(monkeypatch, "cs.CV")
