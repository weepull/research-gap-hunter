"""Fixtures for the integration tier.

This tier runs the *real* pipeline — real Specter2 weights, real Qdrant queries,
real Neo4j Cypher — because every defect the unit suite misses lives in the
interaction between real components and real data. 327 fully-mocked unit tests
run in 0.78s and cannot, even in principle, observe a cluster swallowing half a
domain or a threshold sitting below its own noise floor.

Three safety properties, all load-bearing:

1. **Isolation.** Neo4j work happens in a *separate database* (default
   ``rghintegration``, override with ``NEO4J_TEST_DATABASE``) and Qdrant work in
   *separate collections* (``*_integration``). A test run can therefore never
   touch the real corpus. The isolation is asserted, not assumed: if the test
   database resolves to the same name as the production one, collection ensures
   fail loudly rather than proceeding.

2. **Skip, never fail, when services are down.** A contributor without Docker
   running should see ``skipped``, not a wall of red that says nothing about
   their change. Probes are session-scoped so the cost is paid once.

3. **A fixed fixture corpus, not the live one.** Integration tests against the
   live corpus would be non-deterministic and would silently change meaning on
   every ingestion. ``fixtures/corpus.json`` is hand-labelled and carries both a
   ``true_domain`` (ground truth) and a ``declared_domain`` (what the pre-fix
   pipeline stores, deliberately wrong for three entries).

Note on module-level constant patching: ``pipeline.gap_scorer``,
``pipeline.cross_domain`` and ``vectors.search`` each do
``from vectors.embed import _COLLECTION_LIMITATIONS``, which binds the *value*
at import. Patching only ``vectors.embed`` would leave those three pointing at
the production collections, so every namespace that holds a binding is patched.
"""

import importlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "corpus.json"

# Namespaces holding a by-value binding of the collection-name constants.
_COLLECTION_NAMESPACES = (
    "vectors.embed",
    "vectors.search",
    "pipeline.gap_scorer",
    "pipeline.cross_domain",
)

_TEST_LIMITATIONS = "limitations_integration"
_TEST_FUTURE_DIRECTIONS = "future_directions_integration"


def _test_database() -> str:
    return os.getenv("NEO4J_TEST_DATABASE", "rghintegration").strip() or "rghintegration"


@contextmanager
def _isolated_stores(db_path=None):
    """Point every store binding at the isolated test locations, then restore.

    A context manager rather than fixture-level patching, because scope matters
    here in a way that bit this harness once already. A session-scoped fixture's
    finalizer runs at *end of session*, and `tests/integration/` sorts before
    `tests/test_*.py` — so patching module globals in the session fixture leaked
    the test collection names into the unit suite, and `pytest -m ''` failed three
    tests that passed in either tier alone. This is applied around the corpus load
    and again per test (see _isolate_collections), so nothing survives the
    integration package.
    """
    previous_env = os.environ.get("NEO4J_DATABASE")
    os.environ["NEO4J_DATABASE"] = _test_database()

    import pipeline.batch as batch

    previous_db_path = batch._DB_PATH
    if db_path is not None:
        batch._DB_PATH = db_path

    patched: list[tuple[object, str, object]] = []
    for name in _COLLECTION_NAMESPACES:
        module = importlib.import_module(name)
        for attr, value in (
            ("_COLLECTION_LIMITATIONS", _TEST_LIMITATIONS),
            ("_COLLECTION_FUTURE_DIRECTIONS", _TEST_FUTURE_DIRECTIONS),
        ):
            if hasattr(module, attr):
                patched.append((module, attr, getattr(module, attr)))
                setattr(module, attr, value)
    try:
        yield
    finally:
        for module, attr, value in patched:
            setattr(module, attr, value)
        batch._DB_PATH = previous_db_path
        if previous_env is None:
            os.environ.pop("NEO4J_DATABASE", None)
        else:
            os.environ["NEO4J_DATABASE"] = previous_env


@pytest.fixture(autouse=True)
def _isolate_collections(request):
    """Re-apply store isolation for the duration of each integration test.

    Autouse and function-scoped so it unwinds after every test. Skipped for tests
    that never touch a store, so a pure-unit assertion in this package does not
    pay for it.
    """
    if "loaded_corpus" not in request.fixturenames:
        yield
        return
    db_path = getattr(request.config, "_rgh_integration_db", None)
    with _isolated_stores(db_path):
        yield


# ---------------------------------------------------------------------------
# Service probes — session scoped, skip (never fail) when a service is absent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def neo4j_available() -> str:
    """The isolated test database name, or skip if Neo4j is unreachable."""
    production = os.getenv("NEO4J_DATABASE", "neo4j")
    test_db = _test_database()
    if test_db == production:
        pytest.fail(
            f"NEO4J_TEST_DATABASE ({test_db!r}) must differ from NEO4J_DATABASE "
            f"({production!r}) — integration tests must never write to the real corpus."
        )
    try:
        from graph.populate import get_neo4j_driver

        driver = get_neo4j_driver()
    except Exception as exc:  # noqa: BLE001 — absence is a skip, not a failure
        pytest.skip(f"Neo4j unreachable ({exc.__class__.__name__}); start it to run this tier")
    try:
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE {test_db} IF NOT EXISTS WAIT")
    except Exception as exc:  # noqa: BLE001
        driver.close()
        pytest.skip(f"cannot create isolated database {test_db!r}: {exc}")
    driver.close()
    return test_db


@pytest.fixture(scope="session")
def qdrant_available():
    try:
        from vectors.embed import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant unreachable ({exc.__class__.__name__}); start it to run this tier")
    return client


@pytest.fixture(scope="session")
def ollama_available():
    """Only /explain needs Ollama; kept separate so the rest of the tier runs without it."""
    import urllib.error
    import urllib.request

    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=5).read()
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"Ollama unreachable ({exc.__class__.__name__}); start it to run this test")
    return base


@pytest.fixture(scope="session")
def embedding_model(qdrant_available):
    """Load Specter2 once for the whole session — it is ~440MB of weights."""
    from vectors.embed import load_embedding_model

    return load_embedding_model()


# ---------------------------------------------------------------------------
# Fixture corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixture_papers() -> list[dict]:
    return json.loads(_FIXTURE.read_text())["papers"]


def _resolve_domain(paper: dict) -> str | None:
    """Decide what domain this paper is stored under, the way ingestion should.

    Delegates to the production curation helper when it exists, so this harness
    exercises the real capability rather than a test-local reimplementation.
    Before that helper lands it falls back to the caller's declaration, which is
    exactly the pre-fix behaviour — which is why the ground-truth assertions in
    test_domain_labelling.py fail against pre-fix code.

    Returns None when the paper does not belong in the corpus at all.
    """
    try:
        from pipeline.domains import curate_declared_domain
    except ImportError:
        return paper["declared_domain"]
    return curate_declared_domain(
        declared=paper["declared_domain"],
        title=paper["title"],
        abstract=paper.get("abstract", ""),
    )


@pytest.fixture(scope="session")
def loaded_corpus(
    request, neo4j_available, qdrant_available, embedding_model, fixture_papers,
    tmp_path_factory,
):
    """Load the fixture corpus into the isolated stores through the production path.

    Uses ``_paper_to_row`` and ``_upsert_paper_counting`` — the same functions real
    ingestion uses — so schema and serialisation bugs surface here rather than being
    mocked away. Yields a dict describing what was loaded.

    Store isolation is applied *around the load only*; each test re-applies it via
    the autouse ``_isolate_collections`` fixture. See _isolated_stores for why.
    """
    import sqlite_utils
    from qdrant_client.http.exceptions import UnexpectedResponse

    import pipeline.batch as batch
    from graph.populate import _upsert_paper_counting, create_constraints, get_neo4j_driver
    from vectors.embed import embed_future_directions, embed_limitations

    test_db = neo4j_available
    client = qdrant_available

    db_path = tmp_path_factory.mktemp("integration-sqlite") / "papers.db"
    # Stashed on the config so the per-test fixture can re-apply the same path.
    request.config._rgh_integration_db = db_path

    with _isolated_stores(db_path):
        # --- start from empty stores ---------------------------------------
        for collection in (_TEST_LIMITATIONS, _TEST_FUTURE_DIRECTIONS):
            try:
                client.delete_collection(collection)
            except (UnexpectedResponse, ValueError):
                pass

        driver = get_neo4j_driver()
        with driver.session(database=test_db) as session:
            session.run("MATCH (n) DETACH DELETE n")
        create_constraints(driver)

        # --- load -----------------------------------------------------------
        sqlite_db = sqlite_utils.Database(db_path)
        loaded: list[dict] = []
        excluded: list[str] = []
        for paper in fixture_papers:
            domain = _resolve_domain(paper)
            if domain is None:
                excluded.append(paper["arxiv_id"])
                continue
            record = {
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "year": paper["year"],
                "domain": domain,
                "objectives": paper["objectives"],
                "methods": paper["methods"],
                "datasets": paper["datasets"],
                "evaluation_metrics": paper["evaluation_metrics"],
                "limitations": paper["limitations"],
                "future_directions": paper["future_directions"],
                "raw_json": json.dumps({"source": "integration_fixture"}),
                "ingested_at": "2026-01-01T00:00:00+00:00",
                "extraction_tier": paper["extraction_tier"],
            }
            row = dict(record)
            for field in batch._LIST_FIELDS:
                row[field] = json.dumps(row[field])
            sqlite_db["papers"].insert(row, pk="arxiv_id", replace=True, alter=True)
            with driver.session(database=test_db) as session:
                session.execute_write(lambda tx, p=record: _upsert_paper_counting(tx, p))
            loaded.append(record)
        driver.close()

        embed_limitations()
        embed_future_directions()

        info = {
            "loaded": loaded,
            "excluded": excluded,
            "database": test_db,
            "sqlite_path": db_path,
            "limitations_collection": _TEST_LIMITATIONS,
            "future_directions_collection": _TEST_FUTURE_DIRECTIONS,
        }

    yield info

    # --- teardown: drop the test collections and empty the test database ----
    for collection in (_TEST_LIMITATIONS, _TEST_FUTURE_DIRECTIONS):
        try:
            client.delete_collection(collection)
        except Exception:  # noqa: BLE001 — teardown must not mask a test failure
            pass
    try:
        from graph.populate import get_neo4j_driver as _driver

        driver = _driver()
        with driver.session(database=test_db) as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="session")
def ground_truth(fixture_papers) -> dict[str, str]:
    return {p["arxiv_id"]: p["true_domain"] for p in fixture_papers}
