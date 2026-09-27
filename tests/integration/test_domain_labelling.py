"""Integration proof for PLAN.md item #1 — domain labelling.

These fail against pre-fix code: ``_resolve_domain`` in conftest falls back to
the caller's declared domain when ``pipeline.domains`` does not exist, so the
fixture's three deliberately-wrong declarations survive into the corpus.
"""

import pytest

pytestmark = pytest.mark.integration


def _stored_domains(loaded_corpus) -> dict[str, str]:
    return {p["arxiv_id"]: p["domain"] for p in loaded_corpus["loaded"]}


def test_off_topic_papers_are_excluded_from_the_corpus(loaded_corpus, ground_truth):
    """A paper belonging to neither CV nor MI is a bad search hit, not a domain label.

    9903.00001 (analytic number theory) and 9903.00002 (control theory) mirror the
    six such papers found in the live corpus. Both declare computer_vision.
    """
    off_topic = {pid for pid, dom in ground_truth.items() if dom == "other"}
    stored = set(_stored_domains(loaded_corpus))

    assert off_topic, "fixture must contain at least one off-topic paper to be meaningful"
    leaked = sorted(off_topic & stored)
    assert not leaked, (
        f"off-topic papers were ingested into the corpus: {leaked}. "
        "These are bad search-ingestion hits and must be excluded, not relabelled."
    )


def test_every_stored_paper_carries_its_true_domain(loaded_corpus, ground_truth):
    """A medical paper must not be stored as computer_vision.

    9902.00006 declares computer_vision with unambiguously clinical content —
    the fixture analogue of 2305.17456 in the live corpus, which drove the top
    cross-domain 'discovery' by being a medical sentence labelled CV.
    """
    stored = _stored_domains(loaded_corpus)
    mismatched = {
        pid: (dom, ground_truth[pid]) for pid, dom in stored.items() if dom != ground_truth[pid]
    }
    assert not mismatched, f"stored domain != true domain (stored, true): {mismatched}"


def test_only_known_research_domains_are_stored(loaded_corpus):
    stored = set(_stored_domains(loaded_corpus).values())
    assert stored <= {"computer_vision", "medical_imaging"}, f"unexpected domains: {stored}"


def test_graph_domains_match_sqlite_domains(loaded_corpus):
    """The two stores must agree, since Neo4j drives scoring and SQLite drives /paper."""
    import os
    import sqlite3

    from graph.populate import get_neo4j_driver

    driver = get_neo4j_driver()
    with driver.session(database=loaded_corpus["database"]) as session:
        graph = {
            r["i"]: r["d"]
            for r in session.run("MATCH (p:Paper) RETURN p.arxiv_id AS i, p.domain AS d")
        }
    driver.close()

    conn = sqlite3.connect(loaded_corpus["sqlite_path"])
    rows = dict(conn.execute("SELECT arxiv_id, domain FROM papers").fetchall())

    assert graph == rows, f"graph/SQLite domain drift: graph={graph} sqlite={rows}"


def test_qdrant_payload_domains_match_the_graph(loaded_corpus):
    """Payload domain is copied from Paper.domain at embed time, so it must agree."""
    from vectors.embed import get_qdrant_client

    client = get_qdrant_client()
    points, _ = client.scroll(
        collection_name=loaded_corpus["limitations_collection"], limit=1000, with_payload=True
    )
    payload_domains = {p.payload["domain"] for p in points}
    assert payload_domains <= {"computer_vision", "medical_imaging"}, payload_domains

    expected = {p["domain"] for p in loaded_corpus["loaded"] if p["limitations"]}
    assert payload_domains == expected, f"payload domains {payload_domains} != graph {expected}"
