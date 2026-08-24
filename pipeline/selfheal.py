"""Startup reconciliation between the Neo4j graph and the SQLite paper store.

Neo4j is the scoring source of truth: `_count_papers_in_domain()` counts Paper
nodes, and the discovery engine reads limitations and future directions from the
graph. SQLite holds the full paper records that `/paper/{arxiv_id}` serves.

Those two stores can drift, and the drift is one-directional in practice —
SQLite loses rows that Neo4j still has. It happened once already (documented in
CLAUDE.md: 63 Paper nodes against 44 SQLite rows, which made 13 of 27 gaps cite
at least one paper whose detail endpoint 404'd), and an ephemeral container
filesystem would reproduce it on every redeploy.

The fix has two layers, and this module is the second one:

  1. **Persistence.** `PAPERS_DB_PATH` points the database at a mounted volume
     so the file survives a redeploy.
  2. **Self-heal.** This module. On startup it compares the two stores and
     rebuilds any rows SQLite is missing, straight from graph relationships.

**Which of those is primary depends on the platform, and on the current
deployment target it is this one.** Render's free tier has no persistent disk at
all, and free services spin down after 15 minutes idle and cold-start on the
next request. So there is no volume for layer 1 to point at, SQLite starts empty
on every boot, and this module rebuilds all of it from Neo4j each time. That is
the intended design on that platform, not a workaround: Neo4j is already the
scoring source of truth, so treating SQLite as a derived cache rebuilt from it
is the honest arrangement — and it is what makes a disk-less free tier viable.

The WARNING this logs is therefore **routine on Render and alarming anywhere
else**. On a platform with a mounted volume, a rebuild means the volume did not
do its job and is worth investigating. On Render free it means the container
restarted, which is expected. The log text names the volume because that is the
actionable case; read it with the platform in mind.

**Reconstruction, not re-extraction.** Rows are rebuilt from what the graph
already holds — no LLM call, no network call, fully deterministic. Re-running
the normal ingestion path here would be actively harmful:
`graph/populate.py:_upsert_paper_counting()` only ever MERGEs and never removes
relationships, and `Limitation` nodes are keyed on exact `text`. Ollama
extraction is non-deterministic, so re-extracting a paper that already has
limitations would attach *new* Limitation nodes while the old ones stayed,
leaving that paper reporting both wordings and inflating the corpus with
near-duplicates that then corrupt clustering.

**Known lossy fields.** Neo4j never stored `objectives`, `evaluation_metrics` or
`raw_json`, so those come back empty on a rebuilt row. `raw_json` instead records
its own provenance, so a rebuilt row is visibly reconstructed rather than looking
like a failed extraction.
"""

import json
import logging
import os
from datetime import datetime, timezone

from pipeline.batch import _get_db, _paper_to_row
from pipeline.extractor import PaperExtract

logger = logging.getLogger(__name__)

# Marks a row as rebuilt rather than extracted. Kept as a module constant
# because tests and any future audit query both need to match on it.
RECONSTRUCTION_SOURCE = "reconstructed_from_neo4j"

# Papers per Cypher round trip. The corpus is ~130 papers today, so this is one
# query; the chunking is here so a grown corpus does not build a huge IN list.
_CHUNK = 200

_MISSING_FIELDS_NOTE = (
    "objectives, evaluation_metrics and raw_json are not stored in Neo4j and "
    "cannot be recovered without re-extracting the paper"
)

# Pattern comprehensions rather than several OPTIONAL MATCHes: the latter build
# a cartesian product across limitations x directions x methods x datasets,
# which for a well-populated paper is hundreds of rows to collect(DISTINCT) back
# down again.
_FETCH_QUERY = """
MATCH (p:Paper)
WHERE p.arxiv_id IN $ids
RETURN p.arxiv_id AS arxiv_id,
       p.title AS title,
       p.year AS year,
       p.domain AS domain,
       [(p)-[:REPORTS_LIMITATION]->(l) | l.text] AS limitations,
       [(p)-[:SUGGESTS_FUTURE]->(f) | f.text] AS future_directions,
       [(p)-[:USES_METHOD]->(m) | m.name] AS methods,
       [(p)-[:USES_DATASET]->(d) | d.name] AS datasets,
       [(p)-[r:REPORTS_LIMITATION]->() | r.tier] AS tiers
"""


def _database() -> str:
    """The Neo4j database name.

    The Neo4j Desktop *instance* name (rgh-mvp) is not the database name; the
    database is 'neo4j' unless NEO4J_DATABASE says otherwise.
    """
    return os.getenv("NEO4J_DATABASE", "neo4j")


def selfheal_enabled() -> bool:
    """Whether startup reconciliation should run. On by default.

    A kill switch exists because this is a write path that runs before the app
    serves traffic: if it ever misbehaves in a deployment, turning it off must
    not require a code change.
    """
    return os.getenv("SELFHEAL_ON_STARTUP", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def graph_paper_ids(driver) -> set[str]:
    """Every arxiv_id with a Paper node."""
    with driver.session(database=_database()) as session:
        result = session.run("MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id")
        return {r["arxiv_id"] for r in result if r["arxiv_id"]}


def sqlite_paper_ids(db=None) -> set[str]:
    """Every arxiv_id with a SQLite row. Empty set if the table does not exist."""
    db = db if db is not None else _get_db()
    if "papers" not in db.table_names():
        return set()
    return {
        row["arxiv_id"]
        for row in db["papers"].rows_where(select="arxiv_id")
        if row.get("arxiv_id")
    }


def _row_to_extract(record: dict) -> PaperExtract:
    """Build a PaperExtract from one graph record.

    Tier lives on the REPORTS_LIMITATION relationship rather than on the Paper
    node, so it is read from the first relationship that carries one. A paper
    with no limitations has no tier to recover and falls back to the model
    default.
    """
    tiers = [t for t in (record.get("tiers") or []) if t]
    tier = tiers[0] if tiers else "explicit"

    provenance = {
        "source": RECONSTRUCTION_SOURCE,
        "reconstructed_at": datetime.now(timezone.utc).isoformat(),
        "note": _MISSING_FIELDS_NOTE,
    }

    return PaperExtract(
        arxiv_id=record["arxiv_id"],
        title=record.get("title") or "",
        year=record.get("year") or 0,
        domain=record.get("domain") or "computer_vision",
        # Not recoverable from the graph — see the module docstring.
        objectives=[],
        evaluation_metrics=[],
        methods=[m for m in (record.get("methods") or []) if m],
        datasets=[d for d in (record.get("datasets") or []) if d],
        limitations=[limitation for limitation in (record.get("limitations") or []) if limitation],
        future_directions=[f for f in (record.get("future_directions") or []) if f],
        raw_json=json.dumps(provenance),
        ingested_at=datetime.now(timezone.utc).isoformat(),
        extraction_tier=tier,
    )


def fetch_papers_from_graph(driver, arxiv_ids: list[str]) -> list[PaperExtract]:
    """Reconstruct PaperExtracts for the given ids from graph relationships."""
    if not arxiv_ids:
        return []

    papers: list[PaperExtract] = []
    with driver.session(database=_database()) as session:
        for start in range(0, len(arxiv_ids), _CHUNK):
            chunk = arxiv_ids[start : start + _CHUNK]
            for record in session.run(_FETCH_QUERY, ids=chunk):
                data = dict(record)
                if not data.get("arxiv_id"):
                    continue
                papers.append(_row_to_extract(data))
    return papers


def reconcile_sqlite_from_graph(driver, db=None) -> dict:
    """Rebuild SQLite rows that Neo4j has and SQLite does not.

    Returns a summary dict; never raises for an individual bad row, so one
    unparseable paper cannot stop the rest of the corpus being repaired.

    Only ever *adds* rows. Existing rows are left untouched — a row already in
    SQLite is richer than anything the graph can reconstruct, since the graph
    never stored objectives, evaluation_metrics or raw_json.
    """
    db = db if db is not None else _get_db()

    in_graph = graph_paper_ids(driver)
    in_sqlite = sqlite_paper_ids(db)
    missing = sorted(in_graph - in_sqlite)
    # Reported for visibility only. Rows SQLite has that the graph does not are
    # not an error — /paper/ still serves them — and deleting them would change
    # nothing about scoring while destroying data.
    orphaned = sorted(in_sqlite - in_graph)

    summary = {
        "graph_papers": len(in_graph),
        "sqlite_papers": len(in_sqlite),
        "missing_from_sqlite": len(missing),
        "rebuilt": 0,
        "failed": 0,
        "sqlite_only": len(orphaned),
    }

    if not missing:
        logger.info(
            "Paper stores agree: %d in Neo4j, %d in SQLite. No repair needed.",
            len(in_graph),
            len(in_sqlite),
        )
        return summary

    # Loud on purpose. Reaching this branch means the persistent volume did not
    # do its job, and that is worth investigating even though the app recovers.
    logger.warning(
        "Paper store drift detected: Neo4j has %d papers, SQLite has %d. "
        "Rebuilding %d missing row(s) from graph relationships. This should not "
        "happen when PAPERS_DB_PATH points at a persistent volume — check that "
        "the volume is mounted.",
        len(in_graph),
        len(in_sqlite),
        len(missing),
    )

    papers = fetch_papers_from_graph(driver, missing)
    table = db["papers"]
    for paper in papers:
        try:
            # replace=False: never clobber an existing richer row. alter=True so
            # a store created before extraction_tier existed gains the column
            # rather than failing the insert.
            table.insert(_paper_to_row(paper), pk="arxiv_id", replace=False, alter=True)
            summary["rebuilt"] += 1
        except Exception:  # noqa: BLE001 — one bad row must not stop the repair
            summary["failed"] += 1
            logger.exception("Could not rebuild SQLite row for %s", paper.arxiv_id)

    logger.warning(
        "Paper store repair complete: %d rebuilt, %d failed. Rebuilt rows have "
        "empty objectives/evaluation_metrics and a raw_json marked %r.",
        summary["rebuilt"],
        summary["failed"],
        RECONSTRUCTION_SOURCE,
    )
    return summary
