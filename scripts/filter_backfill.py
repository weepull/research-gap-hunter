"""Apply the Phase 1b extraction filter to the ALREADY-STORED corpus.

Filtering only new ingestions would leave the existing boilerplate in place, and the
existing boilerplate is what actually corrupts current output — `"remains
challenging"` and `"future work includes"` were complete `Limitation` node texts.

**This re-filters stored strings. It never re-extracts.** Re-running extraction on a
paper that already has graph relationships is the documented hazard in CLAUDE.md:
`_upsert_paper_counting` only MERGEs and never removes, and `Limitation` is UNIQUE on
`text`, so a non-deterministic second extraction attaches *new* nodes while the old
ones stay — the paper then reports both wordings and the corpus fills with
near-duplicates that corrupt clustering.

Order of operations matters. Relationships are detached before nodes are deleted,
and Qdrant is rebuilt from the graph afterwards rather than patched, because
`vectors/embed.py` assigns point ids positionally and never deletes.

Usage:
    python scripts/filter_backfill.py --dry-run
    python scripts/filter_backfill.py --apply
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from pipeline.extraction_filter import FIELDS, filter_extractions, log_rejections  # noqa: E402

_REL = {"limitations": ("REPORTS_LIMITATION", "Limitation"),
        "future_directions": ("SUGGESTS_FUTURE", "FutureDirection")}


def _jl(raw):
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or [])
    except (json.JSONDecodeError, TypeError):
        return []


def run(apply: bool) -> int:
    from graph.populate import get_neo4j_driver
    from pipeline.batch import _get_db

    db = _get_db()
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    rows = list(db["papers"].rows)

    plan: dict[str, dict[str, list[str]]] = {}
    all_rejections = []
    reasons = Counter()
    per_field = Counter()

    for row in rows:
        pid = row["arxiv_id"]
        for field in FIELDS:
            items = _jl(row.get(field))
            if not items:
                continue
            kept, rejected = filter_extractions(items, field, pid)
            if rejected:
                plan.setdefault(pid, {})[field] = kept
                all_rejections.extend(rejected)
                for r in rejected:
                    reasons[r.reason] += 1
                    per_field[f"{field}:{r.reason}"] += 1

    total_items = sum(len(_jl(r.get(f))) for r in rows for f in FIELDS)
    print("=" * 92)
    print(f"{'APPLY' if apply else 'DRY RUN — nothing will be written'}")
    print("=" * 92)
    print(f"  papers scanned            : {len(rows)}")
    print(f"  extracted strings scanned : {total_items}")
    print(f"  strings rejected          : {len(all_rejections)} "
          f"({len(all_rejections)/max(1,total_items)*100:.1f}%)")
    print(f"  papers affected           : {len(plan)}")
    print(f"\n  rejections by reason:")
    for reason, n in reasons.most_common():
        print(f"     {reason:<14} {n:>4} ({n/max(1,len(all_rejections))*100:.0f}%)")
    print(f"\n  rejections by field and reason:")
    for key, n in sorted(per_field.items()):
        print(f"     {key:<34} {n:>4}")
    print(f"\n  every rejected string:")
    for r in sorted(all_rejections, key=lambda r: (r.field, r.reason, r.text)):
        print(f"     [{r.field[:3]}/{r.reason:<11}] {r.arxiv_id}  {r.text[:72]!r}")

    if not apply:
        print("\nDry run complete. Re-run with --apply to write.")
        return 0

    # --- 1. SQLite ---------------------------------------------------------
    for pid, fields in plan.items():
        for field, kept in fields.items():
            db.execute(
                f"UPDATE papers SET {field} = ? WHERE arxiv_id = ?",
                [json.dumps(kept), pid],
            )
    db.conn.commit()
    print(f"\nSQLite: updated {sum(len(f) for f in plan.values())} field(s) "
          f"across {len(plan)} paper(s)")

    # --- 2. Neo4j: detach the rejected relationships -----------------------
    driver = get_neo4j_driver()
    detached = 0
    for r in all_rejections:
        rel, label = _REL[r.field]
        with driver.session(database=database) as session:
            summary = session.run(
                f"""
                MATCH (p:Paper {{arxiv_id: $pid}})-[r:{rel}]->(n:{label} {{text: $text}})
                DELETE r
                """,
                pid=r.arxiv_id, text=r.text,
            ).consume()
            detached += summary.counters.relationships_deleted
    print(f"Neo4j: detached {detached} relationship(s)")

    # --- 3. delete nodes left with no paper --------------------------------
    with driver.session(database=database) as session:
        orphans = session.run(
            """
            MATCH (n)
            WHERE (n:Limitation OR n:FutureDirection)
              AND NOT (:Paper)-->(n)
            DETACH DELETE n
            """
        ).consume()
    print(f"Neo4j: deleted {orphans.counters.nodes_deleted} orphaned node(s)")
    driver.close()

    # --- 4. audit log ------------------------------------------------------
    written = log_rejections(all_rejections)
    print(f"Logged {written} rejection(s) to data/rejected_extractions.jsonl")

    # --- 5. rebuild the vector store --------------------------------------
    from vectors.embed import (
        _COLLECTION_FUTURE_DIRECTIONS, _COLLECTION_LIMITATIONS,
        embed_future_directions, embed_limitations, get_qdrant_client,
    )
    client = get_qdrant_client()
    for collection in (_COLLECTION_LIMITATIONS, _COLLECTION_FUTURE_DIRECTIONS):
        try:
            client.delete_collection(collection)
            print(f"dropped Qdrant collection {collection!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"could not drop {collection!r} ({exc.__class__.__name__}); recreating")
    lim = embed_limitations()
    fut = embed_future_directions()
    print(f"re-embedded {lim['embedded']} limitations, {fut['embedded']} future directions")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return run(apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
