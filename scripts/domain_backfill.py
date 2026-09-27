"""One-off corpus curation for PLAN.md #1 — reviewed 2026-09-28.

## What this does and why it is a manifest rather than a computation

`extract_paper()` hardcoded `domain="computer_vision"` and `ingest_from_query()`
never overrode it, so for the whole history of this corpus a paper's domain
recorded *which script ingested it*. A Semantic Scholar keyword search returns
whatever matches the words, and nothing downstream checked, so the 46-paper
"computer vision" corpus contained analytic number theory, atomic physics,
control theory, survey astronomy, music generation, domain-specific NLP, vehicle
motion planning, two language models and a copyright-law paper — plus four
genuine medical-imaging papers.

Option 1A stops new mislabelling. It does nothing for the 127 rows already
stored, which is what this script is for.

**The determinations below were reviewed paper by paper against real titles and
extracted metadata; they are not the keyword classifier's raw output.** That
matters: `pipeline.domains.classify_text` flagged 19 papers as matching neither
domain, and several of those (Flickr30k Entities, SPHINX, PaliGemma, streaming
video understanding, 3D shape matching) are perfectly good computer vision whose
vocabulary the term lists simply do not cover. Trusting it blindly would have
deleted real papers. That brittleness is exactly why the advisor restricted the
heuristic to a non-blocking verifier at ingestion time.

Recording the outcome as an explicit manifest, rather than recomputing it, means
this historical curation is auditable and cannot silently change if those term
lists are ever edited.

## Why papers are removed rather than relabelled

A paper belonging to neither computer vision nor medical imaging is not a domain
label problem — it is a bad search-ingestion hit. Storing it under a third label
would keep it in the corpus, where it would continue to dilute the frequency
denominator of whichever domain it sat in. Removal is the advisor's decision.

## Why this rebuilds Qdrant instead of patching it

`vectors/embed.py:_upsert_records` assigns point ids positionally (`id=i`) over a
Cypher result with **no ORDER BY**, and only ever upserts — nothing deletes. A
corpus that shrinks therefore leaves orphaned points behind, holding stale text
and a stale `domain` payload, still matching queries. Since this curation removes
11 papers, both collections are dropped and rebuilt from the graph rather than
patched in place.

Usage:
    python scripts/domain_backfill.py --dry-run     # report, change nothing
    python scripts/domain_backfill.py --apply
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# The reviewed manifest
# --------------------------------------------------------------------------

# Papers that belong to neither computer vision nor medical imaging. Each is a
# bad search-ingestion hit; the reason is recorded so the call is reviewable.
REMOVE: dict[str, str] = {
    "2309.10305": "Baichuan 2 — a large language model, no vision component",
    "2310.06825": "Mistral 7B — a large language model, no vision component",
    "2401.00920": "Bose-Einstein condensate interference — atomic physics",
    "2402.14135": "eROSITA steep-spectrum AGN survey — observational astronomy",
    "2405.00658": "divisor-bounded multiplicative functions — analytic number theory",
    "2406.14526": "copyrighted-character generation — copyright law analysis",
    "2408.12247": "domain-specific Q&A fine-tuning — NLP, no imaging",
    "2409.00587": "FLUX that Plays Music — audio/music generation",
    "2412.01234": "autonomous-vehicle planning — motion planning and control, not perception",
    "2501.03894": "robust moving-horizon estimation — control theory",
    "2502.01618": "inference-time scaling of LLMs — language-model inference",
}

# Genuine medical-imaging papers stored as computer_vision.
RELABEL: dict[str, tuple[str, str]] = {
    "2305.17456": ("medical_imaging", "Trustworthy Deep Learning for Medical Image Segmentation"),
    "2307.15872": ("medical_imaging", "Cross-dimensional transfer learning in medical image segmentation"),
    "2409.03367": ("medical_imaging", "TBConvL-Net — robust medical image segmentation"),
    "2501.16469": ("medical_imaging", "Object Detection for Medical Image Analysis (RT-DETR)"),
}

# Borderline papers examined and deliberately KEPT as computer_vision, recorded so
# a future session does not re-litigate them or read the classifier's flags as
# unactioned findings.
KEPT_AFTER_REVIEW: dict[str, str] = {
    "1505.04870": "Flickr30k Entities — region-to-phrase grounding in images",
    "2207.10077": "DebiAN — bias mitigation on image classification (Multi-Color MNIST)",
    "2304.09148": "SAM-Adapter — general SAM adaptation (camouflage, shadow); medical is one case among several",
    "2311.07575": "SPHINX — multimodal LLM built on visual embeddings",
    "2401.13601": "MM-LLMs survey — vision-language models, consistent with the other VLM papers here",
    "2405.16009": "streaming long-video understanding — video vision",
    "2407.07726": "PaliGemma — vision-language model",
    "2409.13112": "construction-waste analysis — explicitly an applied computer-vision review",
    "2411.03511": "3D shape surface matching benchmark — geometric vision",
    "2510.16295": "OpenLVLM-MIA — membership inference on large vision-language models",
}


def _print_manifest() -> None:
    print(f"REMOVE ({len(REMOVE)}) — belong to neither research domain:")
    for pid, reason in sorted(REMOVE.items()):
        print(f"    {pid}  {reason}")
    print(f"\nRELABEL ({len(RELABEL)}) — computer_vision -> medical_imaging:")
    for pid, (domain, reason) in sorted(RELABEL.items()):
        print(f"    {pid}  -> {domain}   {reason}")
    print(f"\nKEPT after review ({len(KEPT_AFTER_REVIEW)}) — borderline, deliberately unchanged:")
    for pid, reason in sorted(KEPT_AFTER_REVIEW.items()):
        print(f"    {pid}  {reason}")


def _neo4j_database() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


def run(apply: bool) -> int:
    from graph.populate import get_neo4j_driver
    from pipeline.batch import _get_db
    from pipeline.domains import RESEARCH_DOMAINS
    from vectors.embed import (
        _COLLECTION_FUTURE_DIRECTIONS,
        _COLLECTION_LIMITATIONS,
        embed_future_directions,
        embed_limitations,
        get_qdrant_client,
    )

    _print_manifest()
    mode = "APPLY" if apply else "DRY RUN — nothing will be written"
    print(f"\n=== {mode} ===\n")

    db = _get_db()
    driver = get_neo4j_driver()
    database = _neo4j_database()

    with driver.session(database=database) as session:
        before = {
            r["d"]: r["n"]
            for r in session.run(
                "MATCH (p:Paper) RETURN p.domain AS d, count(*) AS n ORDER BY d"
            )
        }
    sqlite_before = dict(
        db.execute("SELECT domain, count(*) FROM papers GROUP BY domain").fetchall()
    )
    print(f"before — neo4j: {before}   sqlite: {sqlite_before}")

    present = {row[0] for row in db.execute("SELECT arxiv_id FROM papers").fetchall()}
    missing = (set(REMOVE) | set(RELABEL)) - present
    if missing:
        print(f"\nWARNING: manifest ids absent from SQLite (already curated?): {sorted(missing)}")

    if not apply:
        print("\nDry run complete. Re-run with --apply to write.")
        driver.close()
        return 0

    # --- 1. remove off-topic papers ---------------------------------------
    removed = 0
    for pid in sorted(REMOVE):
        with driver.session(database=database) as session:
            summary = session.run(
                "MATCH (p:Paper {arxiv_id: $id}) DETACH DELETE p", id=pid
            ).consume()
            removed += summary.counters.nodes_deleted
        db.execute("DELETE FROM papers WHERE arxiv_id = ?", [pid])
    db.conn.commit()
    print(f"removed {removed} Paper nodes and {len(REMOVE)} SQLite rows")

    # --- 2. delete nodes orphaned by that removal -------------------------
    # Limitation/FutureDirection are keyed on exact text and Method/Dataset on
    # name, so a node can survive with no Paper attached. Scoring joins through
    # Paper and would ignore them, but leaving them is cruft that misreports the
    # graph's size.
    with driver.session(database=database) as session:
        orphans = session.run(
            """
            MATCH (n)
            WHERE (n:Limitation OR n:FutureDirection OR n:Method OR n:Dataset)
              AND NOT (:Paper)-->(n)
            DETACH DELETE n
            """
        ).consume()
    print(f"deleted {orphans.counters.nodes_deleted} orphaned nodes")

    # --- 3. relabel mislabelled papers ------------------------------------
    for pid, (domain, _reason) in sorted(RELABEL.items()):
        assert domain in RESEARCH_DOMAINS, domain
        with driver.session(database=database) as session:
            session.run(
                "MATCH (p:Paper {arxiv_id: $id}) SET p.domain = $domain",
                id=pid,
                domain=domain,
            ).consume()
        db.execute("UPDATE papers SET domain = ? WHERE arxiv_id = ?", [domain, pid])
    db.conn.commit()
    print(f"relabelled {len(RELABEL)} papers")

    # --- 4. drop and rebuild both Qdrant collections ----------------------
    client = get_qdrant_client()
    for collection in (_COLLECTION_LIMITATIONS, _COLLECTION_FUTURE_DIRECTIONS):
        try:
            client.delete_collection(collection)
            print(f"dropped Qdrant collection {collection!r}")
        except Exception as exc:  # noqa: BLE001 — absent is fine, we are recreating it
            print(f"could not drop {collection!r} ({exc.__class__.__name__}); recreating anyway")

    print("re-embedding from the graph…")
    lim = embed_limitations()
    fd = embed_future_directions()
    print(f"embedded {lim['embedded']} limitations, {fd['embedded']} future directions")

    # --- 5. report ---------------------------------------------------------
    with driver.session(database=database) as session:
        after = {
            r["d"]: r["n"]
            for r in session.run(
                "MATCH (p:Paper) RETURN p.domain AS d, count(*) AS n ORDER BY d"
            )
        }
    sqlite_after = dict(
        db.execute("SELECT domain, count(*) FROM papers GROUP BY domain").fetchall()
    )
    print(f"\nafter  — neo4j: {after}   sqlite: {sqlite_after}")
    if after != sqlite_after:
        print("ERROR: the two stores disagree after curation")
        driver.close()
        return 1
    bad = set(after) - set(RESEARCH_DOMAINS)
    if bad:
        print(f"ERROR: unexpected domains remain: {bad}")
        driver.close()
        return 1
    driver.close()
    print("\nOK — stores agree and only known research domains remain.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="report only")
    group.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()
    return run(apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
