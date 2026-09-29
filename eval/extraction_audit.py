"""Extraction audit: 30 random papers, extracted limitations beside the source snippet.

Phase 4(d). Scoring quality is bounded by extraction quality, and extraction has never
been checked against what the papers actually say — only against whether the JSON parsed.
This puts the two side by side so a human can see whether a limitation was really stated.

The snippet is re-fetched from the arXiv PDF through the *same* code path extraction used
(`fetch_full_text`), so what the auditor reads is what the model read, not a different
view of the paper. That costs a PDF download per paper, which is why this is a sampled
audit rather than a full sweep.

`raw_json` holds the pre-filter model output, so the audit also shows which strings the
Phase 1b filter removed and why.

Usage:
    python eval/extraction_audit.py --sample 30
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from eval.common import EVAL_DIR  # noqa: E402

RANDOM_SEED = 20260929
_SNIPPET_CHARS = 1200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--no-fetch", action="store_true",
                        help="skip PDF re-fetch; leaves the snippet column empty")
    args = parser.parse_args()

    from pipeline.batch import _get_db
    from pipeline.extraction_filter import classify
    from pipeline.extractor import fetch_full_text

    db = _get_db()
    rows = [dict(r) for r in db["papers"].rows]
    if not rows:
        print("no papers in the corpus")
        return 1

    sample = random.Random(RANDOM_SEED).sample(rows, min(args.sample, len(rows)))
    print(f"auditing {len(sample)} of {len(rows)} papers "
          f"(seed {RANDOM_SEED}, so the sample is reproducible)")

    out_path = EVAL_DIR / "extraction_audit.csv"
    columns = [
        "arxiv_id", "domain", "title", "tier",
        "limitation_kept", "filtered_out", "filter_reason",
        "source_snippet", "verdict", "notes",
    ]
    written = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, paper in enumerate(sample, 1):
            def _jl(raw):
                try:
                    return json.loads(raw) if isinstance(raw, str) else (raw or [])
                except (json.JSONDecodeError, TypeError):
                    return []

            kept = _jl(paper.get("limitations"))
            raw = {}
            try:
                raw = json.loads(paper.get("raw_json") or "{}")
            except json.JSONDecodeError:
                raw = {}
            pre_filter = raw.get("limitations", []) or []
            removed = [t for t in pre_filter if t not in kept]

            snippet = ""
            if not args.no_fetch:
                try:
                    text, _tier = fetch_full_text(paper["arxiv_id"], abstract="")
                    snippet = " ".join((text or "").split())[:_SNIPPET_CHARS]
                except Exception as exc:  # noqa: BLE001 — an audit row is still useful
                    snippet = f"<fetch failed: {exc.__class__.__name__}>"
            print(f"  [{i}/{len(sample)}] {paper['arxiv_id']} "
                  f"kept={len(kept)} filtered={len(removed)}", flush=True)

            if not kept and not removed:
                writer.writerow({
                    "arxiv_id": paper["arxiv_id"], "domain": paper["domain"],
                    "title": paper.get("title", ""), "tier": paper.get("extraction_tier", ""),
                    "limitation_kept": "<none extracted>", "filtered_out": "",
                    "filter_reason": "", "source_snippet": snippet,
                    "verdict": "", "notes": "",
                })
                written += 1
                continue

            for text in kept:
                writer.writerow({
                    "arxiv_id": paper["arxiv_id"], "domain": paper["domain"],
                    "title": paper.get("title", ""), "tier": paper.get("extraction_tier", ""),
                    "limitation_kept": text, "filtered_out": "", "filter_reason": "",
                    "source_snippet": snippet, "verdict": "", "notes": "",
                })
                written += 1
            for text in removed:
                verdict = classify(text, "limitations")
                writer.writerow({
                    "arxiv_id": paper["arxiv_id"], "domain": paper["domain"],
                    "title": paper.get("title", ""), "tier": paper.get("extraction_tier", ""),
                    "limitation_kept": "", "filtered_out": text,
                    "filter_reason": verdict[0] if verdict else "unknown",
                    "source_snippet": snippet, "verdict": "", "notes": "",
                })
                written += 1

    print(f"\nwrote {written} rows to {out_path}")
    print("Fill `verdict` per row with: stated / implied / not_supported")
    print("  stated        the snippet says this outright")
    print("  implied       defensible reading of the snippet")
    print("  not_supported the snippet does not support it — an extraction error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
