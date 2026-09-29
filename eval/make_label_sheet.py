"""Generate eval/label_sheet.csv — the top 20 gaps per domain, awaiting hand labels.

Phase 4(a) and 4(c). The sheet carries three systems interleaved so the annotator cannot
tell which ranking a row came from while labelling:

    current    the live system
    frequency  frequency_score only, the obvious naive baseline
    random     a fixed-seed shuffle of the same gap pool

All three draw from the *same* pool of scored gaps, so any difference in precision@k is
attributable to the ranking rather than to different candidate sets. Rows are sorted by
gap text, not by system or rank, so the sheet gives no positional hint about which system
proposed a row.

Usage:
    python eval/make_label_sheet.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from eval.common import EVAL_DIR, LABEL_HELP, paper_titles, write_sheet  # noqa: E402
from pipeline.domains import RESEARCH_DOMAINS  # noqa: E402

TOP_K = 20
RANDOM_SEED = 20260929  # fixed so the random baseline is reproducible


def _rows_for(domain: str, gaps: list) -> list[dict]:
    titles = paper_titles(sorted({p for g in gaps for p in g.supporting_papers}))

    rankings = {
        "current": list(gaps),
        "frequency": sorted(gaps, key=lambda g: (-g.frequency_score, g.gap_description)),
    }
    shuffled = list(gaps)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    rankings["random"] = shuffled

    rows = []
    for system, ordered in rankings.items():
        for rank, gap in enumerate(ordered[:TOP_K], 1):
            rows.append({
                "domain": domain,
                "rank": rank,
                "system": system,
                "gap_text": gap.gap_description,
                "tier": gap.tier,
                "score": f"{gap.score:.4f}",
                "frequency_score": f"{gap.frequency_score:.4f}",
                "recency_score": f"{gap.recency_score:.4f}",
                "solution_deficit_score": f"{gap.solution_deficit_score:.4f}",
                "supporting_paper_ids": " ".join(gap.supporting_papers),
                "supporting_paper_titles": " | ".join(
                    titles.get(p, p) for p in gap.supporting_papers
                ),
                "label": "",
                "notes": "",
            })
    return rows


def main() -> int:
    from pipeline.gap_scorer import score_gaps

    all_rows = []
    for domain in RESEARCH_DOMAINS:
        gaps = score_gaps(domain=domain, top_n=500)
        print(f"{domain}: {len(gaps)} gaps scored")
        if not gaps:
            continue
        all_rows.extend(_rows_for(domain, gaps))

    # Sorted by text so the sheet leaks no hint about which system proposed a row.
    all_rows.sort(key=lambda r: (r["domain"], r["gap_text"], r["system"]))

    path = EVAL_DIR / "label_sheet.csv"
    written = write_sheet(all_rows, path)
    unique = len({(r["domain"], r["gap_text"]) for r in all_rows})
    print(f"\nwrote {written} rows ({unique} unique gaps) to {path}")
    print("\nFill the `label` column with exactly one of:")
    for label, help_text in LABEL_HELP.items():
        print(f"  {label:<16} {help_text}")
    print("\nRows are interleaved across three rankings and sorted by gap text, so you "
          "cannot tell which system proposed a row while labelling. A gap appearing in "
          "more than one ranking needs labelling only once — eval/score.py propagates a "
          "label to every row with the same (domain, gap_text).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
