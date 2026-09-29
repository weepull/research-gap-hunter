"""Score a filled label sheet: precision@k per domain and system, with Wilson CIs.

Phase 4(b) and 4(c). Reports nothing unless labels exist, and says so plainly rather than
emitting a zero that could be mistaken for a measurement.

`real_gap` counts as a hit. `generic_filler` and `off_topic` are both misses, but they are
reported separately because they mean different things: filler is an extraction-quality
problem, off-topic is a corpus-rules problem.

Usage:
    python eval/score.py
    python eval/score.py --sheet eval/label_sheet.csv --k 5 10 20
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.common import EVAL_DIR, LABELS, read_sheet, wilson_interval  # noqa: E402

_SYSTEMS = ("current", "frequency", "random")


def _labels_by_gap(rows: list[dict]) -> dict[tuple[str, str], str]:
    """A label applies to a gap, so it propagates to every row with the same text.

    The sheet interleaves three rankings, so the same gap usually appears more than once.
    Requiring it to be labelled three times would triple the annotator's work and invite
    inconsistent labels for identical text.
    """
    out: dict[tuple[str, str], str] = {}
    conflicts: list[tuple[str, str]] = []
    for row in rows:
        label = (row.get("label") or "").strip()
        if not label:
            continue
        key = (row["domain"], row["gap_text"])
        if key in out and out[key] != label:
            conflicts.append(key)
        out[key] = label
    for key in conflicts:
        print(f"  WARNING: conflicting labels for {key[0]} / {key[1][:60]!r}; "
              f"using {out[key]!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", default=str(EVAL_DIR / "label_sheet.csv"))
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    path = Path(args.sheet)
    if "NOT_GROUND_TRUTH" in path.name:
        print(f"REFUSING to score {path.name}: that file is LLM triage, not ground truth. "
              f"It exists to speed up human labelling and must never feed a reported "
              f"metric.")
        return 2
    if not path.exists():
        print(f"No sheet at {path}. Run eval/make_label_sheet.py first.")
        return 1

    rows = read_sheet(path)
    labels = _labels_by_gap(rows)
    unique_gaps = {(r["domain"], r["gap_text"]) for r in rows}

    print("=" * 92)
    print("EVALUATION — precision@k by domain and ranking")
    print("=" * 92)
    print(f"  sheet            : {path}")
    print(f"  rows             : {len(rows)}")
    print(f"  unique gaps      : {len(unique_gaps)}")
    print(f"  labelled gaps    : {len(labels)}")

    if not labels:
        print("\n  NO LABELS PRESENT — nothing is reported.")
        print("  Fill the `label` column in the sheet with one of "
              f"{', '.join(LABELS)} and re-run.")
        print("\n  This is the honest state: the system's mechanism has been tested "
              "extensively,\n  but the QUALITY of its output has not been measured at "
              "all until this sheet\n  is filled in by a human.")
        return 0

    coverage = len(labels) / max(1, len(unique_gaps))
    print(f"  label coverage   : {coverage*100:.0f}% of unique gaps")
    if coverage < 1.0:
        print("  NOTE: coverage is partial, so precision below is computed over the "
              "labelled subset only and is not a complete picture.")

    unknown = {v for v in labels.values()} - set(LABELS)
    if unknown:
        print(f"\n  ERROR: unrecognised labels {sorted(unknown)}; expected "
              f"{list(LABELS)}")
        return 1

    by_domain = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["domain"], row["gap_text"])
        if key in labels:
            by_domain[row["domain"]][row["system"]].append((int(row["rank"]), labels[key]))

    for domain in sorted(by_domain):
        print(f"\n{'-' * 92}\n{domain}\n{'-' * 92}")
        breakdown = defaultdict(int)
        for _rank, label in by_domain[domain].get("current", []):
            breakdown[label] += 1
        total = sum(breakdown.values())
        if total:
            print("  label mix (current ranking): " + "  ".join(
                f"{label}={breakdown[label]} ({breakdown[label]/total*100:.0f}%)"
                for label in LABELS
            ))
        print(f"\n  {'system':<12}{'k':>4}{'hits':>6}{'n':>5}{'precision':>11}"
              f"{'95% CI':>20}")
        for system in _SYSTEMS:
            entries = sorted(by_domain[domain].get(system, []))
            if not entries:
                continue
            for k in args.k:
                window = [label for rank, label in entries if rank <= k]
                if not window:
                    continue
                hits = sum(1 for label in window if label == "real_gap")
                n = len(window)
                low, high = wilson_interval(hits, n)
                print(f"  {system:<12}{k:>4}{hits:>6}{n:>5}{hits/n:>11.3f}"
                      f"{f'[{low:.3f}, {high:.3f}]':>20}")

    print(f"\n{'=' * 92}")
    print("  Wilson intervals, not normal-approximation: n is small and the proportion "
          "is often\n  near 0 or 1, where the normal interval gives bounds outside "
          "[0, 1].")
    print("  Overlapping intervals between systems mean the difference is NOT "
          "established.")
    print("  Single annotator, so there is no inter-annotator agreement figure and no "
          "way to\n  separate a labelling error from a system error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
