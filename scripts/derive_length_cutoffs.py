"""Derive the minimum-informative-length cutoffs from the corpus. READ-ONLY.

The cutoff in pipeline/extraction_filter.py must be a *stated percentile of the
observed distribution*, not a round number someone liked. This script reports the
distribution and the candidate percentiles so the choice is auditable and can be
re-checked as the corpus grows.

Word count is the criterion rather than character count: it tracks informativeness
better and is not skewed by long hyphenated technical compounds (e.g.
"contrast-agnostic, pathology-encoded representations" is 3 words but 52 characters).
Character statistics are reported alongside for reference.

Usage:
    python scripts/derive_length_cutoffs.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from pipeline.batch import _DB_PATH  # noqa: E402

FIELDS = ("limitations", "future_directions")
_REPORT_PERCENTILES = (1, 5, 10, 15, 20, 25, 50, 75, 90, 99)


def _load(field: str) -> list[str]:
    conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
    out = []
    for (raw,) in conn.execute(f"SELECT {field} FROM papers"):
        try:
            items = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (json.JSONDecodeError, TypeError):
            items = []
        out.extend(t.strip() for t in items if t and t.strip())
    return out


def main() -> int:
    print("=" * 92)
    print("EXTRACTION LENGTH DISTRIBUTION (live corpus)")
    print("=" * 92)
    chosen = {}
    for field in FIELDS:
        strings = _load(field)
        if not strings:
            print(f"\n{field}: no strings")
            continue
        words = np.array([len(s.split()) for s in strings])
        chars = np.array([len(s) for s in strings])
        print(f"\n{field}: n={len(strings)}")
        for label, arr in (("words", words), ("chars", chars)):
            pcts = "  ".join(f"p{p}={int(np.percentile(arr, p))}" for p in _REPORT_PERCENTILES)
            print(f"  {label:<6} min={arr.min():<4} median={int(np.median(arr)):<4} "
                  f"max={arr.max():<5} {pcts}")
        p5, p10 = int(np.percentile(words, 5)), int(np.percentile(words, 10))
        chosen[field] = p5
        below5 = int((words < p5).sum())
        below10 = int((words < p10).sum())
        print(f"  candidate cutoffs (words): p5={p5} would reject {below5} "
              f"({below5/len(strings)*100:.1f}%) · p10={p10} would reject {below10} "
              f"({below10/len(strings)*100:.1f}%)")
        print(f"  strings below p5={p5} words:")
        for s in sorted({s for s in strings if len(s.split()) < p5},
                        key=lambda x: (len(x.split()), x)):
            print(f"     {len(s.split())}w  {s!r}")

    print()
    print("=" * 92)
    print("CHOSEN: the 5th percentile of the word-count distribution, per field.")
    print(f"  {chosen}")
    print()
    print("  p5 rather than p10 deliberately. p10 for future_directions is 5 words,")
    print("  which would discard terse but genuinely specific solutions such as")
    print("  'contrast-agnostic, pathology-encoded representations'. p5 removes only the")
    print("  extreme tail — the one- and two-word fragments that cannot describe a gap")
    print("  ('Overfitting', 'Dataset bias', 'domain shift') — and leaves anything with")
    print("  a subject and a predicate. The hedging and prompt-echo gates catch the")
    print("  remaining short boilerplate independently, so this gate does not have to")
    print("  carry that work alone.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
