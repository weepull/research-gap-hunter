"""Measure how much each scoring term actually influences the ranking. READ-ONLY.

The formula's coefficients (0.40 frequency / 0.35 recency / 0.25 solution-deficit)
are **not** the observed influence on the ordering. A term's effect on a ranking
is its weight times how much it *varies* across the result set: a coefficient of
0.40 on a term that only ever moves between 0.02 and 0.20 shifts scores by at most
0.07, while a coefficient of 0.35 on a term spanning the full [0, 1] shifts them
by 0.35.

This script reports both, so the gap between the documented weights and the real
behaviour is a measured number rather than an impression. PLAN.md #4 chose to
document that gap honestly (option 4C) rather than normalise it away (4B), because
normalising makes every score relative to whatever else is in the result set —
adding one paper would move every number with no change in evidence, destroying
the cross-run comparability `_corpus_reference_year` exists to guarantee.

Usage:
    python scripts/measure_weight_influence.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from pipeline.domains import RESEARCH_DOMAINS  # noqa: E402
from pipeline.gap_scorer import score_gaps  # noqa: E402

_WEIGHTS = (("frequency", 0.40), ("recency", 0.35), ("solution_deficit", 0.25))


def _spread(values: np.ndarray) -> float:
    return float(values.max() - values.min()) if values.size else 0.0


def main() -> int:
    for domain in RESEARCH_DOMAINS:
        gaps = score_gaps(domain=domain, top_n=200)
        if len(gaps) < 2:
            print(f"\n{domain}: {len(gaps)} gap(s) — too few to measure influence")
            continue

        terms = {
            "frequency": np.array([g.frequency_score for g in gaps]),
            "recency": np.array([g.recency_score for g in gaps]),
            "solution_deficit": np.array([g.solution_deficit_score for g in gaps]),
        }
        scores = np.array([g.score for g in gaps])

        print(f"\n{'=' * 78}")
        print(f"{domain} — {len(gaps)} gaps")
        print(f"{'=' * 78}")
        print(
            f"{'term':<18}{'weight':>8}{'range':>18}{'spread':>9}"
            f"{'w*spread':>10}{'w*sd':>9}"
        )
        spreads, sds = [], []
        for name, weight in _WEIGHTS:
            values = terms[name]
            spread, sd = _spread(values), float(values.std())
            spreads.append(weight * spread)
            sds.append(weight * sd)
            print(
                f"{name:<18}{weight:>8.2f}"
                f"{f'{values.min():.4f}-{values.max():.4f}':>18}"
                f"{spread:>9.4f}{weight * spread:>10.4f}{weight * sd:>9.4f}"
            )

        spreads_arr, sds_arr = np.array(spreads), np.array(sds)
        print("\neffective influence on the ordering:")
        for i, (name, weight) in enumerate(_WEIGHTS):
            by_spread = spreads_arr[i] / spreads_arr.sum() * 100 if spreads_arr.sum() else 0
            by_sd = sds_arr[i] / sds_arr.sum() * 100 if sds_arr.sum() else 0
            print(
                f"  {name:<18} nominal {weight * 100:>4.0f}%   "
                f"measured {by_spread:>5.1f}% (by spread)   {by_sd:>5.1f}% (by sd)"
            )

        # Does the frequency term change the order at all?
        without_frequency = 0.35 * terms["recency"] + 0.25 * terms["solution_deficit"]
        full_order = list(np.argsort(-scores, kind="stable"))
        reduced_order = list(np.argsort(-without_frequency, kind="stable"))
        top = min(10, len(gaps))
        print(
            f"\n  deleting the frequency term entirely: top-{top} order "
            f"{'UNCHANGED' if full_order[:top] == reduced_order[:top] else 'changes'}, "
            f"full order {'unchanged' if full_order == reduced_order else 'changes'}"
        )

        tied = len(scores) - len(set(np.round(scores, 4)))
        print(f"  exact-score ties: {tied} of {len(gaps)} gaps share a score with another")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
