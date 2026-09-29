"""Re-derive similarity thresholds from measured null distributions. READ-ONLY.

A "null distribution" here is the similarity that *arbitrary* pairs from a
population reach by chance, measured over every such pair in the live corpus
using the stored Specter2 vectors. Below a null's 95th percentile a "match" is
statistically indistinguishable from a random pairing.

This is the method the advisor accepted for `_SOLUTION_THRESHOLDS` and the
cross-domain threshold on 2026-08-23 (PROJECT_HARDENING_PLAN.md A1/A2), which had
no script — it is written down here so re-derivation after ingestion is a command
rather than an archaeology exercise.

**Populations are not interchangeable.** Each threshold governs a different pair
population and must be derived from that one:

  _CLUSTER_THRESHOLDS   limitation x limitation, within a domain
  _SOLUTION_THRESHOLDS  limitation x future-direction, within a domain
  cross-domain          limitation x future-direction, across domains

Conflating them is the specific error this project already made twice: 0.85
"looked conservative" beside the 0.86 cluster threshold although the two measured
different populations, and the cluster threshold was justified by sitting "well
above the median" when the relevant bar is the upper percentile of the null, not
its centre.

**Queries are domain-filtered.** Deriving a within-domain null from an unfiltered
scan mixes in cross-domain pairs and produces a contaminated number. This is the
same bug that was live in `_query_neighbours`.

Usage:
    python scripts/derive_thresholds.py            # report only
    python scripts/derive_thresholds.py --apply    # also write stale constants
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from pipeline.domains import RESEARCH_DOMAINS  # noqa: E402
from vectors.embed import (  # noqa: E402
    _COLLECTION_FUTURE_DIRECTIONS,
    _COLLECTION_LIMITATIONS,
    get_qdrant_client,
)

_PERCENTILES = (50, 90, 95, 99)

# A coded constant is updated only when it drifts from its freshly derived value
# by more than this. Re-deriving on every ingestion and chasing the third decimal
# would make rankings move for no defensible reason — the derived value itself has
# sampling noise (bootstrap 95% CIs on these p95s span roughly +/-0.002), so a
# smaller drift is not distinguishable from resampling the same corpus.
DRIFT_TOLERANCE = 0.002

_TODAY = datetime.now(timezone.utc).date().isoformat()


def _fetch(client, collection: str, domain: str) -> np.ndarray:
    """Unit-normalised vectors for one domain, so a dot product is cosine."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
            ),
            limit=500,
            offset=offset,
            with_vectors=True,
            with_payload=False,
        )
        points.extend(batch)
        if offset is None:
            break
    if not points:
        return np.zeros((0, 768))
    matrix = np.asarray([p.vector for p in points], dtype=float)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _describe(name: str, values: np.ndarray) -> dict:
    if values.size == 0:
        print(f"  {name}: no pairs")
        return {}
    stats = {f"p{p}": float(np.percentile(values, p)) for p in _PERCENTILES}
    print(
        f"  {name}: n={values.size:>7,}  mean={values.mean():.4f}  "
        + "  ".join(f"{k}={v:.4f}" for k, v in stats.items())
        + f"  max={values.max():.4f}"
    )
    return stats


def derive_all() -> dict:
    """Every threshold-governing null, derived from the live store. Read-only.

    Returns {"cluster": {domain: p95}, "solution": {domain: p95},
             "cross_domain": p95, "raw": {...full percentile stats...}}.

    Importable so the guard test and the apply path use exactly this code rather
    than a reimplementation that could drift from it.
    """
    client = get_qdrant_client()
    out = {"cluster": {}, "solution": {}, "cross_domain": None,
           "deficit_anchors": {}, "raw": {}}

    for domain in RESEARCH_DOMAINS:
        vecs = _fetch(client, _COLLECTION_LIMITATIONS, domain)
        if len(vecs) >= 2:
            sims = vecs @ vecs.T
            upper = sims[np.triu_indices(len(vecs), k=1)]
            out["cluster"][domain] = float(np.percentile(upper, 95))
            out["raw"][f"cluster:{domain}"] = {
                "n": int(upper.size),
                **{f"p{p}": float(np.percentile(upper, p)) for p in _PERCENTILES},
            }

        fut = _fetch(client, _COLLECTION_FUTURE_DIRECTIONS, domain)
        if len(vecs) and len(fut):
            pairs = (vecs @ fut.T).ravel()
            out["solution"][domain] = float(np.percentile(pairs, 95))
            # Anchors for the continuous solution-deficit rescaling (A9 Option F).
            # p50 is the centre of pure noise, p99 the point at which a match is
            # near-certainly real; between them the measure is informative.
            out["deficit_anchors"][domain] = (
                float(np.percentile(pairs, 50)),
                float(np.percentile(pairs, 99)),
            )
            out["raw"][f"solution:{domain}"] = {
                "n": int(pairs.size),
                **{f"p{p}": float(np.percentile(pairs, p)) for p in _PERCENTILES},
            }

    both = []
    for source, target in ((RESEARCH_DOMAINS[0], RESEARCH_DOMAINS[1]),
                           (RESEARCH_DOMAINS[1], RESEARCH_DOMAINS[0])):
        lim = _fetch(client, _COLLECTION_LIMITATIONS, source)
        fut = _fetch(client, _COLLECTION_FUTURE_DIRECTIONS, target)
        if len(lim) and len(fut):
            both.append((lim @ fut.T).ravel())
    if both:
        joined = np.concatenate(both)
        out["cross_domain"] = float(np.percentile(joined, 95))
        out["raw"]["cross_domain"] = {
            "n": int(joined.size),
            **{f"p{p}": float(np.percentile(joined, p)) for p in _PERCENTILES},
        }
    return out


def coded_constants() -> dict:
    """The constants currently in the code, in the same shape as derive_all()."""
    from pipeline.cross_domain import _CROSS_DOMAIN_THRESHOLD
    from pipeline.gap_scorer import _CLUSTER_THRESHOLDS, _SOLUTION_THRESHOLDS

    from pipeline.gap_scorer import _DEFICIT_RESCALE_ANCHORS

    return {
        "cluster": dict(_CLUSTER_THRESHOLDS),
        "solution": dict(_SOLUTION_THRESHOLDS),
        "cross_domain": _CROSS_DOMAIN_THRESHOLD,
        "deficit_anchors": {k: tuple(v) for k, v in _DEFICIT_RESCALE_ANCHORS.items()},
    }


def drift_report() -> list[dict]:
    """One row per constant: coded, derived, drift, and whether it must change."""
    derived, coded = derive_all(), coded_constants()
    rows = []
    for kind in ("cluster", "solution"):
        for domain, value in sorted(coded[kind].items()):
            d = derived[kind].get(domain)
            rows.append({
                "name": f'_{kind.upper()}_THRESHOLDS["{domain}"]',
                "kind": kind, "domain": domain, "coded": value, "derived": d,
                "drift": None if d is None else abs(d - value),
                "stale": d is not None and abs(d - value) > DRIFT_TOLERANCE,
            })
    for domain, coded_pair in sorted(coded.get("deficit_anchors", {}).items()):
        derived_pair = derived["deficit_anchors"].get(domain)
        for i, label in enumerate(("p50", "p99")):
            dv = None if derived_pair is None else derived_pair[i]
            cv = coded_pair[i]
            rows.append({
                "name": f'_DEFICIT_RESCALE_ANCHORS["{domain}"][{label}]',
                "kind": "deficit_anchors", "domain": domain, "index": i,
                "coded": cv, "derived": dv,
                "drift": None if dv is None else abs(dv - cv),
                "stale": dv is not None and abs(dv - cv) > DRIFT_TOLERANCE,
            })

    d = derived["cross_domain"]
    rows.append({
        "name": "_CROSS_DOMAIN_THRESHOLD", "kind": "cross_domain", "domain": None,
        "coded": coded["cross_domain"], "derived": d,
        "drift": None if d is None else abs(d - coded["cross_domain"]),
        "stale": d is not None and abs(d - coded["cross_domain"]) > DRIFT_TOLERANCE,
    })
    return rows


def _replace_dict_entry(path: str, section: str, domain: str, old: float,
                        new: float, n: int) -> bool:
    """Rewrite one `"domain": value,  # n=...` line inside a constant dict.

    Anchors on the ASSIGNMENT (`SECTION = {`) rather than the first mention of the
    name. Both dict names appear in explanatory comments above their assignments,
    so anchoring on the bare name silently sliced the wrong region — which is how
    the first attempt at this reported "coded literal not found".

    The trailing `# n=` provenance comment is rewritten too. Leaving a stale sample
    size beside a fresh value would be a number in the codebase that nothing
    measured, which is exactly what this script exists to prevent.
    """
    text = open(path).read()
    anchor = f"{section} = {{"
    if anchor not in text:
        return False
    start = text.index(anchor)
    stop = text.index("}", start)
    block = text[start:stop]

    pattern = re.compile(
        rf'(^[ \t]*"{re.escape(domain)}"[ \t]*:[ \t]*)'
        rf'{re.escape(f"{old:.4f}")}'
        r'([ \t]*,)(?:[ \t]*#[^\n]*)?',
        re.M,
    )
    replacement = (
        rf'\g<1>{new:.4f}\g<2>'
        f'  # null p95 over n={n:,} pairs, derived {_TODAY}'
    )
    updated, count = pattern.subn(replacement, block, count=1)
    if count != 1:
        return False
    open(path, "w").write(text[:start] + updated + text[stop:])
    return True


def _replace_scalar(path: str, marker: str, old: float, new: float, n: int) -> bool:
    """Rewrite a module-level `NAME = value` scalar constant."""
    text = open(path).read()
    if marker not in text:
        return False
    idx = text.index(marker)
    end = text.index("\n", idx)
    line = text[idx:end]
    needle = f"{old:.4f}"
    if needle not in line:
        return False
    new_line = line.replace(needle, f"{new:.4f}", 1)
    new_line += f"  # null p95 over n={n:,} pairs, derived {_TODAY}"
    open(path, "w").write(text[:idx] + new_line + text[end:])
    return True


def _replace_anchor_pair(path: str, domain: str, pair: tuple[float, float],
                         n: int) -> bool:
    """Rewrite both numbers of one _DEFICIT_RESCALE_ANCHORS entry together.

    The two anchors are only meaningful as a pair — rescaling between a fresh p50 and
    a stale p99 would produce a measure anchored to two different corpora — so they
    are always written together even when only one drifted.
    """
    text = open(path).read()
    anchor = "_DEFICIT_RESCALE_ANCHORS = {"
    if anchor not in text:
        return False
    start = text.index(anchor)
    stop = text.index("}", start)
    block = text[start:stop]
    pattern = re.compile(
        rf'(^[ \t]*"{re.escape(domain)}"[ \t]*:[ \t]*)\([^)]*\)([ \t]*,)'
        r'(?:[ \t]*#[^\n]*)?',
        re.M,
    )
    replacement = (
        rf'\g<1>({pair[0]:.4f}, {pair[1]:.4f})\g<2>'
        f'  # null p50/p99 over n={n:,} pairs, derived {_TODAY}'
    )
    updated, count = pattern.subn(replacement, block, count=1)
    if count != 1:
        return False
    open(path, "w").write(text[:start] + updated + text[stop:])
    return True


def apply_stale(rows: list[dict]) -> list[dict]:
    """Write derived values for constants whose drift exceeds DRIFT_TOLERANCE.

    Deliberately the only supported way to change these numbers. Hand-editing is
    how an unmeasured value like the old 0.86 cluster threshold got in and survived.
    """
    raw = derive_all()["raw"]
    applied = []
    for row in rows:
        if not row["stale"]:
            continue
        new = round(row["derived"], 4)
        kind_for_n = "solution" if row["kind"] == "deficit_anchors" else row["kind"]
        key = kind_for_n if row["domain"] is None else f'{kind_for_n}:{row["domain"]}'
        n = raw.get(key, {}).get("n", 0)

        if row["kind"] == "deficit_anchors":
            ok = _replace_anchor_pair(
                "pipeline/gap_scorer.py", row["domain"],
                derive_all()["deficit_anchors"][row["domain"]], n,
            )
        elif row["kind"] in ("cluster", "solution"):
            section = "_CLUSTER_THRESHOLDS" if row["kind"] == "cluster" else "_SOLUTION_THRESHOLDS"
            ok = _replace_dict_entry(
                "pipeline/gap_scorer.py", section, row["domain"], row["coded"], new, n
            )
        else:
            ok = _replace_scalar(
                "pipeline/cross_domain.py", "_CROSS_DOMAIN_THRESHOLD = ",
                row["coded"], new, n,
            )
        if not ok:
            print(f"  FAILED {row['name']}: could not locate the literal to rewrite")
            continue
        applied.append({**row, "new": new, "n": n})
        print(f"  UPDATED {row['name']}: {row['coded']:.4f} -> {new:.4f} "
              f"(drift {row['drift']:.4f} > {DRIFT_TOLERANCE}, n={n:,})")
    return applied


def main() -> int:
    client = get_qdrant_client()

    print("=" * 78)
    print("CLUSTER THRESHOLD null — limitation x limitation, within domain")
    print("  (governs _CLUSTER_THRESHOLDS in pipeline/gap_scorer.py)")
    print("=" * 78)
    cluster_stats = {}
    for domain in RESEARCH_DOMAINS:
        vectors = _fetch(client, _COLLECTION_LIMITATIONS, domain)
        if len(vectors) < 2:
            print(f"  {domain}: fewer than two limitations; cannot derive")
            continue
        sims = vectors @ vectors.T
        upper = sims[np.triu_indices(len(vectors), k=1)]
        cluster_stats[domain] = _describe(f"{domain:<16}", upper)

    print()
    print("=" * 78)
    print("SOLUTION THRESHOLD null — limitation x future-direction, within domain")
    print("  (governs _SOLUTION_THRESHOLDS; reported for reference, not changed here)")
    print("=" * 78)
    for domain in RESEARCH_DOMAINS:
        lim = _fetch(client, _COLLECTION_LIMITATIONS, domain)
        fut = _fetch(client, _COLLECTION_FUTURE_DIRECTIONS, domain)
        if len(lim) == 0 or len(fut) == 0:
            print(f"  {domain}: one side empty; cannot derive")
            continue
        _describe(f"{domain:<16}", (lim @ fut.T).ravel())

    print()
    print("=" * 78)
    print("CROSS-DOMAIN null — limitation x future-direction, across domains")
    print("  (governs find_cross_domain_matches; reported for reference)")
    print("=" * 78)
    both = []
    for source, target in ((RESEARCH_DOMAINS[0], RESEARCH_DOMAINS[1]),
                           (RESEARCH_DOMAINS[1], RESEARCH_DOMAINS[0])):
        lim = _fetch(client, _COLLECTION_LIMITATIONS, source)
        fut = _fetch(client, _COLLECTION_FUTURE_DIRECTIONS, target)
        if len(lim) and len(fut):
            both.append((lim @ fut.T).ravel())
    if both:
        _describe("both directions", np.concatenate(both))

    print()
    print("=" * 78)
    print("Suggested _CLUSTER_THRESHOLDS (null p95 — a 5% per-pair noise rate):")
    for domain, stats in cluster_stats.items():
        if stats:
            print(f'    "{domain}": {stats["p95"]:.4f},')
    print("=" * 78)
    return 0


def _drift_table(apply: bool) -> int:
    rows = drift_report()
    print()
    print("=" * 96)
    print(f"DRIFT vs CODED CONSTANTS  (tolerance {DRIFT_TOLERANCE})")
    print("=" * 96)
    print(f"  {'constant':<48}{'coded':>9}{'derived':>10}{'drift':>9}  action")
    for r in sorted(rows, key=lambda r: r["name"]):
        derived = "n/a" if r["derived"] is None else f"{r['derived']:.4f}"
        drift = "n/a" if r["drift"] is None else f"{r['drift']:.4f}"
        action = "UPDATE" if r["stale"] else "keep"
        print(f"  {r['name']:<48}{r['coded']:>9.4f}{derived:>10}{drift:>9}  {action}")
    stale = [r for r in rows if r["stale"]]
    if not stale:
        print("\n  All constants within tolerance. Nothing to do.")
        return 0
    if not apply:
        print(f"\n  {len(stale)} constant(s) exceed tolerance. Re-run with --apply to write them.")
        return 0
    print(f"\n  Applying {len(stale)} constant(s):")
    apply_stale(rows)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write constants whose drift exceeds DRIFT_TOLERANCE")
    args = parser.parse_args()
    rc = main()
    rc = _drift_table(args.apply) or rc
    raise SystemExit(rc)
