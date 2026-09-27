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
    python scripts/derive_thresholds.py
"""

import sys
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


if __name__ == "__main__":
    raise SystemExit(main())
