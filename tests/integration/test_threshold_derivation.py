"""Derivation guard: every coded threshold must match its measured null.

This is the standing check that no threshold in this project is a hand-picked
number. It is the guard that would have caught `_CLUSTER_THRESHOLD = 0.86`, which
sat below its own noise floor for the whole life of the project because nothing
compared it to anything.

It runs in the integration tier because it needs the live vector store, and it
imports `scripts/derive_thresholds.py` rather than reimplementing the derivation —
a guard computed a second way can pass while the real derivation is wrong.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def drift(loaded_corpus_free_store):
    import derive_thresholds

    return derive_thresholds.drift_report()


@pytest.fixture(scope="module")
def loaded_corpus_free_store():
    """This tier's other tests use an isolated fixture corpus; this one must not.

    A 13-paper fixture cannot produce a meaningful null distribution, and the
    constants in the code are derived from the *real* corpus — so the guard has to
    measure the real corpus too. Skips if the production store is unreachable.
    """
    from vectors.embed import get_qdrant_client

    try:
        client = get_qdrant_client()
        names = {c.name for c in client.get_collections().collections}
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant unreachable ({exc.__class__.__name__})")
    missing = {"limitations", "future_directions"} - names
    if missing:
        pytest.skip(f"production collections absent: {sorted(missing)}")
    return client


def test_every_coded_threshold_is_within_tolerance_of_its_derived_value(drift):
    """Fails when a constant drifts from its measured null p95 by > DRIFT_TOLERANCE.

    Fails against pre-Phase-1a code: _SOLUTION_THRESHOLDS["medical_imaging"] was
    0.8987 against a measured null p95 of ~0.8915 — a drift of ~0.0072, i.e. the
    coded value was *stricter* than its own noise floor, so it under-counted
    addressing solutions and made medical-imaging gaps look more open than they are.
    """
    import derive_thresholds

    stale = [r for r in drift if r["stale"]]
    assert not stale, "\n".join(
        f"{r['name']}: coded {r['coded']:.4f} vs derived {r['derived']:.4f} "
        f"(drift {r['drift']:.4f} > {derive_thresholds.DRIFT_TOLERANCE}) — "
        f"re-run scripts/derive_thresholds.py --apply"
        for r in stale
    )


def test_every_threshold_actually_has_a_derived_counterpart(drift):
    """A constant with no derivable null is a constant nobody can check."""
    undrivable = [r["name"] for r in drift if r["derived"] is None]
    assert not undrivable, f"no null distribution could be derived for: {undrivable}"


def test_cluster_and_solution_nulls_are_measured_on_different_populations(drift):
    """The two families must not be reasoned about as if comparable.

    Conflating them is the documented original error: 0.85 "looked conservative"
    beside the 0.86 cluster threshold although they measure limitation x
    future-direction and limitation x limitation respectively.
    """
    import derive_thresholds

    raw = derive_thresholds.derive_all()["raw"]
    for domain in ("computer_vision", "medical_imaging"):
        ck, sk = f"cluster:{domain}", f"solution:{domain}"
        if ck in raw and sk in raw:
            assert raw[ck]["n"] != raw[sk]["n"] or raw[ck]["p95"] != raw[sk]["p95"], (
                f"{domain}: cluster and solution nulls are identical, which means "
                "they are being measured over the same pairs"
            )
