"""Tests for pipeline/gap_scorer.py — the discovery / gap-scoring core."""

import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

import pipeline.gap_scorer as gs
from pipeline.gap_scorer import (
    GapResult,
    _corpus_reference_year,
    cluster_limitations,
    compute_frequency_score,
    compute_recency_score,
    compute_solution_deficit_score,
    score_gaps,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_model_mock(dim: int = 768) -> MagicMock:
    """Mock SentenceTransformer whose encode() returns a numpy array per input."""
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.array(
        [[0.01] * dim for _ in texts]
    )
    return model


def _make_hit(text: str, score: float) -> types.SimpleNamespace:
    """A single Qdrant point with .score and .payload (payload key per embed.py)."""
    return types.SimpleNamespace(score=score, payload={"limitation_text": text})


def _make_qdrant_query_mock(points: list) -> MagicMock:
    """Mock QdrantClient whose query_points() always returns an object with .points."""
    client = MagicMock()
    client.query_points.return_value = types.SimpleNamespace(points=points)
    return client


def _make_qdrant_batch_mock(points_per_index: list[list]) -> MagicMock:
    """Mock QdrantClient whose query_batch_points() returns one response per input vector.

    points_per_index is index-aligned with the limitations passed to
    cluster_limitations: entry i holds the Qdrant hits for limitation i's vector.
    """
    client = MagicMock()
    client.query_batch_points.return_value = [
        types.SimpleNamespace(points=points) for points in points_per_index
    ]
    return client


def _patch_vector_backends(monkeypatch, client, model) -> None:
    monkeypatch.setattr(gs, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(gs, "load_embedding_model", lambda: model)


# ---------------------------------------------------------------------------
# compute_frequency_score
# ---------------------------------------------------------------------------


def test_compute_frequency_score_basic():
    """frequency = unique papers in cluster / total papers."""
    cluster = [{"text": "x", "paper_ids": ["a", "b"], "years": [2024, 2024]}]
    assert compute_frequency_score(cluster, 4) == 0.5


def test_compute_frequency_score_dedups_across_cluster():
    """A paper reporting two limitations in the cluster counts once."""
    cluster = [
        {"text": "x", "paper_ids": ["a", "b"], "years": [2024, 2024]},
        {"text": "y", "paper_ids": ["b", "c"], "years": [2024, 2024]},
    ]
    # unique papers {a, b, c} = 3 of 6
    assert compute_frequency_score(cluster, 6) == 0.5


def test_compute_frequency_score_caps_at_one():
    """frequency_score is capped at 1.0 even if papers exceed the total."""
    cluster = [{"text": "x", "paper_ids": ["a", "b", "c"], "years": [2024, 2024, 2024]}]
    assert compute_frequency_score(cluster, 2) == 1.0


def test_compute_frequency_score_zero_total():
    """A zero or negative total yields 0.0 rather than dividing by zero."""
    cluster = [{"text": "x", "paper_ids": ["a"], "years": [2024]}]
    assert compute_frequency_score(cluster, 0) == 0.0


def test_compute_frequency_score_range():
    """frequency_score stays within [0.0, 1.0]."""
    cluster = [{"text": "x", "paper_ids": ["a", "b"], "years": [2024, 2024]}]
    score = compute_frequency_score(cluster, 3)
    assert 0.0 <= score <= 1.0


def test_compute_frequency_score_defaults_to_full_weight_without_tiers():
    """Missing tier data weights each paper as 1.0 (backwards-compatible behaviour)."""
    cluster = [{"text": "x", "paper_ids": ["a", "b"], "years": [2024, 2024]}]
    assert compute_frequency_score(cluster, 4) == 0.5  # (1.0 + 1.0) / 4


def test_compute_frequency_score_weights_by_tier():
    """Each paper contributes its tier weight: explicit=1.0, conclusion=0.75, inferred=0.5."""
    cluster = [
        {
            "text": "x",
            "paper_ids": ["a", "b", "c"],
            "years": [2024, 2024, 2024],
            "tiers": ["explicit", "conclusion", "inferred"],
        }
    ]
    # (1.0 + 0.75 + 0.5) / 3 = 0.75
    assert compute_frequency_score(cluster, 3) == 0.75


def test_compute_frequency_score_keeps_strongest_tier_per_paper():
    """A paper appearing with several tiers counts once, at its strongest weight."""
    cluster = [
        {"text": "x", "paper_ids": ["a"], "years": [2024], "tiers": ["inferred"]},
        {"text": "y", "paper_ids": ["a"], "years": [2024], "tiers": ["explicit"]},
    ]
    # Paper 'a' counted once at the explicit weight 1.0 → 1.0 / 2
    assert compute_frequency_score(cluster, 2) == 0.5


# ---------------------------------------------------------------------------
# compute_recency_score
# ---------------------------------------------------------------------------


def test_compute_recency_score_all_recent():
    """All papers within the last 2 years gives 1.0."""
    cluster = [{"text": "x", "paper_ids": ["a", "b"], "years": [2024, 2023]}]
    assert compute_recency_score(cluster, current_year=2024) == 1.0


def test_compute_recency_score_mixed():
    """Half recent, half old gives 0.5."""
    cluster = [
        {"text": "x", "paper_ids": ["a", "b", "c", "d"], "years": [2024, 2023, 2020, 2019]}
    ]
    assert compute_recency_score(cluster, current_year=2024) == 0.5


def test_compute_recency_score_all_same_old_year():
    """All papers in the same old year (edge case) gives 0.0 recent."""
    cluster = [{"text": "x", "paper_ids": ["a", "b"], "years": [2018, 2018]}]
    assert compute_recency_score(cluster, current_year=2024) == 0.0


def test_compute_recency_score_no_year_data_returns_half():
    """Missing year data (None / 0) falls back to 0.5."""
    assert compute_recency_score([{"text": "x", "paper_ids": ["a"], "years": [None]}]) == 0.5
    assert compute_recency_score([{"text": "x", "paper_ids": ["a"], "years": [0]}]) == 0.5


def test_compute_recency_score_range():
    """recency_score stays within [0.0, 1.0]."""
    cluster = [{"text": "x", "paper_ids": ["a", "b", "c"], "years": [2024, 2023, 2010]}]
    score = compute_recency_score(cluster, current_year=2024)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# recency baseline: no hardcoded "now"
# ---------------------------------------------------------------------------


def test_compute_recency_score_window_slides_with_current_year():
    """The 2-year window tracks current_year instead of a fixed baseline.

    Regression guard for the hardcoded current_year=2024 default. The same cluster
    is scored against an advancing baseline; each step should drop the papers that
    have fallen out of the window. A reintroduced 2024 constant freezes this
    sequence and the assertion fails.
    """
    cluster = [
        {"text": "x", "paper_ids": ["a", "b", "c", "d"], "years": [2022, 2023, 2024, 2025]}
    ]
    # The window is an open-ended lower bound: year >= current_year - 1. Advancing
    # the baseline drops one paper at a time off the old end.
    assert compute_recency_score(cluster, current_year=2023) == 1.0  # >=2022: all 4
    assert compute_recency_score(cluster, current_year=2024) == 0.75  # >=2023: 3
    assert compute_recency_score(cluster, current_year=2025) == 0.5  # >=2024: 2
    assert compute_recency_score(cluster, current_year=2026) == 0.25  # >=2025: 1
    assert compute_recency_score(cluster, current_year=2027) == 0.0  # >=2026: none


@pytest.mark.parametrize("year", [2015, 2019, 2024, 2025, 2031])
def test_compute_recency_score_default_baseline_has_no_privileged_year(monkeypatch, year):
    """The *default* baseline tracks the clock rather than a frozen constant.

    This exercises the no-argument path deliberately. Passing current_year
    explicitly would prove nothing: the old hardcoded implementation honoured an
    explicit argument too, so only the default can expose a frozen year. 2024 is
    one parameter among several so it cannot pass by coincidence.
    """
    monkeypatch.setattr(gs, "_current_year", lambda: year)

    at_baseline = [{"text": "x", "paper_ids": ["a"], "years": [year]}]
    assert compute_recency_score(at_baseline) == 1.0

    well_before = [{"text": "x", "paper_ids": ["a"], "years": [year - 5]}]
    assert compute_recency_score(well_before) == 0.0


def test_compute_recency_score_defaults_to_wall_clock(monkeypatch):
    """With no baseline supplied, the fallback is the real current year, not 2024."""
    monkeypatch.setattr(gs, "_current_year", lambda: 2031)
    cluster = [{"text": "x", "paper_ids": ["a", "b"], "years": [2030, 2024]}]
    # only the 2030 paper is inside a 2031 window
    assert compute_recency_score(cluster) == 0.5


# ---------------------------------------------------------------------------
# _corpus_reference_year
# ---------------------------------------------------------------------------


def test_corpus_reference_year_uses_newest_paper(monkeypatch):
    """The baseline is the corpus's leading edge, not the wall clock."""
    monkeypatch.setattr(gs, "_current_year", lambda: 2031)
    limitations = [
        {"text": "x", "paper_ids": ["a"], "years": [2019, 2023]},
        {"text": "y", "paper_ids": ["b"], "years": [2025]},
    ]
    assert _corpus_reference_year(limitations) == 2025


def test_corpus_reference_year_clamps_future_dated_metadata(monkeypatch):
    """A mislabelled future year cannot push the window past every real paper."""
    monkeypatch.setattr(gs, "_current_year", lambda: 2026)
    limitations = [{"text": "x", "paper_ids": ["a", "b"], "years": [2025, 2099]}]
    assert _corpus_reference_year(limitations) == 2026


def test_corpus_reference_year_ignores_missing_years(monkeypatch):
    """None / 0 year entries are skipped rather than treated as year zero."""
    monkeypatch.setattr(gs, "_current_year", lambda: 2031)
    limitations = [{"text": "x", "paper_ids": ["a", "b", "c"], "years": [None, 0, 2023]}]
    assert _corpus_reference_year(limitations) == 2023


def test_corpus_reference_year_falls_back_when_no_year_data(monkeypatch):
    """With no usable years at all, fall back to the current year."""
    monkeypatch.setattr(gs, "_current_year", lambda: 2031)
    assert _corpus_reference_year([]) == 2031
    assert _corpus_reference_year([{"text": "x", "paper_ids": ["a"], "years": []}]) == 2031


def test_score_gaps_anchors_recency_to_corpus_not_wall_clock(monkeypatch):
    """score_gaps derives the baseline from the corpus so an ageing corpus still ranks.

    The corpus tops out at 2025 while 'today' is 2031. Against the wall clock every
    paper would be stale and recency would collapse to 0.0 for every cluster,
    nulling 35% of the formula. Anchored to the corpus, the 2024/2025 cluster still
    scores as recent and stays separable from the 2019 one.
    """
    monkeypatch.setattr(gs, "_current_year", lambda: 2031)
    # Corpus tops out at 2025, so the window is year >= 2024. The 2023 cluster is
    # the discriminating case: it is inside a hardcoded-2024 window (>= 2023) but
    # outside the corpus-anchored one, so a frozen constant fails on it. The 2025
    # cluster catches the opposite error — a wall-clock baseline would zero it.
    recent = [{"text": "recent gap", "paper_ids": ["a"], "years": [2025]}]
    mid = [{"text": "mid gap", "paper_ids": ["b"], "years": [2023]}]
    stale = [{"text": "stale gap", "paper_ids": ["c"], "years": [2019]}]

    monkeypatch.setattr(
        gs,
        "get_all_limitations",
        lambda domain="computer_vision": recent + mid + stale,
    )
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 3)
    monkeypatch.setattr(
        gs, "cluster_limitations", lambda lims, domain=None: [recent, mid, stale]
    )
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])

    results = {r.gap_description: r for r in score_gaps()}

    assert results["recent gap"].recency_score == 1.0  # fails under a wall clock
    assert results["mid gap"].recency_score == 0.0  # fails under a frozen 2024
    assert results["stale gap"].recency_score == 0.0


def test_score_gaps_recency_baseline_is_corpus_wide_not_per_cluster(monkeypatch):
    """An old cluster is not rescued by being measured against its own newest paper."""
    monkeypatch.setattr(gs, "_current_year", lambda: 2031)
    recent = [{"text": "recent gap", "paper_ids": ["a"], "years": [2025]}]
    stale = [{"text": "stale gap", "paper_ids": ["b", "c"], "years": [2018, 2019]}]

    monkeypatch.setattr(
        gs, "get_all_limitations", lambda domain="computer_vision": recent + stale
    )
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 3)
    monkeypatch.setattr(
        gs, "cluster_limitations", lambda lims, domain=None: [recent, stale]
    )
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])

    results = {r.gap_description: r for r in score_gaps()}

    # Per-cluster baselines would give the stale cluster 2019 as its own "now"
    # and score it 1.0. Corpus-wide (2025) correctly scores it 0.0.
    assert results["stale gap"].recency_score == 0.0
    assert results["recent gap"].recency_score == 1.0


# ---------------------------------------------------------------------------
# compute_solution_deficit_score
# ---------------------------------------------------------------------------


def test_compute_solution_deficit_no_future_directions(monkeypatch):
    """No addressing future directions => maximally deficient => 1.0."""
    cluster = [{"text": "slow training", "paper_ids": ["a", "b"], "years": [2024, 2024]}]
    client = _make_qdrant_query_mock([])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    assert compute_solution_deficit_score(cluster) == 1.0


def test_compute_solution_deficit_partial_coverage(monkeypatch):
    """One of two reporting papers addressed => deficit 1 - 1/2 = 0.5."""
    cluster = [{"text": "slow training", "paper_ids": ["a", "b"], "years": [2024, 2024]}]
    # One hit above 0.85, one below — only the first counts. The 0.80 hit would have
    # counted under the old 0.75 threshold but no longer does.
    hits = [_make_hit("use adamw", 0.90), _make_hit("close but weak", 0.80)]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    assert compute_solution_deficit_score(cluster) == 0.5


def test_compute_solution_deficit_clamped_to_zero(monkeypatch):
    """More matches than reporting papers floors at zero, never yielding a negative score."""
    cluster = [{"text": "slow training", "paper_ids": ["a"], "years": [2024]}]
    # Three matches above 0.85 but only one reporting paper. The raw ratio is 3/1,
    # so the score is a negative 1 - 3 = -2 before the [0.0, 1.0] clamp floors it.
    # This is what makes the old min(matches, papers_reporting) cap redundant.
    hits = [_make_hit("fix one", 0.9), _make_hit("fix two", 0.88), _make_hit("fix three", 0.87)]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    assert compute_solution_deficit_score(cluster) == 0.0


def test_compute_solution_deficit_no_papers_returns_one():
    """Empty cluster or no reporting papers => 1.0 without touching Qdrant."""
    assert compute_solution_deficit_score([]) == 1.0
    assert compute_solution_deficit_score([{"text": "x", "paper_ids": [], "years": []}]) == 1.0


def test_compute_solution_deficit_range(monkeypatch):
    """solution_deficit_score stays within [0.0, 1.0]."""
    cluster = [{"text": "x", "paper_ids": ["a", "b", "c"], "years": [2024, 2024, 2024]}]
    client = _make_qdrant_query_mock([_make_hit("sol", 0.9)])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    score = compute_solution_deficit_score(cluster)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _find_addressing_solutions — domain filtering and self-match exclusion
# ---------------------------------------------------------------------------


def _make_fd_hit(
    text: str,
    score: float,
    paper_ids: list[str] | None = None,
    domain: str = "computer_vision",
) -> types.SimpleNamespace:
    """A future-direction hit carrying the full payload embed.py actually writes."""
    return types.SimpleNamespace(
        score=score,
        payload={
            "limitation_text": text,
            "paper_ids": paper_ids if paper_ids is not None else [],
            "domain": domain,
        },
    )


def test_solution_thresholds_sit_at_the_measured_noise_floors():
    """Thresholds are the per-domain 95th percentile of random same-domain pairs.

    Advisor decision A1 (2026-08-23). The previous single 0.85 was below both
    floors: 20.1% of random CV (limitation, future-direction) pairs cleared it
    and 44.8% of random MI pairs did, so a fifth to nearly half of "addressed"
    verdicts were chance.
    """
    # Deliberately NOT pinned to exact literals. These constants are owned by
    # scripts/derive_thresholds.py and are rewritten whenever their drift from the
    # measured null exceeds DRIFT_TOLERANCE — so asserting a literal here would turn
    # every legitimate re-derivation into a unit-test failure, and the temptation
    # would be to "fix" the test by copying whatever the code now says, which proves
    # nothing. The exact values are guarded against the live null distribution by
    # tests/integration/test_threshold_derivation.py. What is asserted here is the
    # structure that must hold regardless of the measured numbers.
    assert set(gs._SOLUTION_THRESHOLDS) == {"computer_vision", "medical_imaging"}
    # Every configured floor must be above the old value, or the fix is a no-op.
    assert all(v > 0.85 for v in gs._SOLUTION_THRESHOLDS.values())
    # And each must be a plausible cosine floor rather than a typo.
    assert all(0.85 < v < 1.0 for v in gs._SOLUTION_THRESHOLDS.values())


def test_solution_threshold_lookup_is_per_domain():
    """Each domain gets its own floor rather than one global constant."""
    for domain, value in gs._SOLUTION_THRESHOLDS.items():
        assert gs._solution_threshold(domain) == value
    assert gs._solution_threshold("computer_vision") != gs._solution_threshold(
        "medical_imaging"
    )


def test_solution_threshold_unknown_domain_is_conservative():
    """An unmeasured domain gets the strictest known floor, not the loosest.

    Guessing low would silently count noise as solutions in a domain whose null
    was never measured; guessing high only under-counts, which is recoverable.
    """
    unknown = gs._solution_threshold("robotics")
    assert unknown == max(gs._SOLUTION_THRESHOLDS.values())
    assert unknown >= max(gs._SOLUTION_THRESHOLDS.values())
    assert unknown > min(gs._SOLUTION_THRESHOLDS.values()) or len(
        set(gs._SOLUTION_THRESHOLDS.values())
    ) == 1


def test_find_addressing_solutions_uses_the_domain_threshold(monkeypatch):
    """A hit between the old 0.85 and the CV floor no longer counts as a solution."""
    hits = [
        _make_fd_hit("just below the CV floor", 0.87, paper_ids=["p9"]),
        _make_fd_hit("clearly above it", 0.95, paper_ids=["p9"]),
    ]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    solutions = gs._find_addressing_solutions("a gap", domain="computer_vision")

    assert solutions == ["clearly above it"]


def test_find_addressing_solutions_medical_floor_is_stricter(monkeypatch):
    """0.89 counts in CV but not in medical imaging, whose null runs higher."""
    hits = [_make_fd_hit("mid-band hit", 0.89, paper_ids=["p9"])]

    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    assert gs._find_addressing_solutions("g", domain="computer_vision") == ["mid-band hit"]

    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    assert gs._find_addressing_solutions("g", domain="medical_imaging") == []


def test_find_addressing_solutions_applies_domain_filter(monkeypatch):
    """The Qdrant query must carry a server-side domain filter.

    Regression guard for cross-domain contamination: the old implementation sent
    no query_filter at all, so a medical-imaging future direction could mark a CV
    gap as solved.
    """
    client = _make_qdrant_query_mock([])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    gs._find_addressing_solutions("a gap", domain="medical_imaging")

    query_filter = client.query_points.call_args.kwargs["query_filter"]
    condition = query_filter.must[0]
    assert condition.key == "domain"
    assert condition.match.value == "medical_imaging"


def test_find_addressing_solutions_excludes_self_authored(monkeypatch):
    """A future direction from a paper that reports the limitation does not count."""
    hits = [
        _make_fd_hit("we leave this to future work", 0.95, paper_ids=["p1"]),
        _make_fd_hit("an independent proposal", 0.95, paper_ids=["p9"]),
    ]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    solutions = gs._find_addressing_solutions(
        "a gap", domain="computer_vision", exclude_paper_ids={"p1"}
    )

    assert solutions == ["an independent proposal"]


def test_find_addressing_solutions_excludes_on_any_shared_paper(monkeypatch):
    """A future direction attached to several papers is excluded if *any* reports it."""
    hits = [_make_fd_hit("shared direction", 0.95, paper_ids=["p9", "p1"])]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    assert (
        gs._find_addressing_solutions(
            "a gap", domain="computer_vision", exclude_paper_ids={"p1"}
        )
        == []
    )


def test_find_addressing_solutions_keeps_third_party_solutions(monkeypatch):
    """Exclusion is targeted — unrelated papers' directions still count."""
    hits = [_make_fd_hit("genuine solution", 0.95, paper_ids=["p7"])]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    assert gs._find_addressing_solutions(
        "a gap", domain="computer_vision", exclude_paper_ids={"p1", "p2"}
    ) == ["genuine solution"]


def test_compute_solution_deficit_ignores_self_authored_solution(monkeypatch):
    """A paper restating its own open problem leaves the gap fully deficient.

    Under the old code this scored 0.0 — the limitation looked completely solved
    because the only 'solution' was the reporting paper's own future work.
    """
    cluster = [{"text": "an open problem", "paper_ids": ["p1"], "years": [2025]}]
    hits = [_make_fd_hit("we plan to address this", 0.95, paper_ids=["p1"])]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    assert compute_solution_deficit_score(cluster) == 1.0


def test_compute_solution_deficit_counts_only_independent_solutions(monkeypatch):
    """With two reporting papers, only the outside proposal reduces the deficit."""
    cluster = [{"text": "an open problem", "paper_ids": ["p1", "p2"], "years": [2025, 2025]}]
    hits = [
        _make_fd_hit("p1's own future work", 0.95, paper_ids=["p1"]),
        _make_fd_hit("outside proposal", 0.95, paper_ids=["p9"]),
    ]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    # 1 independent match against 2 reporting papers => 1 - 1/2
    assert compute_solution_deficit_score(cluster) == 0.5


def test_compute_solution_deficit_threads_domain_to_query(monkeypatch):
    """The cluster's domain reaches the Qdrant filter rather than defaulting."""
    cluster = [{"text": "x", "paper_ids": ["p1"], "years": [2025]}]
    client = _make_qdrant_query_mock([])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    compute_solution_deficit_score(cluster, domain="medical_imaging")

    condition = client.query_points.call_args.kwargs["query_filter"].must[0]
    assert condition.match.value == "medical_imaging"


def test_score_gaps_display_solutions_match_scored_solutions(monkeypatch):
    """proposed_solutions shows exactly what the deficit counted — never a looser set.

    The advisor decision was that both call sites share one filtering policy, so a
    self-authored direction must be absent from the user-facing list too.
    """
    cluster = [{"text": "an open gap", "paper_ids": ["p1"], "years": [2025]}]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 1)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: [cluster])

    hits = [
        _make_fd_hit("p1's own future work", 0.95, paper_ids=["p1"]),
        _make_fd_hit("an independent proposal", 0.95, paper_ids=["p9"]),
    ]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    result = score_gaps()[0]

    assert result.proposed_solutions == ["an independent proposal"]
    assert result.solution_deficit_score == 0.0  # one independent match, one paper


def test_score_gaps_threads_domain_into_solution_search(monkeypatch):
    """score_gaps(domain=...) reaches the future-direction filter."""
    cluster = [{"text": "an open gap", "paper_ids": ["p1"], "years": [2025]}]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 1)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: [cluster])
    client = _make_qdrant_query_mock([])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    score_gaps(domain="medical_imaging")

    for call in client.query_points.call_args_list:
        assert call.kwargs["query_filter"].must[0].match.value == "medical_imaging"


# ---------------------------------------------------------------------------
# cluster_limitations
# ---------------------------------------------------------------------------


def test_cluster_limitations_empty_input():
    """No limitations yields no clusters."""
    assert cluster_limitations([]) == []


def test_cluster_limitations_groups_similar(monkeypatch):
    """Limitations scoring >= the domain threshold against the seed collapse into one cluster."""
    lims = [
        # Longest text → picked as seed first.
        {"text": "convergence is slow on long sequences", "paper_ids": ["a"], "years": [2024]},
        {"text": "training is slow", "paper_ids": ["b"], "years": [2024]},
    ]
    seed_hits = [
        _make_hit("convergence is slow on long sequences", 1.0),
        _make_hit("training is slow", 0.90),
    ]
    client = _make_qdrant_batch_mock([seed_hits, []])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims)

    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_limitations_singletons_below_threshold(monkeypatch):
    """Limitations whose neighbours all score below the domain threshold stay singletons."""
    lims = [
        {"text": "slow convergence", "paper_ids": ["a"], "years": [2024]},
        {"text": "high memory use", "paper_ids": ["b"], "years": [2024]},
    ]
    hits_0 = [_make_hit("high memory use", 0.40)]
    hits_1 = [_make_hit("slow convergence", 0.40)]
    client = _make_qdrant_batch_mock([hits_0, hits_1])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims)

    # Singletons are valid clusters — never discarded.
    assert len(clusters) == 2
    assert all(len(c) == 1 for c in clusters)


def test_cluster_limitations_non_transitive(monkeypatch):
    """A~B and B~C above threshold but A~C below → two clusters, not one.

    Union-find would have chained A→B→C into a single cluster; seed-anchored
    grouping only admits members similar to the seed itself.
    """
    a = {"text": "attention maps degrade badly at high image resolution", "paper_ids": ["p1"], "years": [2024]}
    b = {"text": "attention degrades at high resolution", "paper_ids": ["p2"], "years": [2024]}
    c = {"text": "attention is costly", "paper_ids": ["p3"], "years": [2024]}
    # a is the longest text → first seed. sim(a,b)=0.90, sim(b,c)=0.90, sim(a,c)=0.50.
    hits_a = [_make_hit(a["text"], 1.0), _make_hit(b["text"], 0.90), _make_hit(c["text"], 0.50)]
    hits_b = [_make_hit(b["text"], 1.0), _make_hit(a["text"], 0.90), _make_hit(c["text"], 0.90)]
    hits_c = [_make_hit(c["text"], 1.0), _make_hit(b["text"], 0.90), _make_hit(a["text"], 0.50)]
    client = _make_qdrant_batch_mock([hits_a, hits_b, hits_c])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations([a, b, c])

    assert len(clusters) == 2
    by_size = sorted(clusters, key=len, reverse=True)
    assert {lim["text"] for lim in by_size[0]} == {a["text"], b["text"]}
    assert {lim["text"] for lim in by_size[1]} == {c["text"]}


def test_cluster_limitations_seed_anchored_assignment(monkeypatch):
    """An assigned limitation never joins a later seed's cluster, even above threshold."""
    a = {"text": "gradient noise dominates at very small batch sizes", "paper_ids": ["p1"], "years": [2024]}
    b = {"text": "gradient noise at small batch sizes", "paper_ids": ["p2"], "years": [2024]}
    d = {"text": "noisy gradients", "paper_ids": ["p3"], "years": [2024]}
    # Seed order by length: a, b, d. b is similar to both a and d.
    hits_a = [_make_hit(b["text"], 0.90)]
    hits_b = [_make_hit(a["text"], 0.90), _make_hit(d["text"], 0.90)]
    hits_d = [_make_hit(b["text"], 0.90)]  # b already claimed by a's cluster
    client = _make_qdrant_batch_mock([hits_a, hits_b, hits_d])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations([a, b, d])

    assert len(clusters) == 2
    by_size = sorted(clusters, key=len, reverse=True)
    assert {lim["text"] for lim in by_size[0]} == {a["text"], b["text"]}
    assert {lim["text"] for lim in by_size[1]} == {d["text"]}


def test_cluster_limitations_batch_embedding_called_once(monkeypatch):
    """All texts are embedded in one encode() call and one batched Qdrant query."""
    lims = [
        {"text": f"limitation number {i}", "paper_ids": [f"p{i}"], "years": [2024]}
        for i in range(3)
    ]
    client = _make_qdrant_batch_mock([[], [], []])
    model = _make_model_mock()
    _patch_vector_backends(monkeypatch, client, model)

    cluster_limitations(lims)

    assert model.encode.call_count == 1
    encoded_texts = model.encode.call_args.args[0]
    assert list(encoded_texts) == [lim["text"] for lim in lims]
    assert model.encode.call_args.kwargs.get("batch_size") == 32
    assert client.query_batch_points.call_count == 1
    assert client.query_points.call_count == 0


def test_cluster_limitations_sequential_fallback(monkeypatch):
    """Clients without query_batch_points fall back to one query_points per cached vector."""
    lims = [
        {"text": "slow convergence", "paper_ids": ["a"], "years": [2024]},
        {"text": "high memory use", "paper_ids": ["b"], "years": [2024]},
    ]
    client = MagicMock()
    del client.query_batch_points  # simulate an older client without the batch endpoint
    client.query_points.side_effect = [
        types.SimpleNamespace(points=[]),
        types.SimpleNamespace(points=[]),
    ]
    model = _make_model_mock()
    _patch_vector_backends(monkeypatch, client, model)

    clusters = cluster_limitations(lims)

    assert len(clusters) == 2
    assert client.query_points.call_count == 2
    assert model.encode.call_count == 1  # embeddings still batched and reused


def test_cluster_limitations_partitions_all_inputs(monkeypatch):
    """Every input limitation appears in exactly one cluster."""
    lims = [
        {"text": "aaaa", "paper_ids": ["p1"], "years": [2024]},
        {"text": "bbb", "paper_ids": ["p2"], "years": [2024]},
        {"text": "cc", "paper_ids": ["p3"], "years": [2024]},
    ]
    # a and b are similar (0.9); c matches nothing.
    hits_a = [_make_hit("bbb", 0.9)]
    hits_b = [_make_hit("aaaa", 0.9)]
    hits_c = [_make_hit("bbb", 0.3)]
    client = _make_qdrant_batch_mock([hits_a, hits_b, hits_c])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims)

    total = sum(len(c) for c in clusters)
    assert total == 3
    assert {lim["text"] for c in clusters for lim in c} == {"aaaa", "bbb", "cc"}


# ---------------------------------------------------------------------------
# score_gaps
# ---------------------------------------------------------------------------


def test_score_gaps_empty_limitations(monkeypatch):
    """No limitations => empty result, no clustering or scoring attempted."""
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": [])
    assert score_gaps() == []


def test_score_gaps_formula_weights(monkeypatch):
    """Composite score equals 0.40*freq + 0.35*recency + 0.25*deficit."""
    cluster = [{"text": "slow training", "paper_ids": ["a"], "years": [2024]}]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 5)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: [cluster])
    monkeypatch.setattr(gs, "compute_frequency_score", lambda c, t: 0.6)
    monkeypatch.setattr(gs, "compute_recency_score", lambda c, current_year=2024: 0.4)
    monkeypatch.setattr(gs, "compute_solution_deficit_score", lambda c, **kw: 0.8)
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])

    results = score_gaps()

    assert len(results) == 1
    expected = round(0.40 * 0.6 + 0.35 * 0.4 + 0.25 * 0.8, 4)
    assert results[0].score == expected
    assert results[0].frequency_score == 0.6
    assert results[0].recency_score == 0.4
    assert results[0].solution_deficit_score == 0.8


def test_score_gaps_returns_sorted_list(monkeypatch):
    """Results are GapResults sorted by score descending."""
    c_low = [{"text": "low gap", "paper_ids": ["a"], "years": [2024]}]
    c_high = [{"text": "high gap", "paper_ids": ["b"], "years": [2024]}]

    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": c_low + c_high)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 4)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: [c_low, c_high])
    monkeypatch.setattr(gs, "compute_frequency_score", lambda c, t: 0.9 if c is c_high else 0.1)
    monkeypatch.setattr(gs, "compute_recency_score", lambda c, current_year=2024: 0.5)
    monkeypatch.setattr(gs, "compute_solution_deficit_score", lambda c, **kw: 0.5)
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])

    results = score_gaps()

    assert all(isinstance(r, GapResult) for r in results)
    assert [r.gap_description for r in results] == ["high gap", "low gap"]
    assert results[0].score >= results[1].score


def test_score_gaps_respects_top_n(monkeypatch):
    """Only the top_n highest-scoring gaps are returned."""
    clusters = [
        [{"text": f"lim{i}", "paper_ids": [f"p{i}"], "years": [2024]}] for i in range(5)
    ]
    flat = [c[0] for c in clusters]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": flat)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 5)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: clusters)
    # avoid Qdrant: no addressing solutions => deficit computed without network
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])

    results = score_gaps(top_n=2)

    assert len(results) == 2


def test_score_gaps_uses_the_centroid_nearest_text_as_description(monkeypatch):
    """gap_description is the cluster's centroid-nearest member.

    Replaces test_score_gaps_uses_most_frequent_text_as_description, which asserted
    the old "most frequent text" semantics. Two things were wrong with it. The
    behaviour it pinned could never fire — `Limitation` is UNIQUE on text and
    `get_all_limitations` groups by `l.text`, so every count inside a cluster is
    always 1 and `most_common` always resolved a total tie by insertion order, i.e.
    the longest seed text. And its fixture put two identical texts in one cluster,
    which the graph cannot produce.
    """
    cluster = [
        # The longest text sits deliberately far from the other two, so "longest"
        # and "centroid-nearest" give different answers and the assertion can tell
        # the fix from the bug.
        {"text": "an outlying limitation with by far the longest text in this cluster",
         "paper_ids": ["a"], "years": [2024], "tiers": ["explicit"]},
        {"text": "middle gap", "paper_ids": ["b"], "years": [2024], "tiers": ["explicit"]},
        {"text": "near middle", "paper_ids": ["c"], "years": [2024], "tiers": ["explicit"]},
    ]
    vectors = {
        cluster[0]["text"]: [1.0, 0.0, 0.0],
        cluster[1]["text"]: [0.0, 1.0, 0.0],
        cluster[2]["text"]: [0.0, 0.98, 0.20],
    }
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.array([vectors[t] for t in texts])

    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_contributing_papers", lambda domain: 3)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: [cluster])
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])
    monkeypatch.setattr(gs, "load_embedding_model", lambda: model)
    monkeypatch.setattr(gs, "_embed_texts", lambda m, texts: m.encode(texts).tolist())
    gs._representative_cache.clear()

    results = score_gaps()

    assert results[0].gap_description in {"middle gap", "near middle"}
    assert results[0].gap_description != cluster[0]["text"], (
        "the longest member must not win merely for being longest"
    )
    assert results[0].supporting_papers == ["a", "b", "c"]


def test_score_gaps_collects_proposed_solutions(monkeypatch):
    """proposed_solutions come from addressing future directions for the cluster."""
    cluster = [{"text": "needs solving", "paper_ids": ["a"], "years": [2024]}]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 1)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, domain=None: [cluster])
    monkeypatch.setattr(gs, "compute_solution_deficit_score", lambda c, **kw: 0.0)
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: ["try approach X"])

    results = score_gaps()

    assert results[0].proposed_solutions == ["try approach X"]


# ---------------------------------------------------------------------------
# GapResult model
# ---------------------------------------------------------------------------


def test_gap_result_model_rejects_missing_fields():
    """GapResult requires all fields — missing ones raise ValidationError."""
    with pytest.raises(ValidationError):
        GapResult(gap_description="x", score=0.5)


def test_gap_result_model_accepts_full_payload():
    """A fully-specified GapResult validates and round-trips its fields."""
    gap = GapResult(
        gap_description="x",
        score=0.5,
        frequency_score=0.4,
        recency_score=0.3,
        solution_deficit_score=0.2,
        supporting_papers=["a"],
        proposed_solutions=["b"],
    )
    assert gap.score == 0.5
    assert gap.supporting_papers == ["a"]
    assert gap.proposed_solutions == ["b"]


# ---------------------------------------------------------------------------
# Cluster threshold derivation and the size cap — PLAN.md #3
# ---------------------------------------------------------------------------


def test_cluster_thresholds_are_per_domain_and_above_their_nulls():
    """Each domain's threshold must be its own measured null p95, not a shared guess.

    Fails against pre-fix code, which had a single undrived _CLUSTER_THRESHOLD of
    0.86 — below both domains' null p95 (CV 0.8769, MI 0.8954), so 10.9% of
    arbitrary CV pairs and 28.3% of arbitrary MI pairs cleared it.
    """
    from pipeline.gap_scorer import _CLUSTER_THRESHOLDS, _cluster_threshold

    assert set(_CLUSTER_THRESHOLDS) == {"computer_vision", "medical_imaging"}
    for domain, value in _CLUSTER_THRESHOLDS.items():
        assert _cluster_threshold(domain) == value
        assert value > 0.86, (
            f"{domain} threshold {value} is at or below the old undrived 0.86, "
            "which sat beneath this population's noise floor"
        )


def test_unmeasured_domain_falls_back_to_the_strictest_threshold():
    """Guessing low would silently merge unrelated limitations; guessing high only fragments."""
    from pipeline.gap_scorer import _CLUSTER_THRESHOLDS, _cluster_threshold

    assert _cluster_threshold("a_domain_with_no_measured_null") == max(
        _CLUSTER_THRESHOLDS.values()
    )


@pytest.mark.parametrize(
    "total,expected",
    [(0, 2), (1, 2), (2, 2), (5, 2), (10, 2), (48, 10), (104, 21), (106, 22), (200, 40)],
)
def test_cluster_cap_is_a_share_with_a_floor(total, expected):
    """20% of the domain, rounded up, but never below 2.

    A cap of 1 would mean "no clusters may exist", so the floor is a degeneracy
    guard rather than a tuning dial.
    """
    from pipeline.gap_scorer import _cluster_cap

    assert _cluster_cap(total) == expected


def test_oversized_cluster_splits_instead_of_truncating(monkeypatch):
    """Members past the cap must re-seed their own clusters, never be dropped.

    Fails against pre-fix code, which had no cap at all: 52 of 104 medical-imaging
    limitations landed in a single cluster reported as one gap "supported by" 33
    papers.
    """
    from pipeline.gap_scorer import _cluster_cap, cluster_limitations

    # 10 limitations -> cap 2, so a seed may keep exactly one other member.
    lims = [
        {"text": f"limitation number {i} about a closely related failure mode",
         "paper_ids": [f"p{i}"], "years": [2024], "tiers": ["explicit"]}
        for i in range(10)
    ]
    assert _cluster_cap(len(lims)) == 2

    # Every limitation is highly similar to every other, so without a cap the
    # first seed would absorb all ten.
    all_hits = [
        [_make_hit(lim["text"], 0.99 - 0.001 * j) for j, lim in enumerate(lims)]
        for _ in lims
    ]
    client = _make_qdrant_batch_mock(all_hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims, domain="computer_vision")

    assert all(len(c) <= 2 for c in clusters), [len(c) for c in clusters]
    flattened = [m["text"] for c in clusters for m in c]
    assert len(flattened) == len(lims), "a split must re-seed, not truncate"
    assert sorted(flattened) == sorted(l["text"] for l in lims)


def test_cluster_admits_the_most_similar_candidates_first(monkeypatch):
    """When the cap binds, the tightest subgroup stays with the seed."""
    from pipeline.gap_scorer import cluster_limitations

    lims = [
        {"text": "a seed limitation with the longest text of the group here",
         "paper_ids": ["p0"], "years": [2024], "tiers": ["explicit"]},
        {"text": "barely similar", "paper_ids": ["p1"], "years": [2024], "tiers": ["explicit"]},
        {"text": "extremely similar", "paper_ids": ["p2"], "years": [2024], "tiers": ["explicit"]},
    ]
    # cap for 3 limitations is 2, so exactly one of the two candidates is admitted.
    seed_hits = [
        _make_hit(lims[1]["text"], 0.8800),   # clears the CV threshold, but lower
        _make_hit(lims[2]["text"], 0.9500),   # much closer to the seed
    ]
    client = _make_qdrant_batch_mock([seed_hits, [], []])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims, domain="computer_vision")
    seed_cluster = next(c for c in clusters if c[0]["text"] == lims[0]["text"])

    assert len(seed_cluster) == 2
    assert seed_cluster[1]["text"] == "extremely similar", (
        "the cap must keep the most similar candidate, not whichever Qdrant returned first"
    )


def test_neighbour_query_sends_a_domain_filter(monkeypatch):
    """The neighbour query must narrow by domain server-side.

    Fails against pre-fix code, which sent no filter — the only filtered query in
    the project that didn't — so a mean 42 of every 64 CV hits were other-domain
    points that were fetched, discarded, and crowded genuine same-domain
    neighbours out of the limit window.
    """
    from pipeline.gap_scorer import _query_neighbours

    captured = {}

    class Client:
        def query_batch_points(self, collection_name, requests):
            captured["requests"] = requests
            return [MagicMock(points=[]) for _ in requests]

    _query_neighbours(Client(), [[0.1] * 768], limit=5, domain="medical_imaging")

    request = captured["requests"][0]
    assert request.filter is not None, "no domain filter was sent"
    condition = request.filter.must[0]
    assert condition.key == "domain"
    assert condition.match.value == "medical_imaging"


def test_neighbour_query_without_a_domain_sends_no_filter():
    """The parameter is optional so standalone/diagnostic calls can scan everything."""
    from pipeline.gap_scorer import _query_neighbours

    captured = {}

    class Client:
        def query_batch_points(self, collection_name, requests):
            captured["requests"] = requests
            return [MagicMock(points=[]) for _ in requests]

    _query_neighbours(Client(), [[0.1] * 768], limit=5)
    assert captured["requests"][0].filter is None


# ---------------------------------------------------------------------------
# Cluster representative text — PLAN.md #2
# ---------------------------------------------------------------------------


def _fixed_vector_model(mapping: dict[str, list[float]]) -> MagicMock:
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.array([mapping[t] for t in texts])
    return model


def test_representative_is_the_centroid_nearest_member(monkeypatch):
    """Fails against pre-fix code, which always returned the longest text.

    Measured on the pre-fix live corpus: the description equalled the longest
    member in 27 of 27 CV clusters, because Counter.most_common always saw a total
    tie and resolved it by insertion order — the seed, chosen as the longest string.
    """
    cluster = [
        {"text": "a very long outlying statement that shares little with the others"},
        {"text": "short a"},
        {"text": "short b"},
    ]
    mapping = {
        cluster[0]["text"]: [1.0, 0.0, 0.0],
        cluster[1]["text"]: [0.0, 1.0, 0.0],
        cluster[2]["text"]: [0.0, 0.99, 0.14],
    }
    monkeypatch.setattr(gs, "load_embedding_model", lambda: _fixed_vector_model(mapping))
    monkeypatch.setattr(gs, "_embed_texts", lambda m, texts: m.encode(texts).tolist())
    gs._representative_cache.clear()

    result = gs._cluster_representative_text(cluster)

    assert result in {"short a", "short b"}
    assert result != cluster[0]["text"]


def test_representative_of_a_singleton_needs_no_embedding(monkeypatch):
    """A one-member cluster has a trivial centre; embedding it would be waste."""
    def must_not_load():
        raise AssertionError("a singleton must short-circuit before loading the model")

    monkeypatch.setattr(gs, "load_embedding_model", must_not_load)
    assert gs._cluster_representative_text([{"text": "only one"}]) == "only one"


def test_representative_of_an_empty_cluster_is_empty():
    assert gs._cluster_representative_text([]) == ""


def test_representative_is_independent_of_member_order(monkeypatch):
    """Determinism: reordering a cluster must not change its label."""
    texts = ["alpha limitation", "beta limitation", "gamma limitation"]
    mapping = {
        texts[0]: [1.0, 0.0, 0.0],
        texts[1]: [1.0, 0.0, 0.0],   # identical vectors -> a genuine tie
        texts[2]: [0.0, 1.0, 0.0],
    }
    monkeypatch.setattr(gs, "load_embedding_model", lambda: _fixed_vector_model(mapping))
    monkeypatch.setattr(gs, "_embed_texts", lambda m, texts_: m.encode(texts_).tolist())

    gs._representative_cache.clear()
    forward = gs._cluster_representative_text([{"text": t} for t in texts])
    gs._representative_cache.clear()
    reverse = gs._cluster_representative_text([{"text": t} for t in reversed(texts)])

    assert forward == reverse, "a similarity tie must break lexicographically, not by order"


def test_representative_survives_a_degenerate_centroid(monkeypatch):
    """Mutually opposed members have no centre; the result must still be deterministic."""
    texts = ["aaa opposed", "zzz opposed"]
    mapping = {texts[0]: [1.0, 0.0], texts[1]: [-1.0, 0.0]}
    monkeypatch.setattr(gs, "load_embedding_model", lambda: _fixed_vector_model(mapping))
    monkeypatch.setattr(gs, "_embed_texts", lambda m, t: m.encode(t).tolist())
    gs._representative_cache.clear()

    assert gs._cluster_representative_text([{"text": t} for t in texts]) == "aaa opposed"


# ---------------------------------------------------------------------------
# Ranking order — PLAN.md #2
# ---------------------------------------------------------------------------


def _gap(score, papers, description):
    return gs.GapResult(
        gap_description=description,
        score=score,
        frequency_score=0.0,
        recency_score=0.0,
        solution_deficit_score=0.0,
        supporting_papers=[f"p{i}" for i in range(papers)],
        proposed_solutions=[],
    )


def test_ranking_key_orders_by_score_first():
    entries = [(_gap(0.4, 9, "a"), 2025), (_gap(0.6, 1, "b"), 2019)]
    assert [g.score for g, _ in sorted(entries, key=gs._ranking_key)] == [0.6, 0.4]


def test_ranking_key_breaks_a_tie_by_supporting_paper_count():
    """The better-corroborated of two equally-scored gaps ranks higher."""
    entries = [(_gap(0.5, 1, "aaa"), 2024), (_gap(0.5, 4, "zzz"), 2024)]
    ordered = [g.gap_description for g, _ in sorted(entries, key=gs._ranking_key)]
    assert ordered == ["zzz", "aaa"]


def test_ranking_key_then_breaks_a_tie_by_newest_year():
    entries = [(_gap(0.5, 2, "older"), 2019), (_gap(0.5, 2, "newer"), 2025)]
    ordered = [g.gap_description for g, _ in sorted(entries, key=gs._ranking_key)]
    assert ordered == ["newer", "older"]


def test_ranking_key_is_a_total_order_ending_in_the_description():
    """Otherwise output depends on set/dict iteration order and is irreproducible."""
    entries = [(_gap(0.5, 2, "zebra"), 2024), (_gap(0.5, 2, "apple"), 2024)]
    ordered = [g.gap_description for g, _ in sorted(entries, key=gs._ranking_key)]
    assert ordered == ["apple", "zebra"]


def test_ranking_key_never_uses_description_length():
    """The pre-fix defect exactly: rank inside a tie block tracked text length.

    On the live corpus, ranks 1-3 tied at 0.6065 had descriptions of 72, 70 and 64
    characters, and ranks 4-7 tied at 0.6043 had 50, 44, 41 and 34 — monotonically
    descending. A copyright-law paper held rank 1 for having the longest sentence
    in its tie group.
    """
    long_but_weak = _gap(0.5, 1, "x" * 200)
    short_but_corroborated = _gap(0.5, 5, "short")
    ordered = [
        g.gap_description
        for g, _ in sorted([(long_but_weak, 2024), (short_but_corroborated, 2024)],
                           key=gs._ranking_key)
    ]
    assert ordered[0] == "short", "length must not outrank corroboration"


def test_threshold_constants_are_not_hand_editable_without_a_guard():
    """Every threshold constant must be covered by the derivation guard.

    The guard lives in the integration tier because it needs the live vector store.
    This unit test only checks the *inventory* matches, so adding a new threshold
    without adding it to the derivation report fails here rather than silently
    shipping an unmeasured number — the failure mode that let _CLUSTER_THRESHOLD =
    0.86 survive unchecked for the life of the project.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import derive_thresholds

    coded = derive_thresholds.coded_constants()
    assert set(coded) == {"cluster", "solution", "cross_domain"}, (
        "a threshold family was added or renamed without updating the derivation script"
    )
    assert set(coded["cluster"]) == set(coded["solution"]), (
        "cluster and solution thresholds must cover the same domains"
    )
    assert derive_thresholds.DRIFT_TOLERANCE > 0
