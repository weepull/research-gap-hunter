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
        gs, "cluster_limitations", lambda lims: [recent, mid, stale]
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
        gs, "cluster_limitations", lambda lims: [recent, stale]
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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims: [cluster])

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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims: [cluster])
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
    """Limitations scoring >= 0.86 against the seed collapse into one cluster."""
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
    """Limitations whose neighbours all score below 0.86 stay as singleton clusters."""
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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims: [cluster])
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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims: [c_low, c_high])
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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims: clusters)
    # avoid Qdrant: no addressing solutions => deficit computed without network
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])

    results = score_gaps(top_n=2)

    assert len(results) == 2


def test_score_gaps_uses_most_frequent_text_as_description(monkeypatch):
    """gap_description is the most frequent limitation text in the cluster."""
    cluster = [
        {"text": "repeated gap", "paper_ids": ["a"], "years": [2024]},
        {"text": "repeated gap", "paper_ids": ["b"], "years": [2024]},
        {"text": "rare gap", "paper_ids": ["c"], "years": [2024]},
    ]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 3)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims: [cluster])
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text, **kw: [])

    results = score_gaps()

    assert results[0].gap_description == "repeated gap"
    # all three unique papers collected as supporting evidence
    assert results[0].supporting_papers == ["a", "b", "c"]


def test_score_gaps_collects_proposed_solutions(monkeypatch):
    """proposed_solutions come from addressing future directions for the cluster."""
    cluster = [{"text": "needs solving", "paper_ids": ["a"], "years": [2024]}]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 1)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims: [cluster])
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
