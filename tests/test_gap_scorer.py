"""Tests for pipeline/gap_scorer.py — the discovery / gap-scoring core."""

import types
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

import pipeline.gap_scorer as gs
from pipeline.gap_scorer import (
    GapResult,
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
    # One hit above 0.75, one below — only the first counts.
    hits = [_make_hit("use adamw", 0.90), _make_hit("unrelated", 0.50)]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    assert compute_solution_deficit_score(cluster) == 0.5


def test_compute_solution_deficit_clamped_to_zero(monkeypatch):
    """More matches than reporting papers clamps the score to 0.0, never negative."""
    cluster = [{"text": "slow training", "paper_ids": ["a"], "years": [2024]}]
    hits = [_make_hit("fix one", 0.9), _make_hit("fix two", 0.8), _make_hit("fix three", 0.79)]
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
    client = _make_qdrant_query_mock([_make_hit("sol", 0.8)])
    _patch_vector_backends(monkeypatch, client, _make_model_mock())
    score = compute_solution_deficit_score(cluster)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# cluster_limitations
# ---------------------------------------------------------------------------


def test_cluster_limitations_empty_input():
    """No limitations yields no clusters."""
    assert cluster_limitations([]) == []


def test_cluster_limitations_groups_similar(monkeypatch):
    """Limitations matching each other above 0.82 collapse into one cluster."""
    lims = [
        {"text": "slow convergence", "paper_ids": ["a"], "years": [2024]},
        {"text": "training is slow", "paper_ids": ["b"], "years": [2024]},
    ]
    hits = [_make_hit("slow convergence", 0.95), _make_hit("training is slow", 0.90)]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims)

    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_limitations_singletons_below_threshold(monkeypatch):
    """Limitations whose neighbours all score below 0.82 stay as singletons."""
    lims = [
        {"text": "slow convergence", "paper_ids": ["a"], "years": [2024]},
        {"text": "high memory use", "paper_ids": ["b"], "years": [2024]},
    ]
    hits = [_make_hit("slow convergence", 0.50), _make_hit("high memory use", 0.40)]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims)

    assert len(clusters) == 2
    assert all(len(c) == 1 for c in clusters)


def test_cluster_limitations_partitions_all_inputs(monkeypatch):
    """Every input limitation appears in exactly one cluster."""
    lims = [
        {"text": "a", "paper_ids": ["p1"], "years": [2024]},
        {"text": "b", "paper_ids": ["p2"], "years": [2024]},
        {"text": "c", "paper_ids": ["p3"], "years": [2024]},
    ]
    # a and b are similar (0.9); c matches nothing.
    hits = [_make_hit("a", 0.9), _make_hit("b", 0.9), _make_hit("c", 0.3)]
    client = _make_qdrant_query_mock(hits)
    _patch_vector_backends(monkeypatch, client, _make_model_mock())

    clusters = cluster_limitations(lims)

    total = sum(len(c) for c in clusters)
    assert total == 3
    assert {lim["text"] for c in clusters for lim in c} == {"a", "b", "c"}


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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, min_cluster_size=2: [cluster])
    monkeypatch.setattr(gs, "compute_frequency_score", lambda c, t: 0.6)
    monkeypatch.setattr(gs, "compute_recency_score", lambda c, current_year=2024: 0.4)
    monkeypatch.setattr(gs, "compute_solution_deficit_score", lambda c: 0.8)
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text: [])

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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, min_cluster_size=2: [c_low, c_high])
    monkeypatch.setattr(gs, "compute_frequency_score", lambda c, t: 0.9 if c is c_high else 0.1)
    monkeypatch.setattr(gs, "compute_recency_score", lambda c, current_year=2024: 0.5)
    monkeypatch.setattr(gs, "compute_solution_deficit_score", lambda c: 0.5)
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text: [])

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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, min_cluster_size=2: clusters)
    # avoid Qdrant: no addressing solutions => deficit computed without network
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text: [])

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
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, min_cluster_size=2: [cluster])
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text: [])

    results = score_gaps()

    assert results[0].gap_description == "repeated gap"
    # all three unique papers collected as supporting evidence
    assert results[0].supporting_papers == ["a", "b", "c"]


def test_score_gaps_collects_proposed_solutions(monkeypatch):
    """proposed_solutions come from addressing future directions for the cluster."""
    cluster = [{"text": "needs solving", "paper_ids": ["a"], "years": [2024]}]
    monkeypatch.setattr(gs, "get_all_limitations", lambda domain="computer_vision": cluster)
    monkeypatch.setattr(gs, "_count_papers_in_domain", lambda domain: 1)
    monkeypatch.setattr(gs, "cluster_limitations", lambda lims, min_cluster_size=2: [cluster])
    monkeypatch.setattr(gs, "compute_solution_deficit_score", lambda c: 0.0)
    monkeypatch.setattr(gs, "_find_addressing_solutions", lambda text: ["try approach X"])

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
