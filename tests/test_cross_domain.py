"""Tests for pipeline/cross_domain.py — cross-domain hypothesis matching."""

import json
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import sqlite_utils

import pipeline.cross_domain as cd
from pipeline.cross_domain import (
    CrossDomainMatch,
    explain_match,
    find_cross_domain_matches,
    get_unresolved_gaps,
    ingest_domain_papers,
)
from pipeline.extractor import PaperExtract
from pipeline.gap_scorer import GapResult
from tests.conftest import arxiv_reports

# Every ingest_domain_papers test reaches the P2 admission check; arXiv is answered offline.
pytestmark = pytest.mark.usefixtures("arxiv_admits")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_paper(arxiv_id: str = "2206.01106", domain: str = "computer_vision") -> PaperExtract:
    return PaperExtract(
        arxiv_id=arxiv_id,
        title="A Paper",
        year=2024,
        domain=domain,
        objectives=["obj"],
        methods=["method"],
        datasets=["dataset"],
        evaluation_metrics=["metric"],
        limitations=["a limitation"],
        future_directions=["a future direction"],
        raw_json="{}",
        ingested_at="2026-07-03T00:00:00+00:00",
    )


def _make_gap(
    desc: str = "confidence calibration under distribution shift",
    deficit: float = 0.8,
    papers: tuple = ("cv1", "cv2"),
) -> GapResult:
    return GapResult(
        gap_description=desc,
        score=0.7,
        frequency_score=0.5,
        recency_score=0.5,
        solution_deficit_score=deficit,
        supporting_papers=list(papers),
        proposed_solutions=[],
    )


def _make_fd_hit(text: str, score: float, paper_ids: list | None = None) -> types.SimpleNamespace:
    """A Qdrant future_directions hit (payload keys per vectors/embed.py)."""
    return types.SimpleNamespace(
        score=score,
        payload={
            "limitation_text": text,
            "paper_ids": paper_ids or ["mi1"],
            "domain": "medical_imaging",
        },
    )


def _make_model_mock(dim: int = 768) -> MagicMock:
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.array([[0.01] * dim for _ in texts])
    return model


def _patch_ingest_backends(monkeypatch, db, extract_mock) -> MagicMock:
    """Patch every external backend ingest_domain_papers touches; return the driver mock."""
    driver = MagicMock()
    monkeypatch.setattr(cd, "_get_db", lambda: db)
    monkeypatch.setattr(cd, "extract_paper", extract_mock)
    monkeypatch.setattr(cd, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(cd, "create_constraints", lambda d: None)
    monkeypatch.setattr(cd, "_upsert_paper_counting", MagicMock(return_value={"nodes": 0, "rels": 0}))
    monkeypatch.setattr(cd, "embed_limitations", MagicMock())
    monkeypatch.setattr(cd, "embed_future_directions", MagicMock())
    monkeypatch.setattr(cd.time, "sleep", lambda s: None)
    return driver


# ---------------------------------------------------------------------------
# ingest_domain_papers
# ---------------------------------------------------------------------------


def test_ingest_domain_papers_stores_domain_tag(monkeypatch):
    """The declared domain is passed INTO extraction and persisted as given.

    Premise updated for PLAN.md #1 (option 1A): the domain is no longer patched
    onto the PaperExtract after the fact, because extract_paper now requires and
    validates it. The stub therefore has to honour the argument it is handed,
    which is exactly the contract the real extractor now enforces.
    """
    db = sqlite_utils.Database(memory=True)
    extract_mock = MagicMock(
        side_effect=lambda arxiv_id, domain: _make_paper(domain=domain)
    )
    _patch_ingest_backends(monkeypatch, db, extract_mock)

    result = ingest_domain_papers(["2206.01106"], domain="medical_imaging")

    assert result["ingested"] == 1
    assert result["failed"] == 0
    assert extract_mock.call_args.kwargs["domain"] == "medical_imaging"
    row = list(db["papers"].rows)[0]
    assert row["domain"] == "medical_imaging"
    # List fields serialized as JSON strings, never str(list).
    assert json.loads(row["limitations"]) == ["a limitation"]
    assert json.loads(row["future_directions"]) == ["a future direction"]


def test_ingest_domain_papers_refuses_papers_their_domain_rule_rejects(monkeypatch):
    """An id list is not trusted to belong to the domain it is ingested into (P2).

    Fails against pre-fix code, which extracted every id it was handed: a cs.CL-primary
    paper declared as medical imaging was extracted, stored and written to the graph.
    """
    db = sqlite_utils.Database(memory=True)
    extract_mock = MagicMock(side_effect=lambda arxiv_id, domain: _make_paper(domain=domain))
    driver = _patch_ingest_backends(monkeypatch, db, extract_mock)
    arxiv_reports(monkeypatch, "cs.CL", title="A clinical question answering language model")

    result = ingest_domain_papers(["2401.00001"], domain="medical_imaging")

    assert extract_mock.call_count == 0
    assert result["ingested"] == 0
    assert result.get("rejected") == 1
    assert "papers" not in db.table_names()
    assert driver.session.call_count == 0


def test_ingest_domain_papers_fails_closed_when_arxiv_cannot_be_asked(monkeypatch):
    """No primary category, no admission: raise before extraction or any graph write."""
    from pipeline import arxiv_source

    def unreachable(arxiv_ids):
        raise RuntimeError("arXiv request failed after 4 attempts: timed out")

    db = sqlite_utils.Database(memory=True)
    extract_mock = MagicMock(side_effect=lambda arxiv_id, domain: _make_paper(domain=domain))
    driver = _patch_ingest_backends(monkeypatch, db, extract_mock)
    monkeypatch.setattr(arxiv_source, "fetch_by_ids", unreachable)

    with pytest.raises(RuntimeError, match="arXiv"):
        ingest_domain_papers(["2401.00001"], domain="medical_imaging")
    assert extract_mock.call_count == 0
    assert driver.session.call_count == 0


def test_ingest_domain_papers_populates_graph_and_resyncs_qdrant(monkeypatch):
    """Each ingested paper is written to Neo4j and Qdrant collections are re-synced."""
    db = sqlite_utils.Database(memory=True)
    extract_mock = MagicMock(return_value=_make_paper())
    driver = _patch_ingest_backends(monkeypatch, db, extract_mock)

    ingest_domain_papers(["2206.01106"], domain="medical_imaging")

    assert driver.session.called
    assert cd.embed_limitations.called
    assert cd.embed_future_directions.called


def test_ingest_domain_papers_skips_existing(monkeypatch):
    """Papers already in SQLite are skipped, never re-extracted or re-tagged."""
    db = sqlite_utils.Database(memory=True)
    db["papers"].insert({"arxiv_id": "2206.01106", "domain": "computer_vision"}, pk="arxiv_id")
    extract_mock = MagicMock(side_effect=AssertionError("must not extract existing paper"))
    _patch_ingest_backends(monkeypatch, db, extract_mock)

    result = ingest_domain_papers(["2206.01106"], domain="medical_imaging")

    assert result == {"ingested": 0, "failed": 0, "skipped": 1, "rejected": 0}
    # Domain tag untouched.
    assert list(db["papers"].rows)[0]["domain"] == "computer_vision"
    # No new ingests => no Qdrant re-sync.
    assert not cd.embed_limitations.called


def test_ingest_domain_papers_logs_failures_and_continues(monkeypatch):
    """A failing paper is logged and counted; the batch continues to the next one."""
    db = sqlite_utils.Database(memory=True)
    extract_mock = MagicMock(
        side_effect=[ValueError("boom"), _make_paper(arxiv_id="2307.09254")]
    )
    _patch_ingest_backends(monkeypatch, db, extract_mock)
    failures = []
    monkeypatch.setattr(cd, "_log_failure", lambda aid, reason: failures.append(aid))

    result = ingest_domain_papers(["2206.01106", "2307.09254"], domain="medical_imaging")

    assert result["ingested"] == 1
    assert result["failed"] == 1
    assert failures == ["2206.01106"]
    assert [r["arxiv_id"] for r in db["papers"].rows] == ["2307.09254"]


def test_ingest_domain_papers_empty_list(monkeypatch):
    """An empty id list returns zero counts without touching Qdrant."""
    db = sqlite_utils.Database(memory=True)
    _patch_ingest_backends(monkeypatch, db, MagicMock())

    result = ingest_domain_papers([], domain="medical_imaging")

    assert result == {"ingested": 0, "failed": 0, "skipped": 0, "rejected": 0}
    assert not cd.embed_future_directions.called


# ---------------------------------------------------------------------------
# get_unresolved_gaps
# ---------------------------------------------------------------------------


def test_get_unresolved_gaps_filters_by_deficit(monkeypatch):
    """Only gaps whose deficit is strictly above the domain floor (CV 0.3122) are returned."""
    gaps = [
        _make_gap(desc="resolved", deficit=0.2),
        _make_gap(desc="borderline", deficit=0.3),
        _make_gap(desc="open", deficit=0.5),
    ]
    monkeypatch.setattr(cd, "score_gaps", lambda domain, top_n: gaps)

    unresolved = get_unresolved_gaps("computer_vision")

    assert [gap.gap_description for gap in unresolved] == ["open"]


def test_get_unresolved_gaps_considers_every_gap_not_a_rank_cap(monkeypatch):
    """T1: no top_n rank cap. Every corroborated, unresolved gap may seed matching.

    Replaces the test that asserted top_n was forwarded. Fails against the rank cap,
    which asked score_gaps for 20 and so dropped every gap ranked below 20 — the cap
    that decided MI->CV by a 0.0043 score margin at rank 20 (PLAN_AUDIT_FIX.md R3).
    """
    gaps = [_make_gap(desc=f"g{i}", deficit=0.9) for i in range(25)]
    captured = {}

    def fake_score_gaps(domain, top_n):
        captured.update(domain=domain, top_n=top_n)
        return gaps if top_n is None else gaps[:top_n]

    monkeypatch.setattr(cd, "score_gaps", fake_score_gaps)

    assert len(get_unresolved_gaps("medical_imaging")) == 25
    assert captured == {"domain": "medical_imaging", "top_n": None}


def test_get_unresolved_gaps_requires_corroboration(monkeypatch):
    """T1: a gap seeds cross-domain matching only if >= 2 papers support it.

    Fails against the rank cap, under which a single-source gap with a high deficit
    seeded matches — the whole of MI->CV came from one such gap.
    """
    single = _make_gap(desc="single", deficit=0.9, papers=("p1",))
    corroborated = _make_gap(desc="corroborated", deficit=0.9, papers=("p1", "p2"))
    monkeypatch.setattr(cd, "score_gaps", lambda domain, top_n: [single, corroborated])

    assert [g.gap_description for g in get_unresolved_gaps("medical_imaging")] == ["corroborated"]


def test_the_corroboration_gate_is_the_tier_rule():
    """The gate reuses the ranking's corroboration threshold rather than a second literal."""
    import pipeline.gap_scorer as gs

    assert cd._MIN_SEED_PAPERS is gs._CORROBORATED_MIN_PAPERS or (
        cd._MIN_SEED_PAPERS == gs._CORROBORATED_MIN_PAPERS
    )


# ---------------------------------------------------------------------------
# find_cross_domain_matches
# ---------------------------------------------------------------------------


def test_cross_domain_threshold_sits_at_the_measured_noise_floor():
    """Default threshold is the pooled 95th percentile of random cross-domain pairs.

    Advisor decision A2 (2026-08-23). The previous 0.82 sat *below the median* of
    pure noise — 61.6% of 1,490 randomly paired CV/MI items cleared it.
    """
    import inspect

    default = inspect.signature(
        cd.find_cross_domain_matches
    ).parameters["similarity_threshold"].default

    # Deliberately NOT pinned to a literal. This constant is owned by
    # scripts/derive_thresholds.py and is rewritten whenever its drift from the measured
    # null exceeds DRIFT_TOLERANCE — corpus growth from 31 to 64 CV papers moved it from
    # 0.8792 to 0.8764, and a literal here would turn that legitimate re-derivation into
    # a unit-test failure whose only "fix" is copying whatever the code now says. The
    # exact value is guarded against the live null distribution by
    # tests/integration/test_threshold_derivation.py.
    assert default == cd._CROSS_DOMAIN_THRESHOLD, (
        "the default must be the module constant, not a second copy of the literal"
    )
    # What must hold regardless of the measured value: it sits above the null MEDIAN,
    # which is the error the 0.82 value made (61.6% of random pairs cleared it).
    assert default > 0.83, "must sit above the measured null median"
    assert 0.83 < default < 1.0, "must be a plausible cosine floor"


def test_cross_domain_rejects_pairs_below_the_noise_floor(monkeypatch):
    """A 0.85 pair — comfortably above the old 0.82 — is now correctly refused."""
    gap = GapResult(
        gap_description="a CV gap",
        score=0.5,
        frequency_score=0.5,
        recency_score=0.5,
        solution_deficit_score=0.9,
        supporting_papers=["cv1"],
        proposed_solutions=[],
    )
    monkeypatch.setattr(cd, "get_unresolved_gaps", lambda domain, top_n=20: [gap])
    monkeypatch.setattr(cd, "load_embedding_model", lambda: _make_model_mock())
    monkeypatch.setattr(cd, "_embed_texts", lambda model, texts: [[0.01] * 768])
    client = MagicMock()
    client.query_points.return_value = types.SimpleNamespace(
        points=[
            _make_fd_hit("noise-floor pair", 0.85, ["mi1"]),
            _make_fd_hit("genuinely similar pair", 0.92, ["mi2"]),
        ]
    )
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)

    matches = cd.find_cross_domain_matches()

    assert [m.target_solution for m in matches] == ["genuinely similar pair"]


def test_cross_domain_empty_result_is_valid_output(monkeypatch):
    """Zero matches is honest output, not an error.

    On the current corpus every CV→MI pair scores below the noise floor, so the
    default API path legitimately returns an empty list. It must not raise.
    """
    gap = GapResult(
        gap_description="a CV gap",
        score=0.5,
        frequency_score=0.5,
        recency_score=0.5,
        solution_deficit_score=0.9,
        supporting_papers=["cv1"],
        proposed_solutions=[],
    )
    monkeypatch.setattr(cd, "get_unresolved_gaps", lambda domain, top_n=20: [gap])
    monkeypatch.setattr(cd, "load_embedding_model", lambda: _make_model_mock())
    monkeypatch.setattr(cd, "_embed_texts", lambda model, texts: [[0.01] * 768])
    client = MagicMock()
    client.query_points.return_value = types.SimpleNamespace(
        points=[_make_fd_hit("all below the floor", 0.86, ["mi1"])]
    )
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)

    assert cd.find_cross_domain_matches() == []


def test_find_cross_domain_matches_applies_domain_filter_and_threshold(monkeypatch):
    """Qdrant is queried with a target-domain filter; hits below threshold dropped."""
    gap = _make_gap(papers=("cv1", "cv2"))
    monkeypatch.setattr(cd, "get_unresolved_gaps", lambda domain: [gap])
    hits = [
        _make_fd_hit("uncertainty quantification for segmentation", 0.92, ["mi1", "mi2"]),
        # 0.85 now fails: it is below the 0.8792 noise floor, so it is
        # indistinguishable from a random pairing. It passed under the old 0.82.
        _make_fd_hit("too weak a match", 0.85),
    ]
    client = MagicMock()
    client.query_points.return_value = types.SimpleNamespace(points=hits)
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(cd, "load_embedding_model", lambda: _make_model_mock())

    matches = find_cross_domain_matches()

    # Domain filter applied against the future_directions collection.
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "future_directions"
    condition = kwargs["query_filter"].must[0]
    assert condition.key == "domain"
    assert condition.match.value == "medical_imaging"
    # Only the hit clearing the 0.8792 noise floor survives, fully populated.
    assert len(matches) == 1
    match = matches[0]
    assert isinstance(match, CrossDomainMatch)
    assert match.source_gap == gap.gap_description
    assert match.target_solution == "uncertainty quantification for segmentation"
    assert match.similarity_score == 0.92
    assert match.source_papers == ["cv1", "cv2"]
    assert match.target_papers == ["mi1", "mi2"]
    assert match.source_domain == "computer_vision"
    assert match.target_domain == "medical_imaging"


def test_find_cross_domain_matches_sorted_and_top_n(monkeypatch):
    """Matches are sorted by similarity_score descending and capped at top_n."""
    gaps = [_make_gap(desc="gap one"), _make_gap(desc="gap two")]
    monkeypatch.setattr(cd, "get_unresolved_gaps", lambda domain: gaps)
    client = MagicMock()
    client.query_points.side_effect = [
        types.SimpleNamespace(points=[_make_fd_hit("weaker solution", 0.89)]),
        types.SimpleNamespace(points=[_make_fd_hit("stronger solution", 0.95)]),
    ]
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(cd, "load_embedding_model", lambda: _make_model_mock())

    matches = find_cross_domain_matches()
    assert [m.similarity_score for m in matches] == [0.95, 0.89]

    client.query_points.side_effect = [
        types.SimpleNamespace(points=[_make_fd_hit("weaker solution", 0.89)]),
        types.SimpleNamespace(points=[_make_fd_hit("stronger solution", 0.95)]),
    ]
    top = find_cross_domain_matches(top_n=1)
    assert len(top) == 1
    assert top[0].target_solution == "stronger solution"


def test_find_cross_domain_matches_batches_gap_embeddings(monkeypatch):
    """All gap descriptions are embedded in one encode() call."""
    gaps = [_make_gap(desc="gap one"), _make_gap(desc="gap two")]
    monkeypatch.setattr(cd, "get_unresolved_gaps", lambda domain: gaps)
    client = MagicMock()
    client.query_points.return_value = types.SimpleNamespace(points=[])
    model = _make_model_mock()
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(cd, "load_embedding_model", lambda: model)

    find_cross_domain_matches()

    assert model.encode.call_count == 1
    assert list(model.encode.call_args.args[0]) == ["gap one", "gap two"]


def test_find_cross_domain_matches_no_unresolved_gaps(monkeypatch):
    """No unresolved gaps => empty list, Qdrant never touched."""
    monkeypatch.setattr(cd, "get_unresolved_gaps", lambda domain: [])
    client = MagicMock()
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)

    assert find_cross_domain_matches() == []
    assert not client.query_points.called


def test_find_cross_domain_matches_no_hits_above_threshold(monkeypatch):
    """Gaps with no sufficiently similar target solutions yield no matches."""
    monkeypatch.setattr(cd, "get_unresolved_gaps", lambda domain: [_make_gap()])
    client = MagicMock()
    client.query_points.return_value = types.SimpleNamespace(
        points=[_make_fd_hit("irrelevant", 0.40)]
    )
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(cd, "load_embedding_model", lambda: _make_model_mock())

    assert find_cross_domain_matches() == []


# ---------------------------------------------------------------------------
# explain_match
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# explain_match — always local Ollama, never a hosted API
# ---------------------------------------------------------------------------


def _make_match() -> CrossDomainMatch:
    return CrossDomainMatch(
        source_gap="calibration fails under distribution shift",
        target_solution="uncertainty quantification for segmentation",
        similarity_score=0.9,
        source_papers=["cv1"],
        target_papers=["mi1"],
        source_domain="computer_vision",
        target_domain="medical_imaging",
    )


def test_explain_always_uses_ollama(monkeypatch):
    """There is one backend. A public deployment refuses /explain at the API
    layer instead of routing to a paid API, so this never needs to branch."""
    monkeypatch.setattr(cd, "_call_ollama_text", lambda prompt: "local explanation")

    assert explain_match(_make_match()) == "local explanation"


def test_no_hosted_llm_path_exists():
    """Guard against re-introducing a paid-API branch.

    The hosted path was removed deliberately (cost of anonymous callers), not
    left dormant — a re-added helper would be dead paid-API code until something
    called it, and easy to miss in review.
    """
    assert not hasattr(cd, "_call_hosted_llm_text")
    assert "anthropic" not in cd.__dict__


def test_explain_match_prompt_contains_gap_and_solution(monkeypatch):
    """The Ollama prompt includes both the source gap and the target solution."""
    prompts = []
    monkeypatch.setattr(
        cd, "_call_ollama_text", lambda prompt: prompts.append(prompt) or "An explanation."
    )
    match = CrossDomainMatch(
        source_gap="confidence calibration gap in object detection",
        target_solution="uncertainty quantification for medical segmentation",
        similarity_score=0.85,
        source_papers=["cv1"],
        target_papers=["mi1"],
        source_domain="computer_vision",
        target_domain="medical_imaging",
    )

    explanation = explain_match(match)

    assert explanation == "An explanation."
    assert len(prompts) == 1
    assert "confidence calibration gap in object detection" in prompts[0]
    assert "uncertainty quantification for medical segmentation" in prompts[0]
    # Domains rendered human-readable in the prompt.
    assert "computer vision" in prompts[0]
    assert "medical imaging" in prompts[0]


def test_call_ollama_text_sends_prompt_and_strips(monkeypatch):
    """_call_ollama_text forwards the prompt to ollama.Client.chat and strips output."""
    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "  the explanation  \n"}}
    fake_ollama = types.SimpleNamespace(Client=MagicMock(return_value=fake_client))
    monkeypatch.setitem(__import__("sys").modules, "ollama", fake_ollama)

    result = cd._call_ollama_text("why is this interesting?")

    assert result == "the explanation"
    sent = fake_client.chat.call_args.kwargs["messages"][0]["content"]
    assert sent == "why is this interesting?"


# ---------------------------------------------------------------------------
# verify_pairing — PLAN.md #5, option 5A
# ---------------------------------------------------------------------------


def _stored_point(text: str, vector: list[float], papers: list[str] | None = None):
    return types.SimpleNamespace(
        vector=vector,
        payload={"limitation_text": text, "paper_ids": papers or ["p1"], "domain": "d"},
    )


def _patch_store(monkeypatch, limitations: list, futures: list):
    """Stub Qdrant scroll so verify_pairing sees a known corpus."""
    import pipeline.cross_domain as cd

    def fake_scroll(collection_name, scroll_filter=None, limit=None, offset=None, **kw):
        source = limitations if collection_name == cd._COLLECTION_LIMITATIONS else futures
        return source, None

    client = MagicMock()
    client.scroll.side_effect = fake_scroll
    monkeypatch.setattr(cd, "get_qdrant_client", lambda: client)
    return client


def test_verify_pairing_accepts_a_real_pairing_above_the_threshold(monkeypatch):
    """A pairing that exists and clears the noise floor verifies, with a real score."""
    from pipeline.cross_domain import verify_pairing

    # Identical unit vectors -> similarity 1.0, comfortably above the threshold.
    _patch_store(
        monkeypatch,
        [_stored_point("a real limitation", [1.0, 0.0], ["2304.02643"])],
        [_stored_point("a real future direction", [1.0, 0.0], ["2403.16502"])],
    )

    verified = verify_pairing(
        source_gap="a real limitation",
        target_solution="a real future direction",
        source_domain="computer_vision",
        target_domain="medical_imaging",
    )

    assert verified.similarity_score == pytest.approx(1.0)
    assert verified.similarity_score != 0.0, "the hardcoded 0.0 placeholder must be gone"
    assert verified.source_papers == ["2304.02643"]
    assert verified.target_papers == ["2403.16502"]


def test_verify_pairing_refuses_a_gap_absent_from_the_corpus(monkeypatch):
    """An arbitrary client string is not a finding about the literature."""
    from pipeline.cross_domain import UngroundedPairingError, verify_pairing

    _patch_store(
        monkeypatch,
        [_stored_point("a real limitation", [1.0, 0.0])],
        [_stored_point("a real future direction", [1.0, 0.0])],
    )

    with pytest.raises(UngroundedPairingError, match="not a corpus match"):
        verify_pairing(
            source_gap="Only applies to divisor-bounded multiplicative functions",
            target_solution="a real future direction",
        )


def test_verify_pairing_refuses_a_solution_absent_from_the_corpus(monkeypatch):
    from pipeline.cross_domain import UngroundedPairingError, verify_pairing

    _patch_store(
        monkeypatch,
        [_stored_point("a real limitation", [1.0, 0.0])],
        [_stored_point("a real future direction", [1.0, 0.0])],
    )

    with pytest.raises(UngroundedPairingError, match="not a corpus match"):
        verify_pairing(source_gap="a real limitation", target_solution="invented text")


def test_verify_pairing_refuses_a_pair_below_the_noise_floor(monkeypatch):
    """Both texts real, but the pair is indistinguishable from chance."""
    from pipeline.cross_domain import UngroundedPairingError, verify_pairing

    # Orthogonal vectors -> similarity 0.0, far below the threshold.
    _patch_store(
        monkeypatch,
        [_stored_point("a real limitation", [1.0, 0.0])],
        [_stored_point("a real future direction", [0.0, 1.0])],
    )

    with pytest.raises(UngroundedPairingError, match="below the noise floor"):
        verify_pairing(source_gap="a real limitation", target_solution="a real future direction")


def test_verify_pairing_gates_on_the_same_threshold_the_matcher_admits(monkeypatch):
    """One constant, so the gate and the matcher cannot drift apart."""
    import inspect

    import pipeline.cross_domain as cd

    assert (
        inspect.signature(cd.find_cross_domain_matches)
        .parameters["similarity_threshold"]
        .default
        == cd._CROSS_DOMAIN_THRESHOLD
    )

    _patch_store(
        monkeypatch,
        [_stored_point("lim", [1.0, 0.0])],
        [_stored_point("fut", [1.0, 0.0])],
    )
    verified = cd.verify_pairing(source_gap="lim", target_solution="fut")
    assert verified.threshold == cd._CROSS_DOMAIN_THRESHOLD


def test_verify_pairing_uses_the_solution_threshold_for_a_same_domain_pair(monkeypatch):
    """A same-domain pairing is a different population, so a different null applies."""
    import pipeline.cross_domain as cd
    from pipeline.gap_scorer import _solution_threshold

    _patch_store(
        monkeypatch,
        [_stored_point("lim", [1.0, 0.0])],
        [_stored_point("fut", [1.0, 0.0])],
    )
    verified = cd.verify_pairing(
        source_gap="lim",
        target_solution="fut",
        source_domain="computer_vision",
        target_domain="computer_vision",
    )
    assert verified.threshold == _solution_threshold("computer_vision")


def test_verify_pairing_rejects_empty_input(monkeypatch):
    from pipeline.cross_domain import UngroundedPairingError, verify_pairing

    with pytest.raises(UngroundedPairingError, match="non-empty"):
        verify_pairing(source_gap="   ", target_solution="something")


def test_verify_pairing_runs_before_any_llm_call(monkeypatch):
    """The refusal must be deterministic and server-side, never the model's choice.

    Option 5B — relying on prompt wording to make the model decline — was rejected:
    an 8B model refusing a leading question is weak and untestable. This asserts the
    gate cannot be bypassed by Ollama being reachable.
    """
    import pipeline.cross_domain as cd

    def must_not_run(*a, **k):
        raise AssertionError("verify_pairing must not reach the LLM")

    monkeypatch.setattr(cd, "_call_ollama_text", must_not_run)
    _patch_store(monkeypatch, [], [])

    with pytest.raises(cd.UngroundedPairingError):
        cd.verify_pairing(source_gap="anything", target_solution="anything else")


# ---------------------------------------------------------------------------
# S1 — per-domain deficit floors derived from the solution noise floor
# ---------------------------------------------------------------------------


def _gap_with_deficit(deficit: float) -> GapResult:
    return GapResult(
        gap_description="g", score=0.5, frequency_score=0.1, recency_score=1.0,
        solution_deficit_score=deficit, supporting_papers=["a", "b"], proposed_solutions=[],
    )


def test_the_unresolved_floor_is_per_domain(monkeypatch):
    """A deficit of 0.30 is unresolved in MI (floor 0.2857) but not in CV (0.3122).

    Fails against the single 0.3 floor, under which 0.30 > 0.3 is false in both.
    """
    monkeypatch.setattr(cd, "score_gaps", lambda domain, top_n: [_gap_with_deficit(0.30)])
    assert len(cd.get_unresolved_gaps("medical_imaging")) == 1
    assert len(cd.get_unresolved_gaps("computer_vision")) == 0


def test_each_floor_is_the_solution_noise_floor_on_the_deficit_scale(monkeypatch):
    """"Unresolved" means: no eligible future direction clears the solution threshold.

    Proved through the real deficit computation, not by restating the formula: a
    nearest future direction exactly AT the domain's solution threshold must produce a
    deficit equal to the floor, so the gap is addressed (the floor is a strict `>`);
    one just below it must land above the floor.

    T1 restored the exact-boundary assertion. In S1 the floors were 4-dp literals and
    MI's (0.2857) sat below its exact value (0.285714...), so a nearest exactly on the
    MI threshold counted as unresolved. The floors are now stored at full precision.
    """
    import pipeline.gap_scorer as gs

    assert set(cd._UNRESOLVED_DEFICIT_FLOORS) == set(gs._SOLUTION_THRESHOLDS)
    monkeypatch.setattr(gs, "_cluster_representative_text", lambda cluster: "g")
    cluster = [{"text": "g", "paper_ids": ["a"], "years": [2025]}]
    for domain, threshold in gs._SOLUTION_THRESHOLDS.items():
        floor = cd._unresolved_deficit_floor(domain)
        p50, p99 = gs._DEFICIT_RESCALE_ANCHORS[domain]
        assert floor == pytest.approx(1.0 - (threshold - p50) / (p99 - p50), abs=1e-12)
        for nearest, unresolved in ((threshold, False), (threshold - 0.001, True)):
            monkeypatch.setattr(gs, "_addressing_hits", lambda *a, _s=nearest, **kw: [(_s, "fd", ["b"])])
            deficit = gs.compute_solution_deficit_score(cluster, domain=domain)
            if nearest == threshold:
                assert deficit == pytest.approx(floor, abs=1e-12), (domain, deficit, floor)
            assert (deficit > floor) is unresolved, (domain, nearest, deficit, floor)


def test_an_unknown_domain_gets_the_strictest_floor():
    """Same fallback rule as the other per-domain constants: the strictest known value."""
    assert cd._unresolved_deficit_floor("not_a_domain") == max(cd._UNRESOLVED_DEFICIT_FLOORS.values())
