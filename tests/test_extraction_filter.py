"""Tests for pipeline.extraction_filter (Phase 1b).

Every test here fails against pre-fix code for the simplest reason: the module did
not exist, and the corpus contained `Limitation` nodes whose entire text was
`"remains challenging"` or `"future work includes"`.
"""

import json
import re

import pytest

from pipeline import extraction_filter as ef


# ---------------------------------------------------------------------------
# The prompt-echo constant and its two guards
# ---------------------------------------------------------------------------


def test_prompt_echo_constant_is_not_empty():
    """An empty echo list would silently disable gate (a).

    This is the specific hazard that made the constant explicit rather than parsed
    from the prompt: a runtime-parsed list goes empty the moment the prompt stops
    quoting its examples, and the gate then rejects nothing while still looking alive.
    """
    assert ef.PROMPT_ECHO_PHRASES, "the prompt-echo list must never be empty"
    assert len(ef.PROMPT_ECHO_PHRASES) >= 4


@pytest.mark.parametrize("phrase", ["remains challenging", "future work includes"])
def test_prompt_echo_constant_contains_the_known_echoes(phrase):
    """These two were found as complete Limitation node texts in the live corpus."""
    assert phrase in ef.PROMPT_ECHO_PHRASES


def test_every_quoted_phrase_in_the_prompt_is_covered_by_the_echo_list():
    """A new example added to the prompt must be added to the echo list too.

    The prompt currently quotes no cue phrases at all — that is the Phase 1b fix — so
    this passes over an empty set today. It exists to fail the moment someone
    reintroduces a quoted example without updating PROMPT_ECHO_PHRASES, which is
    exactly how the original leak happened.
    """
    from pipeline.extractor import _EXTRACTION_PROMPT, _TIER_INSTRUCTIONS

    quoted: set[str] = set()
    for text in (*_TIER_INSTRUCTIONS.values(), _EXTRACTION_PROMPT):
        quoted.update(re.findall(r"'([^']{2,60})'", text))
        quoted.update(re.findall(r'"([^"]{2,60})"', text))

    # Ignore JSON keys, the {paper_text} placeholder, and anything that is not a
    # natural-language phrase — the prompt embeds a JSON schema block, so a naive
    # quote scan also picks up punctuation like '": ["'.
    ignored = {
        "objectives", "methods", "datasets", "evaluation_metrics",
        "limitations", "future_directions", "paper_text",
    }
    phrase_like = re.compile(r"^[A-Za-z][A-Za-z \-']*[A-Za-z]$")
    candidates = {
        q for q in quoted
        if q not in ignored and not q.startswith("<") and phrase_like.match(q)
    }

    uncovered = {q for q in candidates if ef.normalise(q) not in ef.PROMPT_ECHO_PHRASES}
    assert not uncovered, (
        f"the extraction prompt quotes example phrases that the echo filter does not "
        f"cover: {sorted(uncovered)}. Add them to PROMPT_ECHO_PHRASES or remove them "
        f"from the prompt."
    )


def test_prompt_contains_no_quotable_cue_phrases():
    """The prompt fix itself: no literal cue phrase is left for the model to echo."""
    from pipeline.extractor import _TIER_INSTRUCTIONS

    for tier, text in _TIER_INSTRUCTIONS.items():
        for phrase in ("remains challenging", "future work includes", "however", "despite"):
            assert f"'{phrase}'" not in text, (
                f"tier {tier!r} still quotes {phrase!r}, which the model will echo back"
            )


# ---------------------------------------------------------------------------
# Gate (a) — prompt echo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "remains challenging", "Remains challenging.", "  FUTURE WORK INCLUDES  ",
    "however", "Limitations", "future work",
])
def test_whole_string_prompt_echoes_are_rejected(text):
    assert ef.classify(text, "limitations")[0] == "prompt_echo"


def test_a_cue_phrase_with_real_content_is_kept():
    """The load-bearing case for whole-string rather than substring matching.

    "remains challenging to process high-resolution images" is a genuine gap in the
    live corpus, reported by two papers. A substring rule would delete it along with
    the bare cue.
    """
    assert ef.classify("remains challenging to process high-resolution images",
                       "limitations") is None
    assert ef.classify(
        "future work includes endowing the model with higher-resolution inputs",
        "future_directions") is None


# ---------------------------------------------------------------------------
# Gate (b) — hedging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "certain aspects may have eluded our scrutiny",
    "recent advances might not be entirely encapsulated",
    "our review is not entirely exhaustive of the field",
])
def test_hedging_substrings_are_rejected_anywhere_in_the_string(text):
    """A paper hedging about its own coverage says nothing about a research gap.

    The bar for this list is high — the phrase must be uninformative no matter what
    surrounds it.
    """
    assert ef.classify(text, "limitations")[0] == "hedging"


@pytest.mark.parametrize("text", [
    "Further research is needed to improve robustness and generalizability",
    "Multi-modal fusion is beyond the scope of this work",
    "Handling long occlusions is left for future work",
])
def test_hedging_framings_that_name_something_specific_are_kept(text):
    """These were rejected by an earlier, over-aggressive substring list.

    The backfill dry run caught it: "Further research is needed to improve the
    robustness and generalizability..." was being discarded even though it names what
    needs improving. Hedged framing around real content is still real content, so
    these phrases only reject when they ARE the whole string.
    """
    assert ef.classify(text, "future_directions") is None


@pytest.mark.parametrize("text", ["further research is needed", "left for future work"])
def test_the_same_framings_alone_are_rejected(text):
    assert ef.classify(text, "future_directions")[0] == "hedging"


@pytest.mark.parametrize("text", ["several limitations", "overcome current limitations"])
def test_whole_string_hedging_is_rejected(text):
    assert ef.classify(text, "future_directions")[0] in {"hedging", "prompt_echo"}


# ---------------------------------------------------------------------------
# Gate (c) — minimum informative length
# ---------------------------------------------------------------------------


def test_min_words_matches_the_documented_percentile():
    """The cutoff is the p5 of the measured word distribution, not a round number."""
    assert set(ef.MIN_WORDS) == set(ef.FIELDS)
    assert all(v >= 2 for v in ef.MIN_WORDS.values())


@pytest.mark.parametrize("text", ["Overfitting", "Dataset bias", "domain shift"])
def test_fragments_below_the_length_cutoff_are_rejected(text):
    reason, detail = ef.classify(text, "limitations")
    assert reason == "too_short"
    assert "p5 cutoff" in detail


def test_a_terse_but_specific_string_survives():
    """p5 was chosen over p10 precisely so this is kept."""
    assert ef.classify("contrast-agnostic, pathology-encoded representations",
                       "future_directions") is None
    assert ef.classify("Poor similarity metric", "limitations") is None


# ---------------------------------------------------------------------------
# Gate (d) — duplicates, parameter-free
# ---------------------------------------------------------------------------


def test_exact_duplicates_within_a_paper_are_removed():
    kept, rejected = ef.filter_extractions(
        ["Slow convergence on long sequences", "slow convergence on long sequences."],
        "limitations", "1234.5678")
    assert len(kept) == 1
    assert [r.reason for r in rejected] == ["duplicate"]


def test_a_contained_string_is_removed_and_the_longer_one_kept():
    # Both strings must clear the length gate, or the shorter one is rejected as
    # too_short before the duplicate check ever sees it.
    kept, rejected = ef.filter_extractions(
        ["accuracy degrades on unseen anatomy",
         "accuracy degrades on unseen anatomy in clinical CT"],
        "limitations", "1234.5678")
    assert kept == ["accuracy degrades on unseen anatomy in clinical CT"]
    assert [r.reason for r in rejected] == ["duplicate"]
    assert "contained in" in rejected[0].detail


def test_distinct_strings_are_both_kept_and_original_order_preserved():
    items = ["Detection fails on small objects", "Inference is slower than real time"]
    kept, rejected = ef.filter_extractions(items, "limitations", "1234.5678")
    assert kept == items, "surviving items must keep their original relative order"
    assert rejected == []


def test_duplicate_detection_uses_no_similarity_threshold():
    """Merely similar strings are NOT merged — no unmeasured constant is involved."""
    items = ["Detection fails on small objects",
             "Detection fails on large objects"]
    kept, _ = ef.filter_extractions(items, "limitations", "1234.5678")
    assert len(kept) == 2


# ---------------------------------------------------------------------------
# Auditability
# ---------------------------------------------------------------------------


def test_every_rejection_is_logged_as_jsonl(tmp_path):
    log = tmp_path / "rejected.jsonl"
    kept = ef.filter_and_log(
        ["remains challenging", "Overfitting",
         "Detection accuracy degrades on small objects"],
        "limitations", "2404.07922", log_path=log)

    assert kept == ["Detection accuracy degrades on small objects"]
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["reason"] for r in records} == {"prompt_echo", "too_short"}
    for record in records:
        assert record["arxiv_id"] == "2404.07922"
        assert record["field"] == "limitations"
        assert record["text"]
        assert record["at"]


def test_filtering_nothing_writes_no_log(tmp_path):
    log = tmp_path / "rejected.jsonl"
    ef.filter_and_log(["Detection accuracy degrades on small objects"],
                      "limitations", "1234.5678", log_path=log)
    assert not log.exists()


def test_empty_input_is_handled():
    assert ef.filter_extractions([], "limitations", "x") == ([], [])


# ---------------------------------------------------------------------------
# Integration with extract_paper
# ---------------------------------------------------------------------------


def test_extract_paper_filters_before_returning(monkeypatch, tmp_path):
    """Boilerplate must never reach the PaperExtract, since Limitation is UNIQUE on text."""
    import pipeline.extractor as extractor

    monkeypatch.setattr(extractor, "fetch_paper_text",
                        lambda a: {"title": "T", "year": 2025, "abstract": "a"})
    monkeypatch.setattr(extractor, "fetch_full_text",
                        lambda a, abstract="": ("text", "conclusion"))
    monkeypatch.setattr(extractor, "call_ollama", lambda prompt: {
        "objectives": [], "methods": [], "datasets": [], "evaluation_metrics": [],
        "limitations": ["remains challenging", "Overfitting",
                        "Detection degrades badly on small distant objects"],
        "future_directions": ["future work includes",
                              "Extend the decoder to exploit temporal context"],
    })
    monkeypatch.setattr(ef, "_LOG_PATH", tmp_path / "rejected.jsonl")

    paper = extractor.extract_paper("2301.00234", domain="computer_vision")

    assert paper.limitations == ["Detection degrades badly on small distant objects"]
    assert paper.future_directions == ["Extend the decoder to exploit temporal context"]
    # raw_json keeps the unfiltered output so the filter stays reversible and auditable.
    raw = json.loads(paper.raw_json)
    assert "remains challenging" in raw["limitations"]
    assert "future work includes" in raw["future_directions"]
