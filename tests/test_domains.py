"""Tests for pipeline.domains — PLAN.md #1 (options 1A + 1C).

Every test here fails against pre-fix code for the simplest possible reason: the
module did not exist, and `extract_paper` stamped every paper computer_vision.
"""

import logging

import pytest

from pipeline.domains import (
    RESEARCH_DOMAINS,
    classify_text,
    curate_declared_domain,
    validate_domain,
    verify_declared_domain,
)


# ---------------------------------------------------------------------------
# 1A — validate_domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", RESEARCH_DOMAINS)
def test_validate_domain_accepts_known_domains(domain):
    assert validate_domain(domain) == domain


def test_validate_domain_strips_surrounding_whitespace():
    assert validate_domain("  computer_vision  ") == "computer_vision"


@pytest.mark.parametrize("bad", ["", "   ", None, "CV", "vision", "other", "Computer_Vision"])
def test_validate_domain_rejects_anything_else(bad):
    """Unknown labels must raise, not fall back to a plausible default."""
    with pytest.raises(ValueError):
        validate_domain(bad)


def test_validate_domain_error_says_what_to_do_with_an_off_topic_paper():
    """The message has to steer toward exclusion, not a third label."""
    with pytest.raises(ValueError, match="excluded from the corpus"):
        validate_domain("number_theory")


# ---------------------------------------------------------------------------
# 1C — classify_text
# ---------------------------------------------------------------------------


def test_classifier_recognises_a_clear_vision_paper():
    verdict = classify_text("YOLOv10: Real-Time End-to-End Object Detection on COCO")
    assert verdict.best == "computer_vision"
    assert verdict.confident


def test_classifier_recognises_a_clear_medical_paper():
    verdict = classify_text(
        "Trustworthy deep learning for medical image segmentation of patient anatomy in clinical CT"
    )
    assert verdict.best == "medical_imaging"
    assert verdict.confident


@pytest.mark.parametrize(
    "text",
    [
        "A Note on Large Sums of Divisor-Bounded Multiplicative Functions",
        "Interference of Two-Dimensional Bose-Einstein Condensates in Micro-Gravity",
        "Robust Moving-horizon Estimation for Nonlinear Systems",
        "Mistral 7B",
    ],
)
def test_classifier_declines_to_guess_on_off_topic_text(text):
    """Real off-topic papers from the live corpus must produce no verdict at all.

    `best is None` has to mean "neither field", never "medical imaging by
    default" — these four were all stored as computer_vision.
    """
    verdict = classify_text(text)
    assert verdict.best is None
    assert verdict.looks_off_topic


def test_classifier_is_deterministic():
    text = "cardiac MRI segmentation in patients"
    assert classify_text(text) == classify_text(text)


def test_segmentation_alone_is_not_medical_evidence():
    """Both fields segment things, so the word cannot carry domain on its own.

    This is why the medical term list is clinical (modality, subject, reader)
    rather than task-shaped: "Segment Anything" is a CV paper full of
    segmentation vocabulary.
    """
    verdict = classify_text("semantic segmentation of objects in natural images")
    assert verdict.best == "computer_vision"


# ---------------------------------------------------------------------------
# 1C — verify_declared_domain is a REPORTER, never an authority
# ---------------------------------------------------------------------------


def test_verifier_is_silent_when_the_declaration_is_corroborated():
    assert verify_declared_domain(
        "computer_vision", title="Object Detection with Transformers on COCO"
    ) is None


def test_verifier_warns_on_a_confident_disagreement():
    warning = verify_declared_domain(
        "computer_vision",
        title="Trustworthy Deep Learning for Medical Image Segmentation",
        abstract="We segment patient anatomy in clinical CT, reviewed by a radiologist.",
    )
    assert warning is not None
    assert "medical_imaging" in warning


def test_verifier_warns_when_neither_domain_is_corroborated():
    warning = verify_declared_domain(
        "computer_vision", title="A Note on Large Sums of Divisor-Bounded Multiplicative Functions"
    )
    assert warning is not None
    assert "belongs in the corpus" in warning


def test_verifier_returns_a_string_and_changes_nothing():
    """The verifier must not be able to alter a domain — it only describes.

    Advisor decision: the heuristic is brittle on this corpus, so letting it
    overrule a human declaration would trade a loud failure mode for a quiet one.
    """
    result = verify_declared_domain("computer_vision", title="clinical MRI of patient anatomy")
    assert isinstance(result, str)
    # there is no return path that yields a domain, by construction
    assert result not in RESEARCH_DOMAINS


def test_extract_paper_logs_a_domain_warning_without_changing_the_domain(monkeypatch, caplog):
    """End-to-end: a mismatch is logged, and the stored domain is still the declared one."""
    import pipeline.extractor as extractor

    monkeypatch.setattr(
        extractor,
        "fetch_paper_text",
        lambda arxiv_id: {
            "title": "Trustworthy Deep Learning for Medical Image Segmentation",
            "year": 2023,
            "abstract": "Segmenting patient anatomy in clinical CT, read by a radiologist.",
        },
    )
    monkeypatch.setattr(extractor, "fetch_full_text", lambda a, abstract="": ("text", "explicit"))
    monkeypatch.setattr(
        extractor,
        "call_ollama",
        lambda prompt: {
            "objectives": [], "methods": [], "datasets": [],
            "evaluation_metrics": [], "limitations": [], "future_directions": [],
        },
    )

    with caplog.at_level(logging.WARNING, logger="pipeline.extractor"):
        paper = extractor.extract_paper("2305.17456", domain="computer_vision")

    assert paper.domain == "computer_vision", "the verifier must not override the declaration"
    assert "Domain check for 2305.17456" in caplog.text
    assert "medical_imaging" in caplog.text


def test_extract_paper_rejects_an_unknown_domain(monkeypatch):
    """Validation happens before any network call, so a bad domain costs nothing."""
    import pipeline.extractor as extractor

    def must_not_run(*a, **k):
        raise AssertionError("validation must precede the Semantic Scholar fetch")

    monkeypatch.setattr(extractor, "fetch_paper_text", must_not_run)

    with pytest.raises(ValueError):
        extractor.extract_paper("2301.00234", domain="astrophysics")


# ---------------------------------------------------------------------------
# curate_declared_domain — the one verb that IS allowed to decide
# ---------------------------------------------------------------------------


def test_curation_excludes_text_matching_neither_domain():
    assert curate_declared_domain(
        "computer_vision", title="A Note on Large Sums of Divisor-Bounded Multiplicative Functions"
    ) is None


def test_curation_corrects_a_confident_mismatch():
    assert curate_declared_domain(
        "computer_vision",
        title="Trustworthy Deep Learning for Medical Image Segmentation",
        abstract="patient anatomy in clinical CT reviewed by a radiologist",
    ) == "medical_imaging"


def test_curation_keeps_the_declaration_when_unsure():
    """A narrow margin is not evidence, so the caller's declaration stands.

    This text scores cv=4 / mi=3 — a one-point lead, below the margin needed for
    confidence. Both fields share most of their machine-learning vocabulary, so a
    narrow win means nothing and the human declaration must survive it.
    """
    text = "MRI reconstruction using convolutional networks and visual features"
    verdict = classify_text(text)
    assert not verdict.confident, "test setup: this text must be a near-tie"
    assert verdict.best == "computer_vision", "test setup: CV must be the narrow leader"

    assert curate_declared_domain("medical_imaging", title=text) == "medical_imaging"
