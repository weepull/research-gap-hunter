"""Tests for pipeline.corpus_rules and pipeline.arxiv_source (Phase 3b).

These fail against pre-fix code because neither module existed: ingestion accepted
whatever a keyword search returned and stamped it with the calling script's domain,
which is how fourteen off-topic papers entered the corpus across two curation passes.
"""

import pytest

from pipeline import corpus_rules as cr
from pipeline.arxiv_source import ArxivCandidate, parse_feed

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.12345v2</id>
    <title>A Detector
      for Small Objects</title>
    <summary>We detect  small objects
      in natural images.</summary>
    <published>2025-01-15T00:00:00Z</published>
    <arxiv:primary_category term="cs.CV"/>
    <category term="cs.CV"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2502.54321</id>
    <title>Language Modelling</title>
    <summary>An LLM.</summary>
    <published>2025-02-20T00:00:00Z</published>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
  </entry>
</feed>"""


# ---------------------------------------------------------------------------
# arXiv feed parsing — pure, so testable without the network
# ---------------------------------------------------------------------------


def test_parse_feed_extracts_primary_category():
    """The primary category is the whole basis of the corpus rules."""
    candidates = parse_feed(_FEED)
    assert [c.primary_category for c in candidates] == ["cs.CV", "cs.CL"]


def test_parse_feed_strips_the_version_suffix():
    """Ids must match what is already stored or skip-already-ingested silently fails."""
    assert parse_feed(_FEED)[0].arxiv_id == "2501.12345"


def test_parse_feed_collapses_whitespace_in_title_and_abstract():
    candidate = parse_feed(_FEED)[0]
    assert candidate.title == "A Detector for Small Objects"
    assert candidate.abstract == "We detect small objects in natural images."


def test_parse_feed_keeps_all_categories_and_year():
    candidate = parse_feed(_FEED)[0]
    assert candidate.categories == ("cs.CV", "cs.LG")
    assert candidate.year == 2025


def test_parse_feed_rejects_unparseable_xml():
    with pytest.raises(ValueError, match="unparseable"):
        parse_feed("<not xml")


def test_parse_feed_tolerates_an_empty_feed():
    assert parse_feed(
        '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    ) == []


# ---------------------------------------------------------------------------
# computer_vision: primary must be cs.CV, cross-listing is not enough
# ---------------------------------------------------------------------------


def test_cv_accepts_cs_cv_primary():
    ok, reason = cr.accept("computer_vision", "x", "cs.CV", allowlist=set())
    assert ok and "cs.CV accepted" in reason


@pytest.mark.parametrize("primary", ["cs.CL", "cs.LG", "eess.IV", "math.NT", "", "cs.AI"])
def test_cv_rejects_any_other_primary(primary):
    """Every paper removed in curation pass 2 was cross-listed cs.CV with a cs.CL primary."""
    ok, reason = cr.accept("computer_vision", "x", primary, allowlist=set())
    assert not ok
    assert "not accepted for computer_vision" in reason


def test_cv_rejects_a_cross_listed_paper_whose_primary_is_elsewhere():
    """Explicitly: cross-listing to cs.CV does not qualify."""
    ok, _ = cr.accept(
        "computer_vision", "2404.07922", "cs.CL",
        title="LaVy: Vietnamese Multimodal Large Language Model",
        allowlist=set(),
    )
    assert not ok


# ---------------------------------------------------------------------------
# medical_imaging: two primaries outright, plus a verified cs.CV path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("primary", ["eess.IV", "physics.med-ph"])
def test_mi_accepts_its_natural_primaries(primary):
    ok, _ = cr.accept("medical_imaging", "x", primary, allowlist=set())
    assert ok


def test_mi_accepts_cs_cv_when_the_keyword_verifier_fires():
    ok, reason = cr.accept(
        "medical_imaging", "x", "cs.CV",
        title="Tumour segmentation in clinical CT",
        abstract="We segment patient anatomy, reviewed by a radiologist.",
        allowlist=set(),
    )
    assert ok and "medical keyword match" in reason


def test_mi_rejects_cs_cv_when_the_verifier_does_not_fire():
    """A general CV paper must not slip into medical imaging just by being cs.CV."""
    ok, reason = cr.accept(
        "medical_imaging", "x", "cs.CV",
        title="Object detection on COCO",
        abstract="Bounding boxes on natural images.",
        allowlist=set(),
    )
    assert not ok and "requires a medical keyword match" in reason


def test_mi_rejects_other_primaries_without_an_allowlist_entry():
    ok, reason = cr.accept("medical_imaging", "x", "cs.CY", allowlist=set())
    assert not ok and "not accepted for medical_imaging" in reason


def test_the_cv_rule_is_not_applied_to_mi():
    """A single-category rule cannot work for a field spanning two primaries.

    Medical imaging is 44 cs.CV and 35 eess.IV in the current corpus, so applying the
    CV rule to it would reject real papers.
    """
    assert cr.PRIMARY_RULES["computer_vision"] != cr.PRIMARY_RULES["medical_imaging"]
    assert "eess.IV" in cr.PRIMARY_RULES["medical_imaging"]
    assert "eess.IV" not in cr.PRIMARY_RULES["computer_vision"]


# ---------------------------------------------------------------------------
# the allowlist
# ---------------------------------------------------------------------------


def test_allowlist_overrides_the_primary_rule():
    ok, reason = cr.accept("medical_imaging", "2402.14815", "cs.CY",
                           allowlist={"2402.14815"})
    assert ok and "allowlisted" in reason


def test_allowlist_file_parses_comments_and_blank_lines(tmp_path):
    path = tmp_path / "allow.txt"
    path.write_text("# a comment\n\n2402.14815  # cs.CY note\n2404.15692\n")
    assert cr.load_allowlist(path) == {"2402.14815", "2404.15692"}


def test_missing_allowlist_file_is_an_empty_set(tmp_path):
    assert cr.load_allowlist(tmp_path / "nope.txt") == set()


def test_the_shipped_allowlist_covers_the_six_known_exemptions():
    """Without these the rules would reject MI papers already judged in scope."""
    shipped = cr.load_allowlist()
    for arxiv_id in ("2402.14815", "2404.15692", "2405.00241",
                     "2501.04614", "2505.03785", "2506.07044"):
        assert arxiv_id in shipped, f"{arxiv_id} missing from data/mi_allowlist.txt"


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------


def test_an_unknown_domain_is_rejected_before_any_rule_runs():
    with pytest.raises(ValueError):
        cr.accept("robotics", "x", "cs.RO", allowlist=set())


def test_accept_always_explains_itself():
    """ingest_log.jsonl records why every paper was taken, not only the failures."""
    for domain, primary in (("computer_vision", "cs.CV"), ("medical_imaging", "eess.IV"),
                            ("computer_vision", "cs.CL")):
        _ok, reason = cr.accept(domain, "x", primary, allowlist=set())
        assert reason.strip(), f"no reason given for {domain}/{primary}"


def test_the_keyword_verifier_never_assigns_a_domain():
    """It gates a declared domain; it does not choose one.

    Same restraint as pipeline.domains.verify_declared_domain — the heuristic is
    brittle enough that letting it decide would trade a loud failure mode for a quiet
    one.
    """
    ok_cv, _ = cr.accept("computer_vision", "x", "cs.CV",
                         title="clinical MRI of patient anatomy", allowlist=set())
    # Medical text does NOT move this paper out of the domain the caller declared.
    assert ok_cv is True
