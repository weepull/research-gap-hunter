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


# ---------------------------------------------------------------------------
# P2 — old-style ids survive feed parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("math/0309136v1", "math/0309136"),
    ("cs/0309136v3", "cs/0309136"),
    ("2301.00234v2", "2301.00234"),
])
def test_parse_feed_keeps_the_whole_arxiv_id(raw, expected):
    """Pre-2007 ids contain a slash. Taking only the last path segment turned
    `math/0309136` into `0309136`, so a by-id lookup could never find it and the
    admission check would reject a paper for a parsing reason, not a rule."""
    feed = _FEED.replace("2502.54321", raw, 1)
    ids = [c.arxiv_id for c in parse_feed(feed)]
    assert expected in ids


# ---------------------------------------------------------------------------
# P2 — one admission validator for every ingestion path
# ---------------------------------------------------------------------------

_KOSMOS = ArxivCandidate("2306.14824", "Kosmos-2: Grounding Multimodal LLMs", "", "cs.CL",
                         ("cs.CL", "cs.CV"), "2023-06-26T00:00:00Z")
_CLINICAL = "Deep learning segmentation of tumours on chest CT and MRI scans for radiology"

_RULE_CASES = [
    ("computer_vision", "cs.CV", "Object detection"),
    ("computer_vision", "cs.CL", "A language model"),
    ("computer_vision", "eess.IV", "Image restoration"),
    ("computer_vision", "", "No category"),
    ("medical_imaging", "eess.IV", "Image restoration"),
    ("medical_imaging", "physics.med-ph", "Dosimetry"),
    ("medical_imaging", "cs.CV", _CLINICAL),
    ("medical_imaging", "cs.CV", "Object detection in street scenes"),
    ("medical_imaging", "cs.LG", _CLINICAL),
]


@pytest.mark.parametrize("domain, primary, title", _RULE_CASES)
def test_admit_decides_exactly_as_accept_does(domain, primary, title):
    """The validator is a coverage fix, not a policy change: same verdict, same reason."""
    candidate = ArxivCandidate("2501.00001", title, "", primary, (primary,), "2025")
    verdict = cr.admit(domain, ["2501.00001"], known={"2501.00001": candidate},
                       allowlist=set())["2501.00001"]
    assert (verdict.accepted, verdict.reason) == cr.accept(
        domain, "2501.00001", primary, title, "", allowlist=set()
    )


def test_admit_with_known_candidates_makes_no_request(monkeypatch):
    from tests.conftest import arxiv_reports

    calls = arxiv_reports(monkeypatch, "cs.CV")
    cr.admit("computer_vision", [_KOSMOS.arxiv_id], known={_KOSMOS.arxiv_id: _KOSMOS},
             allowlist=set())
    assert calls == []


def test_admit_looks_up_unknown_ids_in_one_request(monkeypatch):
    from tests.conftest import arxiv_reports

    calls = arxiv_reports(monkeypatch, "cs.CL")
    verdicts = cr.admit("computer_vision", ["2306.14824", "2401.13601"], allowlist=set())
    assert calls == [["2306.14824", "2401.13601"]]
    assert not any(v.accepted for v in verdicts.values())
    assert all(v.primary_category == "cs.CL" for v in verdicts.values())


def test_admit_keys_verdicts_by_the_callers_id_including_a_version(monkeypatch):
    from tests.conftest import arxiv_reports

    calls = arxiv_reports(monkeypatch, "cs.CV")
    verdicts = cr.admit("computer_vision", ["2301.00234v2"], allowlist=set())
    assert calls == [["2301.00234"]]
    assert verdicts["2301.00234v2"].accepted


def test_admit_rejects_an_id_arxiv_has_no_record_of(monkeypatch):
    from pipeline import arxiv_source

    monkeypatch.setattr(arxiv_source, "fetch_by_ids", lambda ids: {})
    verdict = cr.admit("computer_vision", ["0000.99999"], allowlist=set())["0000.99999"]
    assert not verdict.accepted
    assert "no record" in verdict.reason


def test_admit_does_not_look_up_allowlisted_ids(monkeypatch):
    from tests.conftest import arxiv_reports

    calls = arxiv_reports(monkeypatch, "cs.CL")
    verdict = cr.admit("medical_imaging", ["2401.00001"], allowlist={"2401.00001"})["2401.00001"]
    assert verdict.accepted
    assert calls == []


def test_admit_fails_closed_when_arxiv_cannot_be_asked(monkeypatch):
    from pipeline import arxiv_source

    def unreachable(arxiv_ids):
        raise RuntimeError("arXiv request failed after 4 attempts: timed out")

    monkeypatch.setattr(arxiv_source, "fetch_by_ids", unreachable)
    with pytest.raises(cr.AdmissionUnverifiable):
        cr.admit("computer_vision", ["2501.00001"], allowlist=set())


def test_the_tranche_path_still_rejects_by_the_rule(monkeypatch):
    """Control: the one path that was already gated must behave identically after P2."""
    import importlib

    tranche = importlib.import_module("scripts.ingest_tranche")
    logged = []
    cv = ArxivCandidate("2501.00001", "Object detection", "", "cs.CV", ("cs.CV",), "2025")
    pages = {0: [_KOSMOS, cv]}
    monkeypatch.setattr(tranche, "search_category",
                        lambda category, start=0, max_results=100: pages.get(start, []))
    monkeypatch.setattr(tranche, "_save_checkpoint", lambda state: None)
    monkeypatch.setattr(tranche, "_log_outcome", logged.append)
    monkeypatch.setattr(tranche.corpus_rules, "load_allowlist", lambda path=None: set())

    accepted = tranche._gather_candidates("computer_vision", 5, set(), {})

    assert [c.arxiv_id for c, _ in accepted] == ["2501.00001"]
    assert [(e["arxiv_id"], e["outcome"]) for e in logged] == [("2306.14824", "rule_rejected")]
