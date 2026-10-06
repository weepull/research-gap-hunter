"""Unit tests for scripts/harvest_external_labels.py (V1, 2026-10-06).

Pure parsing and row-building only, on offline fixtures. Pins the phase's rules: raw values
are kept verbatim, missing stays missing (never inferred), a PubMed hit counts only when the
returned record verifiably is the paper, and Semantic Scholar's model-generated
fieldsOfStudy is never requested.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import harvest_external_labels as h  # noqa: E402

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2305.17456v3</id>
    <arxiv:doi>10.1016/j.media.2024.103127</arxiv:doi>
    <arxiv:journal_ref>Medical Image Analysis 94 (2024) 103127</arxiv:journal_ref>
    <arxiv:primary_category term="eess.IV"/>
    <category term="eess.IV"/>
    <category term="cs.CV"/>
    <category term="physics.med-ph"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/math/0309136v1</id>
    <arxiv:primary_category term="cs.CV"/>
    <category term="cs.CV"/>
  </entry>
</feed>"""

_PUBMED = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">38000001</PMID>
      <Article><ArticleTitle>Trustworthy deep learning for medical image segmentation.</ArticleTitle></Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName UI="D006801" MajorTopicYN="N">Humans</DescriptorName></MeshHeading>
        <MeshHeading>
          <DescriptorName UI="D007091" MajorTopicYN="Y">Image Processing, Computer-Assisted</DescriptorName>
          <QualifierName UI="Q000379" MajorTopicYN="N">methods</QualifierName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData><ArticleIdList>
      <ArticleId IdType="pubmed">38000001</ArticleId>
      <ArticleId IdType="doi">10.1016/J.MEDIA.2024.103127</ArticleId>
      <ArticleId IdType="pmc">PMC1234567</ArticleId>
    </ArticleIdList></PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation Status="PubMed-not-MEDLINE" Owner="NLM">
      <PMID Version="1">38000002</PMID>
      <Article><ArticleTitle>An unrelated paper.</ArticleTitle></Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/other</ArticleId></ArticleIdList></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_semantic_scholar_never_requests_model_generated_fields():
    fields = set(h.S2_FIELDS.split(","))
    assert fields == {"venue", "publicationVenue", "externalIds"}
    assert not fields & {"fieldsOfStudy", "s2FieldsOfStudy"}


def test_arxiv_feed_separates_primary_from_cross_lists_and_keeps_doi_and_journal_ref():
    parsed = h.parse_arxiv_feed(_ATOM)
    p = parsed["2305.17456"]
    assert p["primary"] == "eess.IV"
    assert p["cross_lists"] == ["cs.CV", "physics.med-ph"]
    assert p["doi"] == "10.1016/j.media.2024.103127"
    assert p["journal_ref"] == "Medical Image Analysis 94 (2024) 103127"


def test_arxiv_absent_fields_stay_absent_and_old_style_ids_survive():
    p = h.parse_arxiv_feed(_ATOM)["math/0309136"]
    assert p["cross_lists"] == [] and p["doi"] is None and p["journal_ref"] is None


def test_semantic_scholar_venue_is_kept_raw_and_a_null_entry_is_missing():
    payload = [
        {"externalIds": {"DOI": "10.48550/arXiv.2305.17456"}, "venue": "arXiv.org",
         "publicationVenue": {"name": "arXiv.org", "issn": "2331-8422"}},
        None,
    ]
    parsed = h.parse_s2_batch(payload, ["2305.17456", "2401.00001"])
    assert parsed["2305.17456"]["venue"] == "arXiv.org"
    assert json.loads(parsed["2305.17456"]["publication_venue"]) == {"name": "arXiv.org", "issn": "2331-8422"}
    assert parsed["2305.17456"]["doi"] == "10.48550/arXiv.2305.17456"
    assert parsed["2401.00001"] is None


def test_pubmed_records_keep_mesh_verbatim_with_status_and_ids():
    records = h.parse_pubmed_xml(_PUBMED)
    r = records["38000001"]
    assert r["status"] == "MEDLINE" and r["pmc"] == "PMC1234567"
    assert r["mesh"] == [
        {"descriptor": "Humans", "ui": "D006801", "major": False, "qualifiers": []},
        {"descriptor": "Image Processing, Computer-Assisted", "ui": "D007091", "major": True,
         "qualifiers": [{"qualifier": "methods", "ui": "Q000379", "major": False}]},
    ]
    assert records["38000002"]["mesh"] == []


def test_a_pubmed_hit_counts_only_when_it_verifiably_is_the_paper():
    records = h.parse_pubmed_xml(_PUBMED)
    assert h.verify_pubmed_match(records["38000001"], doi="10.1016/j.media.2024.103127", title=None)
    assert h.verify_pubmed_match(records["38000001"], doi=None,
                                 title="Trustworthy Deep Learning for Medical Image Segmentation")
    assert not h.verify_pubmed_match(records["38000002"], doi="10.1016/j.media.2024.103127", title=None)
    assert not h.verify_pubmed_match(records["38000002"], doi=None,
                                     title="Trustworthy Deep Learning for Medical Image Segmentation")


def test_missing_stays_missing_and_every_field_has_a_source_and_fetched_at():
    row = h.build_row(
        "2401.00001",
        arxiv=None, arxiv_at="2026-10-06T00:00:00Z",
        s2=None, s2_at="2026-10-06T00:00:01Z",
        pubmed={"outcome": "lookup_failed"}, pubmed_at="2026-10-06T00:00:02Z",
    )
    for field in h.VALUE_FIELDS:
        assert row[field] == "", field
        assert row[f"{field}_source"] and row[f"{field}_fetched_at"], field
    assert "missing" in row["arxiv_primary_category_source"]
    assert set(row) == {"arxiv_id"} | {f"{f}{s}" for f in h.VALUE_FIELDS for s in ("", "_source", "_fetched_at")}


def test_not_found_in_pubmed_is_a_result_not_a_gap():
    """A completed lookup that finds nothing is recorded as such; a failed one stays empty."""
    found_nothing = h.build_row("x", arxiv=None, arxiv_at="t", s2=None, s2_at="t",
                                pubmed={"outcome": "not_found", "tried": ["doi", "title"]}, pubmed_at="t")
    assert found_nothing["pubmed_indexed"] == "not_found"
    failed = h.build_row("x", arxiv=None, arxiv_at="t", s2=None, s2_at="t",
                         pubmed={"outcome": "lookup_failed"}, pubmed_at="t")
    assert failed["pubmed_indexed"] == ""
