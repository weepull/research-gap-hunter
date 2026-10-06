"""Harvest third-party, human-assigned domain signals for every corpus paper (V1). READ-ONLY.

Collects, per paper, raw values from three external sources:

- **arXiv**: primary category and every cross-list, the author-supplied DOI and journal-ref.
- **Semantic Scholar**: `venue`, `publicationVenue` and `externalIds` (for the DOI). Never
  `fieldsOfStudy` or `s2FieldsOfStudy`, which are model-generated.
- **PubMed**: looked up by DOI, then by title. A hit counts only if the returned record's own
  DOI or title matches; then PMID, PMC id, MEDLINE status and every MeSH heading verbatim.

Writes `eval/external_labels.csv`, one row per paper, with a `_source` and `_fetched_at` for
every value. Missing stays missing: an empty value with a source that says why. Nothing is
inferred, normalised, classified or written anywhere else. No LLM, no classifier, and not the
project's keyword rule.

Raw responses are cached under `data/external_raw/` (gitignored), so a rate-limited run resumes
instead of refetching or dropping papers. HTTP 429 and 5xx back off for up to ~16 minutes per
request.

    python scripts/harvest_external_labels.py
"""

import csv
import json
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "external_raw"
OUT = ROOT / "eval" / "external_labels.csv"

ARXIV_API = "https://export.arxiv.org/api/query"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# venue/publicationVenue are the venue signal; externalIds supplies the DOI for PubMed.
# fieldsOfStudy / s2FieldsOfStudy are model-generated and deliberately never requested.
S2_FIELDS = "venue,publicationVenue,externalIds"

_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_BACKOFF = (30, 60, 120, 240, 480)

VALUE_FIELDS = (
    "arxiv_primary_category", "arxiv_cross_lists", "arxiv_doi", "arxiv_journal_ref",
    "s2_venue", "s2_publication_venue", "s2_doi",
    "pubmed_indexed", "pubmed_match_method", "pubmed_pmid", "pubmed_pmc", "pubmed_status",
    "pubmed_mesh",
)


# --------------------------------------------------------------------------- parsing (pure)

def _arxiv_id(raw_id: str) -> str:
    tail = raw_id.split("/abs/", 1)[1] if "/abs/" in raw_id else raw_id.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", tail)


def parse_arxiv_feed(xml_text: str) -> dict:
    """{arxiv_id: {primary, cross_lists, doi, journal_ref}} from an Atom feed. Raw values."""
    out = {}
    for entry in ET.fromstring(xml_text).findall("atom:entry", _NS):
        pid = _arxiv_id(entry.findtext("atom:id", default="", namespaces=_NS) or "")
        if not pid:
            continue
        prim = entry.find("arxiv:primary_category", _NS)
        primary = prim.get("term") if prim is not None else None
        cats = [c.get("term") for c in entry.findall("atom:category", _NS) if c.get("term")]
        out[pid] = {
            "primary": primary,
            "cross_lists": [c for c in cats if c != primary],
            "doi": (entry.findtext("arxiv:doi", default=None, namespaces=_NS) or None),
            "journal_ref": (entry.findtext("arxiv:journal_ref", default=None, namespaces=_NS) or None),
        }
    return out


def parse_s2_batch(payload: list, requested: list[str]) -> dict:
    """{arxiv_id: {venue, publication_venue (raw JSON), doi} or None when S2 has no record}."""
    out = {}
    for pid, item in zip(requested, payload):
        if item is None:
            out[pid] = None
            continue
        pv = item.get("publicationVenue")
        out[pid] = {
            "venue": item.get("venue") or None,
            "publication_venue": None if pv is None else json.dumps(pv, sort_keys=True, ensure_ascii=False),
            "doi": (item.get("externalIds") or {}).get("DOI"),
        }
    return out


def parse_esearch(payload: dict) -> list[str]:
    return list(payload.get("esearchresult", {}).get("idlist", []))


def parse_pubmed_xml(xml_text: str) -> dict:
    """{pmid: {title, dois, pmc, status, mesh}} — MeSH headings verbatim, with UIs and flags."""
    out = {}
    for art in ET.fromstring(xml_text).findall("PubmedArticle"):
        mc = art.find("MedlineCitation")
        pmid = mc.findtext("PMID")
        title = "".join(mc.find("Article/ArticleTitle").itertext()) if mc.find("Article/ArticleTitle") is not None else ""
        ids = art.findall("PubmedData/ArticleIdList/ArticleId")
        mesh = []
        for mh in mc.findall("MeshHeadingList/MeshHeading"):
            d = mh.find("DescriptorName")
            mesh.append({
                "descriptor": d.text, "ui": d.get("UI"), "major": d.get("MajorTopicYN") == "Y",
                "qualifiers": [{"qualifier": q.text, "ui": q.get("UI"), "major": q.get("MajorTopicYN") == "Y"}
                               for q in mh.findall("QualifierName")],
            })
        out[pmid] = {
            "title": title,
            "dois": [i.text for i in ids if i.get("IdType") == "doi" and i.text],
            "pmc": next((i.text for i in ids if i.get("IdType") == "pmc"), None),
            "status": mc.get("Status"),
            "mesh": mesh,
        }
    return out


def _norm_title(text: str) -> str:
    """For verifying a title match only — never applied to any stored value."""
    t = unicodedata.normalize("NFKD", text or "").lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def verify_pubmed_match(record: dict, doi: str | None, title: str | None) -> bool:
    if doi:
        return doi.lower() in {d.lower() for d in record["dois"]}
    if title:
        return _norm_title(record["title"]) == _norm_title(title)
    return False


def build_row(pid: str, arxiv, arxiv_at: str, s2, s2_at: str, pubmed: dict, pubmed_at: str) -> dict:
    """One CSV row. Each value gets `<field>_source` and `<field>_fetched_at`; missing is empty."""
    row = {"arxiv_id": pid}

    def put(field, value, source, at):
        row[field], row[f"{field}_source"], row[f"{field}_fetched_at"] = value, source, at

    a_src = "arXiv API" if arxiv else "arXiv API: missing (no entry returned)"
    put("arxiv_primary_category", (arxiv or {}).get("primary") or "", a_src, arxiv_at)
    put("arxiv_cross_lists", ";".join((arxiv or {}).get("cross_lists") or []), a_src, arxiv_at)
    for f in ("doi", "journal_ref"):
        v = (arxiv or {}).get(f)
        put(f"arxiv_{f}", v or "", a_src if (not arxiv or v) else f"arXiv API: missing ({f} not provided)", arxiv_at)

    s_src = "Semantic Scholar paper/batch" if s2 else "Semantic Scholar: missing (no record)"
    for f, key in (("s2_venue", "venue"), ("s2_publication_venue", "publication_venue"), ("s2_doi", "doi")):
        v = (s2 or {}).get(key)
        put(f, v or "", s_src if (not s2 or v) else f"Semantic Scholar: missing ({key} empty)", s2_at)

    outcome = pubmed.get("outcome")
    rec = pubmed.get("record") or {}
    if outcome == "found":
        p_src = f"PubMed E-utilities (matched by {pubmed['method']}, verified)"
    elif outcome == "not_found":
        p_src = f"PubMed E-utilities: no verified record (tried {', '.join(pubmed.get('tried', []))})"
    else:
        p_src = "PubMed E-utilities: missing (lookup failed)"
    put("pubmed_indexed", {"found": "yes", "not_found": "not_found"}.get(outcome, ""), p_src, pubmed_at)
    put("pubmed_match_method", pubmed.get("method", "") if outcome == "found" else "", p_src, pubmed_at)
    put("pubmed_pmid", pubmed.get("pmid", "") if outcome == "found" else "", p_src, pubmed_at)
    put("pubmed_pmc", rec.get("pmc") or "", p_src, pubmed_at)
    put("pubmed_status", rec.get("status") or "", p_src, pubmed_at)
    put("pubmed_mesh", json.dumps(rec["mesh"], ensure_ascii=False) if outcome == "found" else "", p_src, pubmed_at)
    return row


# --------------------------------------------------------------------------- network (cached)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _manifest() -> dict:
    path = CACHE / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _cached(name: str, fetch) -> tuple[str, str]:
    """Return (body, fetched_at), fetching once and caching raw bytes for resumption."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path, manifest = CACHE / name, _manifest()
    if path.exists() and name in manifest:
        return path.read_text(), manifest[name]
    body = fetch()
    path.write_text(body)
    manifest[name] = _now()
    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return body, manifest[name]


def _http(url: str, data: bytes | None = None, headers: dict | None = None, pause: float = 0.0) -> str:
    """GET/POST with backoff on 429 and 5xx. Raises after the last wait rather than dropping."""
    for attempt, wait in enumerate((0,) + _BACKOFF):
        if wait:
            print(f"    backing off {wait}s (attempt {attempt + 1})", flush=True)
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "research-gap-hunter/0.1 (academic; V1 label harvest)", **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            time.sleep(pause)
            return body
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            print(f"    HTTP {exc.code} from {url[:70]}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"    network error {exc!r}", flush=True)
    raise RuntimeError(f"gave up after {len(_BACKOFF) + 1} attempts: {url[:90]}")


def harvest(ids: list[str]) -> list[dict]:
    from dotenv import dotenv_values
    s2_key = (dotenv_values(ROOT / ".env").get("SEMANTIC_SCHOLAR_API_KEY") or "").strip()

    arxiv, arxiv_at, failed = {}, {}, []
    for i in range(0, len(ids), 25):
        chunk = ids[i:i + 25]
        name = f"arxiv_{i:03d}.xml"
        try:
            body, at = _cached(name, lambda c=chunk: _http(
                f"{ARXIV_API}?{urllib.parse.urlencode({'id_list': ','.join(c), 'max_results': len(c)})}", pause=3.0))
        except RuntimeError as exc:
            print(f"  arXiv chunk {i} failed: {exc}"); failed += [f"arxiv:{p}" for p in chunk]; continue
        for pid, rec in parse_arxiv_feed(body).items():
            arxiv[pid], arxiv_at[pid] = rec, at
        print(f"  arXiv {i + len(chunk)}/{len(ids)}", flush=True)

    s2, s2_at = {}, _now()
    try:
        body, s2_at = _cached("s2_batch.json", lambda: _http(
            f"{S2_BATCH}?fields={S2_FIELDS}", data=json.dumps({"ids": [f"ARXIV:{p}" for p in ids]}).encode(),
            headers={"Content-Type": "application/json", **({"x-api-key": s2_key} if s2_key else {})}, pause=1.0))
        s2 = parse_s2_batch(json.loads(body), ids)
    except RuntimeError as exc:
        print(f"  Semantic Scholar failed: {exc}"); failed += [f"s2:{p}" for p in ids]

    titles = dict(sqlite3.connect(f"file:{ROOT / 'data' / 'papers.db'}?mode=ro", uri=True)
                  .execute("SELECT arxiv_id, title FROM papers").fetchall())
    rows = []
    for n, pid in enumerate(ids, 1):
        dois = [d for d in ((arxiv.get(pid) or {}).get("doi"), (s2.get(pid) or {}).get("doi")) if d]
        tried, result, at = [], {"outcome": "not_found"}, _now()
        try:
            for method, term in [("doi", f"{d}[doi]") for d in dict.fromkeys(dois)] + [("title", f"\"{titles.get(pid, '')}\"[Title]")]:
                if method == "title" and not titles.get(pid):
                    continue
                tried.append(method)
                key = re.sub(r"[^A-Za-z0-9]+", "_", f"{pid}_{method}_{term}")[:150]
                body, at = _cached(f"pm_search_{key}.json", lambda t=term: _http(
                    f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode({'db': 'pubmed', 'term': t, 'retmode': 'json', 'retmax': 5})}", pause=0.4))
                pmids = parse_esearch(json.loads(body))
                if not pmids:
                    continue
                xml, at = _cached(f"pm_fetch_{'_'.join(pmids)}.xml", lambda p=pmids: _http(
                    f"{EUTILS}/efetch.fcgi?{urllib.parse.urlencode({'db': 'pubmed', 'id': ','.join(p), 'retmode': 'xml'})}", pause=0.4))
                for pmid, rec in parse_pubmed_xml(xml).items():
                    if verify_pubmed_match(rec, doi=term[:-5] if method == "doi" else None,
                                           title=titles.get(pid) if method == "title" else None):
                        result = {"outcome": "found", "method": method, "pmid": pmid, "record": rec}
                        break
                if result["outcome"] == "found":
                    break
            if result["outcome"] != "found":
                result["tried"] = tried
        except RuntimeError as exc:
            print(f"  PubMed {pid} failed: {exc}"); failed.append(f"pubmed:{pid}")
            result = {"outcome": "lookup_failed"}
        rows.append(build_row(pid, arxiv.get(pid), arxiv_at.get(pid, _now()), s2.get(pid), s2_at, result, at))
        if n % 25 == 0:
            print(f"  PubMed {n}/{len(ids)}", flush=True)
    if failed:
        print(f"COULD NOT FETCH: {failed}")
    return rows


def main() -> int:
    ids = sorted(r[0] for r in sqlite3.connect(f"file:{ROOT / 'data' / 'papers.db'}?mode=ro", uri=True)
                 .execute("SELECT arxiv_id FROM papers").fetchall())
    rows = harvest(ids)
    fields = ["arxiv_id"] + [f"{f}{s}" for f in VALUE_FIELDS for s in ("", "_source", "_fetched_at")]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
