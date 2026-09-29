"""arXiv API client for candidate discovery.

Separate from `pipeline/batch.py`'s Semantic Scholar client because the two answer
different questions. Semantic Scholar is used for *metadata about a known paper* (title,
year, abstract). arXiv is used for *finding candidates*, and critically it is the only
one of the two that reports a paper's **primary category** — which is what the corpus
rules in `pipeline/corpus_rules.py` are built on.

Rate limiting: arXiv's terms ask for no more than one request every three seconds. That
is respected by default and is not configurable downward.
"""

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# arXiv asks for one request every three seconds. Not lowered.
_MIN_REQUEST_INTERVAL = 3.0
_MAX_ATTEMPTS = 4
_BACKOFF_BASE = 5.0
# arXiv rejects unhelpful default user agents and requires an Accept it recognises.
_HEADERS = {
    "User-Agent": "research-gap-hunter/0.1 (academic gap analysis; contact via repo)",
    "Accept": "application/atom+xml",
}

_last_request_at = 0.0


@dataclass(frozen=True)
class ArxivCandidate:
    """One paper as arXiv reports it."""

    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    categories: tuple[str, ...]
    published: str

    @property
    def year(self) -> int:
        match = re.match(r"^(\d{4})", self.published or "")
        return int(match.group(1)) if match else 0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_at = time.monotonic()


def _get(params: dict) -> str:
    """One throttled GET with exponential backoff. Returns the response body."""
    url = f"{_API}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        _throttle()
        try:
            request = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            if attempt == _MAX_ATTEMPTS - 1:
                break
            wait = _BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "arXiv request failed (attempt %d/%d): %s; retrying in %.0fs",
                attempt + 1, _MAX_ATTEMPTS, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"arXiv request failed after {_MAX_ATTEMPTS} attempts: {last_error}")


def parse_feed(xml_text: str) -> list[ArxivCandidate]:
    """Parse an arXiv Atom feed into candidates. Pure, so it is testable offline."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"arXiv returned unparseable XML: {exc}") from None

    out: list[ArxivCandidate] = []
    for entry in root.findall("atom:entry", _NS):
        raw_id = (entry.findtext("atom:id", default="", namespaces=_NS) or "")
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        # Strip the version suffix: 2301.00234v2 -> 2301.00234, so ids match what is
        # already stored and the skip-already-ingested check works.
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        if not arxiv_id:
            continue
        primary_el = entry.find("arxiv:primary_category", _NS)
        categories = tuple(
            c.get("term", "") for c in entry.findall("atom:category", _NS) if c.get("term")
        )
        out.append(ArxivCandidate(
            arxiv_id=arxiv_id,
            title=" ".join((entry.findtext("atom:title", default="", namespaces=_NS) or "").split()),
            abstract=" ".join((entry.findtext("atom:summary", default="", namespaces=_NS) or "").split()),
            primary_category=(primary_el.get("term", "") if primary_el is not None else ""),
            categories=categories,
            published=entry.findtext("atom:published", default="", namespaces=_NS) or "",
        ))
    return out


def search_category(
    category: str, start: int = 0, max_results: int = 100,
    sort_by: str = "submittedDate",
) -> list[ArxivCandidate]:
    """Candidates in one arXiv category, newest first by default.

    Queries `cat:<category>` rather than a keyword search. Keyword search is how the
    original corpus acquired analytic number theory and atomic physics papers under
    "computer vision": a keyword query returns whatever matches the words, and nothing
    downstream checked. A category query is a claim arXiv itself makes about the paper.
    """
    body = _get({
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    })
    return parse_feed(body)


def fetch_by_ids(arxiv_ids: list[str]) -> dict[str, ArxivCandidate]:
    """Look up specific ids, 25 at a time. Returns {arxiv_id: candidate}."""
    found: dict[str, ArxivCandidate] = {}
    for i in range(0, len(arxiv_ids), 25):
        chunk = arxiv_ids[i:i + 25]
        body = _get({"id_list": ",".join(chunk), "max_results": len(chunk)})
        for candidate in parse_feed(body):
            found[candidate.arxiv_id] = candidate
    return found
