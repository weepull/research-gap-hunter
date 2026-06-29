"""Extracts structured information from arXiv papers via Ollama llama3.1:8b."""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

_SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
_ARXIV_PDF_BASE = "https://arxiv.org/pdf"
_LOG_PATH = Path("data/failed_extractions.log")

# A browser-like User-Agent — arxiv.org returns 403 for default python-requests UA.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Section headers in priority order — the earliest-priority match wins, so the
# limitations section (most valuable for this project) is preferred when present.
_SECTION_HEADERS = ("limitation", "future work", "conclusion", "discussion")
_MAX_SECTION_CHARS = 4000
# Limitation/conclusion sections live in the last ~30% of a paper; 20 pages
# gives longer papers enough coverage to reach them.
_MAX_PDF_PAGES = 20
# Fraction of a paper (from the end) where limitation/conclusion sections cluster.
_TAIL_FRACTION = 0.4
# Matches a header that looks like an actual section heading: at the start of a
# line, optionally preceded by a section number ("5." / "5"), and ending the line.
_SECTION_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\.?\s+)?(limitation|future work|conclusion|discussion)s?\s*\n",
    re.IGNORECASE,
)
# Explicit-tier headings (limitations / future work) are the strongest signal;
# conclusion / discussion headings are a weaker, second-choice source.
_EXPLICIT_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\.?\s+)?(limitation|future work)s?\s*\n",
    re.IGNORECASE,
)
_CONCLUSION_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\.?\s+)?(conclusion|discussion)s?\s*\n",
    re.IGNORECASE,
)
_EXTRACTION_PROMPT = """\
You are a scientific paper analyst. Extract structured information from the following paper.

Return ONLY valid JSON with these exact keys. No explanation, no markdown, no preamble.

{{
  "objectives": ["<list of research objectives>"],
  "methods": ["<list of algorithms or architectures used>"],
  "datasets": ["<list of datasets mentioned>"],
  "evaluation_metrics": ["<list of metrics used>"],
  "limitations": ["<list of explicit limitation statements — be granular, one limitation per item>"],
  "future_directions": ["<list of future work suggestions from the authors>"]
}}

If limitations are not explicitly stated, return an empty list [] — do not invent limitations.
If future_directions are not explicitly stated, return an empty list [] — do not invent future_directions.

Paper text:
{paper_text}"""

_REQUIRED_KEYS = {"objectives", "methods", "datasets", "evaluation_metrics", "limitations", "future_directions"}


class ExtractionTier(str, Enum):
    """How a paper's limitations were sourced — drives prompt and gap-score weight."""

    EXPLICIT = "explicit"      # a dedicated limitations / future-work section
    CONCLUSION = "conclusion"  # only a conclusion / discussion section
    INFERRED = "inferred"      # no relevant section; fall back to the abstract


# Extra prompt guidance appended for the weaker tiers. EXPLICIT keeps the base
# prompt unchanged (see CLAUDE.md — the explicit prompt structure is fixed).
_TIER_INSTRUCTIONS = {
    ExtractionTier.CONCLUSION.value: (
        "This text is from the conclusion section. Extract implied limitations — "
        "look for phrases like 'however', 'despite', 'remains challenging', "
        "'future work includes', 'we leave X for future'. Be specific."
    ),
    ExtractionTier.INFERRED.value: (
        "Limitations are not explicitly stated. Infer them from what the paper "
        "claims to solve and what it does not address. Be conservative — only "
        "infer clear limitations, not speculative ones."
    ),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperExtract(BaseModel):
    arxiv_id: str
    title: str
    year: int
    domain: str = "computer_vision"
    objectives: list[str]
    methods: list[str]
    datasets: list[str]
    evaluation_metrics: list[str]
    limitations: list[str]
    future_directions: list[str]
    raw_json: str
    ingested_at: str
    extraction_tier: str = "explicit"


def fetch_paper_text(arxiv_id: str) -> dict:
    """Fetch paper metadata from Semantic Scholar with exponential backoff on 429.

    Returns dict with keys: title, year, abstract.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    url = f"{_SEMANTIC_SCHOLAR_BASE}/paper/arXiv:{arxiv_id}"
    params = {"fields": "title,year,abstract,tldr,openAccessPdf"}
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    max_attempts = 3
    for attempt in range(max_attempts):
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 429:
            if attempt == max_attempts - 1:
                response.raise_for_status()
            wait = 2 ** attempt
            logger.warning("Rate limited by Semantic Scholar, retrying in %ds", wait)
            time.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        abstract = data.get("abstract") or ""
        if not abstract and data.get("tldr"):
            abstract = data["tldr"].get("text", "")
        return {
            "title": data.get("title", ""),
            "year": data.get("year") or 0,
            "abstract": abstract,
        }

    # Unreachable but satisfies type checkers
    raise RuntimeError("fetch_paper_text exhausted retries")


def _extract_pdf_text(pdf_bytes: bytes, max_pages: int = _MAX_PDF_PAGES) -> list[str]:
    """Extract text from the first max_pages pages of a PDF, one string per page.

    Uses PyMuPDF (fitz): page.get_text("text") preserves proper newlines and
    handles multi-column layouts better than pdfplumber.
    """
    import fitz  # PyMuPDF; lazy import so the module loads without it present

    pages: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            if index >= max_pages:
                break
            pages.append(page.get_text("text") or "")
    return pages


def _extract_section(text: str) -> str:
    """Return the most relevant section's text, capped at _MAX_SECTION_CHARS chars.

    First looks for an actual section *heading* (header at the start of a line,
    optionally numbered like "5. Limitations"), preferring higher-priority headers
    in _SECTION_HEADERS. If no heading is found, falls back to a plain substring
    search for the header anywhere in the text. Returns "" when nothing matches.
    """
    if not text:
        return ""

    # Prefer a real section heading. Among all heading matches, pick the one with
    # the highest-priority header (lowest index), breaking ties by position.
    best_start: int | None = None
    best_rank: tuple[int, int] | None = None
    for match in _SECTION_HEADING_RE.finditer(text):
        keyword = match.group(1).lower()
        rank = (_SECTION_HEADERS.index(keyword), match.start())
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_start = match.start()

    if best_start is not None:
        return text[best_start : best_start + _MAX_SECTION_CHARS].strip()

    # Fallback: substring search anywhere, still in priority order.
    lowered = text.lower()
    for header in _SECTION_HEADERS:
        idx = lowered.find(header)
        if idx != -1:
            return text[idx : idx + _MAX_SECTION_CHARS].strip()
    return ""


def _page_section_text(
    pages: list[str], index: int, heading_re: re.Pattern = _SECTION_HEADING_RE
) -> str:
    """Return text from the heading on pages[index] through the next page, capped.

    Anchors at the heading match within the page (so leading body text on that
    page is dropped) and appends the following page, since a section often spills
    across a page boundary. Truncated to _MAX_SECTION_CHARS.
    """
    page_text = pages[index]
    match = heading_re.search(page_text)
    start = match.start() if match else 0

    combined = page_text[start:]
    if index + 1 < len(pages):
        combined = combined + "\n" + pages[index + 1]
    return combined[:_MAX_SECTION_CHARS].strip()


def _select_section_from_pages(pages: list[str]) -> tuple[str, str]:
    """Find the most relevant section across pages and classify its extraction tier.

    Searches the tail pages (where limitation/conclusion sections cluster) first,
    then the head pages. Prefers an explicit heading (limitations / future work)
    anywhere over a conclusion / discussion heading. Returns
    ``(section_text, tier)`` where tier is one of ExtractionTier's values:
    "explicit", "conclusion", or "inferred" (the last with empty text).
    """
    if not pages:
        return "", ExtractionTier.INFERRED.value

    total = len(pages)
    tail_start = int(total * (1 - _TAIL_FRACTION))
    # Tail pages first, then the head pages as a fallback over the whole document.
    search_order = list(range(tail_start, total)) + list(range(tail_start))

    for heading_re, tier in (
        (_EXPLICIT_HEADING_RE, ExtractionTier.EXPLICIT.value),
        (_CONCLUSION_HEADING_RE, ExtractionTier.CONCLUSION.value),
    ):
        for index in search_order:
            if heading_re.search(pages[index]):
                return _page_section_text(pages, index, heading_re), tier

    return "", ExtractionTier.INFERRED.value


def fetch_full_text(arxiv_id: str, abstract: str = "") -> tuple[str, str]:
    """Download the arXiv PDF and return its relevant section plus an extraction tier.

    Downloads https://arxiv.org/pdf/{arxiv_id} with a browser-like User-Agent,
    extracts text page by page via PyMuPDF (first 20 pages), and returns
    ``(section_text, tier)`` for the most relevant section (max 4000 chars),
    searching the last 40% of pages first. Falls back to ``(abstract, "inferred")``
    if the PDF cannot be fetched or no relevant section is found. Uses exponential
    backoff, up to 3 attempts.
    """
    url = f"{_ARXIV_PDF_BASE}/{arxiv_id}"
    max_attempts = 3
    pages: list[str] = []

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=_BROWSER_HEADERS, timeout=30)
            response.raise_for_status()
            pages = _extract_pdf_text(response.content)
            break
        except Exception as exc:  # noqa: BLE001 — any failure should retry then fall back
            if attempt == max_attempts - 1:
                logger.warning(
                    "PDF fetch failed for %s after %d attempts (%s); falling back to abstract",
                    arxiv_id,
                    max_attempts,
                    exc,
                )
                return abstract, ExtractionTier.INFERRED.value
            wait = 2 ** attempt
            logger.warning(
                "PDF fetch failed for %s (attempt %d/%d): %s; retrying in %ds",
                arxiv_id,
                attempt + 1,
                max_attempts,
                exc,
                wait,
            )
            time.sleep(wait)

    section, tier = _select_section_from_pages(pages)
    if section:
        return section, tier

    logger.info("No relevant section found in PDF for %s; falling back to abstract", arxiv_id)
    return abstract, ExtractionTier.INFERRED.value


def call_ollama(prompt: str) -> dict:
    """Send prompt to Ollama and return parsed JSON dict.

    Raises ValueError if the response body is not valid JSON or missing required keys.
    """
    import ollama as _ollama  # imported here so the module loads without Ollama running

    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    client = _ollama.Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
        format="json",
    )
    raw_text = response["message"]["content"].strip()

    # Strip markdown code fences if the model wraps output despite instructions
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ollama returned non-JSON output: {raw_text[:200]}") from exc

    missing = _REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"Ollama response missing keys: {missing}")

    return parsed


def _log_failure(arxiv_id: str, reason: str, raw: str) -> None:
    """Append a failure record to data/failed_extractions.log."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with _LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] arxiv_id={arxiv_id} reason={reason}\n")
        fh.write(f"  raw={raw[:500]}\n")


def _build_prompt(paper_text: str, tier: str) -> str:
    """Render the extraction prompt, adding tier-specific guidance for weaker tiers.

    The "explicit" tier renders the base prompt verbatim; "conclusion" and
    "inferred" inject extra instructions just before the paper text.
    """
    prompt = _EXTRACTION_PROMPT.format(paper_text=paper_text)
    instruction = _TIER_INSTRUCTIONS.get(tier, "")
    if instruction:
        prompt = prompt.replace(
            "\nPaper text:\n", f"\n{instruction}\n\nPaper text:\n", 1
        )
    return prompt


def extract_paper(arxiv_id: str) -> PaperExtract:
    """Fetch paper, run LLM extraction, validate with Pydantic, and return PaperExtract.

    On validation failure, logs to data/failed_extractions.log and re-raises.
    """
    paper_meta = fetch_paper_text(arxiv_id)
    # Semantic Scholar supplies metadata (title, year); the body text comes from the
    # PDF's limitations/future-work/conclusion section, falling back to the abstract.
    # The tier records where the text came from and shapes the extraction prompt.
    body_text, tier = fetch_full_text(arxiv_id, abstract=paper_meta["abstract"])
    paper_text = f"Title: {paper_meta['title']}\n\n{body_text}"
    prompt = _build_prompt(paper_text, tier)

    raw_dict = call_ollama(prompt)
    raw_json_str = json.dumps(raw_dict)

    try:
        result = PaperExtract(
            arxiv_id=arxiv_id,
            title=paper_meta["title"],
            year=paper_meta["year"],
            domain="computer_vision",
            objectives=raw_dict.get("objectives", []),
            methods=raw_dict.get("methods", []),
            datasets=raw_dict.get("datasets", []),
            evaluation_metrics=raw_dict.get("evaluation_metrics", []),
            limitations=raw_dict.get("limitations", []),
            future_directions=raw_dict.get("future_directions", []),
            raw_json=raw_json_str,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            extraction_tier=tier,
        )
    except ValidationError as exc:
        _log_failure(arxiv_id, str(exc), raw_json_str)
        raise

    return result
