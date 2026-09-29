"""Extracts structured information from arXiv papers via Ollama llama3.1:8b."""

import json
import logging
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from pipeline.domains import validate_domain, verify_declared_domain
from pipeline.extraction_filter import filter_and_log

load_dotenv()

_SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
_ARXIV_PDF_BASE = "https://arxiv.org/pdf"

# arXiv ids are interpolated straight into outbound URLs, so they are validated
# before use: an id containing "../" or a query string can redirect a fetch to a
# different path on arxiv.org or Semantic Scholar. Covers the modern form
# (2301.00234, optionally versioned) and the pre-2007 form (math/0309136,
# cs.CV/0309136).
_ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$"
)

# Upper bound on a buffered PDF. response.content would read an arbitrarily
# large body into memory before any size check could run.
_MAX_PDF_BYTES = 30 * 1024 * 1024
_PDF_CHUNK_BYTES = 64 * 1024
_LOG_PATH = Path("data/failed_extractions.log")

# A browser-like User-Agent — arxiv.org returns 403 for default python-requests UA.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Semantic Scholar's free-tier limit resets on roughly a 60-second cadence, so a
# retry schedule has to approach that before giving up. The previous 2**attempt
# schedule waited 1s then 2s — it abandoned a paper ~3s after the first 429, far
# inside the window, which is how every entry in failed_extractions.log was lost.
_S2_RETRY_WAITS = (15, 45)  # cumulative 60s across the default 3 attempts
_S2_MAX_RETRY_WAIT = 120  # cap, so a hostile Retry-After cannot stall a batch

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

Every item must be a self-contained statement naming a specific problem or a specific
proposed step. Do not return section headings, connective words, or hedging about how
complete this paper is. If an item would not be understandable on its own, omit it.

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
# These deliberately contain NO quoted example phrases. The previous version listed
# cue words verbatim ("look for phrases like 'remains challenging', 'future work
# includes'"), and llama3.1:8b echoed them straight back: the corpus ended up with
# Limitation nodes whose entire text was one of those cues, which then became cluster
# seeds, gap descriptions and cross-domain match sources. Describing the *kind* of
# clause to look for cannot leak the same way. pipeline/extraction_filter.py rejects
# the known echoes as a second line of defence, since a future prompt edit could
# reintroduce the problem.
_TIER_INSTRUCTIONS = {
    ExtractionTier.CONCLUSION.value: (
        "This text is from the conclusion section, so limitations are usually implied "
        "rather than stated outright. Look for clauses where the authors qualify a "
        "result, contrast it with something they did not achieve, or defer work to a "
        "later paper. Report the substance of each such clause as a self-contained "
        "statement — never the connective or hedging wording itself."
    ),
    ExtractionTier.INFERRED.value: (
        "Limitations are not explicitly stated. Infer them from what the paper claims "
        "to solve and what it does not address. Be conservative — only infer clear "
        "limitations, not speculative ones. Each one must name a specific problem."
    ),
}

# No logging.basicConfig here — a library module must not reconfigure the root
# logger for everything that imports it. The app entrypoint owns that.
logger = logging.getLogger(__name__)


class PaperExtract(BaseModel):
    arxiv_id: str
    title: str
    year: int
    # No default, by advisor decision (PLAN.md #1, option 1A). This field used to
    # default to "computer_vision" here *and* be hardcoded again at the
    # construction site in extract_paper(), so a paper's domain recorded which
    # script ingested it rather than what it was about. Six off-topic papers and
    # four medical papers entered the corpus as computer_vision that way, and
    # domain divides frequency_score, filters every Qdrant query and routes
    # cross-domain matching. A caller that does not know the domain must now fail
    # rather than quietly produce CV rows. See pipeline/domains.py.
    domain: str
    objectives: list[str]
    methods: list[str]
    datasets: list[str]
    evaluation_metrics: list[str]
    limitations: list[str]
    future_directions: list[str]
    raw_json: str
    ingested_at: str
    extraction_tier: str = "explicit"


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """True if arxiv_id is safe to interpolate into an outbound URL.

    See _ARXIV_ID_RE. This is a safety check on untrusted input, not a claim that
    the paper exists — verify real ids against arxiv.org or Semantic Scholar.
    """
    return bool(arxiv_id) and bool(_ARXIV_ID_RE.match(arxiv_id))


def rate_limit_wait_seconds(attempt: int, response=None) -> float:
    """Seconds to wait before retrying a Semantic Scholar 429.

    Prefers the server's own ``Retry-After`` header when it sends one, since that
    is authoritative; otherwise falls back to _S2_RETRY_WAITS. Capped at
    _S2_MAX_RETRY_WAIT so a malformed or hostile header cannot stall a batch.
    """
    # Insist on a real mapping holding a real scalar. Anything looser accepts
    # objects that merely happen to be float()-able and silently produces a
    # nonsense wait.
    headers = getattr(response, "headers", None)
    raw = headers.get("Retry-After") if isinstance(headers, Mapping) else None

    retry_after = None
    if isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
        try:
            retry_after = float(raw)
        except ValueError:
            retry_after = None

    if retry_after is not None and retry_after >= 0:
        return min(retry_after, _S2_MAX_RETRY_WAIT)

    index = min(attempt, len(_S2_RETRY_WAITS) - 1)
    return _S2_RETRY_WAITS[index]


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
            wait = rate_limit_wait_seconds(attempt, response)
            logger.warning("Rate limited by Semantic Scholar, retrying in %ss", wait)
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


def _download_pdf(url: str) -> bytes:
    """Download a PDF, refusing to buffer more than _MAX_PDF_BYTES.

    Streamed with a running byte budget rather than reading response.content,
    which would materialise the whole body in memory before any check could run.
    """
    response = requests.get(url, headers=_BROWSER_HEADERS, timeout=30, stream=True)
    response.raise_for_status()

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=_PDF_CHUNK_BYTES):
        if not chunk:
            continue
        total += len(chunk)
        if total > _MAX_PDF_BYTES:
            raise ValueError(
                f"PDF at {url} exceeds the {_MAX_PDF_BYTES}-byte limit; aborting download"
            )
        chunks.append(chunk)
    return b"".join(chunks)


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
            pages = _extract_pdf_text(_download_pdf(url))
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


def log_extraction_failure(
    arxiv_id: str,
    reason: str,
    raw: str | None = None,
    source: str | None = None,
    log_path: Path | None = None,
) -> None:
    """Append a failure record to the extraction failure log.

    The single implementation behind every failure log in the project. It
    previously existed twice with different signatures — one here taking a raw
    payload, one in pipeline.batch without it — which meant the two formats could
    drift and a caller could not tell which it was invoking. `raw` and `source`
    are optional so both original shapes are expressible.

    `log_path` lets a caller supply its own destination. Callers must pass their
    own module-level path rather than relying on the default: tests redirect the
    log by patching the *calling* module's _LOG_PATH, and a delegate that always
    resolved this module's constant would silently write to the real log instead.
    """
    destination = log_path if log_path is not None else _LOG_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    label = f"{source} " if source else ""
    with destination.open("a", encoding="utf-8") as fh:
        fh.write(f"[{timestamp}] {label}arxiv_id={arxiv_id} reason={reason}\n")
        if raw is not None:
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


def extract_paper(arxiv_id: str, domain: str) -> PaperExtract:
    """Fetch paper, run LLM extraction, validate with Pydantic, and return PaperExtract.

    ``domain`` is **required** and validated against pipeline.domains —
    there is deliberately no default (PLAN.md #1, option 1A). It used to be
    hardcoded to "computer_vision" here, and ``ingest_from_query`` never
    overrode it, so the stored domain reflected the ingesting script rather than
    the paper.

    After extraction the declared domain is checked against the paper's own
    title and abstract by a keyword heuristic, and a **warning is logged** when
    they confidently disagree. That check never changes the stored value: the
    heuristic is brittle on this corpus (SAM is a CV paper thick with
    segmentation vocabulary; "Trustworthy Deep Learning for Medical Image
    Segmentation" is a medical paper thick with CV vocabulary), so letting it
    overrule a human declaration would swap a loud failure mode for a quiet one.

    On validation failure, logs to data/failed_extractions.log and re-raises.
    """
    domain = validate_domain(domain)
    paper_meta = fetch_paper_text(arxiv_id)
    # Semantic Scholar supplies metadata (title, year); the body text comes from the
    # PDF's limitations/future-work/conclusion section, falling back to the abstract.
    # The tier records where the text came from and shapes the extraction prompt.
    body_text, tier = fetch_full_text(arxiv_id, abstract=paper_meta["abstract"])
    paper_text = f"Title: {paper_meta['title']}\n\n{body_text}"
    prompt = _build_prompt(paper_text, tier)

    raw_dict = call_ollama(prompt)
    # raw_json preserves the UNFILTERED model output, so the filter is always
    # reversible from what is stored and an audit can see what was discarded.
    raw_json_str = json.dumps(raw_dict)

    # Deterministic quality gates (Phase 1b). Applied before anything reaches
    # SQLite or the graph, because Limitation is UNIQUE on text and a boilerplate
    # node, once created, is shared by every paper that emits the same string.
    limitations = filter_and_log(
        raw_dict.get("limitations", []), "limitations", arxiv_id
    )
    future_directions = filter_and_log(
        raw_dict.get("future_directions", []), "future_directions", arxiv_id
    )

    try:
        result = PaperExtract(
            arxiv_id=arxiv_id,
            title=paper_meta["title"],
            year=paper_meta["year"],
            domain=domain,
            objectives=raw_dict.get("objectives", []),
            methods=raw_dict.get("methods", []),
            datasets=raw_dict.get("datasets", []),
            evaluation_metrics=raw_dict.get("evaluation_metrics", []),
            limitations=limitations,
            future_directions=future_directions,
            raw_json=raw_json_str,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            extraction_tier=tier,
        )
    except ValidationError as exc:
        log_extraction_failure(arxiv_id, str(exc), raw=raw_json_str)
        raise

    # Non-blocking verifier (PLAN.md #1, option 1C). Reports, never overrides.
    warning = verify_declared_domain(
        declared=domain, title=paper_meta["title"], abstract=paper_meta["abstract"]
    )
    if warning:
        logger.warning("Domain check for %s: %s", arxiv_id, warning)

    return result
