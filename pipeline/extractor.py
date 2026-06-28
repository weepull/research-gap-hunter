"""Extracts structured information from arXiv papers via Ollama llama3.1:8b."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

_SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
_LOG_PATH = Path("data/failed_extractions.log")
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


def extract_paper(arxiv_id: str) -> PaperExtract:
    """Fetch paper, run LLM extraction, validate with Pydantic, and return PaperExtract.

    On validation failure, logs to data/failed_extractions.log and re-raises.
    """
    paper_meta = fetch_paper_text(arxiv_id)
    paper_text = f"Title: {paper_meta['title']}\n\nAbstract:\n{paper_meta['abstract']}"
    prompt = _EXTRACTION_PROMPT.format(paper_text=paper_text)

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
        )
    except ValidationError as exc:
        _log_failure(arxiv_id, str(exc), raw_json_str)
        raise

    return result
