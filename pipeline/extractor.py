"""Extracts structured information from arXiv papers via Ollama llama3.1:8b."""

from pydantic import BaseModel


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


def fetch_paper_text(arxiv_id: str) -> tuple[str, str, int]:
    """Fetch paper abstract and metadata from Semantic Scholar.

    Returns (title, abstract_text, year).
    """
    pass


def call_ollama(paper_text: str) -> dict:
    """Send paper text to Ollama llama3.1:8b and return parsed JSON dict."""
    pass


def extract_paper(arxiv_id: str) -> PaperExtract:
    """Fetch paper, run LLM extraction, validate, and return PaperExtract.

    Logs failed validations to data/failed_extractions.log without raising.
    """
    pass
