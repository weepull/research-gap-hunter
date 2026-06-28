"""Tests for pipeline/extractor.py."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline.extractor import PaperExtract, call_ollama, extract_paper, fetch_paper_text


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

MOCK_PAPER_META = {
    "title": "DINO: Self-Supervised Vision Transformers",
    "year": 2021,
    "abstract": "We present DINO, a self-supervised method for vision.",
}

MOCK_LLM_DICT = {
    "objectives": ["Learn visual representations without labels"],
    "methods": ["Vision Transformer", "self-distillation"],
    "datasets": ["ImageNet"],
    "evaluation_metrics": ["top-1 accuracy", "linear probing"],
    "limitations": ["Requires large compute for pretraining", "Sensitive to augmentation choice"],
    "future_directions": ["Apply to video understanding", "Reduce compute requirements"],
}


def _make_ss_response(data: dict, status: int = 200):
    """Return a mock requests.Response for Semantic Scholar."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = data
    if status != 200:
        mock_resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


def _mock_ollama_client(response_content: str) -> MagicMock:
    """Return a fake ollama.Client whose chat() returns response_content."""
    client = MagicMock()
    client.chat.return_value = {"message": {"content": response_content}}
    mock_ollama_mod = MagicMock()
    mock_ollama_mod.Client.return_value = client
    return mock_ollama_mod


# ---------------------------------------------------------------------------
# fetch_paper_text
# ---------------------------------------------------------------------------


def test_fetch_paper_text_returns_dict():
    """fetch_paper_text should return a dict with title, year, and abstract keys."""
    ss_payload = {
        "title": "Some CV Paper",
        "year": 2023,
        "abstract": "We propose a novel method.",
        "tldr": None,
        "openAccessPdf": None,
    }
    with patch("pipeline.extractor.requests.get", return_value=_make_ss_response(ss_payload)):
        result = fetch_paper_text("2301.00234")

    assert result["title"] == "Some CV Paper"
    assert result["year"] == 2023
    assert "novel method" in result["abstract"]


def test_fetch_paper_text_falls_back_to_tldr():
    """When abstract is empty, fetch_paper_text should use tldr.text as the abstract."""
    ss_payload = {
        "title": "Paper without abstract",
        "year": 2022,
        "abstract": "",
        "tldr": {"text": "Fallback summary from TLDR"},
        "openAccessPdf": None,
    }
    with patch("pipeline.extractor.requests.get", return_value=_make_ss_response(ss_payload)):
        result = fetch_paper_text("2212.09748")

    assert result["abstract"] == "Fallback summary from TLDR"


def test_fetch_paper_text_retries_on_429():
    """fetch_paper_text should retry on HTTP 429 then succeed on the third attempt."""
    rate_limited = _make_ss_response({}, status=429)
    # Don't raise on rate-limited responses — let the retry loop handle them
    rate_limited.raise_for_status.side_effect = None

    success_payload = {
        "title": "Retry Paper",
        "year": 2023,
        "abstract": "Eventually fetched.",
        "tldr": None,
        "openAccessPdf": None,
    }
    success_resp = _make_ss_response(success_payload)

    responses = [rate_limited, rate_limited, success_resp]

    with patch("pipeline.extractor.requests.get", side_effect=responses), \
         patch("pipeline.extractor.time.sleep"):
        result = fetch_paper_text("2303.05499")

    assert result["title"] == "Retry Paper"


def test_fetch_paper_text_raises_after_max_retries():
    """fetch_paper_text should raise after 3 consecutive 429 responses."""
    rate_limited = _make_ss_response({}, status=429)
    rate_limited.raise_for_status.side_effect = Exception("HTTP 429")

    with patch("pipeline.extractor.requests.get", return_value=rate_limited), \
         patch("pipeline.extractor.time.sleep"):
        with pytest.raises(Exception):
            fetch_paper_text("0000.00000")


# ---------------------------------------------------------------------------
# call_ollama — patch sys.modules so the lazy import picks up the mock
# ---------------------------------------------------------------------------


def test_call_ollama_returns_dict():
    """call_ollama should return a dict with all six required keys."""
    mock_ollama = _mock_ollama_client(json.dumps(MOCK_LLM_DICT))
    with patch.dict(sys.modules, {"ollama": mock_ollama}):
        result = call_ollama("some prompt")

    assert set(result.keys()) >= {
        "objectives", "methods", "datasets",
        "evaluation_metrics", "limitations", "future_directions",
    }


def test_call_ollama_raises_on_invalid_json():
    """call_ollama should raise ValueError when Ollama returns non-JSON text."""
    mock_ollama = _mock_ollama_client("Sorry, I cannot do that.")
    with patch.dict(sys.modules, {"ollama": mock_ollama}):
        with pytest.raises(ValueError, match="non-JSON"):
            call_ollama("some prompt")


def test_call_ollama_strips_markdown_fences():
    """call_ollama should strip ```json ... ``` fences before parsing."""
    fenced = f"```json\n{json.dumps(MOCK_LLM_DICT)}\n```"
    mock_ollama = _mock_ollama_client(fenced)
    with patch.dict(sys.modules, {"ollama": mock_ollama}):
        result = call_ollama("some prompt")

    assert "limitations" in result


def test_call_ollama_raises_on_missing_keys():
    """call_ollama should raise ValueError when required keys are absent."""
    incomplete = {"objectives": ["do something"]}  # missing 5 keys
    mock_ollama = _mock_ollama_client(json.dumps(incomplete))
    with patch.dict(sys.modules, {"ollama": mock_ollama}):
        with pytest.raises(ValueError, match="missing keys"):
            call_ollama("some prompt")


# ---------------------------------------------------------------------------
# extract_paper
# ---------------------------------------------------------------------------


def _patch_extract(monkeypatch, llm_dict: dict | None = None, meta: dict | None = None):
    """Monkeypatch fetch_paper_text and call_ollama for extract_paper tests."""
    import pipeline.extractor as mod
    monkeypatch.setattr(mod, "fetch_paper_text", lambda _id: meta or MOCK_PAPER_META)
    monkeypatch.setattr(mod, "call_ollama", lambda _prompt: llm_dict or MOCK_LLM_DICT)


def test_extract_paper_happy_path(monkeypatch):
    """extract_paper should return a valid PaperExtract on success."""
    _patch_extract(monkeypatch)
    result = extract_paper("2301.00234")

    assert isinstance(result, PaperExtract)
    assert result.arxiv_id == "2301.00234"
    assert result.domain == "computer_vision"
    assert len(result.limitations) > 0
    assert result.ingested_at != ""
    assert json.loads(result.raw_json)  # raw_json must be valid JSON


def test_extract_paper_sets_title_and_year(monkeypatch):
    """extract_paper should populate title and year from fetch_paper_text metadata."""
    _patch_extract(monkeypatch)
    result = extract_paper("2303.05499")

    assert result.title == MOCK_PAPER_META["title"]
    assert result.year == MOCK_PAPER_META["year"]


def test_extract_paper_raw_json_contains_limitations(monkeypatch):
    """raw_json field should be deserializable and contain the limitations key."""
    _patch_extract(monkeypatch)
    result = extract_paper("2212.09748")
    parsed = json.loads(result.raw_json)
    assert "limitations" in parsed


def test_extract_paper_logs_on_validation_failure(monkeypatch, tmp_path):
    """On Pydantic ValidationError, extract_paper logs to failed_extractions.log and raises."""
    import pipeline.extractor as mod

    log_file = tmp_path / "failed_extractions.log"
    monkeypatch.setattr(mod, "_LOG_PATH", log_file)
    monkeypatch.setattr(mod, "fetch_paper_text", lambda _id: MOCK_PAPER_META)
    monkeypatch.setattr(mod, "call_ollama", lambda _prompt: MOCK_LLM_DICT)

    # Force PaperExtract to raise by replacing it with a broken subclass inside the module
    original_cls = mod.PaperExtract

    class _AlwaysFails(original_cls):
        def __init__(self, **data):
            # Trigger a real ValidationError by omitting required fields
            super().__init__(
                arxiv_id=data.get("arxiv_id", ""),
                title=data.get("title", ""),
                year=data.get("year", 0),
                objectives=data.get("objectives", []),
                methods=data.get("methods", []),
                datasets=data.get("datasets", []),
                evaluation_metrics=data.get("evaluation_metrics", []),
                limitations=data.get("limitations", []),
                future_directions=data.get("future_directions", []),
                raw_json=data.get("raw_json", ""),
                ingested_at=data.get("ingested_at", ""),
            )
            # Deliberately corrupt the object after construction so the log path is exercised
            raise ValidationError.from_exception_data(  # type: ignore[call-arg]
                title="PaperExtract",
                input_type="python",
                line_errors=[],
            )

    monkeypatch.setattr(mod, "PaperExtract", _AlwaysFails)

    with pytest.raises(ValidationError):
        mod.extract_paper("bad-id")

    assert log_file.exists(), "Failure log should have been written"
    content = log_file.read_text()
    assert "bad-id" in content


# ---------------------------------------------------------------------------
# PaperExtract model
# ---------------------------------------------------------------------------


def test_paper_extract_model_defaults_domain():
    """PaperExtract should default domain to 'computer_vision'."""
    paper = PaperExtract(
        arxiv_id="2301.00234",
        title="Test Paper",
        year=2023,
        objectives=["obj"],
        methods=["method"],
        datasets=["ImageNet"],
        evaluation_metrics=["mAP"],
        limitations=["slow inference"],
        future_directions=["speed up"],
        raw_json="{}",
        ingested_at="2026-06-29T00:00:00+00:00",
    )
    assert paper.domain == "computer_vision"


def test_paper_extract_model_rejects_missing_fields():
    """PaperExtract should raise ValidationError when mandatory fields are absent."""
    with pytest.raises(ValidationError):
        PaperExtract(arxiv_id="x", title="T")  # year, objectives, etc. are missing
