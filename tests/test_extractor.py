"""Tests for pipeline/extractor.py."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pipeline.extractor import (
    PaperExtract,
    _extract_section,
    call_ollama,
    extract_paper,
    fetch_full_text,
    fetch_paper_text,
)


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
# _extract_section — pure section-selection logic
# ---------------------------------------------------------------------------


def test_extract_section_prefers_limitation():
    """When multiple headers exist, the limitations section is preferred."""
    text = "Discussion of stuff. Future Work is broad. Limitations: it is slow."
    result = _extract_section(text)
    assert result.lower().startswith("limitation")


def test_extract_section_finds_future_work():
    """Falls through to 'future work' when no limitations header is present."""
    text = "Methods are described here. Future Work: extend the model to video."
    result = _extract_section(text)
    assert result.lower().startswith("future work")


def test_extract_section_is_case_insensitive():
    """Headers are matched regardless of case (e.g. all-caps CONCLUSION)."""
    text = "intro paragraph CONCLUSION here is the closing summary"
    result = _extract_section(text)
    assert "summary" in result.lower()


def test_extract_section_caps_at_2000_chars():
    """The returned section never exceeds 2000 characters."""
    text = "Limitations " + ("a" * 5000)
    result = _extract_section(text)
    assert len(result) <= 2000


def test_extract_section_returns_empty_without_header():
    """No recognised header (or empty input) yields an empty string."""
    assert _extract_section("Just methods and results, nothing relevant here.") == ""
    assert _extract_section("") == ""


# ---------------------------------------------------------------------------
# fetch_full_text — PDF download + section extraction + fallback
# ---------------------------------------------------------------------------


def _make_pdf_response() -> MagicMock:
    """A mock requests.Response for a successful PDF download."""
    resp = MagicMock()
    resp.content = b"%PDF-1.5 fake bytes"
    resp.raise_for_status.return_value = None
    return resp


def test_fetch_full_text_returns_section(monkeypatch):
    """A successful fetch returns the extracted relevant section, not the abstract."""
    import pipeline.extractor as mod

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _make_pdf_response())
    monkeypatch.setattr(
        mod, "_extract_pdf_text", lambda _b: "Intro. Limitations: the model is slow."
    )

    result = fetch_full_text("2301.00234", abstract="ABSTRACT")
    assert "model is slow" in result.lower()
    assert result != "ABSTRACT"


def test_fetch_full_text_uses_browser_user_agent(monkeypatch):
    """The PDF request must carry a browser-like User-Agent and the arXiv PDF URL."""
    import pipeline.extractor as mod

    captured: dict = {}

    def fake_get(url, headers=None, timeout=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        return _make_pdf_response()

    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod, "_extract_pdf_text", lambda _b: "Limitations: something")

    fetch_full_text("2301.00234", abstract="ABS")

    assert "2301.00234" in captured["url"]
    assert "User-Agent" in captured["headers"]
    assert "Mozilla" in captured["headers"]["User-Agent"]


def test_fetch_full_text_falls_back_to_abstract_on_fetch_failure(monkeypatch):
    """If every download attempt raises, the abstract is returned."""
    import pipeline.extractor as mod

    monkeypatch.setattr(mod.requests, "get", MagicMock(side_effect=Exception("403 Forbidden")))
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    result = fetch_full_text("2301.00234", abstract="FALLBACK ABSTRACT")
    assert result == "FALLBACK ABSTRACT"


def test_fetch_full_text_falls_back_when_no_section(monkeypatch):
    """If the PDF has no recognised section, the abstract is returned."""
    import pipeline.extractor as mod

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _make_pdf_response())
    monkeypatch.setattr(mod, "_extract_pdf_text", lambda _b: "Only intro and methods text.")

    result = fetch_full_text("2301.00234", abstract="ABS FALLBACK")
    assert result == "ABS FALLBACK"


def test_fetch_full_text_retries_then_succeeds(monkeypatch):
    """A transient failure is retried with backoff before a successful fetch."""
    import pipeline.extractor as mod

    attempts = [Exception("temporary"), _make_pdf_response()]

    def fake_get(*a, **k):
        outcome = attempts.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(mod.requests, "get", fake_get)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mod, "_extract_pdf_text", lambda _b: "Conclusion: it works well.")

    result = fetch_full_text("2301.00234", abstract="ABS")
    assert "it works well" in result.lower()
    assert attempts == []  # both queued outcomes consumed


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
    """Monkeypatch fetch_paper_text, fetch_full_text, and call_ollama for extract_paper tests."""
    import pipeline.extractor as mod
    resolved_meta = meta or MOCK_PAPER_META
    monkeypatch.setattr(mod, "fetch_paper_text", lambda _id: resolved_meta)
    monkeypatch.setattr(
        mod, "fetch_full_text", lambda _id, abstract="": resolved_meta["abstract"]
    )
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


def test_extract_paper_uses_full_text_in_prompt(monkeypatch):
    """extract_paper should feed fetch_full_text output (not just abstract) into the LLM."""
    import pipeline.extractor as mod

    monkeypatch.setattr(mod, "fetch_paper_text", lambda _id: MOCK_PAPER_META)
    monkeypatch.setattr(
        mod, "fetch_full_text", lambda _id, abstract="": "Limitations: GPU memory bound."
    )

    captured = {}

    def fake_call_ollama(prompt):
        captured["prompt"] = prompt
        return MOCK_LLM_DICT

    monkeypatch.setattr(mod, "call_ollama", fake_call_ollama)

    extract_paper("2301.00234")

    assert "GPU memory bound" in captured["prompt"]


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
    monkeypatch.setattr(mod, "fetch_full_text", lambda _id, abstract="": abstract)
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
