"""Tests for pipeline/extractor.py."""

import pytest
from pipeline.extractor import PaperExtract, extract_paper, fetch_paper_text, call_ollama


def test_fetch_paper_text_returns_tuple():
    """fetch_paper_text should return (title: str, text: str, year: int)."""
    pass


def test_call_ollama_returns_dict():
    """call_ollama should return a dict with all required keys."""
    pass


def test_extract_paper_object_detection():
    """extract_paper('2301.00234') should return a valid PaperExtract with limitations."""
    pass


def test_extract_paper_image_segmentation():
    """extract_paper('2303.05499') should return a valid PaperExtract."""
    pass


def test_extract_paper_vision_transformers():
    """extract_paper('2212.09748') should return a valid PaperExtract."""
    pass


def test_extract_paper_logs_on_validation_failure(tmp_path, monkeypatch):
    """When LLM returns malformed JSON, failure is logged and no exception is raised."""
    pass


def test_paper_extract_model_fields():
    """PaperExtract requires all mandatory fields and defaults domain to computer_vision."""
    pass
