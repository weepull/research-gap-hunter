"""Integration proof for PLAN.md item #5 — /explain grounding.

Fails against pre-fix code: /explain accepts two arbitrary query strings, builds
a CrossDomainMatch with similarity_score=0.0 hardcoded, and the prompt never sees
the score — so llama3.1:8b confabulated a shared structure between analytic
number theory and histopathology federated learning, and echoed the false
computer_vision label back as fact.
"""

import pytest

pytestmark = pytest.mark.integration


def test_ungrounded_pairing_is_refused_before_the_llm_runs(loaded_corpus):
    """A pairing absent from the corpus must be refused deterministically, server-side.

    Deliberately does NOT depend on Ollama: the point of 5A is that the refusal
    happens before any generation, so this test must pass with Ollama stopped.
    """
    from pipeline.cross_domain import UngroundedPairingError, verify_pairing

    with pytest.raises(UngroundedPairingError):
        verify_pairing(
            source_gap="Only applies to divisor-bounded multiplicative functions",
            target_solution="Investigation of federated learning for histopathology",
            source_domain="computer_vision",
            target_domain="medical_imaging",
        )


def test_pairing_absent_from_the_corpus_is_refused(loaded_corpus):
    """Text that appears nowhere in the corpus cannot be a corpus match."""
    from pipeline.cross_domain import UngroundedPairingError, verify_pairing

    with pytest.raises(UngroundedPairingError):
        verify_pairing(
            source_gap="this sentence appears in no paper in the corpus whatsoever",
            target_solution="nor does this one, by construction",
            source_domain="computer_vision",
            target_domain="medical_imaging",
        )


def test_a_real_cross_domain_match_verifies_and_reports_its_similarity(loaded_corpus):
    """A pairing produced by find_cross_domain_matches must verify successfully."""
    from pipeline.cross_domain import find_cross_domain_matches, verify_pairing

    matches = find_cross_domain_matches(top_n=5)
    if not matches:
        pytest.skip("fixture corpus produced no cross-domain matches above the noise floor")

    match = matches[0]
    verified = verify_pairing(
        source_gap=match.source_gap,
        target_solution=match.target_solution,
        source_domain=match.source_domain,
        target_domain=match.target_domain,
    )
    assert verified.similarity_score == pytest.approx(match.similarity_score, abs=1e-3)
    assert verified.similarity_score > 0.0, "the real similarity must be reported, not 0.0"


def test_explain_response_carries_similarity_and_hypothesis_marker(loaded_corpus, ollama_available):
    """A grounded explanation must ship its evidence and an explicit hypothesis marker."""
    from fastapi.testclient import TestClient

    from pipeline.cross_domain import find_cross_domain_matches

    matches = find_cross_domain_matches(top_n=5)
    if not matches:
        pytest.skip("fixture corpus produced no cross-domain matches above the noise floor")
    match = matches[0]

    from api.main import app

    with TestClient(app) as client:
        response = client.get(
            "/explain",
            params={
                "source_gap": match.source_gap,
                "target_solution": match.target_solution,
                "source": match.source_domain,
                "target": match.target_domain,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["explanation"].strip()
    assert body["similarity_score"] == pytest.approx(match.similarity_score, abs=1e-3)
    assert body["is_hypothesis"] is True
    assert body["grounding"] == "corpus_match"


def test_explain_refuses_ungrounded_pairing_over_http(loaded_corpus):
    """The HTTP surface must refuse, not 500 and not 200 with invented prose."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.get(
            "/explain",
            params={
                "source_gap": "Only applies to divisor-bounded multiplicative functions",
                "target_solution": "Investigation of federated learning for histopathology",
            },
        )

    assert response.status_code == 422, response.text
    assert "noise floor" in response.text or "not a corpus match" in response.text
