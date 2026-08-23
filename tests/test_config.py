"""Tests for pipeline/config.py — the DEMO_MODE deployment switch."""

import pytest

from pipeline.config import allowed_origins, is_demo_mode


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Both variables unset unless a test sets them."""
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)


def test_demo_mode_defaults_off():
    """Local development is the default — an unset var must not enable demo mode."""
    assert is_demo_mode() is False


@pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "on", " true "])
def test_demo_mode_truthy_values(monkeypatch, value):
    monkeypatch.setenv("DEMO_MODE", value)
    assert is_demo_mode() is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "anything-else"])
def test_demo_mode_falsey_values(monkeypatch, value):
    """Anything not explicitly truthy leaves the API fully functional.

    Failing open is correct here: the risk of a typo silently disabling a
    developer's local /ingest outweighs the risk of it silently enabling one,
    since a real deployment sets the variable deliberately.
    """
    monkeypatch.setenv("DEMO_MODE", value)
    assert is_demo_mode() is False


def test_allowed_origins_open_in_local_dev():
    """A dev frontend runs on whatever port is free — do not make them configure it."""
    assert allowed_origins() == ["*"]


def test_allowed_origins_ignores_config_when_not_in_demo_mode(monkeypatch):
    """ALLOWED_ORIGINS is inert locally, so a leftover value cannot break dev."""
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com")
    assert allowed_origins() == ["*"]


def test_allowed_origins_restricted_in_demo_mode(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv(
        "ALLOWED_ORIGINS", "https://rgh.vercel.app, https://www.rgh.vercel.app"
    )

    assert allowed_origins() == ["https://rgh.vercel.app", "https://www.rgh.vercel.app"]


def test_allowed_origins_never_wildcards_in_demo_mode(monkeypatch):
    """The whole point of the demo restriction — "*" must be impossible here."""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://rgh.vercel.app")

    assert "*" not in allowed_origins()


def test_allowed_origins_closed_when_demo_mode_unconfigured(monkeypatch):
    """Misconfiguration fails closed, not open.

    A deploy that turns on demo mode but forgets ALLOWED_ORIGINS should break
    visibly in the browser rather than quietly serving every origin.
    """
    monkeypatch.setenv("DEMO_MODE", "true")

    assert allowed_origins() == []


def test_allowed_origins_tolerates_messy_config(monkeypatch):
    """Trailing commas and stray whitespace are config typos, not empty origins."""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("ALLOWED_ORIGINS", " https://a.example , ,https://b.example, ")

    assert allowed_origins() == ["https://a.example", "https://b.example"]
