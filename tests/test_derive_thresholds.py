"""Unit tests for scripts/derive_thresholds.py's handling of the deficit floors (S1).

The floors are not measured from a null directly. Each is a function of two other
coded constants — the solution threshold (null p95) and the deficit rescale anchors
(null p50/p99) — so these tests pin how the script derives, checks and rewrites them.
derive_all() is replaced with fixed numbers: no live store is touched.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import derive_thresholds as dt  # noqa: E402

_DOM = "computer_vision"


def _derived(threshold=0.8733, anchors=(0.8235, 0.8959)):
    return {
        "cluster": {_DOM: 0.8744}, "solution": {_DOM: threshold},
        "cross_domain": 0.8764, "deficit_anchors": {_DOM: anchors}, "raw": {},
    }


def _coded(floor, threshold=0.8733, anchors=(0.8235, 0.8959)):
    return {
        "cluster": {_DOM: 0.8744}, "solution": {_DOM: threshold},
        "cross_domain": 0.8764, "deficit_anchors": {_DOM: anchors},
        "deficit_floor": {_DOM: floor},
    }


def _floor_row(rows):
    (row,) = [r for r in rows if r["kind"] == "deficit_floor"]
    return row


def test_the_floor_formula():
    assert dt.deficit_floor(0.8733, 0.8235, 0.8959) == pytest.approx(0.3122, abs=1e-4)
    assert dt.deficit_floor(0.8915, 0.8385, 0.9127) == pytest.approx(0.2857, abs=1e-4)


def test_a_floor_in_step_with_its_inputs_is_kept(monkeypatch):
    monkeypatch.setattr(dt, "derive_all", _derived)
    monkeypatch.setattr(dt, "coded_constants",
                        lambda: _coded(dt.deficit_floor(0.8733, 0.8235, 0.8959)))
    row = _floor_row(dt.drift_report())
    assert row["name"] == '_UNRESOLVED_DEFICIT_FLOORS["computer_vision"]'
    assert not row["stale"]


def test_a_floor_out_of_step_with_its_inputs_is_stale(monkeypatch):
    """The old single 0.3 is 0.0122 away from what its inputs imply."""
    monkeypatch.setattr(dt, "derive_all", _derived)
    monkeypatch.setattr(dt, "coded_constants", lambda: _coded(0.3))
    row = _floor_row(dt.drift_report())
    assert row["stale"] and row["derived"] == pytest.approx(0.3122, abs=1e-4)


def test_a_rounded_floor_is_stale(monkeypatch):
    """T1: floors are stored at full precision, so a 4-dp literal is out of step.

    No sampling noise of its own: any difference beyond float precision is
    inconsistency. Fails against S1's FLOOR_TOLERANCE of 0.0001, which accepted the
    rounded 0.2857 that sat below MI's exact 0.285714...
    """
    monkeypatch.setattr(dt, "derive_all", lambda: _derived(threshold=0.8915, anchors=(0.8385, 0.9127)))
    monkeypatch.setattr(dt, "coded_constants",
                        lambda: _coded(0.2857, threshold=0.8915, anchors=(0.8385, 0.9127)))
    row = _floor_row(dt.drift_report())
    assert row["stale"]
    assert dt.FLOOR_TOLERANCE <= 1e-9


def test_the_floor_follows_inputs_that_this_run_will_rewrite(monkeypatch):
    """If the solution threshold is about to be rewritten, the floor must be derived
    from the NEW threshold — otherwise it lags its own inputs by one --apply."""
    monkeypatch.setattr(dt, "derive_all", lambda: _derived(threshold=0.8800))  # stale
    monkeypatch.setattr(dt, "coded_constants", lambda: _coded(0.3122, threshold=0.8733))
    rows = dt.drift_report()
    assert [r for r in rows if r["kind"] == "solution"][0]["stale"]
    assert _floor_row(rows)["derived"] == pytest.approx(
        dt.deficit_floor(0.8800, 0.8235, 0.8959), abs=1e-9)


def test_the_floor_ignores_input_drift_that_will_not_be_written(monkeypatch):
    """Inputs inside DRIFT_TOLERANCE stay coded, so the floor derives from the coded ones."""
    monkeypatch.setattr(dt, "derive_all", lambda: _derived(threshold=0.8738))  # 0.0005
    exact = dt.deficit_floor(0.8733, 0.8235, 0.8959)  # full precision, as stored since T1
    monkeypatch.setattr(dt, "coded_constants", lambda: _coded(exact, threshold=0.8733))
    assert not _floor_row(dt.drift_report())["stale"]


def test_the_locator_rewrites_one_floor_with_its_provenance(tmp_path):
    src = tmp_path / "cross_domain.py"
    src.write_text(
        "# _UNRESOLVED_DEFICIT_FLOORS is explained here first\n"
        "_UNRESOLVED_DEFICIT_FLOORS: dict[str, float] = {\n"
        '    "computer_vision": 0.3000,\n'
        '    "medical_imaging": 0.3000,\n'
        "}\n"
    )
    ok = dt._replace_dict_entry(str(src), "_UNRESOLVED_DEFICIT_FLOORS", "medical_imaging",
                                0.3, 0.2857, 0, comment="from the solution threshold")
    text = src.read_text()
    assert ok
    assert '"medical_imaging": 0.2857,  # from the solution threshold' in text
    assert '"computer_vision": 0.3000,' in text


def test_floors_are_written_and_rewritten_at_full_precision(tmp_path):
    """T1: a floor is written as repr(float), and a full-precision literal can be found
    and rewritten again on the next --apply (the 4-dp matcher could not find it)."""
    src = tmp_path / "cross_domain.py"
    src.write_text(
        "_UNRESOLVED_DEFICIT_FLOORS: dict[str, float] = {\n"
        '    "computer_vision": 0.3122,\n'
        '    "medical_imaging": 0.2857,\n'
        "}\n"
    )
    exact = dt.deficit_floor(0.8915, 0.8385, 0.9127)
    assert dt._replace_floor_entry(str(src), "medical_imaging", 0.2857, exact, "c1")
    assert f'"medical_imaging": {exact!r},  # c1' in src.read_text()
    again = exact + 1e-6
    assert dt._replace_floor_entry(str(src), "medical_imaging", exact, again, "c2")
    text = src.read_text()
    assert f'"medical_imaging": {again!r},  # c2' in text
    assert '"computer_vision": 0.3122,' in text
