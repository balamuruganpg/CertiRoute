"""Presentation must preserve saved metrics, provenance and failure visibility."""

import json
from pathlib import Path
from statistics import mean

import pytest
from fastapi.testclient import TestClient

from certiroute.dashboard.app import create_app
from certiroute.dashboard.data import load_results

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results/gsm8k-sensitivity/results.json"
SHIFT = ROOT / "results/gsm8k-shift/results.json"


def test_dashboard_aggregates_same_study_and_preserves_regret():
    data = load_results(MAIN, SHIFT)
    raw = json.loads(MAIN.read_text())
    for policy in ("cheap-only", "strong-only"):
        expected = [r for r in raw["baselines"] if r["policy"] == policy]
        actual = next(r for r in data["baselines"] if r["policy"] == policy)
        assert actual["quality"] == mean(r["mean_quality"] for r in expected)
        assert actual["cost_per_1000"] == pytest.approx(
            mean(r["mean_cost_usd"] for r in expected) / raw["study"]["test_size"] * 1000
        )
    selected = next(
        r for r in data["operating_points"] if r["epsilon"] == 0.05 and r["calibration_size"] == 500
    )
    assert selected["regret"] == pytest.approx(0.05008030303030303)
    assert selected["savings"] == pytest.approx(0.21062791824991609)
    assert selected["non_rejections"] == 5
    assert selected["regret"] > selected["epsilon"]  # Never round into a passing claim.
    assert data["provenance"]["examples"] == 1319
    assert len(data["operating_points"]) == 16


def test_dashboard_shift_finding_is_read_from_file():
    data = load_results(MAIN, SHIFT)
    transferred = next(
        r
        for r in data["shift"]
        if r["direction"] == "long-to-short" and r["arm"] == "shifted" and r["epsilon"] == 0.05
    )
    recalibrated = next(
        r
        for r in data["shift"]
        if r["direction"] == "long-to-short"
        and r["arm"] == "target-recalibrated"
        and r["epsilon"] == 0.05
    )
    assert transferred["mean_regret"] == pytest.approx(0.20015)
    assert transferred["non_rejection"] is False
    assert recalibrated["mean_regret"] == pytest.approx(0.04848)


def test_dashboard_rejects_mixed_datasets_and_duplicate_cells(tmp_path):
    shift = json.loads(SHIFT.read_text())
    shift["dataset_manifest"]["dataset_sha256"] = "0" * 64
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(shift))
    with pytest.raises(ValueError, match="same dataset"):
        load_results(MAIN, wrong)
    main = json.loads(MAIN.read_text())
    main["summaries"].append(main["summaries"][0])
    wrong.write_text(json.dumps(main))
    with pytest.raises(ValueError, match="operating points"):
        load_results(wrong, SHIFT)


def test_dashboard_routes_and_missing_results_are_explicit(tmp_path):
    with TestClient(create_app(MAIN, SHIFT)) as client:
        assert client.get("/").status_code == 200
        response = client.get("/api/results")
        assert response.status_code == 200
        assert len(response.json()["baselines"]) == 2
        assert "api_key" not in response.text
        for name in ("style.css", "app.js"):
            assert client.get(f"/assets/{name}").status_code == 200
        assert client.get("/assets/../data.py").status_code == 404
    with TestClient(create_app(tmp_path / "missing.json", SHIFT)) as client:
        response = client.get("/api/results")
        assert response.status_code == 503
        assert "unavailable" in response.json()["error"].lower()
        assert str(tmp_path) not in response.text
