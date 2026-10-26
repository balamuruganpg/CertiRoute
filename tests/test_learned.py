"""Learned-score, split-isolation and entirely offline gateway replay checks."""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from certiroute.calibration.artifact import DataProvenance, build_artifact, save_artifact
from certiroute.config import GatewayConfig, Settings, load_config
from certiroute.gateway.app import create_app
from certiroute.router.features import extract_features
from certiroute.router.predictor import (
    LearnedPredictor,
    LearnedWeights,
    load_predictor,
    save_weights,
)
from eval.fit_router import fit_router
from eval.fixtures import FixtureRow, read_fixture
from eval.offline import dataset_for, run, split_rows


@pytest.fixture
def learned(tmp_path):
    original = load_config(Path("configs/offline.yaml"))
    fields = original.model_dump(mode="json")
    fields["router"]["artifact"] = str(tmp_path / "router.json")
    fields["calibration"]["artifact"] = str(tmp_path / "calibration.json")
    config = GatewayConfig.model_validate(fields)
    config_path = tmp_path / "gateway.yaml"
    config_path.write_text(yaml.safe_dump(fields), encoding="utf-8")
    rows, dropped = read_fixture(Path("data/eval/sample.jsonl"), config)
    assert not dropped
    splits = split_rows(rows, 42)
    train = [row.calibration_row() for row in splits["train"]]
    weights = fit_router(train, config, "a" * 64)
    save_weights(weights, config.router.artifact)
    dataset = dataset_for(splits["calibration"], config)
    artifact = build_artifact(
        dataset.rows,
        config,
        DataProvenance(
            sources=dataset.sources,
            sha256="b" * 64,
            collected_at=dataset.collected_at,
        ),
    )
    save_artifact(artifact, config.calibration.artifact)
    return config, config_path, splits, weights, artifact


def test_predictions_are_learned_from_training_labels(learned):
    config, _, splits, weights, _ = learned
    train = [r.calibration_row() for r in splits["train"]]
    assert fit_router(train, config, "a" * 64).sha256 == weights.sha256
    # Flip target labels while holding features fixed. A hardcoded score cannot pass.
    flipped = [
        row.model_copy(update={"q_strong": 1.0, "q_cheap": float(row.q_cheap < row.q_strong)})
        for row in train
    ]
    opposite = LearnedPredictor(fit_router(flipped, config, "c" * 64))
    predictor = load_predictor(config)
    sample = extract_features(splits["test"][0].request)
    assert opposite.score(sample) == pytest.approx(1 - predictor.score(sample), abs=1e-10)
    assert any(abs(weight) > 0.01 for weight in weights.coefficients)


def test_roundtrip_and_weights_tampering(learned):
    config, _, splits, weights, _ = learned
    restored = LearnedWeights.model_validate_json(weights.model_dump_json())
    sample = extract_features(splits["test"][0].request)
    assert LearnedPredictor(restored).score(sample) == load_predictor(config).score(sample)
    fields = json.loads(weights.model_dump_json())
    fields["coefficients"][0] += 1
    with pytest.raises(ValidationError, match="hash mismatch"):
        LearnedWeights.model_validate(fields)


def test_calibration_rejects_training_overlap_including_whitespace(learned):
    config, _, splits, _, artifact = learned
    row = splits["train"][0].calibration_row()
    request = json.loads(json.dumps(row.request))
    request["messages"][0]["content"] = request["messages"][0]["content"].replace(" ", "   ")
    for candidate in (row, row.model_copy(update={"request": request})):
        with pytest.raises(ValueError, match="overlaps router training"):
            build_artifact([candidate], config, artifact.data)


def test_router_update_invalidates_old_calibration(learned):
    config, config_path, splits, _, _ = learned
    replacement = fit_router(
        [r.calibration_row() for r in splits["train"]], config, "d" * 64, l2=0.1
    )
    # Simulate deploying new weights under the same path, keeping the old calibration.
    config.router.artifact.write_text(replacement.model_dump_json(), encoding="utf-8")

    def upstream(request):
        assert json.loads(request.content)["model"] == "fixture-strong-v1"
        return httpx.Response(200, json={"choices": [{"message": {"content": "strong"}}]})

    settings = Settings(config_file=config_path, api_key=SecretStr("test-only"))
    with TestClient(create_app(settings, transport=httpx.MockTransport(upstream))) as client:
        assert client.get("/readyz").status_code == 503
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-only"},
            json=splits["test"][0].request,
        )
        assert response.status_code == 200
        assert response.headers["x-certiroute-tier"] == "strong"


def test_fixture_quality_is_derived_from_recorded_outputs():
    value = json.loads(Path("data/eval/sample.jsonl").read_text().splitlines()[0])
    value["q_cheap"] = 1 - value["q_cheap"]
    with pytest.raises(ValidationError, match="quality disagrees"):
        FixtureRow.model_validate(value)


@pytest.mark.parametrize("public", [False, True], ids=["synthetic", "gsm8k"])
def test_offline_pipeline_and_calibration_cli_without_network(tmp_path, monkeypatch, public):
    async def forbidden(*args, **kwargs):
        raise AssertionError("A real HTTP transport must never be used for fixture evaluation.")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    out = tmp_path / "run"
    result = run(
        Path("data/eval/gsm8k/paired.jsonl" if public else "data/eval/sample.jsonl"),
        Path("configs/offline.gsm8k.yaml" if public else "configs/offline.yaml"),
        Path("configs/pricing.gsm8k.yaml" if public else "configs/pricing.fixture.yaml"),
        out,
    )
    size = 20 if public else 48
    assert result["sizes"] == {"train": size * 2, "calibration": size * 2, "test": size}
    manifest = json.loads((out / "split-manifest.json").read_text())
    sets = [set(ids) for ids in manifest["splits"].values()]
    assert sum(len(ids) for ids in sets) == len(set.union(*sets))
    assert len(result["observations"]) == size * 6
    assert result["lambda_by_epsilon"]["0.01"] == 1.0
    strong = next(s for s in result["summaries"] if s["policy"] == "strong-only")
    for summary in result["summaries"]:
        observations = [
            x
            for x in result["observations"]
            if x["policy"] == summary["policy"] and x["epsilon"] == summary["epsilon"]
        ]
        assert summary["cost_usd"] == pytest.approx(sum(x["cost_usd"] for x in observations))
        assert summary["savings_vs_strong"] == pytest.approx(
            1 - summary["cost_usd"] / strong["cost_usd"]
        )
    command = [
        sys.executable,
        "-m",
        "eval.calibrate",
        "--config",
        str(out / "gateway.yaml"),
        "--input",
        str(out / "calibration.json"),
        "--out",
        str(out / "standalone.json"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    standalone = json.loads((out / "standalone.json").read_text())
    assert standalone["lambda_by_epsilon"] == result["lambda_by_epsilon"]
    assert standalone["router"]["sha256"] == result["router_sha256"]
