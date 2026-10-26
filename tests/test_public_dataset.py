"""Public snapshot integrity, scoring, decoding scope and repeatability."""

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from certiroute.config import GatewayConfig, ScopeConfig, load_config
from certiroute.router.scope import in_calibrated_scope, with_generation_defaults
from eval.fixtures import FixtureRow, last_integer, read_fixture
from eval.import_gsm8k import estimate_tokens
from eval.manifest import read_manifest
from eval.offline import run

DATA = Path("data/eval/gsm8k/paired.jsonl")
CONFIG = Path("configs/offline.gsm8k.yaml")
PRICING = Path("configs/pricing.gsm8k.yaml")


def test_public_snapshot_scoring_and_usage_provenance():
    config = load_config(CONFIG)
    manifest = read_manifest(DATA, config)
    rows, dropped = read_fixture(DATA, config)
    assert len(rows) == 100 and not dropped
    assert manifest.generation["generated_at"] is None
    assert "ESTIMATED" in manifest.token_counts
    for row in rows:
        prompt = row.request["messages"][0]["content"]
        assert prompt.count("Question: ") == 9
        assert row.cheap_usage.prompt_tokens == estimate_tokens(prompt)
        assert row.strong_usage.completion_tokens == estimate_tokens(row.strong_output)
        assert row.cheap_usage.completion_tokens == estimate_tokens(row.cheap_output)
        bad = row.model_dump()
        bad["q_cheap"] = 1 - row.q_cheap
        with pytest.raises(ValidationError, match="quality disagrees"):
            FixtureRow.model_validate(bad)
    assert last_integer("Work: 20,000 / 10 = 2,000.") == "2000"
    assert last_integer("No answer") is None
    assert last_integer("Answer: 007") is None  # Upstream literal parser rejects leading zeros.
    # Preserve upstream quirks explicitly; do not call this a decimal/sign-aware verifier.
    assert last_integer("-12.50") == "50"


def test_snapshot_rejects_tampering_and_model_mismatch(tmp_path):
    config = load_config(CONFIG)
    local = tmp_path / "paired.jsonl"
    shutil.copyfile(DATA, local)
    shutil.copyfile(DATA.with_suffix(".manifest.json"), local.with_suffix(".manifest.json"))
    local.write_bytes(local.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(local, config)
    fields = config.model_dump(mode="json")
    fields["models"]["cheap"]["model"] = "different-model"
    with pytest.raises(ValueError, match="model identities"):
        read_manifest(DATA, GatewayConfig.model_validate(fields))


def test_fixed_stop_scope_and_old_artifact_serialization():
    scope = load_config(CONFIG).scope
    request = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    normalized = with_generation_defaults(request, scope)
    assert normalized["stop"] == ["Question"]
    assert in_calibrated_scope(normalized, scope)
    assert not in_calibrated_scope({**normalized, "stop": ["Other"]}, scope)
    assert not in_calibrated_scope({**normalized, "stop": None}, scope)
    old_scope = ScopeConfig()
    assert "stop" not in old_scope.model_dump(mode="json")
    assert "stop" not in json.loads(old_scope.model_dump_json())
    assert not in_calibrated_scope(
        {**with_generation_defaults(request, old_scope), "stop": ["Question"]}, old_scope
    )


def test_repeat_run_has_identical_weights_thresholds_and_metrics(tmp_path):
    first = run(DATA, CONFIG, PRICING, tmp_path / "first")
    second = run(DATA, CONFIG, PRICING, tmp_path / "second")
    for field in ("split_manifest_sha256", "router_sha256", "lambda_by_epsilon", "summaries"):
        assert first[field] == second[field]
    assert first["sizes"] == {"train": 40, "calibration": 40, "test": 20}
