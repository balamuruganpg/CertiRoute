"""Question-only leakage guards, G1 statistics and offline end-to-end agreement."""

import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

import httpx
import pytest

from certiroute.config import GatewayConfig, RouterConfig, load_config
from certiroute.provenance import overlaps, prompt_signature
from certiroute.router.predictor import load_predictor, save_weights
from eval.fit_router import fit_router
from eval.fixtures import read_fixture
from eval.manifest import read_manifest
from eval.repeated import (
    ScoredRow,
    StudyConfig,
    draw_split,
    evaluate_draw,
    mean_interval,
    policy_metrics,
    run,
)


def test_question_signatures_remove_only_bound_prefix():
    with pytest.raises(ValueError, match="hash-bound"):
        RouterConfig(kind="length_v1", signature_prefix="unbound")
    config = load_config(Path("data/eval/gsm8k-full/gateway-template.yaml"))
    prefix = config.router.signature_prefix

    def request(question):
        return {"messages": [{"role": "user", "content": prefix + question}]}

    left, right = request("Question: How many apples?"), request("Question: What is Bob's age?")
    assert overlaps(prompt_signature(left), prompt_signature(right))
    assert not overlaps(prompt_signature(left, prefix), prompt_signature(right, prefix))
    assert prompt_signature(left, prefix) == prompt_signature(
        request("Question:  How   many apples?"), prefix
    )
    with pytest.raises(ValueError, match="bound signature prefix"):
        prompt_signature({"messages": [{"role": "user", "content": "Missing prefix"}]}, prefix)


def test_full_snapshot_and_signature_binding(tmp_path):
    config = load_config(Path("data/eval/gsm8k-full/gateway-template.yaml"))
    path = Path("data/eval/gsm8k-full/paired.jsonl")
    manifest = read_manifest(path, config)
    rows, dropped = read_fixture(path, config)
    assert len(rows) == 1319 and not dropped and not manifest.excluded_source_indices
    fields = config.model_dump(mode="json")
    fields["router"]["artifact"] = str(tmp_path / "router.json")
    config = GatewayConfig.model_validate(fields)
    weights = fit_router([r.calibration_row() for r in rows[:40]], config, "a" * 64)
    save_weights(weights, config.router.artifact)
    predictor = load_predictor(config)
    with pytest.raises(ValueError, match="overlaps"):
        predictor.check_independent(rows[0].request)
    fields["router"]["signature_prefix"] = ""
    mismatched = GatewayConfig.model_validate(fields)
    with pytest.raises(ValueError, match="mismatch"):
        load_predictor(mismatched)
    with pytest.raises(ValueError, match="do not match"):
        read_manifest(path, mismatched)


def test_partitions_are_reproducible_disjoint_and_complete():
    for repeat in range(20):
        calibration, test = draw_split(60, 40, 42, repeat)
        assert (calibration, test) == draw_split(60, 40, 42, repeat)
        assert len(calibration) == 40 and len(test) == 20
        assert set(calibration).isdisjoint(test)
        assert set(calibration + test) == set(range(60))
    assert draw_split(60, 40, 42, 0) != draw_split(60, 40, 42, 1)
    with pytest.raises(ValueError):
        draw_split(60, 60, 42, 0)


def test_thresholds_do_not_read_test_outcomes_and_random_mix_is_exact():
    calibration = [ScoredRow(str(i), i / 30, 1 - i / 30, float(i % 2), 1, 1, 10) for i in range(30)]
    test = [ScoredRow("a", 0.9, 0.2, 0, 1, 1, 10), ScoredRow("b", 0.4, 0.8, 1, 1, 2, 20)]
    study = StudyConfig(repeats=2)
    before = evaluate_draw(calibration, test, [0.1, 0.2], study)
    after = evaluate_draw(
        calibration, [replace(r, q_cheap=1 - r.q_cheap) for r in test], [0.1, 0.2], study
    )
    assert [r.get("threshold") for r in before] == [r.get("threshold") for r in after]
    mixture = policy_metrics(test, [0.25, 0.25])
    assert mixture["regret"] == 0.125
    assert mixture["quality"] == 0.875
    assert mixture["cost_usd"] == 23.25
    assert mixture["savings_vs_strong"] == pytest.approx(0.225)


def test_confidence_intervals_are_over_draws_and_bonferroni_widens_them():
    values = [0.01, 0.03, 0.02, 0.04] * 250
    ordinary = mean_interval(values)
    adjusted = mean_interval(values, comparisons=4)
    assert adjusted[0] < ordinary[0] < mean(values) < ordinary[1] < adjusted[1]
    assert mean(ordinary) == pytest.approx(mean(values))
    assert mean_interval([0.0] * 1000, bounds=(0, 1)) == [0, 0]


def test_repeated_study_offline_gateway_agreement_and_repeatability(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External networking is forbidden.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    arguments = (
        Path("data/eval/gsm8k/paired.jsonl"),
        Path("configs/offline.gsm8k.yaml"),
        Path("configs/pricing.gsm8k.yaml"),
    )
    study = StudyConfig(repeats=8)
    first = run(*arguments, tmp_path / "first", study)
    second = run(*arguments, tmp_path / "second", study)
    assert first["summaries"] == second["summaries"]
    assert first["paired_comparisons"] == second["paired_comparisons"]
    assert first["hashes"] == second["hashes"]
    assert first["sizes"] == {"train": 40, "pool": 60, "calibration": 40, "test": 20}
    splits = json.loads((tmp_path / "first/split-manifest.json").read_text())
    assert set(splits["train_ids"]).isdisjoint(splits["pool_ids"])
    assert len(first["summaries"]) == 16
    assert "agree" in first["gateway_check"]
    for row in first["summaries"]:
        assert row["repeats"] == 8
        if row["epsilon"] is not None:
            assert row["non_rejection"] == (row["regret_ci_bonferroni"][0] <= row["epsilon"])
