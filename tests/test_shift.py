"""Leakage, shift failure/recalibration and reproducible offline experiment checks."""

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from eval.repeated import ScoredRow
from eval.shift import ShiftConfig, evaluate_transfer, partition_domains, question_length, run


def test_question_length_excludes_shared_demonstrations():
    prefix = "Question: demonstration\nAnswer: 42\n\n"
    request = {"messages": [{"role": "user", "content": prefix + "Question: How many?\nAnswer:"}]}
    assert question_length(request, prefix) == len("How many?")
    with pytest.raises(ValueError, match="prefix"):
        question_length(request, "wrong prefix")


def test_shift_partitions_are_disjoint_repeatable_and_label_free():
    domains = {"short": list(range(30)), "long": list(range(30, 60))}
    study = ShiftConfig(repeats=3, calibration_size=10, test_size=10)
    first = partition_domains(domains, study, 0)
    assert first == partition_domains(domains, study, 0)
    assert first != partition_domains(domains, study, 1)
    all_indices = []
    for partition in first.values():
        assert len(partition["calibration"]) == len(partition["test"]) == 10
        all_indices.extend(partition["calibration"] + partition["test"])
    assert len(all_indices) == len(set(all_indices))
    with pytest.raises(ValueError, match="enough"):
        partition_domains(domains, ShiftConfig(calibration_size=20, test_size=20), 0)


def test_shift_can_break_crc_and_target_recalibration_can_restore_it():
    source = [ScoredRow(f"s{i}", 0.9, 0, 1, 1, 1, 10) for i in range(100)]
    target = [ScoredRow(f"t{i}", 0.9, 0, 0, 1, 1, 10) for i in range(100)]
    target_test = [replace(r, id="test" + r.id) for r in target]
    source_test = [replace(r, id="test" + r.id) for r in source]
    rows = evaluate_transfer(source, source_test, target, target_test, [0.05])
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["source-control"]["regret"] == 0
    assert by_arm["shifted"]["regret"] == 1
    assert by_arm["target-recalibrated"]["regret"] == 0
    changed = evaluate_transfer(
        source,
        source_test,
        target,
        [replace(r, q_cheap=1) for r in target_test],
        [0.05],
    )
    assert [r["threshold"] for r in rows] == [r["threshold"] for r in changed]


def test_shift_run_is_offline_reproducible_and_records_every_arm(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External networking is forbidden")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    args = (
        Path("data/eval/gsm8k/paired.jsonl"),
        Path("configs/offline.gsm8k.yaml"),
        Path("configs/pricing.gsm8k.yaml"),
    )
    study = ShiftConfig(repeats=3, calibration_size=5, test_size=5)
    first = run(*args, tmp_path / "a", study)
    second = run(*args, tmp_path / "b", study)
    assert first["summaries"] == second["summaries"]
    assert first["hashes"] == second["hashes"]
    assert len(first["summaries"]) == first["risk_ci_family_size"] == 24
    splits = json.loads((tmp_path / "a/split-manifest.json").read_text())
    assert set(splits["train_ids"]).isdisjoint(splits["pool_ids"])
    assert set(splits["domains"]["short"]).isdisjoint(splits["domains"]["long"])
    for summary in first["summaries"]:
        assert summary["mean_within_target"] == (summary["mean_regret"] <= summary["epsilon"])
        assert summary["non_rejection"] == (
            summary["regret_ci_bonferroni"][0] <= summary["epsilon"]
        )
