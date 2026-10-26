"""Calibration controls, cached metrics, nested draws and seeded sweep reproducibility."""

import gzip
import itertools
import json
import random
from pathlib import Path
from statistics import mean

import httpx
import pytest

from certiroute.calibration.crc import crc_lambda
from eval.repeated import ScoredRow, policy_metrics
from eval.sensitivity import (
    SensitivityConfig,
    empirical_lambda,
    evaluate_cell,
    nested_partition,
    run,
)
from eval.sensitivity import (
    TestCurve as Curve,
)


def test_empirical_control_preserves_ties_and_removes_only_correction():
    scores, losses = [0.1, 0.5, 0.5, 0.9], [0, 1, 0, 1]
    assert empirical_lambda(scores, losses, 0.25) == 0.5
    assert crc_lambda(scores, losses, 0.25) == 0.9
    rng = random.Random(64)
    for _ in range(100):
        scores = [rng.choice([0, 0.3, 0.8, 1]) for _ in range(20)]
        losses = [rng.random() for _ in scores]
        for epsilon in (0.01, 0.05, 0.1, 0.5):
            threshold = empirical_lambda(scores, losses, epsilon)
            assert threshold <= crc_lambda(scores, losses, epsilon)
            assert (
                mean(loss * (score > threshold) for score, loss in zip(scores, losses, strict=True))
                <= epsilon
            )
            for candidate in {0.0, 1.0, *scores}:
                if candidate < threshold:
                    assert (
                        mean(
                            loss * (score > candidate)
                            for score, loss in zip(scores, losses, strict=True)
                        )
                        > epsilon
                    )


def test_small_exhaustive_negative_control_exhibits_optimistic_risk():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    losses = [1, 0, 1, 0, 1, 0]
    observed = {"CRC": [], "no-CRC": []}
    for calibration in itertools.combinations(range(6), 2):
        test = [i for i in range(6) if i not in calibration]
        for name, method in (("CRC", crc_lambda), ("no-CRC", empirical_lambda)):
            threshold = method(
                [scores[i] for i in calibration], [losses[i] for i in calibration], 0.1
            )
            observed[name].append(mean(losses[i] * (scores[i] > threshold) for i in test))
    assert mean(observed["CRC"]) == 0
    assert mean(observed["no-CRC"]) > 0.1


def test_cached_curve_agrees_with_direct_metric_accounting():
    rows = [
        ScoredRow(str(i), score, 0, cheap, strong, cc, sc)
        for i, (score, cheap, strong, cc, sc) in enumerate(
            [(0, 1, 0, 0.1, 2), (0.5, 0, 1, 0.3, 4), (0.5, 0.2, 0.8, 0.1, 3), (1, 1, 1, 0.4, 5)]
        )
    ]
    curve = Curve(rows, [r.score for r in rows])
    for threshold in (-1, 0, 0.2, 0.5, 0.99, 1):
        expected = policy_metrics(rows, [float(r.score > threshold) for r in rows])
        assert curve.at(threshold) == pytest.approx(expected, abs=1e-12)
    for policy in ("CRC", "no-CRC", "random-CRC"):
        left = evaluate_cell(rows, curve, [0.1], [r.score for r in rows], policy)
        altered = Curve(
            [ScoredRow(r.id, r.score, 0, 0, 1, 2, 5) for r in rows], [r.score for r in rows]
        )
        right = evaluate_cell(rows, altered, [0.1], [r.score for r in rows], policy)
        assert left[0]["threshold"] == right[0]["threshold"]


def test_nested_partitions_keep_test_fixed_and_unused_rows_out():
    study = SensitivityConfig(calibration_sizes=[5, 10, 20], test_size=12)
    candidates, test = nested_partition(60, study, 42, 0)
    assert (candidates, test) == nested_partition(60, study, 42, 0)
    assert len(test) == 12 and len(candidates) == 48
    for size in study.calibration_sizes:
        assert set(candidates[:size]).isdisjoint(test)
        assert len(set(candidates[:size]) | set(test)) == size + 12
    assert set(candidates[:5]) < set(candidates[:10]) < set(candidates[:20])
    with pytest.raises(ValueError):
        nested_partition(25, study, 42, 0)


def test_small_sweep_is_offline_reproducible_and_does_not_pool_seed_uncertainty(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden.")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    args = (
        Path("data/eval/gsm8k/paired.jsonl"),
        Path("configs/offline.gsm8k.yaml"),
        Path("configs/pricing.gsm8k.yaml"),
    )
    study = SensitivityConfig(
        training_seeds=[17, 42],
        ranking_seeds=[0, 1],
        calibration_sizes=[10, 20],
        test_size=20,
        repeats=3,
    )
    first = run(*args, tmp_path / "first", study)
    second = run(*args, tmp_path / "second", study)
    for key in ("draws_sha256", "cells", "summaries", "advantages", "paired_comparisons"):
        assert first[key] == second[key]
    assert len(first["cells"]) == 2 * 2 * 4 * 4
    assert first["risk_ci_family_size"] == len(first["cells"])
    for row in first["summaries"]:
        assert row["mean_regret"] == mean(s["mean_regret"] for s in row["by_training_seed"])
        assert (
            "aggregate_ci" not in row
        )  # Seeds reuse data; no spurious pooled confidence interval.
    for seed in study.training_seeds:
        split = json.loads((tmp_path / f"first/seed-{seed}/split-manifest.json").read_text())
        assert set(split["train_ids"]).isdisjoint(split["pool_ids"])
    with gzip.open(tmp_path / "first/draws.jsonl.gz", "rt") as source:
        draws = [json.loads(line) for line in source]
    assert len(draws) == 6
    assert len(draws[0]["metrics"]) == 2 * 4 * 4
