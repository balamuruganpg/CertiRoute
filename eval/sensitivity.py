"""Offline G2/G6 sensitivity: calibration sizes, training/ranking seeds, no-CRC control."""

import argparse
import gzip
import hashlib
import json
import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, variance
from typing import Any, Self

import yaml
from pydantic import Field, model_validator

from certiroute.calibration.crc import crc_lambda
from certiroute.config import ConfigModel, GatewayConfig, load_config
from certiroute.provenance import content_hash
from certiroute.router.features import extract_features
from certiroute.router.predictor import LearnedPredictor, save_weights
from eval.fit_router import fit_router
from eval.fixtures import read_fixture, read_pricing
from eval.manifest import read_manifest
from eval.offline import code_provenance, dataset_for, split_rows, write_json
from eval.repeated import ScoredRow, draw_split, mean_interval, quantile

POLICIES = ("CRC", "no-CRC", "random-CRC")
METRICS = ("regret", "quality", "cost_usd", "savings_vs_strong", "strong_fraction")
DRAW_COLUMNS = ["calibration_size", "policy", "ranking_seed", "epsilon", "threshold", *METRICS]


class SensitivityConfig(ConfigModel):
    training_seeds: list[int] = Field(default_factory=lambda: [17, 42, 73, 101, 137])
    ranking_seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    calibration_sizes: list[int] = Field(default_factory=lambda: [50, 100, 200, 500])
    test_size: int = Field(default=264, gt=0)
    repeats: int = Field(default=1000, ge=2)
    confidence: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        for values in (self.training_seeds, self.ranking_seeds, self.calibration_sizes):
            if not values or len(values) != len(set(values)) or any(v < 0 for v in values):
                raise ValueError("Seeds and sizes must be nonempty, unique, nonnegative lists.")
        if min(self.calibration_sizes) == 0 or self.calibration_sizes != sorted(
            self.calibration_sizes
        ):
            raise ValueError("Calibration sizes must be positive and increasing.")
        return self


def empirical_lambda(scores: list[float], regrets: list[float], epsilon: float) -> float:
    """Negative control: smallest threshold satisfying sum(loss)/n <= epsilon.

    Keeps the same candidates, strict routing inequality and labels as CRC;
    only removes its finite-sample correction. Never a serving calibration method.
    """
    if (
        not scores
        or len(scores) != len(regrets)
        or not math.isfinite(epsilon)
        or not 0 <= epsilon <= 1
    ):
        raise ValueError("Require nonempty matched inputs and epsilon in [0,1].")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores + regrets):
        raise ValueError("Scores and losses must be finite in [0,1].")
    pairs = sorted(zip(scores, regrets, strict=True))
    ordered_scores = [s for s, _ in pairs]
    ordered_regrets = [r for _, r in pairs]
    candidates = sorted({0.0, 1.0, *scores})
    left, right = 0, len(candidates)
    while left < right:
        middle = (left + right) // 2
        start = bisect_right(ordered_scores, candidates[middle])
        if math.fsum(ordered_regrets[start:]) / len(scores) <= epsilon:
            right = middle
        else:
            left = middle + 1
    return candidates[left] if left < len(candidates) else 1.0


class TestCurve:
    """Cache test metrics for a ranking, preserving ties and strict > routing."""

    def __init__(self, rows: list[ScoredRow], scores: list[float]) -> None:
        if not rows or len(rows) != len(scores):
            raise ValueError("Expected matching nonempty rows and scores.")
        ordered = sorted(zip(scores, rows, strict=True), key=lambda pair: pair[0])
        self.scores = [score for score, _ in ordered]
        self.n = len(rows)
        self.strong_cost = math.fsum(row.strong_cost for row in rows)
        if self.strong_cost <= 0:
            raise ValueError("Strong cost must be positive.")
        self.strong_quality = math.fsum(row.q_strong for row in rows)
        self.loss = [0.0] * (self.n + 1)
        self.quality_delta = [0.0] * (self.n + 1)
        self.cost_delta = [0.0] * (self.n + 1)
        for index in range(self.n - 1, -1, -1):
            row = ordered[index][1]
            self.loss[index] = self.loss[index + 1] + row.regret
            self.quality_delta[index] = self.quality_delta[index + 1] + row.q_cheap - row.q_strong
            self.cost_delta[index] = self.cost_delta[index + 1] + row.cheap_cost - row.strong_cost

    def at(self, threshold: float) -> dict[str, float]:
        index = bisect_right(self.scores, threshold)
        cost = self.strong_cost + self.cost_delta[index]
        return {
            "regret": self.loss[index] / self.n,
            "quality": (self.strong_quality + self.quality_delta[index]) / self.n,
            "cost_usd": cost,
            "savings_vs_strong": 1 - cost / self.strong_cost,
            "strong_fraction": index / self.n,
        }


def nested_partition(
    pool_size: int, study: SensitivityConfig, training_seed: int, repeat: int
) -> tuple[list[int], list[int]]:
    candidates, test = draw_split(pool_size, pool_size - study.test_size, training_seed, repeat)
    if max(study.calibration_sizes) > len(candidates):
        raise ValueError("Largest calibration size plus fixed test size exceeds the pool.")
    return candidates, test


def evaluate_cell(
    calibration: list[ScoredRow],
    test_curve: TestCurve,
    epsilons: list[float],
    scores: list[float],
    policy: str,
) -> list[dict[str, float]]:
    if policy not in POLICIES:
        raise ValueError("Unknown sensitivity policy.")
    calibrate = empirical_lambda if policy == "no-CRC" else crc_lambda
    regrets = [row.regret for row in calibration]
    return [
        {"epsilon": epsilon, "threshold": threshold, **test_curve.at(threshold)}
        for epsilon in epsilons
        for threshold in [calibrate(scores, regrets, epsilon)]
    ]


def summarize_cell(
    measurements: list[dict[str, float]],
    epsilon: float,
    study: SensitivityConfig,
    comparisons: int,
) -> dict[str, Any]:
    risks = [row["regret"] for row in measurements]
    adjusted = mean_interval(risks, study.confidence, comparisons, (0, 1))
    return {
        **{"mean_" + field: mean(row[field] for row in measurements) for field in METRICS},
        "regret_ci_bonferroni": adjusted,
        "regret_variance": variance(risks),
        "threshold_variance": variance(row["threshold"] for row in measurements),
        "regret_quantiles": {str(q): quantile(risks, q) for q in (0.05, 0.5, 0.95)},
        "fraction_exceeding_epsilon": mean(float(r > epsilon) for r in risks),
        "non_rejection": adjusted[0] <= epsilon,
    }


def aggregate(
    cells: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    study: SensitivityConfig,
    epsilons: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    advantages = []
    for n_cal in study.calibration_sizes:
        for epsilon in epsilons:
            for policy in POLICIES:
                selected = [
                    r
                    for r in cells
                    if r["calibration_size"] == n_cal
                    and r["epsilon"] == epsilon
                    and r["policy"] == policy
                ]
                by_seed = [
                    {
                        "training_seed": seed,
                        **{
                            field: mean(r[field] for r in selected if r["training_seed"] == seed)
                            for field in (
                                "mean_regret",
                                "mean_quality",
                                "mean_cost_usd",
                                "mean_savings_vs_strong",
                                "fraction_exceeding_epsilon",
                            )
                        },
                    }
                    for seed in study.training_seeds
                ]
                summaries.append(
                    {
                        "calibration_size": n_cal,
                        "epsilon": epsilon,
                        "policy": policy,
                        **{
                            field: mean(r[field] for r in by_seed)
                            for field in by_seed[0]
                            if field != "training_seed"
                        },
                        "seed_mean_regret_range": [
                            min(r["mean_regret"] for r in by_seed),
                            max(r["mean_regret"] for r in by_seed),
                        ],
                        "seed_mean_savings_range": [
                            min(r["mean_savings_vs_strong"] for r in by_seed),
                            max(r["mean_savings_vs_strong"] for r in by_seed),
                        ],
                        "non_rejections": sum(r["non_rejection"] for r in selected),
                        "cell_count": len(selected),
                        "by_training_seed": by_seed,
                    }
                )
            pairs = [
                r for r in comparisons if r["calibration_size"] == n_cal and r["epsilon"] == epsilon
            ]
            advantages.append(
                {
                    "calibration_size": n_cal,
                    "epsilon": epsilon,
                    "mean_savings_advantage": mean(r["mean_savings_advantage"] for r in pairs),
                    "seed_pair_range": [
                        min(r["mean_savings_advantage"] for r in pairs),
                        max(r["mean_savings_advantage"] for r in pairs),
                    ],
                    "positive_pairs": sum(r["mean_savings_advantage"] > 0 for r in pairs),
                    "positive_adjusted_ci_pairs": sum(
                        r["paired_ci_bonferroni"][0] > 0 for r in pairs
                    ),
                    "pair_count": len(pairs),
                }
            )
    return summaries, advantages


def run(
    input_path: Path,
    config_path: Path,
    pricing_path: Path,
    out: Path,
    study: SensitivityConfig | None = None,
) -> dict[str, Any]:
    study = study or SensitivityConfig()
    config = load_config(config_path)
    if config.router.kind != "logistic_surface_v1" or any(
        not str(p.base_url.host).endswith(".invalid")
        for p in (config.models.cheap, config.models.strong)
    ):
        raise ValueError("Use a learned router and replay-only model identities.")
    manifest = read_manifest(input_path, config)
    rows, dropped = read_fixture(input_path, config)
    pricing = read_pricing(pricing_path, config)
    pool_size = len(rows) - int(0.4 * len(rows))
    nested_partition(pool_size, study, study.training_seeds[0], 0)
    out.mkdir(parents=True, exist_ok=False)
    epsilons = config.calibration.allowed_epsilons
    risk_family = (
        len(study.training_seeds)
        * len(study.calibration_sizes)
        * len(epsilons)
        * (2 + len(study.ranking_seeds))
    )
    paired_family = (
        len(study.training_seeds)
        * len(study.calibration_sizes)
        * len(epsilons)
        * len(study.ranking_seeds)
    )
    cells: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    seed_artifacts: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    with (out / "draws.jsonl.gz").open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for training_seed in study.training_seeds:
                print(
                    f"Training seed {training_seed}: fitting once; "
                    "evaluating all sizes/rankings...",
                    flush=True,
                )
                seed_dir = out / f"seed-{training_seed}"
                seed_dir.mkdir()
                fields = config.model_dump(mode="json")
                fields["router"]["artifact"] = str((seed_dir / "router.json").resolve())
                seed_config = GatewayConfig.model_validate(fields)
                split = split_rows(rows, training_seed)
                train, pool = split["train"], split["calibration"] + split["test"]
                dataset = dataset_for(train, seed_config, manifest)
                write_json(seed_dir / "train.json", dataset.model_dump(mode="json"))
                weights = fit_router(
                    dataset.rows,
                    seed_config,
                    hashlib.sha256((seed_dir / "train.json").read_bytes()).hexdigest(),
                    training_seed,
                )
                save_weights(weights, seed_config.router.artifact)
                predictor = LearnedPredictor(weights)
                scored = []
                for row in pool:
                    predictor.check_independent(row.request)
                    scored.append(
                        ScoredRow(
                            row.id,
                            predictor.score(extract_features(row.request)),
                            0,
                            row.q_cheap,
                            row.q_strong,
                            pricing.models[config.models.cheap.model].cost(row.cheap_usage),
                            pricing.models[config.models.strong.model].cost(row.strong_usage),
                        )
                    )
                rankings = {
                    seed: [
                        int(content_hash(["random-score-v1", seed, row.id])[:13], 16) / 2**52
                        for row in pool
                    ]
                    for seed in study.ranking_seeds
                }
                write_json(
                    seed_dir / "pool-scores.json",
                    {"rows": [asdict(r) for r in scored], "random_rankings": rankings},
                )
                split_manifest = {
                    "training_seed": training_seed,
                    "train_ids": [r.id for r in train],
                    "pool_ids": [r.id for r in pool],
                    "dropped_ids": dropped,
                    "partition_algorithm": "eval.repeated.draw_split(pool_size, "
                    "pool_size-test_size, training_seed, repeat); use prefixes of calibration "
                    "candidates and identical trailing test indices across sizes.",
                    "test_size": study.test_size,
                    "calibration_sizes": study.calibration_sizes,
                }
                write_json(seed_dir / "split-manifest.json", split_manifest)
                groups: dict[tuple[int, str, int | None, float], list[dict[str, float]]] = (
                    defaultdict(list)
                )
                baseline_groups: dict[str, list[dict[str, float]]] = defaultdict(list)
                for repeat in range(study.repeats):
                    candidates, test_indices = nested_partition(
                        len(pool), study, training_seed, repeat
                    )
                    test = [scored[i] for i in test_indices]
                    learned_curve = TestCurve(test, [r.score for r in test])
                    random_curves = {
                        seed: TestCurve(test, [values[i] for i in test_indices])
                        for seed, values in rankings.items()
                    }
                    metrics: list[list[Any]] = []
                    for name, threshold in (
                        ("cheap-only", -1.0),
                        ("strong-only", 1.0),
                        ("fixed-threshold", 0.5),
                    ):
                        baseline_groups[name].append(learned_curve.at(threshold))
                    for n_cal in study.calibration_sizes:
                        indices = candidates[:n_cal]
                        calibration = [scored[i] for i in indices]
                        learned_scores = [r.score for r in calibration]
                        settings: list[tuple[str, int | None, TestCurve, list[float]]] = [
                            ("CRC", None, learned_curve, learned_scores),
                            ("no-CRC", None, learned_curve, learned_scores),
                        ]
                        settings += [
                            ("random-CRC", seed, random_curves[seed], [values[i] for i in indices])
                            for seed, values in rankings.items()
                        ]
                        for policy, ranking_seed, curve, scores in settings:
                            measurements = evaluate_cell(
                                calibration, curve, epsilons, scores, policy
                            )
                            for measurement in measurements:
                                epsilon = measurement["epsilon"]
                                groups[(n_cal, policy, ranking_seed, epsilon)].append(measurement)
                                metrics.append(
                                    [
                                        n_cal,
                                        policy,
                                        ranking_seed,
                                        epsilon,
                                        measurement["threshold"],
                                        *(measurement[field] for field in METRICS),
                                    ]
                                )
                    record = {
                        "training_seed": training_seed,
                        "repeat": repeat,
                        "partition_sha256": content_hash([candidates, test_indices]),
                        "metrics": metrics,
                    }
                    stream.write(
                        (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
                    )
                for (n_cal, policy, ranking_seed, epsilon), measurements in groups.items():
                    cells.append(
                        {
                            "training_seed": training_seed,
                            "calibration_size": n_cal,
                            "policy": policy,
                            "ranking_seed": ranking_seed,
                            "epsilon": epsilon,
                            **summarize_cell(measurements, epsilon, study, risk_family),
                        }
                    )
                for n_cal in study.calibration_sizes:
                    for epsilon in epsilons:
                        learned = groups[(n_cal, "CRC", None, epsilon)]
                        for ranking_seed in study.ranking_seeds:
                            baseline = groups[(n_cal, "random-CRC", ranking_seed, epsilon)]
                            differences = [
                                a["savings_vs_strong"] - b["savings_vs_strong"]
                                for a, b in zip(learned, baseline, strict=True)
                            ]
                            paired.append(
                                {
                                    "training_seed": training_seed,
                                    "ranking_seed": ranking_seed,
                                    "calibration_size": n_cal,
                                    "epsilon": epsilon,
                                    "mean_savings_advantage": mean(differences),
                                    "paired_ci_bonferroni": mean_interval(
                                        differences, study.confidence, paired_family
                                    ),
                                }
                            )
                baselines.extend(
                    {
                        "training_seed": training_seed,
                        "policy": name,
                        **{
                            "mean_" + field: mean(row[field] for row in values) for field in METRICS
                        },
                    }
                    for name, values in baseline_groups.items()
                )
                seed_artifacts.append(
                    {
                        "training_seed": training_seed,
                        "router_sha256": weights.sha256,
                        "train_size": len(train),
                        "pool_size": len(pool),
                        "hashes": {
                            name: hashlib.sha256((seed_dir / name).read_bytes()).hexdigest()
                            for name in ("train.json", "pool-scores.json", "split-manifest.json")
                        },
                    }
                )
    summaries, advantages = aggregate(cells, paired, study, epsilons)
    result: dict[str, Any] = {
        "kind": "offline_calibration_seed_sensitivity",
        "created_at": datetime.now(UTC).isoformat(),
        "study": study.model_dump(mode="json"),
        "epsilons": epsilons,
        "config": config.model_dump(mode="json"),
        "dataset_manifest": manifest.model_dump(mode="json"),
        "pricing": pricing.model_dump(mode="json"),
        "code": code_provenance(),
        "pricing_sha256": hashlib.sha256(pricing_path.read_bytes()).hexdigest(),
        "draws_sha256": hashlib.sha256((out / "draws.jsonl.gz").read_bytes()).hexdigest(),
        "draw_columns": DRAW_COLUMNS,
        "seed_artifacts": seed_artifacts,
        "risk_ci_family_size": risk_family,
        "paired_ci_family_size": paired_family,
        "cells": cells,
        "paired_comparisons": paired,
        "summaries": summaries,
        "advantages": advantages,
        "baselines": baselines,
        "inference": "Per-cell normal Monte Carlo CIs use independent draws conditional on "
        "that training set, finite pool and ranking. Bonferroni risk family includes all seeds, "
        "rankings, sizes, epsilons and three policies. Paired savings CIs adjust across all seed "
        "pairs, sizes and epsilons. Seed ranges are descriptive, "
        "not population confidence intervals. "
        "Overlapping training/pool sets and nested sizes are never counted as independent samples.",
        "limitations": manifest.limitations
        + [
            "Benchmark rows were used in prior experiments; "
            "this is sensitivity analysis, not a new holdout.",
            "Only five configured training partitions and five rankings in the default study; "
            "no generalization claim across arbitrary datasets or model families.",
            "The no-CRC policy is an evaluation-only negative control, "
            "not a deployable calibrated artifact.",
            "Risk bounds are marginal expectations, not guarantees on every split. "
            "Non-rejection is not proof.",
            "Token counts and historical costs remain estimates. "
            "No live provider calls or billed spend.",
        ],
    }
    write_json(out / "results.json", result)
    (out / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(f"Saved sensitivity results and report to {out}", flush=True)
    return result


def render_report(result: dict[str, Any]) -> str:
    study = result["study"]
    lines = [
        "# GSM8K calibration and seed sensitivity",
        "",
        f"Training seeds {study['training_seeds']}; random-ranking seeds {study['ranking_seeds']}; "
        f"{study['repeats']} draws per training seed; fixed test size {study['test_size']}.",
        "Same full prompts, learner, seven features, hyperparameters and epsilon grid. "
        "Each predictor is fitted once before all calibration/test draws.",
        "",
        "Calibration prefixes are nested and test questions stay identical across sizes in a draw.",
        "Tables average training-seed means equally; "
        "random CRC also averages ranking seeds equally.",
        "Ranges show sensitivity across seed means, not confidence intervals for new data.",
        "",
        "## Calibration-size sweep and no-CRC control",
        "",
        "| n_cal | Epsilon | Policy | Mean regret | Training-seed regret range | "
        "Estimated savings | Splits > epsilon |",
        "|---:|---:|---|---:|---|---:|---:|",
    ]
    for row in result["summaries"]:
        low, high = row["seed_mean_regret_range"]
        lines.append(
            f"| {row['calibration_size']} | {row['epsilon']} | {row['policy']} | "
            f"{row['mean_regret']:.5f} | [{low:.5f}, {high:.5f}] | "
            f"{row['mean_savings_vs_strong']:.2%} | {row['fraction_exceeding_epsilon']:.1%} |"
        )
    lines += [
        "",
        "## Learned ranking versus random rankings",
        "",
        "Positive values favor the learned ranking at the same target epsilon; "
        "attained risks differ.",
        "The range covers every training/ranking-seed pair, after averaging its draws.",
        "",
        "| n_cal | Epsilon | Mean savings advantage (pp) | Seed-pair range (pp) | "
        "Positive pairs | Positive adjusted-CI pairs |",
        "|---:|---:|---:|---|---:|---:|",
    ]
    for row in result["advantages"]:
        low, high = row["seed_pair_range"]
        lines.append(
            f"| {row['calibration_size']} | {row['epsilon']} | "
            f"{100 * row['mean_savings_advantage']:.3f} | [{100 * low:.3f}, {100 * high:.3f}] | "
            f"{row['positive_pairs']}/{row['pair_count']} | "
            f"{row['positive_adjusted_ci_pairs']}/{row['pair_count']} |"
        )
    crc = [r for r in result["cells"] if r["policy"] == "CRC"]
    empirical = [r for r in result["cells"] if r["policy"] == "no-CRC"]
    lines += [
        "",
        "## Interpretation and plots",
        "",
        f"CRC non-rejections: {sum(r['non_rejection'] for r in crc)}/{len(crc)} cells. "
        f"No-CRC rejections: {sum(not r['non_rejection'] for r in empirical)}"
        f"/{len(empirical)} cells. "
        "The criterion is adjusted lower mean-regret CI <= epsilon; non-rejection is not proof.",
        "The negative control chooses the smallest lambda with sum(loss)/n <= epsilon. "
        "CRC uses (sum(loss)+1)/(n+1) <= epsilon. Both route cheap strictly when score > lambda.",
        "",
        "![Calibration-size sensitivity](plots/calibration-sweep.png)",
        "",
        "![Risk versus estimated cost](plots/risk-cost.png)",
        "",
        "![Ranking advantage across training seeds](plots/seed-advantage.png)",
        "",
        "## Reproducibility and limitations",
        "",
        result["inference"],
        "",
        f"Risk CI family: {result['risk_ci_family_size']} cells; paired family: "
        f"{result['paired_ci_family_size']} comparisons. Normal intervals are approximate.",
        "Every threshold/metric is in draws.jsonl.gz, using draw_columns from results.json. "
        "Partition checksums, seed-specific ordered IDs, frozen scores and weights permit replay.",
        "Unused calibration candidates are not added to the fixed test set for smaller sizes.",
        "Means, variances, quantiles, per-cell CIs and pair comparisons are in results.json.",
        "Plots use the saved aggregates; PNG/SVG/PDF and plotting provenance are in plots/.",
        "",
        "Model identities: " + "; ".join(result["dataset_manifest"]["models"].values()),
        "Prompt: " + result["dataset_manifest"]["prompt_template"],
        "Decoding metadata: `" + json.dumps(result["dataset_manifest"]["generation"]) + "`.",
        "Tokens: " + result["dataset_manifest"]["token_counts"],
        "Pricing: " + result["pricing"]["description"],
        "",
        *("- " + item for item in result["limitations"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval/gsm8k-full/paired.jsonl"))
    parser.add_argument(
        "--config", type=Path, default=Path("data/eval/gsm8k-full/gateway-template.yaml")
    )
    parser.add_argument("--pricing", type=Path, default=Path("configs/pricing.gsm8k.yaml"))
    parser.add_argument("--study", type=Path, default=Path("configs/study.sensitivity.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study = SensitivityConfig.model_validate(yaml.safe_load(args.study.read_text(encoding="utf-8")))
    run(args.input, args.config, args.pricing, args.out, study)


if __name__ == "__main__":
    main()
