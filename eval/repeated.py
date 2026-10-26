"""Offline G1: freeze one router, repeat calibration/test splits, compare baselines."""

import argparse
import asyncio
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist, mean, stdev
from typing import Any

import yaml
from pydantic import Field, model_validator

from certiroute.calibration.artifact import DataProvenance, build_artifact, save_artifact
from certiroute.calibration.crc import crc_lambda
from certiroute.config import ConfigModel, GatewayConfig, load_config
from certiroute.provenance import content_hash
from certiroute.router.features import extract_features
from certiroute.router.predictor import LearnedPredictor, save_weights
from eval.fit_router import fit_router
from eval.fixtures import read_fixture, read_pricing
from eval.manifest import read_manifest
from eval.offline import code_provenance, compare, dataset_for, split_rows, write_json


class StudyConfig(ConfigModel):
    repeats: int = Field(default=1000, ge=2)
    seed: int = Field(default=42, ge=0)
    calibration_size: int | None = Field(default=None, gt=0)
    fixed_threshold: float = Field(default=0.5, ge=0, le=1)
    random_probabilities: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    confidence: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def check_probabilities(self) -> "StudyConfig":
        values = self.random_probabilities
        if not values or len(values) != len(set(values)) or any(not 0 < p < 1 for p in values):
            raise ValueError("Random probabilities must be distinct and strictly between 0 and 1.")
        return self


@dataclass(frozen=True)
class ScoredRow:
    id: str
    score: float
    random_score: float
    q_cheap: float
    q_strong: float
    cheap_cost: float
    strong_cost: float

    @property
    def regret(self) -> float:
        return max(0.0, self.q_strong - self.q_cheap)


def draw_split(size: int, n_cal: int, seed: int, repeat: int) -> tuple[list[int], list[int]]:
    if not 0 < n_cal < size:
        raise ValueError("Calibration and test must both be nonempty.")
    order = list(range(size))
    random.Random(int(content_hash(["g1-shuffle-v1", seed, repeat]), 16)).shuffle(order)
    return order[:n_cal], order[n_cal:]


def policy_metrics(rows: list[ScoredRow], probabilities: list[float]) -> dict[str, float]:
    """Realized metrics for 0/1 policies; exact routing expectations for random mixes."""
    if not rows or len(rows) != len(probabilities):
        raise ValueError("Need matching nonempty rows and probabilities.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid cheap-routing probability.")
    pairs = list(zip(rows, probabilities, strict=True))
    strong_cost = math.fsum(row.strong_cost for row in rows)
    if strong_cost <= 0:
        raise ValueError("Positive strong baseline cost required.")
    cost = math.fsum(p * row.cheap_cost + (1 - p) * row.strong_cost for row, p in pairs)
    return {
        "quality": math.fsum(p * row.q_cheap + (1 - p) * row.q_strong for row, p in pairs)
        / len(rows),
        "regret": math.fsum(p * row.regret for row, p in pairs) / len(rows),
        "cost_usd": cost,
        "savings_vs_strong": 1 - cost / strong_cost,
        "strong_fraction": 1 - math.fsum(probabilities) / len(rows),
    }


def evaluate_draw(
    calibration: list[ScoredRow], test: list[ScoredRow], epsilons: list[float], study: StudyConfig
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def add(policy: str, probabilities: list[float], **parameters: Any) -> None:
        results.append({"policy": policy, **parameters, **policy_metrics(test, probabilities)})

    add("cheap-only", [1.0] * len(test))
    add("strong-only", [0.0] * len(test))
    add(
        "fixed-threshold",
        [float(r.score > study.fixed_threshold) for r in test],
        threshold=study.fixed_threshold,
    )
    for probability in study.random_probabilities:
        add("random-mix-expected", [probability] * len(test), probability=probability)
    regrets = [r.regret for r in calibration]
    for policy, attribute in (("CertiRoute", "score"), ("random-score-CRC", "random_score")):
        for epsilon in epsilons:
            threshold = crc_lambda([getattr(r, attribute) for r in calibration], regrets, epsilon)
            add(
                policy,
                [float(getattr(r, attribute) > threshold) for r in test],
                epsilon=epsilon,
                threshold=threshold,
            )
    return results


def mean_interval(
    values: list[float],
    confidence: float = 0.95,
    comparisons: int = 1,
    bounds: tuple[float, float] | None = None,
) -> list[float]:
    """Approximate normal Monte Carlo CI, conditional on the fixed training set/pool.

    Independent random partitions, not individual prompts, are the sampling units.
    This interval does not estimate population uncertainty from acquiring new data.
    """
    if len(values) < 2 or comparisons < 1:
        raise ValueError("Need at least two draws and a positive comparison count.")
    center = mean(values)
    z = NormalDist().inv_cdf(1 - (1 - confidence) / (2 * comparisons))
    half = z * stdev(values) / math.sqrt(len(values))
    lower, upper = center - half, center + half
    if bounds:
        lower, upper = max(bounds[0], lower), min(bounds[1], upper)
    return [lower, upper]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize(
    draws: list[dict[str, Any]], epsilons: list[float], study: StudyConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, float | None, float | None], list[dict[str, Any]]] = defaultdict(list)
    for draw in draws:
        for row in draw["policies"]:
            groups[(row["policy"], row.get("epsilon"), row.get("probability"))].append(row)
    summaries = []
    for (policy, epsilon, probability), rows in groups.items():
        risks = [r["regret"] for r in rows]
        adjusted = mean_interval(risks, study.confidence, len(epsilons), (0, 1))
        summaries.append(
            {
                "policy": policy,
                "epsilon": epsilon,
                "probability": probability,
                "repeats": len(rows),
                "mean_regret": mean(risks),
                "regret_ci": mean_interval(risks, study.confidence, bounds=(0, 1)),
                "regret_ci_bonferroni": adjusted,
                "regret_quantiles": {str(p): quantile(risks, p) for p in (0.05, 0.5, 0.95)},
                "max_regret": max(risks),
                "fraction_exceeding_epsilon": {
                    str(e): sum(risk > e for risk in risks) / len(risks) for e in epsilons
                },
                "non_rejection": adjusted[0] <= epsilon if epsilon is not None else None,
                "mean_quality": mean(r["quality"] for r in rows),
                "mean_cost_usd": mean(r["cost_usd"] for r in rows),
                "mean_savings_vs_strong": mean(r["savings_vs_strong"] for r in rows),
                "mean_strong_fraction": mean(r["strong_fraction"] for r in rows),
            }
        )
    comparisons = []
    for epsilon in epsilons:
        learned = groups[("CertiRoute", epsilon, None)]
        baseline = groups[("random-score-CRC", epsilon, None)]
        differences = [
            a["savings_vs_strong"] - b["savings_vs_strong"]
            for a, b in zip(learned, baseline, strict=True)
        ]
        comparisons.append(
            {
                "epsilon": epsilon,
                "mean_savings_advantage": mean(differences),
                "paired_ci_bonferroni": mean_interval(differences, study.confidence, len(epsilons)),
            }
        )
    return summaries, comparisons


def run(
    input_path: Path,
    config_path: Path,
    pricing_path: Path,
    out: Path,
    study: StudyConfig | None = None,
) -> dict[str, Any]:
    study = study or StudyConfig()
    config = load_config(config_path)
    if config.router.kind != "logistic_surface_v1":
        raise ValueError("G1 study requires the frozen learned predictor.")
    if any(
        not str(p.base_url.host).endswith(".invalid")
        for p in (config.models.cheap, config.models.strong)
    ):
        raise ValueError("Public-cache studies require .invalid replay endpoints.")
    manifest = read_manifest(input_path, config)
    pricing = read_pricing(pricing_path, config)
    rows, dropped = read_fixture(input_path, config)
    initial = split_rows(rows, study.seed)
    train, pool = initial["train"], initial["calibration"] + initial["test"]
    n_cal = study.calibration_size or (2 * len(pool) // 3)
    draw_split(len(pool), n_cal, study.seed, 0)  # Validate before writing outputs.
    out.mkdir(parents=True, exist_ok=False)
    fields = config.model_dump(mode="json")
    fields["router"]["artifact"] = str((out / "router.json").resolve())
    fields["calibration"]["artifact"] = str((out / "calibration-artifact.json").resolve())
    config = GatewayConfig.model_validate(fields)
    config_file = out / "gateway.yaml"
    config_file.write_text(yaml.safe_dump(fields, sort_keys=False), encoding="utf-8")
    train_dataset = dataset_for(train, config, manifest)
    write_json(out / "train.json", train_dataset.model_dump(mode="json"))
    weights = fit_router(
        train_dataset.rows,
        config,
        hashlib.sha256((out / "train.json").read_bytes()).hexdigest(),
        study.seed,
    )
    save_weights(weights, config.router.artifact)
    predictor = LearnedPredictor(weights)
    scored = []
    for row in pool:
        predictor.check_independent(row.request)
        scored.append(
            ScoredRow(
                id=row.id,
                score=predictor.score(extract_features(row.request)),
                random_score=int(content_hash(["random-score-v1", study.seed, row.id])[:13], 16)
                / 2**52,
                q_cheap=row.q_cheap,
                q_strong=row.q_strong,
                cheap_cost=pricing.models[config.models.cheap.model].cost(row.cheap_usage),
                strong_cost=pricing.models[config.models.strong.model].cost(row.strong_usage),
            )
        )
    write_json(out / "pool-scores.json", [asdict(row) for row in scored])
    split_manifest = {
        "train_ids": [r.id for r in train],
        "pool_ids": [r.id for r in pool],
        "dropped_ids": dropped,
        "seed": study.seed,
        "train_assignment": "First floor(0.4*N) by SHA256([seed, full messages]); fixed once.",
        "partition_algorithm": "Random(int(SHA256(['g1-shuffle-v1', seed, repeat]),16)).shuffle "
        "on pool index order; first n_cal calibration, remainder test.",
        "n_cal": n_cal,
        "n_test": len(pool) - n_cal,
        "signature_prefix_sha256": content_hash(config.router.signature_prefix),
    }
    write_json(out / "split-manifest.json", split_manifest)
    draws: list[dict[str, Any]] = []
    epsilons = config.calibration.allowed_epsilons
    for repeat in range(study.repeats):
        cal_indices, test_indices = draw_split(len(pool), n_cal, study.seed, repeat)
        draws.append(
            {
                "repeat": repeat,
                "partition_sha256": content_hash([cal_indices, test_indices]),
                "policies": evaluate_draw(
                    [scored[i] for i in cal_indices],
                    [scored[i] for i in test_indices],
                    epsilons,
                    study,
                ),
            }
        )
    with (out / "draws.jsonl").open("x", encoding="utf-8") as target:
        for draw in draws:
            target.write(json.dumps(draw, sort_keys=True, allow_nan=False) + "\n")
    summaries, comparisons = summarize(draws, epsilons, study)

    # Verify the precomputed G1 path against real gateway routing on the first draw.
    cal_indices, test_indices = draw_split(len(pool), n_cal, study.seed, 0)
    calibration = dataset_for([pool[i] for i in cal_indices], config, manifest)
    write_json(out / "calibration.json", calibration.model_dump(mode="json"))
    write_json(
        out / "test.json",
        dataset_for([pool[i] for i in test_indices], config, manifest).model_dump(mode="json"),
    )
    artifact = build_artifact(
        calibration.rows,
        config,
        DataProvenance(
            sources=calibration.sources,
            collected_at=calibration.collected_at,
            sha256=hashlib.sha256((out / "calibration.json").read_bytes()).hexdigest(),
        ),
    )
    save_artifact(artifact, config.calibration.artifact)
    expected_thresholds = {
        str(row["epsilon"]): row["threshold"]
        for row in draws[0]["policies"]
        if row["policy"] == "CertiRoute"
    }
    if artifact.lambda_by_epsilon != expected_thresholds:
        raise ValueError("Gateway artifact and precomputed CRC thresholds disagree.")
    replay_summaries, observations = asyncio.run(
        compare([pool[i] for i in test_indices], config, config_file, pricing)
    )
    scores_by_id = {row.id: row.score for row in scored}
    for observation in observations:
        if observation["policy"] == "CertiRoute":
            threshold = expected_thresholds[str(observation["epsilon"])]
            expected_tier = "cheap" if scores_by_id[observation["id"]] > threshold else "strong"
            if observation["tier"] != expected_tier:
                raise ValueError("Gateway and precomputed routing decisions disagree.")
    for actual in replay_summaries:
        expected = next(
            item
            for item in draws[0]["policies"]
            if item["policy"] == actual["policy"] and item.get("epsilon") == actual["epsilon"]
        )
        for field in ("quality", "regret", "cost_usd", "savings_vs_strong", "strong_fraction"):
            if not math.isclose(actual[field], expected[field], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Gateway and precomputed evaluation disagree: {field}")
    write_json(
        out / "gateway-check.json", {"summaries": replay_summaries, "observations": observations}
    )
    result = {
        "kind": "conditional_finite_pool_g1",
        "created_at": datetime.now(UTC).isoformat(),
        "study": study.model_dump(mode="json"),
        "epsilons": epsilons,
        "sizes": {
            "train": len(train),
            "pool": len(pool),
            "calibration": n_cal,
            "test": len(pool) - n_cal,
        },
        "config": config.model_dump(mode="json"),
        "dataset_manifest": manifest.model_dump(mode="json"),
        "pricing": pricing.model_dump(mode="json"),
        "router_sha256": weights.sha256,
        "first_draw_calibration_id": artifact.calibration_id,
        "hashes": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("draws.jsonl", "pool-scores.json", "split-manifest.json")
        },
        "pricing_sha256": hashlib.sha256(pricing_path.read_bytes()).hexdigest(),
        "code": code_provenance(),
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "gateway_check": "All first-draw CRC thresholds and routing decisions agree exactly; "
        "baseline/CertiRoute metrics agree within 1e-12.",
        "inference": "Approximate normal Monte Carlo confidence intervals over independent "
        "partitions conditional on this fixed training set and finite pool. Bonferroni across "
        "the epsilon grid separately for each CRC policy and paired-comparison family. "
        "These intervals exclude uncertainty from sampling new datasets or retraining.",
        "limitations": manifest.limitations
        + [
            "Fewer than 1000 calibration examples; this implements G1 on the available pool, "
            "not the planned larger-data study or the other G2-G9 experiments.",
            "Non-rejection (adjusted lower CI <= epsilon) is not proof of validity. "
            "No live exchangeability or production guarantee is established.",
            "Random-mix rows integrate Bernoulli routing analytically: their exceedance rates "
            "refer to conditional expected regret on each split, not sampled routing coins.",
            "The old pilot has already exposed some of these benchmark rows; new features "
            "and hyperparameters were not tuned here. This is not a pristine new holdout.",
            "The random-score ranking uses one fixed seed; intervals do not include variation "
            "across alternative random rankings. Normal Monte Carlo CIs are approximate.",
        ],
    }
    write_json(out / "results.json", result)
    report = render_report(result)
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# GSM8K repeated-split study",
        "",
        f"{result['study']['repeats']} independent partitions; sizes: {result['sizes']}.",
        "One frozen logistic surface-feature predictor; no tuning on calibration/test outcomes.",
        "Real cached outputs; estimated tokens and historical modeled costs; zero API calls.",
        "",
        "## Calibrated policies",
        "",
        "CI is Bonferroni-adjusted Monte Carlo uncertainty for mean regret, "
        "conditional on the pool.",
        "Non-rejection follows EVALUATION.md; it is not evidence of deployment safety.",
        "",
        "| Policy | Epsilon | Mean regret | Adjusted CI | Splits > epsilon | Accuracy | "
        "Savings | Strong fraction | Non-rejection |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in result["summaries"]:
        if row["epsilon"] is None:
            continue
        low, high = row["regret_ci_bonferroni"]
        lines.append(
            f"| {row['policy']} | {row['epsilon']} | {row['mean_regret']:.5f} | "
            f"[{low:.5f}, {high:.5f}] | "
            f"{row['fraction_exceeding_epsilon'][str(row['epsilon'])]:.1%} | "
            f"{row['mean_quality']:.2%} | {row['mean_savings_vs_strong']:.2%} | "
            f"{row['mean_strong_fraction']:.2%} | {row['non_rejection']} |"
        )
    lines += [
        "",
        "## Uncalibrated baselines",
        "",
        "Random mixes below are exact expectations over routing coins, not calibrated policies.",
        "The fixed-threshold baseline routes cheap iff score > "
        f"{result['study']['fixed_threshold']}.",
        "",
        "| Policy | Cheap probability | Mean regret | Accuracy | Savings |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["summaries"]:
        if row["epsilon"] is not None:
            continue
        probability = row["probability"] if row["probability"] is not None else "—"
        lines.append(
            f"| {row['policy']} | {probability} | "
            f"{row['mean_regret']:.5f} | "
            f"{row['mean_quality']:.2%} | {row['mean_savings_vs_strong']:.2%} |"
        )
    lines += [
        "",
        "## Learned versus random ranking",
        "",
        "Both rankings use CRC at the same target epsilon, not necessarily the same attained risk.",
        "Random-score CRC uses a fixed label-independent hash ranking; it does not fit labels.",
        "Positive advantage means greater savings for the learned router on paired partitions.",
        "",
        "| Epsilon | Savings advantage (percentage points) | Adjusted paired CI (pp) |",
        "|---|---:|---|",
    ]
    for row in result["paired_comparisons"]:
        low, high = row["paired_ci_bonferroni"]
        lines.append(
            f"| {row['epsilon']} | {100 * row['mean_savings_advantage']:.3f} | "
            f"[{100 * low:.3f}, {100 * high:.3f}] |"
        )
    lines += [
        "",
        "## Provenance and interpretation",
        "",
        result["inference"],
        "",
        result["gateway_check"],
        "",
        "Every draw, threshold and partition checksum is in draws.jsonl; risk quantiles, "
        "exceedance fractions, mean total test costs, configurations and hashes "
        "are in results.json.",
        "Partitions can be reconstructed from split-manifest.json and the pinned algorithm.",
        "",
        "Models: " + "; ".join(result["dataset_manifest"]["models"].values()) + ".",
        "Prompt: " + result["dataset_manifest"]["prompt_template"],
        "Decoding metadata: `" + json.dumps(result["dataset_manifest"]["generation"]) + "`.",
        "Pricing: " + result["pricing"]["description"],
        "Pricing source: " + str(result["pricing"]["source"]),
        "",
        "Limitations:",
        "",
        *("- " + s for s in result["limitations"]),
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
    parser.add_argument("--study", type=Path, default=Path("configs/study.gsm8k.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study = StudyConfig.model_validate(yaml.safe_load(args.study.read_text(encoding="utf-8")))
    run(args.input, args.config, args.pricing, args.out, study)


if __name__ == "__main__":
    main()
