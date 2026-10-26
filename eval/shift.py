"""Offline within-GSM8K covariate shift with source controls and target recalibration."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from pydantic import Field

from certiroute.calibration.crc import crc_lambda
from certiroute.config import ConfigModel, load_config
from certiroute.provenance import content_hash
from certiroute.router.features import extract_features
from certiroute.router.predictor import LearnedPredictor, save_weights
from eval.fit_router import fit_router
from eval.fixtures import read_fixture, read_pricing
from eval.manifest import read_manifest
from eval.offline import code_provenance, dataset_for, split_rows, write_json
from eval.repeated import ScoredRow, mean_interval, policy_metrics, quantile


class ShiftConfig(ConfigModel):
    seed: int = Field(default=42, ge=0)
    repeats: int = Field(default=1000, ge=2)
    calibration_size: int = Field(default=100, gt=0)
    test_size: int = Field(default=100, gt=0)
    confidence: float = Field(default=0.95, gt=0, lt=1)


def question_length(request: dict[str, object], prefix: str) -> int:
    """Measure only the question, never answers or the shared demonstrations."""
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Expected one GSM8K user message.")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("Expected one GSM8K user message.")
    text = message.get("content")
    if not isinstance(text, str) or not text.startswith(prefix):
        raise ValueError("Question does not match the bound prefix.")
    text = text[len(prefix) :]
    # Older pilot configs have no bound prefix; the final question is still explicit.
    question = text.rsplit("Question:", 1)[-1].removesuffix("Answer:").strip()
    if not question:
        raise ValueError("Empty question.")
    return len(question)


def partition_domains(
    domains: dict[str, list[int]], study: ShiftConfig, repeat: int
) -> dict[str, dict[str, list[int]]]:
    partitions = {}
    seen: set[int] = set()
    for domain, indices in domains.items():
        if len(indices) < study.calibration_size + study.test_size:
            raise ValueError(f"Not enough {domain} examples for disjoint calibration and test.")
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError("Domain indices must be unique and disjoint.")
        seen.update(indices)
        order = list(indices)
        random.Random(
            int(content_hash(["shift-partition-v1", study.seed, repeat, domain]), 16)
        ).shuffle(order)
        partitions[domain] = {
            "calibration": order[: study.calibration_size],
            "test": order[study.calibration_size : study.calibration_size + study.test_size],
        }
    return partitions


def evaluate_transfer(
    source_cal: list[ScoredRow],
    source_test: list[ScoredRow],
    target_cal: list[ScoredRow],
    target_test: list[ScoredRow],
    epsilons: list[float],
) -> list[dict[str, Any]]:
    results = []
    for epsilon in epsilons:
        source_threshold = crc_lambda(
            [r.score for r in source_cal], [r.regret for r in source_cal], epsilon
        )
        target_threshold = crc_lambda(
            [r.score for r in target_cal], [r.regret for r in target_cal], epsilon
        )
        for arm, threshold, test in (
            ("source-control", source_threshold, source_test),
            ("shifted", source_threshold, target_test),
            ("target-recalibrated", target_threshold, target_test),
        ):
            results.append(
                {
                    "arm": arm,
                    "epsilon": epsilon,
                    "threshold": threshold,
                    **policy_metrics(test, [float(r.score > threshold) for r in test]),
                }
            )
    return results


def run(
    input_path: Path,
    config_path: Path,
    pricing_path: Path,
    out: Path,
    study: ShiftConfig | None = None,
) -> dict[str, Any]:
    study = study or ShiftConfig()
    config = load_config(config_path)
    if config.router.kind != "logistic_surface_v1":
        raise ValueError("Shift study requires the learned predictor.")
    if any(
        not str(p.base_url.host).endswith(".invalid")
        for p in (config.models.cheap, config.models.strong)
    ):
        raise ValueError("Offline studies require .invalid replay endpoints.")
    manifest = read_manifest(input_path, config)
    pricing = read_pricing(pricing_path, config)
    rows, dropped = read_fixture(input_path, config)
    initial = split_rows(rows, study.seed)
    train, pool = initial["train"], initial["calibration"] + initial["test"]
    boundary = median(question_length(r.request, config.router.signature_prefix) for r in train)
    lengths = [question_length(r.request, config.router.signature_prefix) for r in pool]
    domains = {
        "short": [i for i, n in enumerate(lengths) if n <= boundary],
        "long": [i for i, n in enumerate(lengths) if n > boundary],
    }
    partition_domains(domains, study, 0)  # Fail before creating partial artifacts.
    out.mkdir(parents=True, exist_ok=False)
    training = dataset_for(train, config, manifest)
    write_json(out / "train.json", training.model_dump(mode="json"))
    weights = fit_router(
        training.rows,
        config,
        hashlib.sha256((out / "train.json").read_bytes()).hexdigest(),
        study.seed,
    )
    save_weights(weights, out / "router.json")
    predictor = LearnedPredictor(weights)
    scored = []
    for row in pool:
        predictor.check_independent(row.request)
        scored.append(
            ScoredRow(
                row.id,
                predictor.score(extract_features(row.request)),
                0.0,
                row.q_cheap,
                row.q_strong,
                pricing.models[config.models.cheap.model].cost(row.cheap_usage),
                pricing.models[config.models.strong.model].cost(row.strong_usage),
            )
        )
    write_json(out / "pool-scores.json", [asdict(row) for row in scored])
    write_json(
        out / "split-manifest.json",
        {
            "train_ids": [r.id for r in train],
            "pool_ids": [r.id for r in pool],
            "dropped_ids": dropped,
            "question_lengths": lengths,
            "training_median_question_characters": boundary,
            "domains": {name: [pool[i].id for i in indices] for name, indices in domains.items()},
            "domain_indices": domains,
            "rule": "Short: question characters <= training median; long: > median. "
            "Boundary and training assignment never use pool labels. "
            "Both transfer directions reported.",
            "partition_algorithm": "Random(int(SHA256("
            "['shift-partition-v1', seed, repeat, domain]),16)).shuffle(domain_indices); "
            "first n_cal calibration, next n_test test; remainder unused.",
        },
    )
    epsilons = config.calibration.allowed_epsilons
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    with (out / "draws.jsonl").open("x", encoding="utf-8") as handle:
        for repeat in range(study.repeats):
            partitions = partition_domains(domains, study, repeat)
            policies = []
            for source, target in (("short", "long"), ("long", "short")):
                arms = evaluate_transfer(
                    [scored[i] for i in partitions[source]["calibration"]],
                    [scored[i] for i in partitions[source]["test"]],
                    [scored[i] for i in partitions[target]["calibration"]],
                    [scored[i] for i in partitions[target]["test"]],
                    epsilons,
                )
                for arm in arms:
                    arm["direction"] = f"{source}-to-{target}"
                    groups[(arm["direction"], arm["arm"], arm["epsilon"])].append(arm)
                policies.extend(arms)
            handle.write(
                json.dumps(
                    {
                        "repeat": repeat,
                        "partitions": partitions,
                        "policies": policies,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    family = len(groups)
    summaries = []
    for (direction, arm_name, epsilon), observations in groups.items():
        risks = [r["regret"] for r in observations]
        interval = mean_interval(risks, study.confidence, family, (0, 1))
        summaries.append(
            {
                "direction": direction,
                "arm": arm_name,
                "epsilon": epsilon,
                "mean_regret": mean(risks),
                "regret_ci_bonferroni": interval,
                "mean_within_target": mean(risks) <= epsilon,
                "non_rejection": interval[0] <= epsilon,
                "fraction_exceeding_epsilon": mean(float(r > epsilon) for r in risks),
                "regret_quantiles": {str(p): quantile(risks, p) for p in (0.05, 0.5, 0.95)},
                "mean_quality": mean(r["quality"] for r in observations),
                "mean_cost_usd": mean(r["cost_usd"] for r in observations),
                "mean_savings_vs_strong": mean(r["savings_vs_strong"] for r in observations),
                "mean_strong_fraction": mean(r["strong_fraction"] for r in observations),
                "mean_threshold": mean(r["threshold"] for r in observations),
            }
        )
    result = {
        "kind": "offline_within_gsm8k_length_shift",
        "study": study.model_dump(mode="json"),
        "epsilons": epsilons,
        "sizes": {"train": len(train), **{name: len(ids) for name, ids in domains.items()}},
        "training_median_question_characters": boundary,
        "domain_diagnostics": {
            name: {
                "mean_question_characters": mean(lengths[i] for i in indices),
                **policy_metrics([scored[i] for i in indices], [1.0] * len(indices)),
            }
            for name, indices in domains.items()
        },
        "config": config.model_dump(mode="json"),
        "dataset_manifest": manifest.model_dump(mode="json"),
        "pricing": pricing.model_dump(mode="json"),
        "router_sha256": weights.sha256,
        "hashes": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in (
                "train.json",
                "router.json",
                "pool-scores.json",
                "split-manifest.json",
                "draws.jsonl",
            )
        },
        "code": code_provenance(),
        "risk_ci_family_size": family,
        "summaries": summaries,
        "inference": "Approximate normal Monte Carlo CIs over independent partition draws, "
        "conditional on one fixed training set and finite pools. "
        "Bonferroni across both directions, "
        "three arms and all epsilons. Arms share rows; they are not independent datasets. "
        "Mean <= target is descriptive; lower adjusted CI <= target is the plan's non-rejection "
        "criterion, not proof of validity. Individual splits can exceed epsilon.",
        "limitations": manifest.limitations
        + [
            "Question length is a label-free covariate, not a verified difficulty label. "
            "This within-GSM8K stress test does not complete S1's cross-dataset domain study.",
            "One training seed and previously analyzed benchmark rows; "
            "not a new untouched holdout.",
            "Target recalibration uses a separate, fully labeled target sample; no online drift "
            "detector, audit-delay simulation or audit budget is evaluated. "
            "Calibration acquisition "
            "cost is excluded from reported routed-generation savings.",
            "Real frozen outputs; estimated token counts and historical modeled prices, not bills. "
            "No live production traffic, provider calls or production guarantee.",
        ],
    }
    write_json(out / "results.json", result)
    (out / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(f"Saved shift results to {out}")
    return result


def render_report(result: dict[str, Any]) -> str:
    study = result["study"]
    lines = [
        "# GSM8K question-length distribution shift",
        "",
        f"{study['repeats']} draws, seed {study['seed']}, "
        f"{study['calibration_size']} calibration and {study['test_size']} "
        "test examples per domain.",
        f"Sizes: {result['sizes']}. Training-only median question length: "
        f"{result['training_median_question_characters']:g} characters.",
        "Short questions are at or below that boundary; long questions are above it. "
        "The eight shared demonstrations are excluded from length, but remain in router features.",
        "One frozen predictor. Both directions, all epsilons and all arms were fixed before "
        "inspecting this experiment's outcomes. No threshold uses test labels.",
        "",
        "Source-control calibrates and tests on disjoint source examples. Shifted reuses that "
        "threshold on target examples. Target-recalibrated calibrates on separate target examples "
        "and evaluates on the identical target test set. Unused domain rows remain excluded.",
        "",
        "## Empirical results",
        "",
        "Regret is clipped per question before averaging. Within target reports the measured "
        "mean, without a confidence allowance. Rejected means the adjusted lower CI is above "
        "epsilon; this is distinct from one empirical mean or split exceeding epsilon.",
        "",
        "| Direction | Arm | Epsilon | Mean regret | Adjusted CI | Within target? | "
        "Rejected? | Splits > target | Estimated savings |",
        "|---|---|---:|---:|---|---|---|---:|---:|",
    ]
    for row in result["summaries"]:
        low, high = row["regret_ci_bonferroni"]
        lines.append(
            f"| {row['direction']} | {row['arm']} | {row['epsilon']:g} | "
            f"{row['mean_regret']:.5f} | [{low:.5f}, {high:.5f}] | "
            f"{'yes' if row['mean_within_target'] else 'no'} | "
            f"{'no' if row['non_rejection'] else 'yes'} | "
            f"{row['fraction_exceeding_epsilon']:.1%} | {row['mean_savings_vs_strong']:.2%} |"
        )
    lines += [
        "",
        "## Domain diagnostics",
        "",
        "Computed after fixing the split rule; these labels did not select the domains.",
        "",
        "| Domain | Mean question characters | Cheap-only regret | Cheap accuracy |",
        "|---|---:|---:|---:|",
    ]
    for domain, row in result["domain_diagnostics"].items():
        lines.append(
            f"| {domain} | {row['mean_question_characters']:.1f} | "
            f"{row['regret']:.5f} | {row['quality']:.2%} |"
        )
    lines += [
        "",
        "## Interpretation and reproducibility",
        "",
        result["inference"],
        "",
        f"Adjusted risk family: {result['risk_ci_family_size']} cells. "
        "Per-draw partitions, thresholds and metrics are in draws.jsonl; ordered IDs, "
        "lengths and group membership in split-manifest.json; scores and weights are frozen.",
        "",
        "Model identities: " + "; ".join(result["dataset_manifest"]["models"].values()),
        "Prompt: " + result["dataset_manifest"]["prompt_template"],
        "Decoding metadata: `" + json.dumps(result["dataset_manifest"]["generation"]) + "`.",
        "Tokens: " + result["dataset_manifest"]["token_counts"],
        "Pricing: " + result["pricing"]["description"],
        "",
        "## Limitations",
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
    parser.add_argument("--study", type=Path, default=Path("configs/study.shift.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study = ShiftConfig.model_validate(yaml.safe_load(args.study.read_text(encoding="utf-8")))
    run(args.input, args.config, args.pricing, args.out, study)


if __name__ == "__main__":
    main()
