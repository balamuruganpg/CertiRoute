"""Task-stratified CRC evaluation on GSM8K, ARC-Challenge and executable MBPP."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from certiroute.calibration.artifact import (
    CalibrationRow,
    DataProvenance,
    build_artifact,
    save_artifact,
)
from certiroute.config import GatewayConfig, load_config
from certiroute.provenance import content_hash, overlaps, prompt_signature
from certiroute.router.features import extract_features
from certiroute.router.policy import route
from certiroute.router.predictor import LearnedPredictor, save_weights
from certiroute.router.scope import with_generation_defaults
from eval.benchmarks import BenchmarkRow, portfolio_metrics, read_benchmark
from eval.fit_router import fit_router
from eval.offline import code_provenance, write_json
from eval.repeated import (
    ScoredRow,
    StudyConfig,
    draw_split,
    evaluate_draw,
    mean_interval,
    summarize,
)

EPSILONS = [0.01, 0.02, 0.05, 0.10]


def load_task(name: str, data: Path) -> tuple[list[BenchmarkRow], dict[str, Any], GatewayConfig]:
    config = load_config(Path("configs/offline.gsm8k.yaml"))
    if name == "gsm8k":
        path = Path("data/eval/gsm8k-full/paired.jsonl")
        # Discover the existing snapshot name without changing its format.
        if not path.exists():
            candidates = list(path.parent.glob("*.jsonl"))
            if len(candidates) != 1:
                raise ValueError("Ambiguous GSM8K dataset")
            path = candidates[0]
        manifest = json.loads(path.with_suffix(".manifest.json").read_bytes())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
            raise ValueError("GSM8K checksum mismatch")
        rows = []
        for line in path.read_bytes().splitlines():
            record = json.loads(line)
            record.pop("source")
            rows.append(BenchmarkRow(task="gsm8k", **record))
        cfg = config.model_dump(mode="json")
        cfg["router"]["signature_prefix"] = manifest.get("signature_prefix", "")
        manifest["pricing"] = {
            "cheap": [0.24, 0.24],
            "strong": [10, 30],
            "as_of": "2024-06-26",
            "description": "Historical modeled USD/M",
        }
        return rows, manifest, GatewayConfig.model_validate(cfg)
    rows, manifest = read_benchmark(data / (name + ".jsonl"))
    cfg = config.model_dump(mode="json")
    for tier in ("cheap", "strong"):
        cfg["models"][tier]["model"] = manifest["models"][tier]
    cfg["scope"] = {"max_chars": 32000, "max_tokens": 1024, "temperature": 0, "top_p": 1}
    cfg["quality_metric"] = {
        "name": "mbpp_base_tests" if name == "mbpp" else "routerbench_arc_published_exact_match",
        "version": "1",
    }
    return rows, manifest, GatewayConfig.model_validate(cfg)


def run_task(name: str, data: Path, out: Path, repeats: int, seed: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    rows, manifest, config = load_task(name, data)
    retained: list[BenchmarkRow] = []
    signatures: list[frozenset[str]] = []
    dropped: list[str] = []
    for row in rows:
        signature = prompt_signature(row.request, config.router.signature_prefix)
        if any(overlaps(signature, previous) for previous in signatures):
            dropped.append(row.id)
        else:
            retained.append(row)
            signatures.append(signature)
    ordered = sorted(retained, key=lambda row: content_hash([seed, row.request["messages"]]))
    train_size = int(0.4 * len(ordered))
    train, pool = ordered[:train_size], ordered[train_size:]
    n_cal = len(pool) // 2
    config = config.model_copy(
        update={
            "router": config.router.model_copy(update={"artifact": out / "router.json"}),
            "calibration": config.calibration.model_copy(
                update={
                    "artifact": out / "calibration.json",
                    "default_epsilon": 0.05,
                    "allowed_epsilons": EPSILONS,
                }
            ),
        }
    )
    training = [
        CalibrationRow(request=r.request, q_cheap=r.q_cheap, q_strong=r.q_strong) for r in train
    ]
    weights = fit_router(training, config, content_hash([r.model_dump() for r in train]), seed)
    save_weights(weights, config.router.artifact)
    predictor = LearnedPredictor(weights)
    scores = []
    for row in pool:
        predictor.check_independent(row.request)
        costs = {}
        for tier in ("cheap", "strong"):
            usage = getattr(row, tier + "_usage")
            price = manifest["pricing"][tier]
            costs[tier] = (
                usage.prompt_tokens * price[0] + usage.completion_tokens * price[1]
            ) / 1e6
        scores.append(
            ScoredRow(
                row.id,
                predictor.score(extract_features(row.request)),
                int(content_hash(["random-rank", seed, row.id]), 16) / (2**256),
                row.q_cheap,
                row.q_strong,
                costs["cheap"],
                costs["strong"],
            )
        )
    study = StudyConfig(repeats=repeats, seed=seed)
    draws: list[dict[str, Any]] = []
    for repeat in range(repeats):
        cal, test = draw_split(len(pool), n_cal, seed, repeat)
        policies = evaluate_draw(
            [scores[i] for i in cal], [scores[i] for i in test], EPSILONS, study
        )
        for policy in policies:
            policy["cost_usd"] /= len(test)
        draws.append(
            {"repeat": repeat, "partition_sha256": content_hash([cal, test]), "policies": policies}
        )
    cal, test = draw_split(len(pool), n_cal, seed, 0)
    artifact = build_artifact(
        [
            CalibrationRow(
                request=pool[i].request, q_cheap=pool[i].q_cheap, q_strong=pool[i].q_strong
            )
            for i in cal
        ],
        config,
        DataProvenance(
            sources=[name, manifest["dataset_sha256"]],
            sha256=content_hash([pool[i].model_dump() for i in cal]),
            collected_at=datetime(2026, 10, 26, tzinfo=UTC),
        ),
    )
    save_artifact(artifact, config.calibration.artifact)
    for epsilon in EPSILONS:
        expected = next(
            p
            for p in draws[0]["policies"]
            if p["policy"] == "CertiRoute" and p["epsilon"] == epsilon
        )
        if artifact.threshold(epsilon) != expected["threshold"]:
            raise ValueError("Artifact and evaluation thresholds differ")
        for i in test:
            decision = route(
                with_generation_defaults(pool[i].request, config.scope),
                epsilon=epsilon,
                config=config,
                artifact=artifact,
                predictor=predictor,
            )
            wanted = "cheap" if scores[i].score > artifact.threshold(epsilon) else "strong"
            if decision.tier != wanted:
                raise ValueError("Gateway and evaluation decisions differ")
    write_json(out / "config.json", config.model_dump(mode="json"))
    write_json(out / "pool-scores.json", [asdict(row) for row in scores])
    write_json(
        out / "split-manifest.json",
        {
            "train_ids": [r.id for r in train],
            "pool_ids": [r.id for r in pool],
            "dropped_ids": dropped,
            "seed": seed,
            "calibration_size": n_cal,
            "test_size": len(pool) - n_cal,
            "first_calibration_ids": [pool[i].id for i in cal],
            "first_test_ids": [pool[i].id for i in test],
        },
    )
    with (out / "draws.jsonl").open("x", encoding="utf-8") as f:
        for draw in draws:
            f.write(json.dumps(draw) + "\n")
    summaries, _ = summarize(draws, EPSILONS, study)
    result = {
        "task": name,
        "sizes": {
            "raw": len(rows),
            "train": len(train),
            "calibration": n_cal,
            "test": len(pool) - n_cal,
        },
        "dataset_manifest": manifest,
        "summaries": summaries,
        "gateway_check": "All first-draw thresholds and decisions match.",
        "router_sha256": weights.sha256,
    }
    write_json(out / "results.json", result)
    return {**result, "draws": draws}


def run(data: Path, out: Path, repeats: int = 1000, seed: int = 42) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("At least two repeats required")
    out.mkdir(parents=True, exist_ok=False)
    tasks = [run_task(name, data, out / name, repeats, seed) for name in ("gsm8k", "arc", "mbpp")]
    combined = []
    for repeat in range(repeats):
        policies = []
        for index, original in enumerate(tasks[0]["draws"][repeat]["policies"]):
            parts = []
            for task in tasks:
                policy = task["draws"][repeat]["policies"][index]
                strong = next(
                    p for p in task["draws"][repeat]["policies"] if p["policy"] == "strong-only"
                )
                parts.append({**policy, "strong_cost_usd": strong["cost_usd"]})
            metric = portfolio_metrics(parts)
            policies.append(
                {
                    **original,
                    **metric,
                    "strong_fraction": sum(p["strong_fraction"] for p in parts) / 3,
                }
            )
        combined.append({"repeat": repeat, "policies": policies})
    summaries, _ = summarize(combined, EPSILONS, StudyConfig(repeats=repeats, seed=seed))
    tasks.append({"task": "equal-task portfolio", "summaries": summaries, "draws": combined})
    # A single declared 4-task x 4-epsilon family for the headline CRC validity checks.
    for task in tasks:
        for summary in task["summaries"]:
            if summary["policy"] == "CertiRoute":
                values = [
                    next(
                        p["regret"]
                        for p in d["policies"]
                        if p["policy"] == "CertiRoute" and p["epsilon"] == summary["epsilon"]
                    )
                    for d in task["draws"]
                ]
                ci = mean_interval(values, comparisons=16, bounds=(0, 1))
                summary["regret_ci_family16"] = ci
                summary["non_rejection_family16"] = ci[0] <= summary["epsilon"]
        del task["draws"]
    result = {
        "kind": "task_stratified_finite_pool_crc",
        "repeats": repeats,
        "seed": seed,
        "epsilons": EPSILONS,
        "tasks": tasks,
        "cost_unit": "USD per request",
        "mixture": "Equal task weights, separate frozen router/calibration per task.",
        "inference": "Normal Monte Carlo intervals conditional on fixed pools/training; "
        "16-comparison Bonferroni family. Not population confidence.",
        "code": code_provenance(),
    }
    write_json(out / "results.json", result)
    lines = [
        "# Multi-benchmark CRC study",
        "",
        "Frozen real pairs; historical modeled costs, not bills. Target regret: 5%.",
        "",
        "| Task | Policy | Quality | Regret | USD/1k | Savings |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        for p in task["summaries"]:
            if p["policy"] not in ("cheap-only", "strong-only", "CertiRoute"):
                continue
            if p["policy"] == "CertiRoute" and p["epsilon"] != 0.05:
                continue
            lines.append(
                f"| {task['task']} | {p['policy']} | {p['mean_quality']:.2%} | "
                f"{p['mean_regret']:.3%} | {1000 * p['mean_cost_usd']:.3f} | "
                f"{p['mean_savings_vs_strong']:.2%} |"
            )
    lines += [
        "",
        "40% training; remaining 60% split equally into calibration/test on each of "
        f"{repeats:,} seeded draws. Full epsilon grid, random-ranking CRC, fixed threshold "
        "and random mixtures are in results.json. Costs per request are averaged before "
        "portfolio savings are calculated. See docs/MULTIBENCH.md for provenance, "
        "scoring, split rules and limitations.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eval/multibench"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.out, args.repeats, args.seed)
    print(f"Saved multi-benchmark report to {args.out}")


if __name__ == "__main__":
    main()
