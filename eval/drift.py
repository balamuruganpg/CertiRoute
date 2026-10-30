"""Replay finite-window detection, strong fallback and disjoint target recalibration."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from certiroute.calibration.crc import crc_lambda
from certiroute.monitor.drift import score_shift
from certiroute.provenance import content_hash
from eval.offline import code_provenance, write_json
from eval.repeated import ScoredRow, mean_interval, policy_metrics


def run(source: Path, out: Path, repeats: int = 1000) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("Need at least two draws")
    raw = (source / "pool-scores.json").read_bytes()
    split_raw = (source / "split-manifest.json").read_bytes()
    evidence = json.loads((source / "results.json").read_bytes())
    for name, data in [("pool-scores.json", raw), ("split-manifest.json", split_raw)]:
        if evidence["hashes"][name] != hashlib.sha256(data).hexdigest():
            raise ValueError("Source study evidence hash mismatch")
    rows = [ScoredRow(**r) for r in json.loads(raw)]
    splits = json.loads(split_raw)
    domains = splits["domain_indices"]
    if any(len(set(indices)) < 300 for indices in domains.values()):
        raise ValueError("Need 300 distinct rows per domain")
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for source_domain, target_domain in [("long", "short"), ("short", "long")]:
        draws = []
        for repeat in range(repeats):
            partitions = {}
            for domain, indices in domains.items():
                order = list(indices)
                random.Random(int(content_hash(["monitor-v1", 42, repeat, domain]), 16)).shuffle(
                    order
                )
                partitions[domain] = [order[:100], order[100:200], order[200:300]]
            src_cal, reference, control = [[rows[i] for i in p] for p in partitions[source_domain]]
            detection, audit, holdout = [[rows[i] for i in p] for p in partitions[target_domain]]
            alarm = score_shift(
                [r.score for r in reference], [r.score for r in detection], alpha=0.05, windows=2
            )
            false_alarm = score_shift(
                [r.score for r in reference], [r.score for r in control], alpha=0.05, windows=2
            )
            old = crc_lambda([r.score for r in src_cal], [r.regret for r in src_cal], 0.05)
            new = crc_lambda([r.score for r in audit], [r.regret for r in audit], 0.05)
            before = policy_metrics(holdout, [float(r.score > old) for r in holdout])
            after = policy_metrics(holdout, [float(r.score > new) for r in holdout])
            deployed = after if alarm["detected"] else before
            draws.append(
                {
                    "repeat": repeat,
                    "detected": alarm["detected"],
                    "false_alarm": false_alarm["detected"],
                    "detector": alarm,
                    "before": before,
                    "recalibrated": after,
                    "detector_policy": deployed,
                    "source_threshold": old,
                    "target_threshold": new,
                    "audit_incremental_cost_usd": sum(r.cheap_cost for r in audit),
                    "audit_total_pair_cost_usd": sum(r.cheap_cost + r.strong_cost for r in audit),
                    "partition_sha256": content_hash(partitions),
                }
            )
            if repeat == 0:
                write_json(
                    out / (source_domain + "-to-" + target_domain + "-splits.json"),
                    {
                        "reference": [r.id for r in reference],
                        "source_calibration": [r.id for r in src_cal],
                        "no_shift_control": [r.id for r in control],
                        "target_detection": [r.id for r in detection],
                        "target_audit": [r.id for r in audit],
                        "target_holdout": [r.id for r in holdout],
                    },
                )
        summary: dict[str, Any] = {
            "direction": source_domain + "-to-" + target_domain,
            "detection_rate": mean(float(d["detected"]) for d in draws),
            "false_alarm_rate": mean(float(d["false_alarm"]) for d in draws),
            "detection_delay_requests_if_detected": 100,
            "additional_audit_requests_before_reactivation": 100,
            "mean_audit_pair_cost_usd": mean(d["audit_total_pair_cost_usd"] for d in draws),
        }
        for arm in ("before", "recalibrated", "detector_policy"):
            risks = [d[arm]["regret"] for d in draws]
            ci = mean_interval(risks, comparisons=6, bounds=(0, 1))
            summary[arm] = {
                "mean_regret": mean(risks),
                "regret_ci_family6": ci,
                "non_rejection": ci[0] <= 0.05,
                "mean_savings": mean(d[arm]["savings_vs_strong"] for d in draws),
            }
        results.append(summary)
        with (out / (source_domain + "-to-" + target_domain + ".jsonl")).open("x") as f:
            for draw in draws:
                f.write(json.dumps(draw) + "\n")
    result = {
        "epsilon": 0.05,
        "repeats": repeats,
        "seed": 42,
        "summaries": results,
        "detector": "Two-sample DKW, alpha=.05 across two predeclared windows; n=100 each.",
        "source_hashes": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("pool-scores.json", "split-manifest.json", "results.json")
        },
        "code": code_provenance(),
        "limitations": [
            "Finite-pool simulated abrupt within-GSM8K length shift.",
            "No production monitor or continuous-stream guarantee.",
            "Detection window serves stale policy; alert switches to strong while "
            "100 separate target pairs are acquired. Before/after uses later holdout.",
            "Recalibration is fresh/unselected target data, not alarm-window labels.",
            "Score drift can miss conditional quality drift; periodic label audits "
            "remain necessary.",
        ],
    }
    write_json(out / "results.json", result)
    lines = [
        "# Drift detection and recalibration replay",
        "",
        "| Shift | Detection | No-shift alarms | Before regret | Recalibrated regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in results:
        lines.append(
            f"| {s['direction']} | {s['detection_rate']:.1%} | {s['false_alarm_rate']:.1%} | "
            f"{s['before']['mean_regret']:.3%} | {s['recalibrated']['mean_regret']:.3%} |"
        )
    lines += [
        "",
        "Target: 5%. An alert is evaluated after 100 requests; acquire 100 further paired "
        "audits while serving strong, then evaluate on 100 disjoint target requests. "
        "All splits, costs, detector-policy outcomes and adjusted intervals are saved. "
        "See docs/DRIFT.md; a missed alert does not trigger recalibration.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/gsm8k-shift"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    run(args.source, args.out, args.repeats)
    print(f"Saved drift replay to {args.out}")


if __name__ == "__main__":
    main()
