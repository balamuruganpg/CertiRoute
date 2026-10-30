"""Publish an allowlisted live snapshot and verify its metrics without provider calls."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from eval.live import summarize_calls
from eval.offline import write_json


def select(value: dict[str, Any], fields: str) -> dict[str, Any]:
    return {key: value[key] for key in fields.split() if key in value}


def export_snapshot(source: Path, out: Path) -> None:
    raw = (source / "results.json").read_bytes()
    original = json.loads(raw)
    if original.get("status") != "complete":
        raise ValueError("Only a complete live study can be published")
    out.mkdir(parents=True, exist_ok=False)
    report = select(
        original,
        "status created_at summaries acquisition upstream_attempts "
        "budget_charged_usd_including_prior_runs calibration_id limitations",
    )
    spec = original["spec"]
    report["spec"] = select(
        spec,
        "train calibration test seed epsilon max_tokens pricing_as_of pricing_sources pricing_note",
    )
    for tier in ("cheap", "strong"):
        if tier in spec:
            report["spec"][tier] = select(spec[tier], "base_url model input cached_input output")
    report["execution"] = select(
        original.get("execution", {}),
        "interval_seconds acquisition_429_retries resumed reused_acquisition_responses",
    )
    with (out / "observations.jsonl").open("x", encoding="utf-8") as stream:
        for item in original["observations"]:
            observation = select(
                item,
                "id policy phase status ok output latency_ms cost_usd cost_known quality regret",
            )
            attempts = []
            for attempt in item["attempts"]:
                saved = select(
                    attempt,
                    "tier ok cost_usd reserved_usd cost_known started_at "
                    "status cost_basis returned_model latency_ms retry_after_seconds",
                )
                usage = attempt.get("usage")
                saved["usage"] = (
                    select(usage, "prompt_tokens completion_tokens total_tokens cost_in_usd_ticks")
                    if isinstance(usage, dict)
                    else None
                )
                if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens_details"), dict):
                    saved["usage"]["prompt_tokens_details"] = select(
                        usage["prompt_tokens_details"], "cached_tokens"
                    )
                attempts.append(saved)
            observation["attempts"] = attempts
            stream.write(json.dumps(observation, allow_nan=False) + "\n")
    splits = json.loads((source / "splits.json").read_bytes())
    write_json(out / "splits.json", select(splits, "mode train calibration test source_sha256"))
    report["publication"] = {
        "private_result_sha256": hashlib.sha256(raw).hexdigest(),
        "observations_sha256": hashlib.sha256(
            (out / "observations.jsonl").read_bytes()
        ).hexdigest(),
        "note": "Frozen live outputs and metrics; account paths, request IDs and credential "
        "configuration omitted. This snapshot is not an executable provider config.",
    }
    write_json(out / "results.json", report)


def verify_snapshot(root: Path) -> None:
    report = json.loads((root / "results.json").read_bytes())
    raw = (root / "observations.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != report["publication"]["observations_sha256"]:
        raise ValueError("Observation checksum mismatch")
    observations = [json.loads(line) for line in raw.splitlines()]
    splits = json.loads((root / "splits.json").read_bytes())
    test_ids = splits["test"]
    policies = {"cheap-only", "strong-only", "CertiRoute"}
    groups: dict[str, dict[str, Any]] = {}
    for item in observations:
        for attempt in item["attempts"]:
            usage = attempt.get("usage") or {}
            prices = report["spec"][attempt["tier"]]
            if not attempt["cost_known"]:
                cost = attempt["reserved_usd"]
            elif attempt["cost_basis"] == "xai_reported_usd_ticks":
                cost = usage["cost_in_usd_ticks"] / 1e10
            elif attempt["cost_basis"] in {
                "actual_usage_times_recorded_cache_prices",
                "uncached_input_upper_bound",
            }:
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if not 0 <= cached <= usage["prompt_tokens"]:
                    raise ValueError("Invalid cache usage")
                cost = (
                    (usage["prompt_tokens"] - cached) * prices["input"]
                    + cached * prices["cached_input"]
                    + usage["completion_tokens"] * prices["output"]
                ) / 1e6
            else:
                raise ValueError("Unknown cost basis in public snapshot")
            if not math.isclose(cost, attempt["cost_usd"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Attempt cost does not match usage/prices/reservation")
        if not math.isclose(
            item["cost_usd"],
            sum(a["cost_usd"] for a in item["attempts"]),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("Observation cost does not match attempts")
        if item["phase"] == "test":
            group = groups.setdefault(item["id"], {})
            if item["policy"] in group:
                raise ValueError("Duplicate policy observation")
            group[item["policy"]] = item
    if len(test_ids) != len(set(test_ids)) or set(groups) != set(test_ids):
        raise ValueError("Held-out test IDs differ from frozen split")
    if set(test_ids) & set(splits["train"] + splits["calibration"]):
        raise ValueError("Held-out test split overlaps acquisition")
    for group in groups.values():
        if set(group) != policies:
            raise ValueError("Each test prompt requires all three policies")
        strong = group["strong-only"]
        for item in group.values():
            regret = max(0.0, strong["quality"] - item["quality"]) if strong["ok"] else None
            if regret != item["regret"]:
                raise ValueError("Regret does not match paired policy quality")
    for policy, summary in report["summaries"].items():
        values = [o for o in observations if o["phase"] == "test" and o["policy"] == policy]
        computed = summarize_calls(values)
        computed["quality"] = sum(v["quality"] for v in values) / len(values)
        regrets = [v["regret"] for v in values if v["regret"] is not None]
        computed["regret"] = sum(regrets) / len(regrets) if regrets else None
        computed["regret_denominator"] = len(regrets)
        for key, expected in summary.items():
            actual = computed[key]
            if expected is None and actual is None:
                continue
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Metric mismatch: {policy} / {key}")
    # Re-score published test outputs against the original public GSM8K references.
    from eval.fixtures import last_integer
    from eval.multibench import load_task

    rows, _, _ = load_task("gsm8k", Path("data/eval/multibench"))
    answers = {row.id: row.expected for row in rows}
    for item in observations:
        if item["phase"] == "test":
            quality = float(item["ok"] and last_integer(item["output"]) == answers[item["id"]])
            if quality != item["quality"]:
                raise ValueError("Published quality does not match frozen output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_snapshot(args.verify)
        print("Frozen live metrics and test-output quality verified offline; no provider calls.")
    elif args.source and args.out:
        export_snapshot(args.source, args.out)
        print(f"Published snapshot: {args.out}")
    else:
        parser.error("Use --verify SNAPSHOT, or --source PRIVATE_RUN --out NEW_SNAPSHOT")


if __name__ == "__main__":
    main()
