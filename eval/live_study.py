"""Collect fresh calibration pairs, then measure three policies against real providers.

Default is a local plan only. --execute sends paid requests, reads keys locally,
and requires a persistent budget journal. Never reuse offline model-pair artifacts.
"""

import argparse
import asyncio
import json
import math
import os
import random
import secrets
import time
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values
from pydantic import SecretStr

from certiroute.calibration.artifact import (
    CalibrationRow,
    DataProvenance,
    build_artifact,
    save_artifact,
)
from certiroute.config import GatewayConfig, Settings
from certiroute.gateway.app import create_app
from certiroute.provenance import content_hash, overlaps, prompt_signature
from certiroute.router.predictor import save_weights
from eval.fit_router import fit_router
from eval.fixtures import last_integer
from eval.live import Budget, MeteredTransport, summarize_calls
from eval.live_recovery import ResumeError, acquire_with_retry, load_checkpoint
from eval.multibench import load_task
from eval.offline import code_provenance, write_json


class AcquisitionError(ValueError):
    """Actionable diagnostics built only from local labels and HTTP status codes."""

    def __init__(self, tier: str, status: int | None, cost_known: bool) -> None:
        provider = {"cheap": "Groq", "strong": "xAI"}[tier]
        hint = "Acquisition stopped; incomplete pairs cannot be used for calibration."
        if status == 402:
            hint = "Check provider credits and API-key spending limits before retrying."
        elif status in (401, 403):
            hint = "Check the local provider key and its access permissions."
        elif status == 429:
            hint = "Provider rate limit reached; review quota and pacing before retrying."
        elif status is None:
            hint = "No HTTP response received; check connectivity or provider availability."
        elif not cost_known:
            hint = "Provider usage/cost is missing; the budget reservation has been retained."
        self.diagnostic = {"provider": provider, "http_status": status, "hint": hint}
        super().__init__(
            f"{provider}: HTTP {status if status is not None else 'unavailable'}. {hint}"
        )


def validate_spec(spec: dict[str, Any]) -> None:
    if spec["strong"]["model"] != "grok-4.20-0309-non-reasoning":
        raise ValueError("The budgeted study requires the fixed xAI non-reasoning model")
    if (
        not 0 < spec["budget_usd"] <= 2.50
        or spec["total_authorized_usd"] != 2.99
        or spec["fee_reserve_usd"] != 0.49
    ):
        raise ValueError("Study must preserve the $2.99 authorization and $0.49 fee buffer")
    if (
        type(spec["pilot_requests"]) is not int
        or not 10 <= spec["pilot_requests"] <= 20
        or spec["pilot_requests"] % 2
        or not 0 < spec["pilot_budget_usd"] <= 0.10
    ):
        raise ValueError("Pilot requires 10-20 even calls and at most $0.10")
    if not all(
        isinstance(spec[k], int) and spec[k] >= 20 for k in ("train", "calibration", "test")
    ):
        raise ValueError("Need at least 20 prompts in each disjoint split")
    if not 0 < spec["epsilon"] <= 1 or not 1 <= spec["max_tokens"] <= 1024:
        raise ValueError("Invalid risk or output bound")
    if not math.isfinite(spec["interval_seconds"]) or spec["interval_seconds"] < 0:
        raise ValueError("Invalid pacing")
    for tier, host, path in (
        ("cheap", "api.groq.com", "/openai/v1"),
        ("strong", "api.x.ai", "/v1"),
    ):
        endpoint = urlparse(spec[tier]["base_url"])
        if (
            endpoint.scheme != "https"
            or endpoint.hostname != host
            or endpoint.username
            or endpoint.path != path
            or endpoint.query
            or endpoint.fragment
            or endpoint.port not in (None, 443)
        ):
            raise ValueError("Unexpected provider endpoint")
        for kind in ("input", "output", "cached_input"):
            value = spec[tier][kind]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Confirmed finite positive {tier} {kind} prices required (USD/M tokens)"
                )
        if spec[tier]["cached_input"] > spec[tier]["input"]:
            raise ValueError("Cached price cannot exceed the uncached reservation price")


async def preflight(spec: dict[str, Any], keys: dict[str, str]) -> None:
    age = (date.today() - date.fromisoformat(spec["pricing_as_of"])).days
    if not 0 <= age <= 7:
        raise ValueError("Refresh provider pricing from official sources (max age 7 days)")
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        for tier in ("cheap", "strong"):
            provider = spec[tier]
            response = await client.get(
                provider["base_url"] + "/models", headers={"Authorization": "Bearer " + keys[tier]}
            )
            if response.status_code != 200:
                raise AcquisitionError(tier, response.status_code, False)
            models = response.json()["data"]
            model = next((m for m in models if m["id"] == provider["model"]), None)
            if model is None:
                raise ValueError("Configured model is unavailable; no completions sent")
            # xAI's /models verifies key-specific access; published prices
            # are a dated snapshot, not an OpenRouter-style pricing catalog.


async def study(
    spec: dict[str, Any],
    out: Path,
    budget: Budget,
    pilot_requests: int | None = None,
    *,
    resume: bool = False,
    interval_seconds: float | None = None,
) -> dict[str, Any]:
    if pilot_requests is not None and (
        type(pilot_requests) is not int or not 10 <= pilot_requests <= 20 or pilot_requests % 2
    ):
        raise ValueError("Pilot requires 10-20 even calls")
    env = dotenv_values(".env")
    keys = {
        tier: os.environ.get(spec[tier]["api_key_env"]) or env.get(spec[tier]["api_key_env"]) or ""
        for tier in ("cheap", "strong")
    }
    if not all(keys.values()):
        raise ValueError("Configure both provider keys locally before executing")
    if resume and pilot_requests:
        raise ResumeError("Resume is supported only for full-study acquisition")
    interval = spec["interval_seconds"] if interval_seconds is None else interval_seconds
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("Invalid pacing")
    rows, manifest, source_config = load_task("gsm8k", Path("data/eval/multibench"))
    retained = []
    signatures: list[frozenset[str]] = []
    for row in rows:
        sig = prompt_signature(row.request, source_config.router.signature_prefix)
        if not any(overlaps(sig, previous) for previous in signatures):
            retained.append(row)
            signatures.append(sig)
    ordered = sorted(retained, key=lambda r: content_hash([spec["seed"], r.request["messages"]]))
    count = (
        pilot_requests // 2
        if pilot_requests
        else sum(spec[k] for k in ("train", "calibration", "test"))
    )
    if len(ordered) < count:
        raise ValueError("Insufficient distinct prompts")
    selected = ordered[:count]
    boundary = count if pilot_requests else spec["train"] + spec["calibration"]
    config = GatewayConfig.model_validate(
        {
            "models": {
                tier: {k: spec[tier][k] for k in ("base_url", "model", "api_key_env")}
                for tier in ("cheap", "strong")
            },
            "router": {
                "kind": "logistic_surface_v1",
                "artifact": str(out / "router.json"),
                "signature_prefix": source_config.router.signature_prefix,
            },
            "calibration": {
                "artifact": str(out / "calibration.json"),
                "allowed_epsilons": [spec["epsilon"]],
                "default_epsilon": spec["epsilon"],
            },
            "quality_metric": {"name": "routellm_gsm8k_last_integer", "version": "1"},
            "scope": {
                "max_chars": 16000,
                "max_tokens": spec["max_tokens"],
                "temperature": 0,
                "top_p": 1,
            },
        }
    )
    splits = {
        "mode": "pilot" if pilot_requests else "full",
        "train": [r.id for r in selected[: spec["train"]]],
        "calibration": [r.id for r in selected[spec["train"] : boundary]],
        "test": [r.id for r in selected[boundary:]],
        "source_sha256": manifest["dataset_sha256"],
        "spec": spec,
    }
    measurements = load_checkpoint(out, splits, budget) if resume else []
    if resume:
        try:
            saved_config = GatewayConfig.model_validate_json((out / "gateway.json").read_bytes())
            if saved_config != config:
                raise ResumeError("Saved gateway scope differs; original files preserved")
        except (OSError, ValueError) as exc:
            raise ResumeError(
                "Saved gateway config is missing or differs; original files preserved"
            ) from exc
    reusable = {(m["id"], m["policy"]): m for m in measurements if m["ok"] and m["cost_known"]}
    execution = {
        "interval_seconds": interval,
        "acquisition_429_retries": 0 if pilot_requests else 3,
        "resumed": resume,
        "reused_acquisition_responses": len(reusable),
        "journal": str(budget.path.resolve()),
    }
    await preflight(spec, keys)
    if not resume:
        write_json(out / "gateway.json", config.model_dump(mode="json"))
        write_json(out / "splits.json", splits)
    with (out / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**execution, "started_at": datetime.now(UTC).isoformat()}) + "\n")
    if resume:
        print(
            f"Reusing {len(reusable)} successful acquisition responses; "
            "prior failed attempts and charges retained.",
            flush=True,
        )
    providers = {
        urlparse(spec[t]["base_url"]).hostname: {**spec[t], "tier": t} for t in ("cheap", "strong")
    }
    meter = MeteredTransport(httpx.AsyncHTTPTransport(retries=0), budget, providers)
    collected: list[CalibrationRow] = []
    gateway_key = secrets.token_urlsafe(32)
    settings = Settings(config_file=out / "gateway.json", api_key=SecretStr(gateway_key))

    def payload(row: Any) -> dict[str, Any]:
        return {
            "messages": row.request["messages"],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": spec["max_tokens"],
            "stream": False,
        }

    async def call(
        client: httpx.AsyncClient,
        url: str,
        body: dict[str, Any],
        key: str,
        identifier: str,
        policy: str,
        phase: str,
    ) -> dict[str, Any]:
        await asyncio.sleep(interval)
        start, index = time.perf_counter(), len(meter.calls)
        output, status, ok = "", None, False
        try:
            response = await client.post(url, json=body, headers={"Authorization": "Bearer " + key})
            status = response.status_code
            if response.is_success:
                value = response.json()["choices"][0]["message"]["content"]
                if isinstance(value, str):
                    output, ok = value, True
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            # Never print upstream body/exception text: it can contain sensitive data.
            pass
        attempts = meter.calls[index:]
        if budget.halted:
            raise ValueError("Unexpected provider charge; halt before reporting complete")
        # Budget exhaustion causes zero attempts. Stop rather than fabricate a provider error.
        if not attempts:
            raise ValueError("No metered attempt; stopped before exceeding budget")
        observation = {
            "id": identifier,
            "policy": policy,
            "phase": phase,
            "status": status,
            "ok": ok,
            "output": output,
            "latency_ms": 1000 * (time.perf_counter() - start),
            "cost_usd": math.fsum(a["cost_usd"] for a in attempts),
            "cost_known": all(a["cost_known"] for a in attempts),
            "attempts": attempts,
        }
        measurements.append(observation)
        with (out / "responses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(observation) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return observation

    # The same metered transport is used for direct policies and gateway upstream requests.
    # Its pool may be closed by gateway lifespan; httpx close is idempotent.
    async with httpx.AsyncClient(transport=meter, timeout=120) as direct:
        for row_index, row in enumerate(selected[:boundary]):
            qualities = {}
            for tier in ("cheap", "strong"):
                observation = reusable.get((row.id, tier + "-only"))
                if observation is None:
                    observation = await acquire_with_retry(
                        partial(
                            call,
                            direct,
                            spec[tier]["base_url"] + "/chat/completions",
                            {**payload(row), "model": spec[tier]["model"]},
                            keys[tier],
                            row.id,
                            tier + "-only",
                            "acquisition",
                        ),
                        tier,
                        retries=0 if pilot_requests else 3,
                    )
                if not observation["ok"] or not observation["cost_known"]:
                    raise AcquisitionError(tier, observation["status"], observation["cost_known"])
                qualities[tier] = float(last_integer(observation["output"]) == row.expected)
                observation["quality"] = qualities[tier]
            collected.append(
                CalibrationRow(
                    request=payload(row), q_cheap=qualities["cheap"], q_strong=qualities["strong"]
                )
            )
            if (row_index + 1) % 10 == 0:
                print(
                    f"Acquisition {row_index + 1}/{boundary} pairs; cumulative "
                    f"charged/reserved ${budget.charged:.4f}.",
                    flush=True,
                )
        if pilot_requests:
            summaries = {}
            for tier in ("cheap", "strong"):
                values = [m for m in measurements if m["policy"] == tier + "-only"]
                summaries[tier + "-only"] = {
                    **summarize_calls(values),
                    "quality": sum(m["quality"] for m in values) / len(values),
                }
            # Average observed rates, with every routed test request conservatively
            # assumed to call both models. This projection is not a billing bound.
            mean_pair_cost = sum(s["cost_usd"] / s["requests"] for s in summaries.values())
            projected = mean_pair_cost * (spec["train"] + spec["calibration"] + 2 * spec["test"])
            return {
                "status": "pilot_complete",
                "created_at": datetime.now(UTC).isoformat(),
                "spec": spec,
                "execution": execution,
                "spec_sha256": content_hash(spec),
                "journal": str(budget.path.resolve()),
                "request_count": pilot_requests,
                "summaries": summaries,
                "upstream_attempts": summarize_calls(meter.calls),
                "observations": measurements,
                "budget_charged_usd_including_prior_runs": budget.charged,
                "pilot_charged_usd": budget.charged - budget.initial_charged,
                "projected_full_study_usd": projected,
                "eligible_for_full_study": projected <= spec["budget_usd"] - budget.charged,
                "limitations": [
                    "Pilot only: no router training, CRC guarantee or CertiRoute comparison.",
                    "Projection uses pilot mean costs; actual charges may differ.",
                    "xAI USD ticks are provider-reported charges; "
                    "token estimates are the fallback.",
                ],
            }
        print("Acquisition complete. Fitting router and calibrating CRC threshold...", flush=True)
        weights = fit_router(
            collected[: spec["train"]],
            config,
            content_hash([r.model_dump() for r in collected[: spec["train"]]]),
            spec["seed"],
        )
        save_weights(weights, config.router.artifact)
        calibration = collected[spec["train"] :]
        artifact = build_artifact(
            calibration,
            config,
            DataProvenance(
                sources=["fresh-live-pairs"],
                sha256=content_hash([r.model_dump() for r in calibration]),
                collected_at=datetime.now(UTC),
            ),
        )
        save_artifact(artifact, config.calibration.artifact)
        print(
            f"Calibration complete. Starting held-out testing: {spec['test']} prompts "
            f"x 3 policies; {interval:g}s pacing before each request. "
            "The study is not finished until the final measurements-saved message.",
            flush=True,
        )
        app = create_app(settings, transport=meter)
        # create_app reads the same local keys from env/.env; none are written to config.
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway.local", timeout=240
            ) as gateway:
                rng = random.Random(spec["seed"])
                for test_index, row in enumerate(selected[boundary:], start=1):
                    order = ["cheap", "strong", "auto"]
                    rng.shuffle(order)
                    observed = {}
                    for tier in order:
                        if tier == "auto":
                            item = await call(
                                gateway,
                                "/v1/chat/completions",
                                {**payload(row), "model": "certiroute/auto"},
                                gateway_key,
                                row.id,
                                "CertiRoute",
                                "test",
                            )
                        else:
                            item = await call(
                                direct,
                                spec[tier]["base_url"] + "/chat/completions",
                                {**payload(row), "model": spec[tier]["model"]},
                                keys[tier],
                                row.id,
                                tier + "-only",
                                "test",
                            )
                        item["quality"] = float(
                            item["ok"] and last_integer(item["output"]) == row.expected
                        )
                        observed[tier] = item
                        print(
                            f"Test {test_index}/{spec['test']} | {item['policy']} | "
                            f"HTTP {item['status']} | cumulative charged/reserved "
                            f"${budget.charged:.4f}.",
                            flush=True,
                        )
                    for item in observed.values():
                        item["regret"] = (
                            max(0.0, observed["strong"]["quality"] - item["quality"])
                            if observed["strong"]["ok"]
                            else None
                        )
    summaries = {}
    for policy in ("cheap-only", "strong-only", "CertiRoute"):
        values = [m for m in measurements if m["phase"] == "test" and m["policy"] == policy]
        regrets = [v["regret"] for v in values if v["regret"] is not None]
        summaries[policy] = {
            **summarize_calls(values),
            "quality": sum(v["quality"] for v in values) / len(values),
            "regret": sum(regrets) / len(regrets) if regrets else None,
            "regret_denominator": len(regrets),
        }
    result = {
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "spec": spec,
        "execution": execution,
        "summaries": summaries,
        "acquisition": summarize_calls([m for m in measurements if m["phase"] == "acquisition"]),
        "budget_charged_usd_including_prior_runs": budget.charged,
        "upstream_attempts": summarize_calls([a for m in measurements for a in m["attempts"]]),
        "observations": measurements,
        "calibration_id": artifact.calibration_id,
        "code": code_provenance(),
        "limitations": [
            "Full acquisition retries HTTP 429 up to 3 times; all attempts retained. "
            "Test requests are not retried; gateway fallback remains enabled.",
            "Concurrency=1; nonstreaming latency excludes pacing and acquisition backoff.",
            "Gateway runs in-process over ASGI; excludes network ingress latency.",
            "Unknown costs retain full reservation; costs are not invoices.",
            "Regret excludes prompts whose strong reference request failed.",
            "One fresh calibration draw is descriptive, not a CRC validity test.",
        ],
    }
    return result


def validate_pilot_report(report: dict[str, Any], spec: dict[str, Any], charged: float) -> None:
    """Fail closed unless the same pair/scope passed a complete, affordable pilot."""
    if (
        report.get("status") != "pilot_complete"
        or report.get("spec_sha256") != content_hash(spec)
        or report.get("request_count") != spec["pilot_requests"]
        or report.get("eligible_for_full_study") is not True
    ):
        raise ValueError("A successful matching pilot report is required before the full study")
    summaries = report.get("summaries", {})
    for policy in ("cheap-only", "strong-only"):
        s = summaries.get(policy, {})
        if (
            s.get("requests") != spec["pilot_requests"] // 2
            or s.get("error_rate") != 0
            or s.get("unknown_cost_requests") != 0
        ):
            raise ValueError("The pilot must have successful, metered requests for both providers")
    projected = report.get("projected_full_study_usd")
    prior = report.get("budget_charged_usd_including_prior_runs")
    if (
        not isinstance(projected, (int, float))
        or not math.isfinite(projected)
        or projected < 0
        or projected > spec["budget_usd"] - charged
        or not isinstance(prior, (int, float))
        or not math.isfinite(prior)
        or prior < 0
        or prior > charged
    ):
        raise ValueError(
            "The pilot projection exceeds the remaining budget or journal is inconsistent"
        )


def save_run_result(out: Path, result: dict[str, Any]) -> None:
    """Preserve old incomplete reports while atomically replacing the latest result."""
    pending = out / f"results-pending-{time.time_ns()}.json"
    write_json(pending, result)
    target = out / "results.json"
    if target.exists():
        (out / f"results-before-resume-{time.time_ns()}.json").write_bytes(target.read_bytes())
    pending.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/live.study.json"))
    parser.add_argument("--out", type=Path, default=Path("results/live-measurement"))
    parser.add_argument("--journal", type=Path, default=Path("build/live-budget.jsonl"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful full-run acquisition responses in --out",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=15,
        help="Delay before each logical request (default: 15); excludes latency",
    )
    parser.add_argument(
        "--confirm-account-pricing",
        action="store_true",
        help="Full run only: confirm account rates/debits do not exceed pilot estimates, "
        "and account fees fit the $0.49 reserve",
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        help="Successful matching pilot results.json; required for full execution",
    )
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds < 0:
        parser.error("--interval-seconds must be finite and nonnegative")
    if args.resume and (args.mode != "full" or not args.out.is_dir()):
        parser.error("--resume requires --mode full and the original existing --out folder")
    spec = json.loads(args.config.read_bytes())
    try:
        validate_spec(spec)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.execute:
        count = (
            spec["pilot_requests"]
            if args.mode == "pilot"
            else 2 * (spec["train"] + spec["calibration"]) + 3 * spec["test"]
        )
        print(
            f"NOT RUN: {args.mode}; {count} planned provider calls"
            + (" plus possible gateway fallback attempts. " if args.mode == "full" else ". ")
            + f"Pilot cap ${spec['pilot_budget_usd']:.2f}; cumulative API ceiling "
            + f"${spec['budget_usd']:.2f}; $0.49 retained for fees. "
            + "Use --execute after configuring local keys. Full execution requires --pilot-report."
        )
        return
    if args.mode == "full" and args.pilot_report is None:
        parser.error("Full execution requires --pilot-report from a successful pilot")
    if args.mode == "full" and not args.confirm_account_pricing:
        parser.error(
            "Review pilot account charges and rates, then use --confirm-account-pricing; "
            "published prices alone do not establish your account's charges"
        )
    if args.out.exists() and not args.resume:
        parser.error(
            "Output folder already exists; choose a new --out and preserve the budget journal"
        )
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    lock = args.journal.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        budget = Budget(
            args.journal,
            spec["budget_usd"],
            run_limit=spec["pilot_budget_usd"] if args.mode == "pilot" else None,
        )
        if args.mode == "full":
            report = json.loads(args.pilot_report.read_bytes())
            if report.get("journal") != str(args.journal.resolve()):
                parser.error("Preserve the pilot's budget journal for full execution")
            try:
                validate_pilot_report(report, spec, budget.charged)
            except ValueError as exc:
                parser.error(str(exc))
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
        initial_attempts = len(budget.entries)
        try:
            result = asyncio.run(
                study(
                    spec,
                    args.out,
                    budget,
                    pilot_requests=spec["pilot_requests"] if args.mode == "pilot" else None,
                    resume=args.resume,
                    interval_seconds=args.interval_seconds,
                )
            )
        except ResumeError as exc:
            raise SystemExit(str(exc)) from None
        except Exception as exc:
            failure = exc.diagnostic if isinstance(exc, AcquisitionError) else None
            # Rejected setup must not relabel previous evidence (including a completed
            # report). A resume can replace it only after making a new metered attempt.
            if not args.resume or len(budget.entries) > initial_attempts:
                save_run_result(
                    args.out,
                    {
                        "status": "incomplete",
                        "error_type": type(exc).__name__,
                        "failure": failure,
                        "budget_charged_usd": budget.charged,
                        "note": (
                            "Inspect local configuration and journal; no live result is claimed."
                        ),
                    },
                )
            raise SystemExit(
                (str(exc) + " " if isinstance(exc, AcquisitionError) else "")
                + "Study incomplete; saved partial evidence. No secrets logged."
            ) from None
        result["account_pricing_confirmed_by_operator"] = (
            args.mode == "full" and args.confirm_account_pricing
        )
        save_run_result(args.out, result)
        print(f"Live measurements saved to {args.out}; charged/reserved ${budget.charged:.4f}")
        if args.mode == "pilot":
            print(
                f"Pilot complete; projected full study ${result['projected_full_study_usd']:.4f}. "
                f"Eligible: {result['eligible_for_full_study']}. "
                "Stopped for review; no continuation."
            )
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
