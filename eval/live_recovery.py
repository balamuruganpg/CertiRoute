"""Bounded acquisition retries and validated reuse of already paid responses."""

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from eval.live import Budget


class ResumeError(ValueError):
    """Safe local diagnostic; never includes upstream text or credentials."""


async def acquire_with_retry(
    operation: Callable[[], Awaitable[dict[str, Any]]], tier: str, retries: int
) -> dict[str, Any]:
    """Retry HTTP 429 only during full-run acquisition; every attempt stays logged."""
    for attempt in range(retries + 1):
        observation = await operation()
        if observation["status"] != 429 or attempt == retries:
            return observation
        retry_after = max(a.get("retry_after_seconds", 0) for a in observation["attempts"])
        delay = max(60 * 2**attempt, retry_after)
        if delay > 300:
            # Long quota windows need operator review, not rapid retry loops.
            return observation
        print(
            f"{tier}: HTTP 429; waiting {delay:.0f}s before acquisition retry "
            f"{attempt + 1}/{retries}.",
            flush=True,
        )
        while delay > 0:
            await asyncio.sleep(min(delay, 60))
            delay -= min(delay, 60)
    raise AssertionError("Unreachable retry state")


def load_checkpoint(out: Path, splits: dict[str, Any], budget: Budget) -> list[dict[str, Any]]:
    """Accept an acquisition prefix only, with every charge in the shared ledger.

    Failed attempts do not advance the prefix. Reuse depends on transport success,
    never answer quality. No test observations may have been collected yet.
    """
    try:
        if (out / "router.json").exists() or (out / "calibration.json").exists():
            raise ResumeError("Resume is limited to acquisition before artifacts or test execution")
        if json.loads((out / "splits.json").read_bytes()) != splits:
            raise ResumeError("Resume requires identical data, splits, models and study config")
        if (out / "results.json").exists():
            if json.loads((out / "results.json").read_bytes()).get("status") != "incomplete":
                raise ResumeError("Cannot resume a completed study")
        path = out / "responses.jsonl"
        observations = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expected = [
            (identifier, tier + "-only")
            for identifier in splits["train"] + splits["calibration"]
            for tier in ("cheap", "strong")
        ]
        cursor, seen = 0, set()
        for item in observations:
            if (
                item["phase"] != "acquisition"
                or cursor >= len(expected)
                or (item["id"], item["policy"]) != expected[cursor]
            ):
                raise ResumeError(
                    "Resume supports acquisition prefixes only; test data cannot be reused"
                )
            if not isinstance(item["output"], str) or not item["attempts"]:
                raise ResumeError("Invalid saved acquisition response")
            for attempt in item["attempts"]:
                ident = attempt["id"]
                if (
                    ident in seen
                    or ident not in budget.settled
                    or attempt["tier"] + "-only" != item["policy"]
                    or budget.entries[ident] != attempt["cost_usd"]
                ):
                    raise ResumeError("Saved responses do not match the original budget journal")
                seen.add(ident)
            if not math.isclose(
                item["cost_usd"],
                sum(a["cost_usd"] for a in item["attempts"]),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ResumeError("Saved response costs are inconsistent")
            if item["ok"] and item["cost_known"]:
                if item["status"] != 200:
                    raise ResumeError("Invalid successful acquisition status")
                cursor += 1
        return observations
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ResumeError(
            "Missing or malformed acquisition checkpoint; original files preserved"
        ) from exc
