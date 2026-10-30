"""Shared multi-benchmark records and deterministic scoring helpers."""

import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import Field

from certiroute.calibration.artifact import CalibrationRow
from eval.fixtures import TokenCounts


def answer_letter(output: str) -> str | None:
    text = output.strip()
    match = re.fullmatch(r"\(?([A-D])\)?[.)]?", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"(?:The answer is|Answer:)\s*\(?([A-D])\)?[.]?", text)
    return match.group(1) if match else None


def extract_python(output: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", output, re.DOTALL)
    return "\n".join(blocks) if blocks else output


class BenchmarkRow(CalibrationRow):
    id: str
    task: str
    expected: str = ""
    cheap_output: str
    strong_output: str
    cheap_usage: TokenCounts
    strong_usage: TokenCounts
    tests: list[str] = Field(default_factory=list)
    test_setup: str = ""


def read_benchmark(path: Path) -> tuple[list[BenchmarkRow], dict[str, Any]]:
    raw = path.read_bytes()
    manifest: dict[str, Any] = json.loads(path.with_suffix(".manifest.json").read_bytes())
    if hashlib.sha256(raw).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    rows = [BenchmarkRow.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    if not rows:
        raise ValueError("Empty dataset")
    return rows, manifest


def portfolio_metrics(tasks: list[dict[str, float]]) -> dict[str, float]:
    """Equal task weights; input costs are per request, not per differently sized test set."""
    if not tasks or any(t["strong_cost_usd"] <= 0 for t in tasks):
        raise ValueError("Positive strong costs required")
    result = {
        key: mean(t[key] for t in tasks)
        for key in ("quality", "regret", "cost_usd", "strong_cost_usd")
    }
    result["savings_vs_strong"] = 1 - result["cost_usd"] / result["strong_cost_usd"]
    return result
