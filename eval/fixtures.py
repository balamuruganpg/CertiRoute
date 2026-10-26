"""Validated paired outputs, deterministic quality scoring and explicit pricing."""

import ast
import re
from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from certiroute.calibration.artifact import CalibrationRow
from certiroute.config import ConfigModel, GatewayConfig
from certiroute.provenance import overlaps, prompt_signature
from certiroute.router.scope import in_calibrated_scope, with_generation_defaults


class TokenCounts(ConfigModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class FixtureRow(CalibrationRow):
    id: str
    source: Literal["synthetic_arithmetic_v1", "routellm_gsm8k_v1"]
    expected: str
    cheap_output: str
    strong_output: str
    cheap_usage: TokenCounts
    strong_usage: TokenCounts

    def quality(self, output: str) -> float:
        if self.source == "routellm_gsm8k_v1":
            return float(last_integer(output) == self.expected)
        return float(output.strip() == self.expected.strip())

    @model_validator(mode="after")
    def check_labels(self) -> Self:
        if self.q_cheap != self.quality(self.cheap_output):
            raise ValueError("Cheap quality disagrees with output scoring.")
        if self.q_strong != self.quality(self.strong_output):
            raise ValueError("Strong quality disagrees with output scoring.")
        return self

    def calibration_row(self) -> CalibrationRow:
        return CalibrationRow(request=self.request, q_cheap=self.q_cheap, q_strong=self.q_strong)


class TokenPrice(ConfigModel):
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)

    def cost(self, usage: TokenCounts) -> float:
        return (
            usage.prompt_tokens * self.input_per_million
            + usage.completion_tokens * self.output_per_million
        ) / 1_000_000


def last_integer(output: str) -> str | None:
    """RouteLLM's GSM8K scorer: last unsigned integer after removing commas.

    This deliberately preserves the upstream metric, including its limitations
    for decimals and negative answers. It is not a general numeric verifier.
    """
    numbers = re.findall(r"\d+", output.replace(",", ""))
    if not numbers:
        return None
    try:
        return str(ast.literal_eval(numbers[-1]))
    except (SyntaxError, ValueError):
        return None


class Pricing(ConfigModel):
    as_of: date
    currency: Literal["USD"]
    synthetic: bool
    description: str = "Invented fixture rates; not provider prices."
    source: str | None = None
    models: dict[str, TokenPrice]


def read_pricing(path: Path, config: GatewayConfig) -> Pricing:
    pricing = Pricing.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if set(pricing.models) != {config.models.cheap.model, config.models.strong.model}:
        raise ValueError("Pricing model IDs do not match config.")
    return pricing


def read_fixture(path: Path, config: GatewayConfig) -> tuple[list[FixtureRow], list[str]]:
    """Deduplicate before splitting, without examining labels or outcomes."""
    rows: list[FixtureRow] = []
    signatures: list[frozenset[str]] = []
    dropped: list[str] = []
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = FixtureRow.model_validate_json(line)
        metric = (
            "exact_match"
            if row.source == "synthetic_arithmetic_v1"
            else "routellm_gsm8k_last_integer"
        )
        if config.quality_metric.name != metric or config.quality_metric.version != "1":
            raise ValueError("Dataset quality metric does not match config.")
        if row.id in ids:
            raise ValueError("Duplicate record ID.")
        ids.add(row.id)
        request = with_generation_defaults(row.request, config.scope)
        if not in_calibrated_scope(request, config.scope):
            raise ValueError("Fixture contains an out-of-scope prompt.")
        signature = prompt_signature(request, config.router.signature_prefix)
        if any(overlaps(signature, previous) for previous in signatures):
            dropped.append(row.id)
        else:
            rows.append(row)
            signatures.append(signature)
    if len(rows) < 10:
        raise ValueError("Need at least ten distinct fixture prompts.")
    if len({row.source for row in rows}) != 1:
        raise ValueError("Mixed dataset sources are not supported.")
    return rows, dropped
