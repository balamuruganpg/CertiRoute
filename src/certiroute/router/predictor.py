"""Learned logistic predictor with JSON weights and no runtime ML dependency."""

import math
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from certiroute.config import ConfigModel, GatewayConfig, MetricConfig, RouterConfig, ScopeConfig
from certiroute.provenance import content_hash, model_identity, overlaps, prompt_signature
from certiroute.router.features import FEATURE_NAMES


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


class Predictor(Protocol):
    identity: dict[str, str]

    def score(self, features: list[float]) -> float: ...

    def check_independent(self, request: dict[str, object]) -> None: ...


class LearnedWeights(ConfigModel):
    version: Literal["logistic_surface_v1"] = "logistic_surface_v1"
    sha256: str
    feature_names: tuple[str, ...] = FEATURE_NAMES
    coefficients: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    training_signatures: tuple[tuple[str, ...], ...]
    training_data_sha256: str
    seed: int
    iterations: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    l2: float = Field(ge=0)
    models: dict[str, str]
    quality_metric: MetricConfig
    scope: ScopeConfig
    signature_prefix: str = ""

    @model_serializer(mode="wrap")
    def serialize_weights(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if not self.signature_prefix:
            result.pop("signature_prefix", None)
        return result

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json", exclude={"sha256"}))

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("Feature version mismatch.")
        if any(
            len(values) != len(FEATURE_NAMES)
            for values in (self.coefficients, self.means, self.scales)
        ) or any(s <= 0 for s in self.scales):
            raise ValueError("Invalid feature dimensions or scales.")
        if not self.training_signatures or self.sha256 != self.digest():
            raise ValueError("Missing training provenance or model hash mismatch.")
        return self


class LearnedPredictor:
    def __init__(self, weights: LearnedWeights) -> None:
        self.weights = weights
        self.identity = {"version": weights.version, "sha256": weights.sha256}
        self.training_signatures = tuple(frozenset(s) for s in weights.training_signatures)

    def score(self, features: list[float]) -> float:
        values = [
            coef * (x - mean) / scale
            for x, coef, mean, scale in zip(
                features,
                self.weights.coefficients,
                self.weights.means,
                self.weights.scales,
                strict=True,
            )
        ]
        return sigmoid(self.weights.intercept + math.fsum(values))

    def check_independent(self, request: dict[str, object]) -> None:
        signature = prompt_signature(request, self.weights.signature_prefix)
        if any(overlaps(signature, train) for train in self.training_signatures):
            raise ValueError("Calibration/test prompt overlaps router training data.")


class LengthPredictor:
    """Legacy explicit baseline, retained to read earlier fixture configurations."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self.identity = {
            "version": "length_v1",
            "sha256": content_hash({"kind": "length_v1", "length_scale": config.length_scale}),
        }

    def score(self, features: list[float]) -> float:
        return 1.0 / (1.0 + features[0] / self.config.length_scale)

    def check_independent(self, request: dict[str, object]) -> None:
        pass


def load_predictor(config: GatewayConfig) -> Predictor:
    if config.router.kind == "length_v1":
        return LengthPredictor(config.router)
    weights = LearnedWeights.model_validate_json(config.router.artifact.read_bytes())
    if (
        weights.models != model_identity(config)
        or weights.scope != config.scope
        or weights.quality_metric != config.quality_metric
        or weights.signature_prefix != config.router.signature_prefix
    ):
        raise ValueError("Learned router model/metric/scope mismatch.")
    return LearnedPredictor(weights)


def save_weights(weights: LearnedWeights, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(weights.model_dump_json(indent=2) + "\n")
