"""Strict threshold routing with conservative scope and artifact failure handling."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from certiroute.calibration.artifact import CalibrationArtifact
from certiroute.config import GatewayConfig
from certiroute.router.features import extract_features
from certiroute.router.predictor import Predictor, load_predictor
from certiroute.router.scope import in_calibrated_scope


@dataclass(frozen=True)
class RoutingDecision:
    tier: Literal["cheap", "strong"]
    epsilon: float
    reason: str
    scope: Literal["in", "out", "bypass"] = "out"
    score: float | None = None
    calibration_id: str | None = None
    degraded: bool = False


def route(
    request: dict[str, object],
    *,
    epsilon: float,
    config: GatewayConfig,
    artifact: CalibrationArtifact | None,
    predictor: Predictor | None = None,
) -> RoutingDecision:
    if artifact is None:
        return RoutingDecision("strong", epsilon, "calibration_unavailable")
    try:
        predictor = predictor or load_predictor(config)
        artifact.validate_runtime(config, predictor)
        if not in_calibrated_scope(request, config.scope):
            return RoutingDecision("strong", epsilon, "out_of_scope")
        score = predictor.score(extract_features(request))
        if not isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid router score.")
        tier: Literal["cheap", "strong"] = (
            "cheap" if score > artifact.threshold(epsilon) else "strong"
        )
        return RoutingDecision(tier, epsilon, "calibrated", "in", score, artifact.calibration_id)
    except Exception:
        # Never expose request text/secrets or turn a router failure into cheap traffic.
        return RoutingDecision("strong", epsilon, "routing_unavailable")
