"""Content-addressed CRC artifacts bound to the serving policy configuration."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from certiroute.calibration.crc import crc_lambda
from certiroute.calibration.losses import regret
from certiroute.config import ConfigModel, GatewayConfig, MetricConfig, ScopeConfig
from certiroute.provenance import content_hash as content_hash
from certiroute.provenance import model_identity as model_identity
from certiroute.provenance import overlaps, prompt_signature
from certiroute.router.features import extract_features
from certiroute.router.predictor import Predictor, load_predictor
from certiroute.router.scope import in_calibrated_scope, with_generation_defaults


def router_identity(config: GatewayConfig, predictor: Predictor | None = None) -> dict[str, str]:
    return (predictor or load_predictor(config)).identity


class CalibrationRow(ConfigModel):
    request: dict[str, object]
    q_strong: float = Field(ge=0, le=1)
    q_cheap: float = Field(ge=0, le=1)


class DataProvenance(ConfigModel):
    sources: list[str] = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    collected_at: AwareDatetime


class CalibrationArtifact(ConfigModel):
    schema_version: Literal[1] = 1
    calibration_id: str
    method: Literal["crc"] = "crc"
    loss: Literal["clipped_regret_v1"] = "clipped_regret_v1"
    B: float = Field(default=1.0, ge=1, le=1)
    n: int = Field(gt=0)
    lambda_by_epsilon: dict[str, float] = Field(min_length=1)
    router: dict[str, str]
    models: dict[str, str]
    quality_metric: MetricConfig
    scope: ScopeConfig
    data: DataProvenance
    created_at: AwareDatetime
    expires_at: AwareDatetime

    def digest(self) -> str:
        return "cal_" + content_hash(self.model_dump(mode="json", exclude={"calibration_id"}))

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        grid = []
        for key, threshold in self.lambda_by_epsilon.items():
            epsilon = float(key)
            if not 0 < epsilon <= 1 or key != str(epsilon) or not 0 <= threshold <= 1:
                raise ValueError("Invalid epsilon key or threshold.")
            if epsilon < 1 / (self.n + 1) and threshold != 1:
                raise ValueError("Too few calibration points for a nontrivial threshold.")
            grid.append((epsilon, threshold))
        ordered = sorted(grid)
        if any(left[1] < right[1] for left, right in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("Thresholds must not increase as epsilon increases.")
        if self.expires_at <= self.created_at:
            raise ValueError("Expiry must follow creation.")
        if self.calibration_id != self.digest():
            raise ValueError("Calibration content hash mismatch.")
        return self

    def threshold(self, epsilon: float) -> float:
        return self.lambda_by_epsilon[str(epsilon)]

    def validate_runtime(self, config: GatewayConfig, predictor: Predictor | None = None) -> None:
        now = datetime.now(UTC)
        if not self.created_at <= now < self.expires_at:
            raise ValueError("Calibration is expired or future-dated.")
        if self.models != model_identity(config) or self.router != router_identity(
            config, predictor
        ):
            raise ValueError("Model or router identity mismatch.")
        if self.quality_metric != config.quality_metric or self.scope != config.scope:
            raise ValueError("Metric or generation/scope settings mismatch.")
        if set(self.lambda_by_epsilon) != {str(e) for e in config.calibration.allowed_epsilons}:
            raise ValueError("Calibrated epsilon grid does not match config.")


def load_artifact(
    path: Path, config: GatewayConfig, predictor: Predictor | None = None
) -> CalibrationArtifact:
    artifact = CalibrationArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    artifact.validate_runtime(config, predictor)
    return artifact


def build_artifact(
    rows: list[CalibrationRow],
    config: GatewayConfig,
    provenance: DataProvenance,
    *,
    valid_days: int = 7,
) -> CalibrationArtifact:
    """Calibrate a fixed scorer from paired observations, never from test outcomes."""
    if not rows or valid_days <= 0:
        raise ValueError("Nonempty calibration rows and positive validity days are required.")
    predictor = load_predictor(config)
    scores, regrets = [], []
    seen: set[str] = set()
    signatures: list[frozenset[str]] = []
    for row in rows:
        request = with_generation_defaults(row.request, config.scope)
        if not in_calibrated_scope(request, config.scope):
            raise ValueError("Calibration request is outside the configured scope.")
        predictor.check_independent(request)
        prompt_hash = content_hash(request["messages"])
        signature = prompt_signature(request, config.router.signature_prefix)
        if prompt_hash in seen or any(overlaps(signature, old) for old in signatures):
            raise ValueError("Duplicate calibration prompt; deduplicate before calibration.")
        seen.add(prompt_hash)
        signatures.append(signature)
        scores.append(predictor.score(extract_features(request)))
        regrets.append(regret(row.q_strong, row.q_cheap))
    now = datetime.now(UTC)
    fields: dict[str, object] = {
        "schema_version": 1,
        "method": "crc",
        "loss": "clipped_regret_v1",
        "B": 1.0,
        "n": len(rows),
        "lambda_by_epsilon": {
            str(e): crc_lambda(scores, regrets, e) for e in config.calibration.allowed_epsilons
        },
        "router": router_identity(config, predictor),
        "models": model_identity(config),
        "quality_metric": config.quality_metric.model_dump(mode="json"),
        "scope": config.scope.model_dump(mode="json"),
        "data": provenance.model_dump(mode="json"),
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(days=valid_days)).isoformat().replace("+00:00", "Z"),
    }
    fields["calibration_id"] = "cal_" + content_hash(fields)
    return CalibrationArtifact.model_validate(fields)


def save_artifact(artifact: CalibrationArtifact, path: Path) -> None:
    """Write a new artifact only; existing artifacts are immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(artifact.model_dump_json(indent=2) + "\n")
