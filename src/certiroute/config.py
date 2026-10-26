"""Environment secrets and validated YAML configuration; no network side effects."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CERTIROUTE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    config_file: Path = Path("configs/gateway.yaml")
    api_key: SecretStr = Field(default=SecretStr("certiroute-local"), min_length=1)
    admin_api_key: SecretStr | None = None


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProviderConfig(ConfigModel):
    base_url: HttpUrl
    model: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    connect_timeout: float = Field(default=5, gt=0)
    read_timeout: float = Field(default=60, gt=0)
    total_timeout: float = Field(default=120, gt=0)


class ModelsConfig(ConfigModel):
    cheap: ProviderConfig
    strong: ProviderConfig


class CalibrationConfig(ConfigModel):
    artifact: Path
    allowed_epsilons: list[float] = Field(min_length=1)
    default_epsilon: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if any(not 0 < epsilon <= 1 for epsilon in self.allowed_epsilons):
            raise ValueError("Allowed epsilons must be finite and in (0, 1].")
        if len(set(self.allowed_epsilons)) != len(self.allowed_epsilons):
            raise ValueError("Allowed epsilons must be unique.")
        if self.default_epsilon not in self.allowed_epsilons:
            raise ValueError("Default epsilon must belong to the allowed grid.")
        return self


class AuditConfig(ConfigModel):
    sample_rate: float = Field(default=0.02, ge=0, le=1)


class SafetyConfig(ConfigModel):
    fail_open_to: Literal["strong"] = "strong"
    log_prompt_text: bool = False


class RouterConfig(ConfigModel):
    kind: Literal["length_v1", "logistic_surface_v1"] = "logistic_surface_v1"
    artifact: Path = Path("artifacts/router.json")
    length_scale: float = Field(default=1000, gt=0)
    signature_prefix: str = ""

    @model_validator(mode="after")
    def bind_signature_rule(self) -> Self:
        if self.signature_prefix and self.kind != "logistic_surface_v1":
            raise ValueError("A signature prefix requires a learned, hash-bound router.")
        return self


class ScopeConfig(ConfigModel):
    max_chars: int = Field(default=16000, gt=0)
    temperature: Literal[0] = 0
    max_tokens: int = Field(default=256, gt=0)
    top_p: Literal[1] = 1
    stop: list[str] | None = Field(default=None, min_length=1, max_length=4)

    @model_serializer(mode="wrap")
    def serialize_scope(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        # Keep hashes of existing no-stop artifacts compatible.
        result: dict[str, object] = handler(self)
        if self.stop is None:
            result.pop("stop", None)
        return result


class MetricConfig(ConfigModel):
    name: str = "exact_match"
    version: str = "1"


class GatewayConfig(ConfigModel):
    models: ModelsConfig
    calibration: CalibrationConfig
    audit: AuditConfig = Field(default_factory=AuditConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    quality_metric: MetricConfig = Field(default_factory=MetricConfig)


def load_config(path: Path) -> GatewayConfig:
    """Reject missing/malformed config rather than silently changing policy."""
    with path.open(encoding="utf-8") as source:
        return GatewayConfig.model_validate(yaml.safe_load(source))
