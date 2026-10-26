"""Hash-bound provenance for public paired-output snapshots."""

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from certiroute.config import ConfigModel, GatewayConfig, MetricConfig, ScopeConfig
from certiroute.provenance import model_identity


class SourceFile(ConfigModel):
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(ConfigModel):
    source: Literal["routellm_gsm8k_v1"]
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at: AwareDatetime
    models: dict[str, str]
    quality_metric: MetricConfig
    scope: ScopeConfig
    source_files: dict[str, SourceFile]
    selection: str
    selected_source_indices: list[int]
    prompt_template: str
    generation: dict[str, Any]
    token_counts: str
    limitations: list[str]
    signature_prefix: str = ""
    excluded_source_indices: list[int] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_manifest(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if not self.signature_prefix:
            result.pop("signature_prefix", None)
            if not self.excluded_source_indices:
                result.pop("excluded_source_indices", None)
        return result


def read_manifest(path: Path, config: GatewayConfig) -> DatasetManifest:
    manifest = DatasetManifest.model_validate_json(
        path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    if manifest.dataset_sha256 != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Dataset checksum does not match manifest.")
    if (
        manifest.models != model_identity(config)
        or manifest.scope != config.scope
        or manifest.quality_metric != config.quality_metric
        or manifest.signature_prefix != config.router.signature_prefix
    ):
        raise ValueError("Dataset model identities, metric or generation scope do not match.")
    return manifest
