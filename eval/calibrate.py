"""Calibrate from frozen, scored paired outputs; makes no provider calls."""

import argparse
import hashlib
from pathlib import Path

from pydantic import AwareDatetime, Field

from certiroute.calibration.artifact import (
    CalibrationRow,
    DataProvenance,
    build_artifact,
    model_identity,
    save_artifact,
)
from certiroute.config import ConfigModel, MetricConfig, ScopeConfig, load_config


class PairedDataset(ConfigModel):
    models: dict[str, str]
    quality_metric: MetricConfig
    scope: ScopeConfig
    sources: list[str] = Field(min_length=1)
    collected_at: AwareDatetime
    rows: list[CalibrationRow] = Field(min_length=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gateway.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="Frozen paired-output JSON")
    parser.add_argument("--out", type=Path, required=True, help="New immutable artifact path")
    parser.add_argument("--valid-days", type=int, default=7)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        raw = args.input.read_bytes()
        dataset = PairedDataset.model_validate_json(raw)
        if (
            dataset.models != model_identity(config)
            or dataset.quality_metric != config.quality_metric
            or dataset.scope != config.scope
        ):
            raise ValueError("Dataset model identities, metric or scope do not match config.")
        provenance = DataProvenance(
            sources=dataset.sources,
            sha256=hashlib.sha256(raw).hexdigest(),
            collected_at=dataset.collected_at,
        )
        artifact = build_artifact(dataset.rows, config, provenance, valid_days=args.valid_days)
        save_artifact(artifact, args.out)
    except (OSError, ValueError) as exc:
        # Validation details can contain prompt text: do not echo them to logs.
        parser.exit(
            2, f"Calibration failed ({type(exc).__name__}); check inputs and output path.\n"
        )
    print(f"Saved {artifact.calibration_id} (n={artifact.n}) to {args.out}")


if __name__ == "__main__":
    main()
