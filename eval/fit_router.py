"""Fit a small logistic classifier on D_router only; no external dependencies/calls."""

import argparse
import hashlib
import math
from pathlib import Path

from certiroute.calibration.artifact import CalibrationRow
from certiroute.config import GatewayConfig, load_config
from certiroute.provenance import content_hash, model_identity, overlaps, prompt_signature
from certiroute.router.features import FEATURE_NAMES, extract_features
from certiroute.router.predictor import LearnedWeights, save_weights, sigmoid
from certiroute.router.scope import in_calibrated_scope, with_generation_defaults
from eval.calibrate import PairedDataset


def fit_router(
    rows: list[CalibrationRow],
    config: GatewayConfig,
    data_sha256: str,
    seed: int = 42,
    *,
    iterations: int = 1500,
    learning_rate: float = 0.1,
    l2: float = 0.01,
) -> LearnedWeights:
    """Standardized features + L2 logistic loss, deterministic full-batch descent.

    Positive label means zero clipped regret, including cases where cheap wins.
    All scaler statistics and coefficients are fitted exclusively on these rows.
    Hyperparameters are fixed in advance; no calibration/test feedback is used.
    """
    if iterations <= 0 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid optimization settings.")
    features, labels = [], []
    signatures: list[frozenset[str]] = []
    for row in rows:
        request = with_generation_defaults(row.request, config.scope)
        if not in_calibrated_scope(request, config.scope):
            raise ValueError("Training request is outside the configured scope.")
        signature = prompt_signature(request, config.router.signature_prefix)
        if any(overlaps(signature, previous) for previous in signatures):
            raise ValueError("Duplicate or near-duplicate training prompt.")
        signatures.append(signature)
        features.append(extract_features(request))
        labels.append(int(row.q_cheap >= row.q_strong))
    if set(labels) != {0, 1}:
        raise ValueError("Training needs both zero-regret and positive-regret examples.")
    n, p = len(features), len(FEATURE_NAMES)
    means = [math.fsum(x[j] for x in features) / n for j in range(p)]
    scales = [
        math.sqrt(math.fsum((x[j] - means[j]) ** 2 for x in features) / n) or 1.0 for j in range(p)
    ]
    standardized = [[(x[j] - means[j]) / scales[j] for j in range(p)] for x in features]
    coefficients, intercept = [0.0] * p, 0.0
    for _ in range(iterations):
        errors = [
            sigmoid(intercept + math.fsum(c * v for c, v in zip(coefficients, x, strict=True))) - y
            for x, y in zip(standardized, labels, strict=True)
        ]
        coefficients = [
            coefficients[j]
            - learning_rate
            * (
                math.fsum(e * x[j] for e, x in zip(errors, standardized, strict=True)) / n
                + l2 * coefficients[j]
            )
            for j in range(p)
        ]
        intercept -= learning_rate * math.fsum(errors) / n
    fields: dict[str, object] = {
        "version": "logistic_surface_v1",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": coefficients,
        "intercept": intercept,
        "means": means,
        "scales": scales,
        "training_signatures": [sorted(s) for s in signatures],
        "training_data_sha256": data_sha256,
        "seed": seed,
        "iterations": iterations,
        "learning_rate": learning_rate,
        "l2": l2,
        "models": model_identity(config),
        "quality_metric": config.quality_metric.model_dump(),
        "scope": config.scope.model_dump(),
    }
    if config.router.signature_prefix:
        fields["signature_prefix"] = config.router.signature_prefix
    fields["sha256"] = content_hash(fields)
    return LearnedWeights.model_validate(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gateway.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="D_router PairedDataset JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    raw = args.input.read_bytes()
    dataset = PairedDataset.model_validate_json(raw)
    if (
        dataset.models != model_identity(config)
        or dataset.quality_metric != config.quality_metric
        or dataset.scope != config.scope
    ):
        parser.error("Training dataset model/metric/scope mismatch.")
    weights = fit_router(dataset.rows, config, hashlib.sha256(raw).hexdigest(), args.seed)
    save_weights(weights, args.out)
    print(f"Saved learned router {weights.sha256} to {args.out}")


if __name__ == "__main__":
    main()
