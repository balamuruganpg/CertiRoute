"""Project published study aggregates into one comparable dashboard payload."""

import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def load_results(main_path: Path, shift_path: Path) -> dict[str, Any]:
    main_bytes, shift_bytes = main_path.read_bytes(), shift_path.read_bytes()
    main, shift = json.loads(main_bytes), json.loads(shift_bytes)
    if (
        main["kind"] != "offline_calibration_seed_sensitivity"
        or shift["kind"] != "offline_within_gsm8k_length_shift"
    ):
        raise ValueError("Expected sensitivity and shift study results.")
    for field in ("dataset_sha256", "models", "quality_metric", "prompt_template"):
        if main["dataset_manifest"][field] != shift["dataset_manifest"][field]:
            raise ValueError("Studies must use the same dataset, models and scoring protocol.")
    if main["pricing"] != shift["pricing"]:
        raise ValueError("Studies must use the same pricing scenario.")
    study = main["study"]
    seeds = study["training_seeds"]
    n_test = study["test_size"]
    if n_test <= 0:
        raise ValueError("Positive test size required for cost normalization.")

    def metrics(row: dict[str, Any]) -> dict[str, float]:
        values = {
            "quality": float(row["mean_quality"]),
            "regret": float(row["mean_regret"]),
            "cost_per_1000": float(row["mean_cost_usd"]) / n_test * 1000,
            "savings": float(row["mean_savings_vs_strong"]),
        }
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("Non-finite result metric.")
        if not 0 <= values["quality"] <= 1 or not 0 <= values["regret"] <= 1:
            raise ValueError("Quality and regret must be bounded.")
        return values

    baselines = []
    for policy in ("cheap-only", "strong-only"):
        rows = [r for r in main["baselines"] if r["policy"] == policy]
        if sorted(r["training_seed"] for r in rows) != sorted(seeds):
            raise ValueError("Baseline must cover each training seed exactly once.")
        averaged = {
            key: mean(r[key] for r in rows)
            for key in (
                "mean_quality",
                "mean_regret",
                "mean_cost_usd",
                "mean_savings_vs_strong",
            )
        }
        baselines.append({"policy": policy, **metrics(averaged)})
    points = []
    expected = {(n, e) for n in study["calibration_sizes"] for e in main["epsilons"]}
    seen = set()
    for row in main["summaries"]:
        if row["policy"] != "CRC":
            continue
        key = (row["calibration_size"], row["epsilon"])
        if key in seen:
            raise ValueError("Duplicate operating points.")
        seen.add(key)
        points.append(
            {
                "policy": "CertiRoute",
                "calibration_size": key[0],
                "epsilon": key[1],
                "non_rejections": row["non_rejections"],
                "cell_count": row["cell_count"],
                "fraction_exceeding_epsilon": row["fraction_exceeding_epsilon"],
                **metrics(row),
            }
        )
    if seen != expected:
        raise ValueError("Incomplete operating points.")
    manifest = main["dataset_manifest"]
    return {
        "baselines": baselines,
        "operating_points": points,
        "epsilons": main["epsilons"],
        "calibration_sizes": study["calibration_sizes"],
        "shift": shift["summaries"],
        "shift_study": shift["study"],
        "provenance": {
            "examples": len(manifest["selected_source_indices"]),
            "training_seeds": len(seeds),
            "draws": len(seeds) * study["repeats"],
            "models": {
                tier: identity.split("::")[-1] for tier, identity in manifest["models"].items()
            },
            "pricing_as_of": main["pricing"]["as_of"],
            "main_sha256": hashlib.sha256(main_bytes).hexdigest(),
            "shift_sha256": hashlib.sha256(shift_bytes).hexdigest(),
            "dataset_sha256": manifest["dataset_sha256"],
        },
    }
