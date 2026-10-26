"""Train, calibrate and compare policies using only local frozen paired outputs."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import platform
import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import SecretStr

from certiroute.calibration.artifact import DataProvenance, build_artifact, save_artifact
from certiroute.config import GatewayConfig, Settings, load_config
from certiroute.gateway.app import create_app
from certiroute.provenance import content_hash, model_identity
from certiroute.router.predictor import load_predictor, save_weights
from certiroute.router.scope import with_generation_defaults
from eval.calibrate import PairedDataset
from eval.fit_router import fit_router
from eval.fixtures import FixtureRow, Pricing, TokenCounts, read_fixture, read_pricing
from eval.manifest import DatasetManifest, read_manifest


def split_rows(rows: list[FixtureRow], seed: int) -> dict[str, list[FixtureRow]]:
    # Assignment depends only on prompt + predeclared seed, never outcomes/labels.
    ordered = sorted(rows, key=lambda r: content_hash([seed, r.request["messages"]]))
    train_end, cal_end = int(0.4 * len(rows)), int(0.8 * len(rows))
    return {
        "train": ordered[:train_end],
        "calibration": ordered[train_end:cal_end],
        "test": ordered[cal_end:],
    }


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def dataset_for(
    rows: list[FixtureRow], config: GatewayConfig, manifest: DatasetManifest | None = None
) -> PairedDataset:
    return PairedDataset(
        models=model_identity(config),
        quality_metric=config.quality_metric,
        scope=config.scope,
        sources=([rows[0].source, manifest.dataset_sha256] if manifest else [rows[0].source]),
        collected_at=manifest.frozen_at if manifest else datetime(2026, 10, 23, tzinfo=UTC),
        rows=[row.calibration_row() for row in rows],
    )


def replay_key(request: dict[str, object], config: GatewayConfig) -> str:
    normalized = with_generation_defaults(request, config.scope)
    return content_hash({k: v for k, v in normalized.items() if k not in ("model", "stream")})


async def compare(
    rows: list[FixtureRow],
    config: GatewayConfig,
    config_path: Path,
    pricing: Pricing,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exercise the real gateway against a network-free recorded-output transport.

    This is offline replay, not a live-provider evaluation or the G9 audit study.
    Unknown requests fail; there is no path that falls through to real networking.
    """
    records = {replay_key(row.request, config): row for row in rows}
    predictor = load_predictor(config)
    for row in rows:
        predictor.check_independent(row.request)

    def replay(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        row = records[replay_key(body, config)]
        tier = next(
            (
                tier
                for tier in ("cheap", "strong")
                if (
                    body["model"] == getattr(config.models, tier).model
                    and str(request.url)
                    == str(getattr(config.models, tier).base_url).rstrip("/") + "/chat/completions"
                )
            ),
            None,
        )
        if tier is None:
            raise ValueError("Unexpected replay provider/model.")
        output = row.cheap_output if tier == "cheap" else row.strong_output
        usage = row.cheap_usage if tier == "cheap" else row.strong_usage
        return httpx.Response(
            200,
            json={
                "id": f"fixture-{row.id}-{tier}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": output},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    **usage.model_dump(),
                    "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                },
            },
        )

    # Ephemeral local authentication, not an upstream API key or a stored secret.
    local_token = secrets.token_urlsafe(32)
    app = create_app(
        Settings(config_file=config_path, api_key=SecretStr(local_token)),
        transport=httpx.MockTransport(replay),
    )
    observations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    strong_cost = math.fsum(
        pricing.models[config.models.strong.model].cost(r.strong_usage) for r in rows
    )
    if strong_cost <= 0:
        raise ValueError("Strong baseline cost must be positive for savings comparison.")
    policies: list[tuple[str, str, float | None]] = [
        ("cheap-only", "cheap", None),
        ("strong-only", "strong", None),
    ]
    policies += [("CertiRoute", "auto", e) for e in config.calibration.allowed_epsilons]
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://offline"
        ) as client:
            if (await client.get("/readyz")).status_code != 200:
                raise ValueError("Gateway failed to load learned router/calibration.")
            for policy, mode, epsilon in policies:
                selected: list[dict[str, Any]] = []
                for row in rows:
                    headers = {"Authorization": f"Bearer {local_token}"}
                    if epsilon is not None:
                        headers["x-certiroute-epsilon"] = str(epsilon)
                    response = await client.post(
                        "/v1/chat/completions",
                        headers=headers,
                        json={**row.request, "model": f"certiroute/{mode}", "stream": False},
                    )
                    response.raise_for_status()
                    tier = response.headers["x-certiroute-tier"]
                    if mode == "auto" and response.headers["x-certiroute-reason"] != "calibrated":
                        raise ValueError("Replay did not use the calibrated path.")
                    body = response.json()
                    output = body["choices"][0]["message"]["content"]
                    if output != (row.cheap_output if tier == "cheap" else row.strong_output):
                        raise ValueError("Gateway replay did not preserve recorded output.")
                    quality = row.quality(output)
                    usage = TokenCounts.model_validate(
                        {key: body["usage"][key] for key in ("prompt_tokens", "completion_tokens")}
                    )
                    cost = pricing.models[body["model"]].cost(usage)
                    observation = {
                        "id": row.id,
                        "policy": policy,
                        "epsilon": epsilon,
                        "tier": tier,
                        "quality": quality,
                        "regret": max(0.0, row.q_strong - quality),
                        "cost_usd": cost,
                        "score": response.headers["x-certiroute-score"],
                        "calibration_id": response.headers["x-certiroute-calibration-id"],
                    }
                    observations.append(observation)
                    selected.append(observation)
                total_cost = math.fsum(x["cost_usd"] for x in selected)
                summaries.append(
                    {
                        "policy": policy,
                        "epsilon": epsilon,
                        "n": len(rows),
                        "quality": math.fsum(x["quality"] for x in selected) / len(rows),
                        "regret": math.fsum(x["regret"] for x in selected) / len(rows),
                        "cost_usd": total_cost,
                        "savings_vs_strong": 1 - total_cost / strong_cost,
                        "strong_fraction": sum(x["tier"] == "strong" for x in selected) / len(rows),
                    }
                )
    return summaries, observations


def code_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    paths = sorted(
        [
            *root.joinpath("src").rglob("*.py"),
            *root.joinpath("eval").glob("*.py"),
            root / "pyproject.toml",
            root / "requirements-eval.lock",
            root / "requirements-plot.lock",
        ]
    )
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        git_sha = git.stdout.strip() if git.returncode == 0 else None
    except FileNotFoundError:
        git_sha = None
    return {
        "git_sha": git_sha,
        "source_sha256": content_hash(files),
        "files": files,
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("fastapi", "httpx", "pydantic", "PyYAML")
        },
    }


def run(
    input_path: Path, config_path: Path, pricing_path: Path, out: Path, seed: int = 42
) -> dict[str, Any]:
    original = load_config(config_path)
    if original.router.kind != "logistic_surface_v1":
        raise ValueError("Offline pilot requires the learned router.")
    pricing = read_pricing(pricing_path, original)
    rows, dropped = read_fixture(input_path, original)
    public = rows[0].source == "routellm_gsm8k_v1"
    dataset_manifest = read_manifest(input_path, original) if public else None
    if not public and (
        original.models.cheap.model != "fixture-cheap-v1"
        or original.models.strong.model != "fixture-strong-v1"
    ):
        raise ValueError("Synthetic artifacts must use fixture model identities.")
    if public and any(
        not str(provider.base_url.host).endswith(".invalid")
        for provider in (original.models.cheap, original.models.strong)
    ):
        raise ValueError("Public-cache artifacts must use replay-only .invalid identities.")
    splits = split_rows(rows, seed)
    out.mkdir(parents=True, exist_ok=False)
    config_fields = original.model_dump(mode="json")
    config_fields["router"]["artifact"] = str((out / "router.json").resolve())
    config_fields["calibration"]["artifact"] = str((out / "calibration-artifact.json").resolve())
    config = GatewayConfig.model_validate(config_fields)
    generated_config = out / "gateway.yaml"
    generated_config.write_text(yaml.safe_dump(config_fields, sort_keys=False), encoding="utf-8")
    for name, split in splits.items():
        write_json(
            out / f"{name}.json",
            dataset_for(split, config, dataset_manifest).model_dump(mode="json"),
        )
    train_hash = hashlib.sha256((out / "train.json").read_bytes()).hexdigest()
    weights = fit_router(
        [row.calibration_row() for row in splits["train"]], config, train_hash, seed
    )
    save_weights(weights, config.router.artifact)
    cal_hash = hashlib.sha256((out / "calibration.json").read_bytes()).hexdigest()
    cal_dataset = dataset_for(splits["calibration"], config, dataset_manifest)
    artifact = build_artifact(
        [row.calibration_row() for row in splits["calibration"]],
        config,
        DataProvenance(
            sources=cal_dataset.sources,
            sha256=cal_hash,
            collected_at=cal_dataset.collected_at,
        ),
    )
    save_artifact(artifact, config.calibration.artifact)
    summaries, observations = asyncio.run(
        compare(splits["test"], config, generated_config, pricing)
    )
    manifest = {
        "seed": seed,
        "fraction_train_cal_test": [0.4, 0.4, 0.2],
        "assignment": "sort SHA256(seed, messages), never labels",
        "near_duplicate_rule": "token-trigram Jaccard >= 0.9 before splitting; "
        "remove only the exact router.signature_prefix when configured",
        "dropped_ids": dropped,
        "splits": {name: [row.id for row in split] for name, split in splits.items()},
    }
    write_json(out / "split-manifest.json", manifest)
    result: dict[str, Any] = {
        "kind": "public_paired_offline_gateway_replay"
        if public
        else "synthetic_fixture_offline_gateway_replay",
        "dataset_manifest": dataset_manifest.model_dump(mode="json") if dataset_manifest else None,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "pricing_sha256": hashlib.sha256(pricing_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": content_hash(manifest),
        "seed": seed,
        "sizes": {name: len(split) for name, split in splits.items()},
        "config": config.model_dump(mode="json"),
        "pricing": pricing.model_dump(mode="json"),
        "router_sha256": weights.sha256,
        "calibration_id": artifact.calibration_id,
        "lambda_by_epsilon": artifact.lambda_by_epsilon,
        "code": code_provenance(),
        "summaries": summaries,
        "observations": observations,
        "limitations": (
            dataset_manifest.limitations
            if dataset_manifest
            else [
                "Synthetic outputs and token counts; hypothetical USD prices; "
                "zero billed API spend.",
                "Deliberately learnable arithmetic fixture; "
                "no real-model savings or quality claim.",
            ]
        )
        + [
            "Single held-out split, not G1 repeated-split validity or live G9.",
            "Zero billed API spend; cost covers the selected generation only, not collection "
            "or router hosting. No live latency was measured.",
        ],
    }
    write_json(out / "results.json", result)
    report = [
        "# Offline paired quality–cost comparison",
        "",
        (
            "Real frozen GSM8K model outputs; estimated token usage and historical modeled costs. "
            "No external API calls."
            if public
            else "Synthetic outputs, invented token counts and hypothetical prices. "
            "No external API calls."
        ),
        "",
        f"Seed {seed}; train/calibration/test = "
        f"{len(splits['train'])}/{len(splits['calibration'])}/{len(splits['test'])}.",
        "Costs are totals over the held-out set, not per-request costs.",
        "",
        "| Policy | Epsilon | Quality | Mean regret | Estimated USD | "
        "Savings vs strong | Strong fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        report.append(
            f"| {item['policy']} | {item['epsilon'] if item['epsilon'] is not None else '—'} | "
            f"{item['quality']:.2%} | {item['regret']:.4f} | {item['cost_usd']:.8f} | "
            f"{item['savings_vs_strong']:.2%} | {item['strong_fraction']:.2%} |"
        )
    report += [
        "",
        "One small held-out split is descriptive; it does not validate the CRC guarantee.",
        "The epsilon grid was fixed before fitting. All operating points are reported.",
        f"With n={artifact.n}, epsilons below {1 / (artifact.n + 1):.6f} force always-strong.",
        "",
        f"Cheap model: `{config.models.cheap.model}`; "
        f"strong model: `{config.models.strong.model}`.",
        f"Quality metric: `{config.quality_metric.name}` v{config.quality_metric.version}.",
        f"Pricing as of {pricing.as_of}: {pricing.description}",
        *([f"Pricing source: {pricing.source}"] if pricing.source else []),
        "",
        "| Model | Input USD / million tokens | Output USD / million tokens |",
        "|---|---:|---:|",
        *(
            f"| {name} | {price.input_per_million} | {price.output_per_million} |"
            for name, price in pricing.models.items()
        ),
        "",
        f"Dataset SHA256: `{result['dataset_sha256']}`.",
        "See results.json for configuration, split, code, dependency and per-request provenance.",
        "",
    ]
    if dataset_manifest:
        report += [
            "Paired-output source: " + dataset_manifest.source_files["gsm8k_responses.csv"].url,
            "Generation script: " + dataset_manifest.source_files["generate_responses.py"].url,
            "",
            dataset_manifest.prompt_template,
            f"Full frozen requests and responses: {input_path.as_posix()}.",
            "Generation metadata: `"
            + json.dumps(dataset_manifest.generation, sort_keys=True)
            + "`.",
            dataset_manifest.token_counts,
            dataset_manifest.selection,
            "",
        ]
    exceeded = [
        f"epsilon {item['epsilon']}: observed regret {item['regret']:.4f}"
        for item in summaries
        if item["epsilon"] is not None and item["regret"] > item["epsilon"]
    ]
    if exceeded:
        report += [
            "Held-out regret exceeded epsilon at " + "; ".join(exceeded) + ". "
            "CRC controls an expectation over calibration/test draws, not every split. "
            "These observations do not establish validity or cost savings at the target risk.",
            "",
        ]
    report += ["Limitations:", "", *("- " + item for item in result["limitations"]), ""]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval/sample.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/offline.yaml"))
    parser.add_argument("--pricing", type=Path, default=Path("configs/pricing.fixture.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/offline-fixture"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.config, args.pricing, args.out, args.seed)


if __name__ == "__main__":
    main()
