"""Render the multi-benchmark comparison from saved metrics, without new experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402


def render(source: Path, out: Path) -> None:
    raw = source.read_bytes()
    data = json.loads(raw)
    out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    colors = ["#8997a8", "#24364b", "#008573"]
    for index, policy in enumerate(["cheap-only", "strong-only", "CertiRoute"]):
        selected = [
            next(
                p
                for p in t["summaries"]
                if p["policy"] == policy and (policy != "CertiRoute" or p["epsilon"] == 0.05)
            )
            for t in data["tasks"]
        ]
        x = [i + (index - 1) * 0.24 for i in range(4)]
        for axis, field, scale in zip(
            axes, ["mean_quality", "mean_cost_usd", "mean_regret"], [100, 1000, 100], strict=True
        ):
            axis.bar(
                x,
                [p[field] * scale for p in selected],
                width=0.23,
                color=colors[index],
                label=policy,
            )
    for axis, label in zip(
        axes, ["Quality (%)", "Modeled USD / 1,000 requests", "Clipped regret (%)"], strict=True
    ):
        axis.set_xticks(range(4), ["GSM8K", "ARC", "MBPP", "Equal-task\nportfolio"])
        axis.set_title(label, loc="left", fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.15)
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower left", frameon=False)
    axes[2].axhline(5, linestyle="--", color="#b44b37", linewidth=1, label="5% target")
    axes[2].legend(frameon=False)
    fig.suptitle(
        "Three benchmarks · frozen real responses · 1,000 CRC partitions",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "epsilon": 0.05,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.input, args.out)


if __name__ == "__main__":
    main()
