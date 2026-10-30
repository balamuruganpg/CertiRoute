"""Render README evidence from published snapshots; no inference or new experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402


def render(root: Path, out: Path) -> None:
    sources = {
        name: root / folder / "results.json"
        for name, folder in (
            ("live", "live-gsm8k-v1"),
            ("offline", "multibench-v1"),
            ("drift", "drift-v1"),
        )
    }
    data = {name: json.loads(path.read_bytes()) for name, path in sources.items()}
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "text.color": "#152940",
            "axes.labelcolor": "#152940",
            "xtick.color": "#43576c",
            "ytick.color": "#43576c",
        }
    )
    policies = ["cheap-only", "strong-only", "CertiRoute"]
    colors = ["#94a3b8", "#334e70", "#007e80"]
    live = data["live"]["summaries"]
    saving = 100 * (1 - live["CertiRoute"]["cost_usd"] / live["strong-only"]["cost_usd"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.72, bottom=0.2, wspace=0.6)
    for ax, field, scale, title, limit in zip(
        axes,
        ["quality", "cost_usd", "regret"],
        [100, 1, 100],
        ["Answer accuracy (%)", "Recorded USD / 100 requests", "Clipped regret (%)"],
        [110, 0.10, 65],
        strict=True,
    ):
        values = [live[p][field] * scale for p in policies]
        ax.barh(policies, values, color=colors, height=0.55)
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=16)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
        for i, value in enumerate(values):
            label = f"${value:.4f}" if field == "cost_usd" else f"{value:.2f}%"
            ax.text(value + limit * 0.025, i, label, va="center", fontsize=10)
    axes[2].axvline(5, color="#b64135", linestyle="--", linewidth=1.5)
    axes[2].text(5.8, 2.48, "5% target", color="#b64135", fontsize=9)
    fig.text(0.035, 0.92, "LIVE GSM8K  /  GROQ + xAI", fontsize=19, fontweight="bold")
    fig.text(
        0.035,
        0.82,
        f"{saving:.2f}% recorded cost savings; "
        f"{live['CertiRoute']['regret'] * 100:.2f}% regret exceeds the 5% target.",
        fontsize=12,
    )
    fig.text(
        0.035,
        0.045,
        "100 held-out prompts per policy · regret over 99 valid strong references · "
        "one unknown-cost strong request retains its reservation",
        fontsize=9,
        color="#526779",
    )
    fig.savefig(out / "live-comparison.png", dpi=160, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for ax, summary in zip(axes, data["drift"]["summaries"], strict=True):
        values = [summary[k]["mean_regret"] * 100 for k in ("before", "recalibrated")]
        ax.barh(
            ["Transfer unchanged", "Target recalibration"],
            values,
            color=["#be5b45", "#007e80"],
            height=0.5,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 24)
        ax.axvline(5, color="#334e70", linestyle="--", label="5% target")
        ax.set_title(summary["direction"].replace("-", " ").title(), loc="left", fontweight="bold")
        for i, value in enumerate(values):
            ax.text(value + 0.4, i, f"{value:.3f}%", va="center")
        ax.set_xlabel("Mean clipped regret (%)")
        ax.legend(loc="lower right", frameon=False)
    fig.suptitle(
        "Length shift + independent recalibration  /  1,000 offline draws", fontweight="bold"
    )
    fig.savefig(out / "drift-recovery.png", dpi=160, facecolor="white")
    plt.close(fig)

    write = {
        "inputs": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Display only; no new experiment. Live and offline model pairs differ.",
    }
    (out / "provenance.json").write_text(json.dumps(write, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.out)


if __name__ == "__main__":
    main()
