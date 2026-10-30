"""Render saved sensitivity aggregates with Matplotlib; no gateway or network imports."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from statistics import mean
from typing import Any, cast

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402
import numpy as np  # type: ignore[import-not-found]  # noqa: E402

COLORS = {"CRC": "#2463a6", "no-CRC": "#d05035", "random-CRC": "#687585"}
LABELS = {
    "CRC": "Learned + CRC",
    "no-CRC": "Learned, no correction",
    "random-CRC": "Random rankings + CRC",
}


def render(input_path: Path, out: Path) -> None:
    payload = input_path.read_bytes()
    result = json.loads(payload)
    if result["kind"] != "offline_calibration_seed_sensitivity":
        raise ValueError("Expected sensitivity results.")
    sizes, epsilons = result["study"]["calibration_sizes"], result["epsilons"]
    seeds = result["study"]["training_seeds"]
    summaries = {(r["calibration_size"], r["epsilon"], r["policy"]): r for r in result["summaries"]}
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "certiroute-sensitivity-v1",
        }
    )

    def save(figure: Any, stem: str) -> None:
        for extension in ("png", "svg", "pdf"):
            metadata = (
                {"Date": None}
                if extension == "svg"
                else ({"CreationDate": None, "ModDate": None} if extension == "pdf" else {})
            )
            figure.savefig(
                out / f"{stem}.{extension}", dpi=180, bbox_inches="tight", metadata=metadata
            )
        plt.close(figure)

    rows = (len(epsilons) + 1) // 2
    fig, axes = plt.subplots(
        rows, 2, figsize=(12, 3.5 * rows + 1), layout="constrained", squeeze=False
    )
    for ax, epsilon in zip(axes.flat, epsilons, strict=False):
        for policy, color in COLORS.items():
            selected = [summaries[(n, epsilon, policy)] for n in sizes]
            ax.plot(
                sizes,
                [r["mean_regret"] for r in selected],
                "o-",
                color=color,
                label=LABELS[policy],
                linewidth=2,
                markersize=5,
            )
            ax.fill_between(
                sizes,
                [r["seed_mean_regret_range"][0] for r in selected],
                [r["seed_mean_regret_range"][1] for r in selected],
                color=color,
                alpha=0.12,
            )
        ax.axhline(epsilon, color="#292929", linestyle=":", linewidth=1.5, label="Target epsilon")
        ax.set(title=f"Target ε = {epsilon:g}", xscale="log", ylim=(0, None))
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.grid(alpha=0.16)
    for ax in list(axes.flat)[len(epsilons) :]:
        ax.set_visible(False)
    fig.suptitle("Calibration size and realized risk", fontsize=17)
    fig.supxlabel(
        "Calibration examples · shading: range of training-seed means (not a CI)", y=0.055
    )
    fig.supylabel("Mean clipped regret")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    cast(Any, fig.get_layout_engine()).set(rect=(0, 0.065, 1, 0.935))
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    save(fig, "calibration-sweep")

    rows = (len(sizes) + 1) // 2
    fig, axes = plt.subplots(
        rows, 2, figsize=(12, 3.5 * rows + 1), layout="constrained", squeeze=False
    )
    cheap = [r for r in result["baselines"] if r["policy"] == "cheap-only"]
    cheap_risk = mean(r["mean_regret"] for r in cheap)
    cheap_cost = 1 - mean(r["mean_savings_vs_strong"] for r in cheap)
    for ax, size in zip(axes.flat, sizes, strict=False):
        ax.plot(
            [0, cheap_risk], [1, cheap_cost], "--", color="#aaaaaa", label="Random-mix reference"
        )
        for policy, color in COLORS.items():
            selected = [summaries[(size, epsilon, policy)] for epsilon in epsilons]
            ax.plot(
                [r["mean_regret"] for r in selected],
                [1 - r["mean_savings_vs_strong"] for r in selected],
                "o-",
                color=color,
                linewidth=2,
                markersize=5,
                label=LABELS[policy],
            )
            if policy == "CRC":
                for index, row in enumerate(selected):
                    ax.annotate(
                        f"ε={row['epsilon']:g}",
                        (row["mean_regret"], 1 - row["mean_savings_vs_strong"]),
                        xytext=(5, 9 if index % 2 == 0 else -15),
                        textcoords="offset points",
                        fontsize=8,
                        color=color,
                    )
        ax.set(
            title=f"Calibration size = {size}", xlim=(-0.005, cheap_risk * 1.07), ylim=(-0.02, 1.08)
        )
        ax.grid(alpha=0.16)
    for ax in list(axes.flat)[len(sizes) :]:
        ax.set_visible(False)
    fig.suptitle("Risk versus estimated generation cost", fontsize=17)
    fig.supxlabel(
        "Mean clipped regret · lines connect all predeclared epsilon operating points", y=0.055
    )
    fig.supylabel("Mean cost / strong-only cost (lower is cheaper)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    cast(Any, fig.get_layout_engine()).set(rect=(0, 0.065, 1, 0.935))
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    save(fig, "risk-cost")

    values = {}
    for epsilon in epsilons:
        matrix = []
        for seed in seeds:
            row = []
            for size in sizes:
                savings = [
                    next(
                        s["mean_savings_vs_strong"]
                        for s in summaries[(size, epsilon, policy)]["by_training_seed"]
                        if s["training_seed"] == seed
                    )
                    for policy in ("CRC", "random-CRC")
                ]
                row.append(100 * (savings[0] - savings[1]))
            matrix.append(row)
        values[epsilon] = np.asarray(matrix)
    maximum = max(0.1, max(float(np.abs(v).max()) for v in values.values()))
    rows = (len(epsilons) + 1) // 2
    fig, axes = plt.subplots(
        rows, 2, figsize=(12, 3.5 * rows + 1), layout="constrained", squeeze=False
    )
    for ax, epsilon in zip(axes.flat, epsilons, strict=False):
        grid = values[epsilon]
        chart = ax.imshow(grid, cmap="RdBu", vmin=-maximum, vmax=maximum, aspect="auto")
        ax.set(
            title=f"Target ε = {epsilon:g}",
            xticks=range(len(sizes)),
            xticklabels=sizes,
            yticks=range(len(seeds)),
            yticklabels=seeds,
        )
        for i, j in np.ndindex(grid.shape):
            ax.text(
                j,
                i,
                f"{grid[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(grid[i, j]) > maximum * 0.6 else "#222222",
            )
    for ax in list(axes.flat)[len(epsilons) :]:
        ax.set_visible(False)
    fig.suptitle("Does the learned ranking outperform random rankings?", fontsize=16)
    fig.supxlabel("Calibration examples · positive values favor learned CRC; random seeds averaged")
    fig.supylabel("Training-partition seed")
    fig.colorbar(
        chart, ax=list(axes.flat), shrink=0.85, label="Savings advantage (percentage points)"
    )
    save(fig, "seed-advantage")

    provenance = {
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")},
        "figures": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
        },
        "note": "Plots use saved aggregates only. "
        "Shading/ranges across seeds are descriptive, not CIs.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved PNG, SVG and PDF figures to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Saved sensitivity results.json")
    parser.add_argument("--out", type=Path, required=True, help="New directory for plots")
    args = parser.parse_args()
    render(args.input, args.out)


if __name__ == "__main__":
    main()
