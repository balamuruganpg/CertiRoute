"""Plot saved shift results; separate optional Matplotlib environment, fully offline."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402


def render(input_path: Path, out: Path) -> None:
    payload = input_path.read_bytes()
    result = json.loads(payload)
    if result["kind"] != "offline_within_gsm8k_length_shift":
        raise ValueError("Expected shift results.")
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"svg.hashsalt": "certiroute-shift-v1", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    upper = max(r["regret_ci_bonferroni"][1] for r in result["summaries"]) * 1.1
    upper = max(upper, max(result["epsilons"]) * 1.1)
    for ax, direction in zip(axes, ("short-to-long", "long-to-short"), strict=True):
        ax.plot([0, upper], [0, upper], ":", color="#333333", label="Target regret")
        for arm, label, color, marker in (
            ("source-control", "Source control", "#687585", "s"),
            ("shifted", "Shifted, no recalibration", "#d05035", "o"),
            ("target-recalibrated", "Target recalibration", "#2463a6", "^"),
        ):
            selected = sorted(
                (r for r in result["summaries"] if r["direction"] == direction and r["arm"] == arm),
                key=lambda r: r["epsilon"],
            )
            ax.errorbar(
                [r["epsilon"] for r in selected],
                [r["mean_regret"] for r in selected],
                yerr=[
                    [r["mean_regret"] - r["regret_ci_bonferroni"][0] for r in selected],
                    [r["regret_ci_bonferroni"][1] - r["mean_regret"] for r in selected],
                ],
                color=color,
                marker=marker,
                label=label,
                capsize=3,
                linewidth=1.7,
            )
        ax.set(
            title=direction.replace("-", " "),
            xlabel="Target regret (epsilon)",
            xlim=(0, max(result["epsilons"]) * 1.06),
            ylim=(-0.005, upper),
        )
        ax.set_xticks(result["epsilons"])
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Mean clipped regret")
    fig.suptitle("GSM8K question-length shift · adjusted Monte Carlo intervals", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    for extension in ("png", "svg", "pdf"):
        metadata = (
            {"Date": None}
            if extension == "svg"
            else {"CreationDate": None, "ModDate": None}
            if extension == "pdf"
            else {}
        )
        fig.savefig(
            out / f"shift-risk.{extension}", dpi=180, bbox_inches="tight", metadata=metadata
        )
    plt.close(fig)
    provenance = {
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")},
        "figures": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved shift plots to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.input, args.out)


if __name__ == "__main__":
    main()
