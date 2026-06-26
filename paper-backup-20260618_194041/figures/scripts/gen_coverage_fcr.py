"""Generate coverage-vs-FCR gate-ladder figure."""

from __future__ import annotations

import numpy as np

from bench.plot_coverage_fcr import build_coverage_table

from paper_style import ARTIFACTS, COLORS, RUNG_LABELS, RUNGS, load_json, save_pdf, setup_matplotlib


def _rate_error_bars(part, metric: str) -> list[list[float]]:
    lower = (part[metric] - part[f"{metric}_ci_low"]).clip(lower=0).fillna(0)
    upper = (part[f"{metric}_ci_high"] - part[metric]).clip(lower=0).fillna(0)
    return [lower.to_list(), upper.to_list()]


def generate() -> list:
    payload = load_json("combined_results")
    table = build_coverage_table(payload)
    ladder = table[table["section"] == "gate_ladder"].copy()

    plt = setup_matplotlib()
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=True)

    for panel, (ax, subset) in enumerate(zip(axes, ["FULL", "MAIN"])):
        part = ladder[ladder["subset"] == subset].sort_values("rung_index")
        x = np.arange(len(part))
        ax.bar(
            x,
            part["coverage"],
            width=0.62,
            color=COLORS["light_blue"],
            edgecolor=COLORS["blue"],
            linewidth=0.6,
            label="Coverage" if panel == 0 else None,
            zorder=1,
        )
        ax.errorbar(
            x,
            part["FCR"],
            yerr=_rate_error_bars(part, "FCR"),
            color=COLORS["red"],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            capsize=2.5,
            label="FCR, exact 95% CI" if panel == 0 else None,
            zorder=3,
        )
        ax.plot(
            x,
            part["known_positive_recall"],
            color=COLORS["green"],
            marker="s",
            markersize=3.4,
            linewidth=1.4,
            label="TPR" if panel == 0 else None,
            zorder=4,
        )
        ax.text(
            0.02,
            0.96,
            f"({chr(97 + panel)}) {subset}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([RUNG_LABELS[r] for r in RUNGS], rotation=25, ha="right")
        ax.set_xlabel("Gate rung")
        ax.set_ylim(-0.03, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color="#cccccc", alpha=0.55, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.04))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.28, top=0.86, wspace=0.13)
    out_path = save_pdf(fig, "fig_coverage_fcr.pdf")
    plt.close(fig)
    return [out_path]


def main() -> int:
    paths = generate()
    for path in paths:
        print(path)
    print(f"source={ARTIFACTS['combined_results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
