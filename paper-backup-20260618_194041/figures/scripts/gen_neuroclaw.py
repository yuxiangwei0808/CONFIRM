"""Generate NeuroClaw vs CONFIRM comparison figure."""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, load_json, rate_ci_error, save_pdf, setup_matplotlib


def _bar_with_ci(ax, x: float, stat: dict, color: str, label: str | None, width: float = 0.32):
    rate = float(stat["rate"])
    bar = ax.bar(
        [x],
        [rate],
        width=width,
        color=color,
        edgecolor="black",
        linewidth=0.4,
        yerr=rate_ci_error(rate, stat["ci95_exact"]),
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
        label=label,
        zorder=3,
    )[0]
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        min(rate + 0.045, 1.035),
        f"{stat['count']}/{stat['denominator']}",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    return bar


def generate() -> list:
    head = load_json("neuroclaw")
    layer = load_json("confirm_layer")

    panels = [
        (
            "Head-to-head shared set",
            [
                ("NeuroClaw", head["neuroclaw_TPR"], head["neuroclaw_FCR"], COLORS["gray"]),
                ("CONFIRM", head["confirm_TPR_on_shared_set"], head["confirm_FCR_on_shared_set"], COLORS["blue"]),
            ],
        ),
        (
            "Post-hoc CONFIRM layer",
            [
                ("NeuroClaw", layer["neuroclaw_alone_TPR"], layer["neuroclaw_alone_FCR"], COLORS["gray"]),
                (
                    "NeuroClaw+CONFIRM",
                    layer["neuroclaw_confirm_layer_TPR"],
                    layer["neuroclaw_confirm_layer_FCR"],
                    COLORS["green"],
                ),
            ],
        ),
    ]

    plt = setup_matplotlib()
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    metric_x = np.array([0.0, 1.0])
    offsets = [-0.18, 0.18]

    for panel_idx, (ax, (panel_label, methods)) in enumerate(zip(axes, panels)):
        for method_idx, (method_label, tpr, fcr, color) in enumerate(methods):
            legend_label = method_label if panel_idx == 0 else None
            _bar_with_ci(ax, metric_x[0] + offsets[method_idx], tpr, color, legend_label)
            _bar_with_ci(ax, metric_x[1] + offsets[method_idx], fcr, color, None)

        ax.text(
            0.02,
            0.96,
            f"({chr(97 + panel_idx)}) {panel_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
        ax.set_xticks(metric_x)
        ax.set_xticklabels(["TPR", "FCR"])
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color="#cccccc", alpha=0.55, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Rate")
    axes[1].annotate(
        "FCR 0.33 -> 0.00",
        xy=(metric_x[1] + offsets[1], 0.02),
        xytext=(metric_x[1] - 0.36, 0.50),
        arrowprops={"arrowstyle": "->", "color": COLORS["red"], "linewidth": 1.0},
        color=COLORS["red"],
        fontsize=7,
        ha="left",
        va="center",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=COLORS["green"], edgecolor="black", linewidth=0.4))
    labels.append("NeuroClaw+CONFIRM")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.54, 1.04),
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.84, wspace=0.12)
    out_path = save_pdf(fig, "fig_neuroclaw.pdf")
    plt.close(fig)
    return [out_path]


def main() -> int:
    for path in generate():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
