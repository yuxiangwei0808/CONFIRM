"""Create the main-text cumulative CONFIRM gate figure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


STAGE_LABELS = (
    "Execution +\nprovenance",
    "Multiple\ntesting",
    "Confound\nchecks",
    "Power",
    "Stability",
    "Replication",
)

SCIENTIFIC_SERIES = {
    "confirm": ("Literature supports claim", "#0F4D92", "o"),
    "abstain": ("Literature supports abstention", "#B64342", "s"),
}

CONTROL_ROWS = (
    ("synthetic_constructed", "Synthetic stress"),
    ("external_constructed_NACC", "AD/aging\nrandom labels"),
    ("external_constructed_ds000030", "Psychosis\nrandom labels"),
)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series(
    rows: list[dict[str, str]],
    *,
    stratum: str,
    disposition: str,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["stratum"] == stratum
        and row["reference_disposition"] == disposition
    ]
    selected.sort(key=lambda row: int(row["stage_index"]))
    if len(selected) != len(STAGE_LABELS):
        raise ValueError(
            f"Expected {len(STAGE_LABELS)} rows for {stratum}/{disposition}, "
            f"found {len(selected)}"
        )
    return selected


def _write_source_data(
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    selected: list[dict[str, str]] = []
    for disposition in SCIENTIFIC_SERIES:
        selected.extend(
            _series(
                rows,
                stratum="scientific_literature",
                disposition=disposition,
            )
        )
    for stratum, _ in CONTROL_ROWS:
        selected.extend(_series(rows, stratum=stratum, disposition="abstain"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


def plot(rows: list[dict[str, str]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )

    figure, (axis_science, axis_controls) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": (1.42, 1.0)},
    )
    x = np.arange(len(STAGE_LABELS))

    for disposition, (label, color, marker) in SCIENTIFIC_SERIES.items():
        selected = _series(
            rows,
            stratum="scientific_literature",
            disposition=disposition,
        )
        totals = np.asarray([int(row["claim_count"]) for row in selected])
        counts = np.asarray([int(row["pass_count"]) for row in selected])
        rates = 100.0 * counts / totals
        intervals = np.asarray(
            [_wilson(int(count), int(total)) for count, total in zip(counts, totals)]
        )
        yerr = np.vstack(
            (
                rates - 100.0 * intervals[:, 0],
                100.0 * intervals[:, 1] - rates,
            )
        )
        axis_science.errorbar(
            x,
            rates,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.8,
            elinewidth=0.8,
            capsize=2.0,
            label=label,
            zorder=3,
        )
        axis_science.annotate(
            f"{counts[-1]}/{totals[-1]}",
            xy=(x[-1], rates[-1]),
            xytext=(
                x[-1] + 0.18,
                rates[-1] + (3.0 if disposition == "confirm" else 2.7),
            ),
            color=color,
            fontsize=6.6,
            fontweight="bold",
            ha="left",
            va="center",
        )

    axis_science.set_xlim(-0.25, 5.65)
    axis_science.set_ylim(0, 108)
    axis_science.set_yticks((0, 25, 50, 75, 100))
    axis_science.set_ylabel("Cumulative gate-pass rate (%)")
    axis_science.set_xticks(x, STAGE_LABELS)
    axis_science.tick_params(axis="x", labelsize=6.2, pad=3)
    axis_science.tick_params(axis="y", labelsize=6.5)
    axis_science.set_title(
        "Scientific literature references",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        pad=6,
    )
    axis_science.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.93),
        fontsize=6.7,
        handlelength=2.5,
    )
    axis_science.axvspan(4.7, 5.3, color="#EEEEEE", zorder=0)
    for value in (25, 50, 75):
        axis_science.axhline(value, color="#E3E3E3", linewidth=0.55, zorder=0)

    control_counts: list[list[int]] = []
    control_totals: list[int] = []
    for stratum, _ in CONTROL_ROWS:
        selected = _series(rows, stratum=stratum, disposition="abstain")
        control_counts.append([int(row["pass_count"]) for row in selected])
        control_totals.append(int(selected[0]["claim_count"]))
    counts_array = np.asarray(control_counts)
    totals_array = np.asarray(control_totals)[:, None]
    rates_array = counts_array / totals_array
    control_cmap = LinearSegmentedColormap.from_list(
        "control_pass",
        ("#F7F7F7", "#F6CFCB", "#B64342"),
    )
    axis_controls.imshow(
        rates_array,
        cmap=control_cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(counts_array.shape[0]):
        for column_index in range(counts_array.shape[1]):
            rate = rates_array[row_index, column_index]
            text_color = "white" if rate >= 0.58 else "#272727"
            axis_controls.text(
                column_index,
                row_index,
                f"{counts_array[row_index, column_index]}/{control_totals[row_index]}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
                fontweight="bold" if column_index == 5 else "normal",
            )
    axis_controls.set_xticks(x, STAGE_LABELS)
    axis_controls.set_yticks(
        np.arange(len(CONTROL_ROWS)),
        [label for _, label in CONTROL_ROWS],
    )
    axis_controls.tick_params(axis="x", labelsize=6.2, pad=3)
    axis_controls.tick_params(axis="y", labelsize=6.3, length=0, pad=5)
    axis_controls.set_title(
        "Constructed controls",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        pad=6,
    )
    for spine in axis_controls.spines.values():
        spine.set_visible(False)
    axis_controls.set_xticks(np.arange(-0.5, len(STAGE_LABELS), 1), minor=True)
    axis_controls.set_yticks(np.arange(-0.5, len(CONTROL_ROWS), 1), minor=True)
    axis_controls.grid(which="minor", color="white", linewidth=1.5)
    axis_controls.tick_params(which="minor", bottom=False, left=False)
    axis_controls.text(
        0,
        -0.24,
        "Cells show passing / total; darker shading indicates a higher pass rate.",
        transform=axis_controls.transAxes,
        fontsize=6.1,
        color="#4D4D4D",
        ha="left",
        va="top",
    )

    axis_science.text(
        -0.12,
        1.06,
        "a",
        transform=axis_science.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )
    axis_controls.text(
        -0.22,
        1.06,
        "b",
        transform=axis_controls.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )

    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.22,
        top=0.88,
        wspace=0.38,
    )
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-ladder",
        default="review-stage/neuroclaimbench-v2.1/gate-attribution/gate_ladder.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="paper/figures/fig_confirm_gate_ladder",
    )
    args = parser.parse_args()
    rows = _read_rows(Path(args.gate_ladder))
    output_prefix = Path(args.output_prefix)
    _write_source_data(rows, output_prefix.with_name(f"{output_prefix.name}_source.csv"))
    plot(rows, output_prefix)
    print(f"Wrote {output_prefix}.svg/.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
