"""Plot coverage-vs-FCR tradeoffs from a combined benchmark result JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

RUNGS = ["exec_only", "+confound", "+power", "+multiverse", "+replication"]
SUBSETS = ("FULL", "MAIN")


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def _ci_values(row: dict[str, Any], metric: str) -> tuple[float, float]:
    values = row.get(f"{metric}_ci95_exact", row.get(f"{metric}_ci95", [math.nan, math.nan]))
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return math.nan, math.nan
    return _as_float(values[0]), _as_float(values[1])


def _summary_for_subset(payload: dict[str, Any], subset: str) -> dict[str, Any]:
    subset_key = subset.lower()
    summary = payload.get(f"summary_{subset_key}")
    if isinstance(summary, dict):
        return summary

    by_authority = payload.get("metrics_by_label_authority", {})
    if isinstance(by_authority, dict):
        authority_summary = by_authority.get(subset_key, {}).get("summary")
        if isinstance(authority_summary, dict):
            return authority_summary

    if subset == "FULL":
        summary = payload.get("summary", payload.get("metrics_exact_ci"))
        if isinstance(summary, dict):
            return summary
    raise KeyError(f"No gate-ladder summary found for subset {subset}")


def _risk_rows_for_subset(payload: dict[str, Any], subset: str) -> list[dict[str, Any]]:
    subset_key = subset.lower()
    rows = payload.get(f"risk_coverage_{subset_key}")
    if isinstance(rows, list):
        return rows
    if subset == "FULL":
        rows = payload.get("risk_coverage", [])
        if isinstance(rows, list):
            return rows
    return []


def _metric_columns(row: dict[str, Any], metric: str) -> dict[str, Any]:
    lo, hi = _ci_values(row, metric)
    return {
        metric: _as_float(row.get(metric)),
        f"{metric}_ci_low": lo,
        f"{metric}_ci_high": hi,
        f"{metric}_count": row.get(f"{metric}_count"),
        f"{metric}_denominator": row.get(f"{metric}_denominator"),
    }


def build_coverage_table(payload: dict[str, Any]) -> pd.DataFrame:
    """Return one row per plotted ladder rung or risk-coverage alpha point."""

    rows: list[dict[str, Any]] = []
    for subset in SUBSETS:
        summary = _summary_for_subset(payload, subset)
        for rung_index, rung in enumerate(RUNGS):
            if rung not in summary:
                continue
            metrics = summary[rung]
            rows.append(
                {
                    "section": "gate_ladder",
                    "subset": subset,
                    "rung": rung,
                    "rung_index": rung_index,
                    "alpha": math.nan,
                    **_metric_columns(metrics, "FCR"),
                    **_metric_columns(metrics, "coverage"),
                    **_metric_columns(metrics, "known_positive_recall"),
                }
            )
        for risk_row in _risk_rows_for_subset(payload, subset):
            alpha = _as_float(risk_row.get("alpha"))
            rows.append(
                {
                    "section": "risk_coverage",
                    "subset": subset,
                    "rung": f"alpha={alpha:g}",
                    "rung_index": math.nan,
                    "alpha": alpha,
                    **_metric_columns(risk_row, "FCR"),
                    **_metric_columns(risk_row, "coverage"),
                    **_metric_columns(risk_row, "known_positive_recall"),
                }
            )
    return pd.DataFrame(rows)


def _rate_error_bars(df: pd.DataFrame, metric: str) -> list[list[float]]:
    lower = (df[metric] - df[f"{metric}_ci_low"]).clip(lower=0).fillna(0)
    upper = (df[f"{metric}_ci_high"] - df[metric]).clip(lower=0).fillna(0)
    return [lower.to_list(), upper.to_list()]


def _plot_gate_ladder(ax: Any, df: pd.DataFrame, subset: str) -> None:
    from matplotlib.ticker import PercentFormatter

    ladder = df[(df["section"] == "gate_ladder") & (df["subset"] == subset)].sort_values("rung_index")
    x = ladder["rung_index"].to_numpy()
    ax.bar(x, ladder["coverage"], width=0.72, color="#d9e7f5", label="Coverage", zorder=1)
    ax.errorbar(
        x,
        ladder["FCR"],
        yerr=_rate_error_bars(ladder, "FCR"),
        color="#b33a3a",
        marker="o",
        linewidth=2,
        capsize=4,
        label="FCR (exact 95% CI)",
        zorder=3,
    )
    ax.plot(
        x,
        ladder["known_positive_recall"],
        color="#1f6f5b",
        marker="s",
        linewidth=2,
        label="Known-positive recall",
        zorder=4,
    )
    ax.set_title(subset)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels(RUNGS, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", color="#cccccc", alpha=0.45, linewidth=0.8)
    ax.set_axisbelow(True)


def _plot_risk_coverage(ax: Any, df: pd.DataFrame) -> None:
    from matplotlib.ticker import PercentFormatter

    colors = {"FULL": "#5b6fb4", "MAIN": "#c2762f"}
    risk = df[df["section"] == "risk_coverage"].sort_values(["subset", "alpha"], ascending=[True, False])
    for subset in SUBSETS:
        part = risk[risk["subset"] == subset]
        if part.empty:
            continue
        ax.errorbar(
            part["coverage"],
            part["FCR"],
            yerr=_rate_error_bars(part, "FCR"),
            marker="o",
            linewidth=2,
            capsize=4,
            color=colors[subset],
            label=subset,
        )
        labels = (
            part.groupby(["coverage", "FCR"], dropna=False)["alpha"]
            .apply(lambda values: ",".join(f"{value:.2f}" for value in values))
            .reset_index()
        )
        for _, row in labels.iterrows():
            ax.annotate(
                row["alpha"],
                (row["coverage"], row["FCR"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
                color=colors[subset],
            )
    ax.set_title("Risk-coverage alpha sweep")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("FCR")
    x_max = min(1.0, max(0.35, float(risk["coverage_ci_high"].max()) * 1.15))
    y_max = min(1.0, max(0.20, float(risk["FCR_ci_high"].max()) * 1.25))
    ax.set_xlim(-0.02, x_max)
    ax.set_ylim(-0.01, y_max)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(color="#cccccc", alpha=0.45, linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")


def write_figure(table: pd.DataFrame, out_path: Path) -> None:
    cache_root = Path(tempfile.gettempdir()) / "confirm-matplotlib"
    mpl_config = cache_root / "mpl"
    xdg_cache = cache_root / "xdg"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.25, 1.0],
        left=0.07,
        right=0.98,
        top=0.86,
        bottom=0.08,
        hspace=0.45,
        wspace=0.18,
    )
    ax_full = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[0, 1], sharey=ax_full)
    ax_risk = fig.add_subplot(grid[1, :])

    _plot_gate_ladder(ax_full, table, "FULL")
    _plot_gate_ladder(ax_main, table, "MAIN")
    _plot_risk_coverage(ax_risk, table)

    ax_full.set_ylabel("Rate")
    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.925))
    fig.suptitle("Coverage vs FCR across confirmation gates", fontsize=14, fontweight="bold", y=0.985)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def generate(combined_results: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """Generate the coverage-vs-FCR PNG and tidy CSV table."""

    combined_path = Path(combined_results)
    output_dir = Path(out_dir)
    payload = json.loads(combined_path.read_text(encoding="utf-8"))
    table = build_coverage_table(payload)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "coverage_vs_fcr_table.csv"
    figure_path = output_dir / "coverage_vs_fcr.png"
    table.to_csv(table_path, index=False)
    write_figure(table, figure_path)
    return figure_path, table_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("combined_results", help="Path to combined_benchmark_results.json")
    parser.add_argument("--out-dir", required=True, help="Directory for coverage_vs_fcr.png and CSV table")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    figure_path, table_path = generate(args.combined_results, args.out_dir)
    print(f"wrote {figure_path}")
    print(f"wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
