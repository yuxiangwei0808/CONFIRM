"""Shared plotting and LaTeX helpers for CONFIRM paper figures."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "figures"

ARTIFACTS = {
    "combined_results": ROOT / "review-stage" / "round5-combat" / "combined_benchmark_results.json",
    "combined_audit": ROOT / "review-stage" / "round5-combat" / "combined_benchmark_audit.csv",
    "negatives_results": ROOT / "review-stage" / "negatives-expansion" / "negatives_expansion_results.json",
    "negatives_audit": ROOT / "review-stage" / "negatives-expansion" / "negatives_expansion_audit.csv",
    "neuroclaw": ROOT / "review-stage" / "round5-neuroclaw" / "neuroclaw_comparison.json",
    "confirm_layer": ROOT / "review-stage" / "confirm-layer" / "confirm_layer_result.json",
    "multillm": ROOT / "review-stage" / "agentic-multillm" / "agentic_multillm_summary_full_sweep_v2.json",
}

RUNGS = ["exec_only", "+confound", "+power", "+multiverse", "+replication"]
RUNG_LABELS = {
    "exec_only": "exec-only",
    "+confound": "+confound",
    "+power": "+power",
    "+multiverse": "+multiverse",
    "+replication": "+replication",
}

# Okabe-Ito / colorblind-friendly colors.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#666666",
    "light_blue": "#CFE8F3",
}


def setup_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "confirm-paper-matplotlib"
    mpl_config = cache_root / "mpl"
    xdg_cache = cache_root / "xdg"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    return plt


def load_json(name: str) -> dict[str, Any]:
    path = ARTIFACTS[name]
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_pdf(fig: Any, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / filename
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    return out_path


def rate_ci_error(rate: float, ci: list[float] | tuple[float, float]) -> list[list[float]]:
    lo, hi = float(ci[0]), float(ci[1])
    return [[max(rate - lo, 0.0)], [max(hi - rate, 0.0)]]


def rate_ci_text(rate: float, lo: float, hi: float, count: Any, denominator: Any, digits: int = 3) -> str:
    return f"{float(rate):.{digits}f} [{float(lo):.{digits}f}, {float(hi):.{digits}f}] ({count}/{denominator})"


def stat_rate(stat: dict[str, Any], metric: str | None = None) -> float:
    if metric and metric in stat:
        return float(stat[metric])
    if "rate" in stat:
        return float(stat["rate"])
    count = float(stat["count"] if "count" in stat else stat.get("FCR_count"))
    denominator = float(stat["denominator"] if "denominator" in stat else stat.get("FCR_denominator"))
    if denominator == 0:
        return math.nan
    return count / denominator


def stat_ci(stat: dict[str, Any], metric: str | None = None) -> tuple[float, float]:
    key = f"{metric}_ci95_exact" if metric else "ci95_exact"
    if key not in stat and "ci95_exact" in stat:
        key = "ci95_exact"
    ci = stat[key]
    return float(ci[0]), float(ci[1])


def stat_count_denominator(stat: dict[str, Any], metric: str | None = None) -> tuple[Any, Any]:
    if metric:
        return stat[f"{metric}_count"], stat[f"{metric}_denominator"]
    return stat["count"], stat["denominator"]


def stat_text(stat: dict[str, Any], metric: str | None = None, digits: int = 3) -> str:
    rate = stat_rate(stat, metric)
    lo, hi = stat_ci(stat, metric)
    count, denominator = stat_count_denominator(stat, metric)
    return rate_ci_text(rate, lo, hi, count, denominator, digits=digits)


def stat_rate_ci_only(stat: dict[str, Any], metric: str | None = None, digits: int = 3) -> str:
    rate = stat_rate(stat, metric)
    lo, hi = stat_ci(stat, metric)
    return f"{rate:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def stat_count_text(stat: dict[str, Any], metric: str | None = None) -> str:
    count, denominator = stat_count_denominator(stat, metric)
    return f"{count}/{denominator}"


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_float(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "--"
    x = float(value)
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
