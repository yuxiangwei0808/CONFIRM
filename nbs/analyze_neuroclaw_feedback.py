"""Compare what an existing agent reports with and without CONFIRM's feedback.

The three regimes are cumulative stages of one pipeline, not competing systems.
The agent drafts claims and CONFIRM gates them; whatever it abstains on is then
revised under one of two feedback signals:

  no feedback        the agent's drafted claims as CONFIRM first gated them
  self-critique      the agent revises from its own panel critique, with the
                     gate-specific diagnosis withheld
  CONFIRM diagnosis  the agent revises from CONFIRM's typed gate localization

Because the feedback arms only ever see the abstentions, every count is reported
against the same denominator of executed drafts, so the regimes are comparable.
Holdout support re-checks each confirmed claim on evidence that played no part in
producing it, which is the metric a coached arm cannot game.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROTOCOL_VERSION = "neuroclaimbench-v2.1-neuroclaw-feedback-analysis-v2"

FEEDBACK_ARMS = ("self_critique", "confirm_diagnosis")
ROW_ORDER = ("no_feedback", "self_critique", "confirm_diagnosis")
ROW_LABELS = {
    "no_feedback": "No feedback",
    "self_critique": "+ agent self-critique",
    "confirm_diagnosis": "+ CONFIRM diagnosis",
}
ROW_COLORS = {
    "no_feedback": "#B07AA1",
    "self_critique": "#69737D",
    "confirm_diagnosis": "#133C66",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    supported = lambda row: row["state"].get("internally_supported_candidate_ids") or []
    return {
        "parents": len(rows),
        "recovered": sum(1 for row in rows if supported(row)),
        "supported_candidates": sum(len(supported(row)) for row in rows),
        "evaluated": sum(row["state"].get("current_data_evaluated_count") or 0 for row in rows),
        "llm_calls": sum(row["llm_calls"] for row in rows),
        "errors": sum(1 for row in rows if row["error"]),
    }


def _collect(main_dir: Path, safety_dir: Path) -> dict[str, Any]:
    baseline = json.loads((main_dir / "parents_summary.json").read_text())
    holdout_path = main_dir / "holdout_audit.json"
    holdout = json.loads(holdout_path.read_text())["arms"] if holdout_path.exists() else {}

    executed = baseline.get("executed_count") or baseline["drafted_count"]
    confirmed = baseline["confirmed_without_feedback"]

    rows: list[dict[str, Any]] = [
        {
            "regime": "no_feedback",
            "regime_label": ROW_LABELS["no_feedback"],
            "executed_drafts": executed,
            "confirmed_total": confirmed,
            "recovered_from_abstentions": "",
            "llm_calls": "",
            "candidates_per_100_calls": "",
            "holdout_supported_total": holdout.get("no_feedback", {}).get("holdout_supported", ""),
            "false_confirmations": 0,
        }
    ]
    for arm in FEEDBACK_ARMS:
        arm_rows = _read_jsonl(main_dir / f"arm_{arm}.jsonl")
        if not arm_rows:
            continue
        metrics = _arm_metrics(arm_rows)
        safety_rows = _read_jsonl(safety_dir / f"arm_{arm}.jsonl")
        safety = _arm_metrics(safety_rows) if safety_rows else None
        arm_holdout = holdout.get(arm, {}).get("holdout_supported")
        base_holdout = holdout.get("no_feedback", {}).get("holdout_supported")
        rows.append(
            {
                "regime": arm,
                "regime_label": ROW_LABELS[arm],
                "executed_drafts": executed,
                "confirmed_total": confirmed + metrics["recovered"],
                "recovered_from_abstentions": metrics["recovered"],
                "llm_calls": metrics["llm_calls"],
                "candidates_per_100_calls": round(
                    100 * metrics["supported_candidates"] / max(metrics["llm_calls"], 1), 1
                ),
                "holdout_supported_total": (
                    base_holdout + arm_holdout
                    if isinstance(base_holdout, int) and isinstance(arm_holdout, int)
                    else ""
                ),
                "false_confirmations": (
                    safety["recovered"] if safety else ""
                ),
            }
        )
    rows.sort(key=lambda row: ROW_ORDER.index(row["regime"]))
    return {
        "baseline": baseline,
        "holdout": holdout,
        "rows": rows,
        "safety_denominator": (
            _arm_metrics(_read_jsonl(safety_dir / f"arm_{FEEDBACK_ARMS[0]}.jsonl"))["parents"]
            if (safety_dir / f"arm_{FEEDBACK_ARMS[0]}.jsonl").exists()
            else 0
        ),
    }


def _plot(data: dict[str, Any], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.it": "Arial:italic",
            "mathtext.rm": "Arial",
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    rows = data["rows"]
    executed = rows[0]["executed_drafts"]
    panels: list[tuple[str, str, list[tuple[str, float, str]]]] = [
        (
            "Claims reported",
            f"of {executed} executed drafts",
            [(r["regime"], r["confirmed_total"], str(r["confirmed_total"])) for r in rows],
        )
    ]
    if all(isinstance(r["holdout_supported_total"], int) for r in rows):
        panels.append(
            (
                "Confirmed on held-out evidence",
                "evidence unused during search",
                [
                    (r["regime"], r["holdout_supported_total"], str(r["holdout_supported_total"]))
                    for r in rows
                ],
            )
        )

    # The safety arm is all zeros in every regime, so it reads as a statement
    # rather than an empty third panel of bars.
    safety_note = ""
    if all(isinstance(r["false_confirmations"], int) for r in rows) and data["safety_denominator"]:
        if all(r["false_confirmations"] == 0 for r in rows):
            safety_note = (
                f"No regime supported any of the {data['safety_denominator']} "
                "site-confounded controls."
            )

    figure, axes = plt.subplots(1, len(panels), figsize=(2.55 * len(panels) + 0.5, 2.3))
    if len(panels) == 1:
        axes = [axes]

    for axis, (title, subtitle, values) in zip(axes, panels):
        positions = range(len(values))
        axis.bar(
            list(positions),
            [v for _r, v, _t in values],
            width=0.62,
            color=[ROW_COLORS[r] for r, _v, _t in values],
            edgecolor="white",
            linewidth=0.6,
        )
        headroom = max([v for _r, v, _t in values] + [1]) * 1.22
        axis.set_ylim(0, headroom)
        for pos, (_r, value, text) in zip(positions, values):
            axis.text(pos, value + headroom * 0.03, text, ha="center", va="bottom", fontsize=8)
        axis.set_xticks([])
        axis.set_title(f"{title}\n{subtitle}, higher is better", fontsize=8, pad=6)
        axis.set_yticks([])
        axis.spines["left"].set_visible(False)

    for axis, letter in zip(axes, "abc"):
        axis.text(
            -0.04,
            1.14,
            letter,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    figure.tight_layout(rect=(0, 0.20 if safety_note else 0.12, 1, 1))

    handles = [
        Patch(facecolor=ROW_COLORS[regime], edgecolor="white", label=ROW_LABELS[regime])
        for regime in ROW_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.075 if safety_note else 0.0),
        columnspacing=1.6,
        handlelength=1.1,
    )
    if safety_note:
        figure.text(0.5, 0.005, safety_note, ha="center", va="bottom", fontsize=7.5)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def _latex_table(data: dict[str, Any]) -> str:
    rows = data["rows"]
    executed = rows[0]["executed_drafts"]
    controls = data["safety_denominator"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{CONFIRM feedback applied to an existing neuroimaging",
        r"agent.} NeuroClaw-adapted drafts every claim and CONFIRM gates it; the two",
        r"feedback regimes then revise only the claims it abstained on, so all three",
        r"rows share the same denominator of executed drafts. Both regimes use the",
        r"same model, budget, validator, and cumulative multiplicity policy, and",
        r"differ only in the feedback signal. Held-out support re-checks each",
        r"reported claim on partitions that played no part in producing it.}",
        r"\label{tab:neuroclaw_feedback}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Reporting regime & Claims reported & Recovered & LLM calls & "
        r"Held-out support & Unsafe (controls) \\",
        r"\midrule",
    ]
    for row in rows:
        recovered = row["recovered_from_abstentions"] or "--"
        calls = f"{row['llm_calls']:,}" if isinstance(row["llm_calls"], int) else "--"
        lines.append(
            f"{row['regime_label']} & {row['confirmed_total']}/{executed} & {recovered} & "
            f"{calls} & {row['holdout_supported_total']} & "
            f"{row['false_confirmations']}/{controls} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    data = _collect(Path(args.main_dir), Path(args.safety_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = data["rows"]

    with (out_dir / "feedback_arms.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "feedback_arms.json").write_text(
        json.dumps({"protocol_version": PROTOCOL_VERSION, **data}, indent=2, sort_keys=True, default=str)
    )
    _plot(data, Path(args.figure_prefix))
    table = _latex_table(data)
    (out_dir / "tab_neuroclaw_feedback.tex").write_text(table)
    if args.paper_table:
        Path(args.paper_table).write_text(table)

    executed = rows[0]["executed_drafts"]
    header = (
        f"{'regime':24}{'reported':>12}{'rate':>8}{'recovered':>11}"
        f"{'calls':>8}{'cand/100':>10}{'holdout':>9}{'false-conf':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        rate = f"{100 * row['confirmed_total'] / executed:.1f}%"
        print(
            f"{row['regime_label']:24}"
            f"{str(row['confirmed_total']) + '/' + str(executed):>12}{rate:>8}"
            f"{str(row['recovered_from_abstentions']):>11}{str(row['llm_calls']):>8}"
            f"{str(row['candidates_per_100_calls']):>10}"
            f"{str(row['holdout_supported_total']):>9}{str(row['false_confirmations']):>11}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-dir",
        default="review-stage/neuroclaimbench-v2.1/neuroclaw-feedback-v1",
    )
    parser.add_argument(
        "--safety-dir",
        default="review-stage/neuroclaimbench-v2.1/neuroclaw-feedback-safety-v1",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/neuroclaw-feedback-v1",
    )
    parser.add_argument("--figure-prefix", default="paper/figures/fig_neuroclaw_feedback")
    parser.add_argument("--paper-table", default="paper/figures/tab_neuroclaw_feedback.tex")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
