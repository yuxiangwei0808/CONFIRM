"""Retrofit CONFIRM onto an existing neuroimaging agent and measure what changes.

NeuroClaw-adapted decides what it would report from the frozen evidence. CONFIRM
can then be attached in two ways: as a downstream veto, where a claim is reported
only if the agent supports it and the gates pass, or as a substitute, where the
gates replace the agent's reporting decision. Both arms score identical claims
and identical evidence, so the difference isolates the governance layer.

Scope is internal: scalar scientific literature references plus the synthetic
controls. External-transfer strata are excluded, and the 24 brain-wide claims are
outside NeuroClaw-adapted's scalar-only coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from confirm.verdict import classify_support

PROTOCOL_VERSION = "neuroclaimbench-v2.1-retrofit-composition-v1"

AGENT_METHOD = "neuroclaw_adapted_judge"
AGENT_LABEL = "NeuroClaw-adapted"

#: Scored strata retained. External-transfer strata are deliberately excluded.
METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("recovery", "internal_scientific", "confirm", "Literature recovery"),
    ("unsafe_support", "internal_scientific", "abstain", "Unsafe literature support"),
    ("synthetic_support", "synthetic_control", "abstain", "Synthetic false confirmations"),
)

ARM_COLORS = {
    "agent_alone": "#B07AA1",
    "veto_screen": "#4B8BBE",
    "veto_full": "#133C66",
    "substitute_screen": "#4B8BBE",
    "substitute_full": "#133C66",
}
ARM_HATCH = {"substitute_screen": "//", "substitute_full": "//"}


def _load_agent(path: Path) -> tuple[dict[str, bool], dict[str, tuple[str, str]]]:
    decisions: dict[str, bool] = {}
    meta: dict[str, tuple[str, str]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] != AGENT_METHOD:
                continue
            if row["available"].strip().lower() != "true":
                continue
            decisions[row["task_id"]] = row["supported"].strip().lower() == "true"
            meta[row["task_id"]] = (row["stratum"], row["reference_disposition"])
    return decisions, meta


def _load_confirm(path: Path, task_ids: set[str]) -> dict[str, dict[str, bool]]:
    tiers = {"screen": "discovery", "full": "confirmed"}
    out: dict[str, dict[str, bool]] = {name: {} for name in tiers}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id not in task_ids:
                continue
            gates = record["gate_results"]["verdict"]["gates"]
            for name, tier in tiers.items():
                out[name][task_id] = classify_support(gates, tier).supported
    return out


def _arms(
    agent: dict[str, bool],
    confirm: dict[str, dict[str, bool]],
) -> dict[str, tuple[str, Callable[[str], bool]]]:
    return {
        "agent_alone": (f"{AGENT_LABEL} alone", lambda t: agent[t]),
        "veto_screen": (
            "+ CONFIRM-Screen (veto)",
            lambda t: agent[t] and confirm["screen"][t],
        ),
        "veto_full": (
            "+ CONFIRM-Full (veto)",
            lambda t: agent[t] and confirm["full"][t],
        ),
        "substitute_screen": (
            "CONFIRM-Screen (substitute)",
            lambda t: confirm["screen"][t],
        ),
        "substitute_full": (
            "CONFIRM-Full (substitute)",
            lambda t: confirm["full"][t],
        ),
    }


def _rows(
    arms: dict[str, tuple[str, Callable[[str], bool]]],
    meta: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_id, (label, predicate) in arms.items():
        for metric, stratum, disposition, metric_label in METRICS:
            task_ids = [t for t, m in meta.items() if m == (stratum, disposition)]
            rows.append(
                {
                    "arm": arm_id,
                    "arm_label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "supported_count": sum(predicate(t) for t in task_ids),
                    "denominator": len(task_ids),
                }
            )
    return rows


def _plot(rows: list[dict[str, Any]], output_prefix: Path) -> None:
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
    arm_order = ["agent_alone", "veto_screen", "veto_full", "substitute_screen", "substitute_full"]
    labels = {row["arm"]: row["arm_label"] for row in rows}
    lookup = {(row["arm"], row["metric"]): row for row in rows}

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(6.6, 2.5),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.25], "wspace": 0.18},
    )
    positions = range(len(arm_order))

    for axis, (metric, _stratum, _disposition, metric_label) in zip(axes, METRICS):
        denominator = lookup[(arm_order[0], metric)]["denominator"]
        values = [lookup[(arm, metric)]["supported_count"] for arm in arm_order]
        axis.barh(
            list(positions),
            values,
            height=0.62,
            color=[ARM_COLORS[arm] for arm in arm_order],
            hatch=[ARM_HATCH.get(arm, "") for arm in arm_order],
            edgecolor="white",
            linewidth=0.6,
        )
        headroom = max(max(values), 1) * 1.28
        axis.set_xlim(0, headroom)
        for index, value in zip(positions, values):
            axis.text(
                value + headroom * 0.03,
                index,
                str(value),
                va="center",
                ha="left",
                fontsize=7.5,
                color="#222222",
            )
        direction = "higher is better" if metric == "recovery" else "lower is better"
        axis.set_title(f"{metric_label}\n$n$={denominator}, {direction}", fontsize=8, pad=6)
        axis.set_xticks([])
        axis.spines["bottom"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(list(positions))
    axes[0].set_yticklabels([labels[arm] for arm in arm_order], fontsize=8)
    axes[0].invert_yaxis()

    for axis, letter in zip(axes, "abc"):
        axis.text(
            -0.02,
            1.14,
            letter,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent, meta = _load_agent(Path(args.joined_decisions))
    meta = {
        task_id: value
        for task_id, value in meta.items()
        if value[0] in {"internal_scientific", "synthetic_control"}
    }
    agent = {task_id: value for task_id, value in agent.items() if task_id in meta}
    confirm = _load_confirm(Path(args.task_outcomes), set(agent))

    missing = [task_id for task_id in agent if task_id not in confirm["full"]]
    if missing:
        raise KeyError(f"No execution record for {len(missing)} scored tasks")

    rows = _rows(_arms(agent, confirm), meta)
    fieldnames = list(rows[0])
    for target in (out_dir / "retrofit_composition.csv", Path(args.figure_prefix + "_source.csv")):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _plot(rows, Path(args.figure_prefix))

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "agent_method": AGENT_METHOD,
        "joined_decisions": str(args.joined_decisions),
        "task_outcomes": str(args.task_outcomes),
        "scored_task_count": len(agent),
        "stratum_counts": dict(Counter(f"{s}:{d}" for s, d in meta.values())),
        "excluded": {
            "external_transfer_strata": "out of scope for this comparison",
            "brainwide_claims": "NeuroClaw-adapted is scalar-only (method_not_applicable)",
        },
        "interpretation_restrictions": [
            "Both arms score identical claims and identical frozen evidence.",
            "Veto composition cannot exceed the agent's own recovery by construction.",
            "Synthetic controls carry construction ground truth; literature references are adjudicated.",
            "NeuroClaw-adapted is a persona adaptation, not the released system.",
        ],
    }
    (out_dir / "retrofit_composition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    width = max(len(row["arm_label"]) for row in rows)
    print(f"scored tasks: {len(agent)}")
    for arm in ("agent_alone", "veto_screen", "veto_full", "substitute_screen", "substitute_full"):
        cells = " ".join(
            f"{m[3]} {next(r for r in rows if r['arm'] == arm and r['metric'] == m[0])['supported_count']}"
            f"/{next(r for r in rows if r['arm'] == arm and r['metric'] == m[0])['denominator']}"
            for m in METRICS
        )
        label = next(r["arm_label"] for r in rows if r["arm"] == arm)
        print(f"  {label:{width}}  {cells}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joined-decisions",
        default="review-stage/neuroclaimbench-v2.1/claim-evaluation-baselines-v2/joined_decisions.csv",
    )
    parser.add_argument(
        "--task-outcomes",
        default="review-stage/neuroclaimbench-v2.1/results/task_outcomes.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/retrofit-composition-v1",
    )
    parser.add_argument("--figure-prefix", default="paper/figures/fig_retrofit_composition")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
