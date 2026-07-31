"""Create paper figures and tables for CONFIRM baselines and evidence tiers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


TIER_LABELS = {
    "discovery": "CONFIRM-Screen",
    "replicated": "CONFIRM-Replicate",
    "confirmed": "CONFIRM-Full",
}
FULL_METHOD_ORDER = (
    "direct_llm_judge",
    "discovery",
    "replicated",
    "confirmed",
)
FULL_METHOD_LABELS = {
    "direct_llm_judge": "Direct LLM judge",
    **TIER_LABELS,
}
# Methods that must all be present for a task to count toward the common scalar
# set. CONFIRM appears here once, as the joined baseline decisions carry a single
# CONFIRM row per task rather than one row per evidence policy.
SCALAR_METHOD_ORDER = (
    "direct_llm_judge",
    "conventional_significance",
    "confirm",
)
# Display order for the common scalar panel. The evidence policies are scored on
# the same common tasks from the per-task tier decisions, so the panel shows the
# same method set as the synthetic panel.
SCALAR_PANEL_ORDER = (
    "direct_llm_judge",
    "conventional_significance",
    "discovery",
    "replicated",
    "confirmed",
)
SCALAR_METHOD_LABELS = {
    "direct_llm_judge": "Direct LLM judge",
    "conventional_significance": "Significance filter",
    "confirm": "CONFIRM-Full",
    **TIER_LABELS,
}
SYNTHETIC_METHOD_ORDER = (
    "direct_llm_judge",
    "conventional_significance",
    "discovery",
    "replicated",
    "confirmed",
)
SYNTHETIC_METHOD_LABELS = {
    "direct_llm_judge": "Direct LLM judge",
    "conventional_significance": "Significance filter",
    **TIER_LABELS,
}
RECOVERY_COLOR = "#247B6B"
UNSAFE_COLOR = "#C8574D"
SYNTHETIC_COLOR = "#9D4E48"
METHOD_COLORS = {
    "direct_llm_judge": "#69737D",
    "conventional_significance": "#9A8064",
    "discovery": "#4B8BBE",
    "replicated": "#2D6A9F",
    "confirmed": "#133C66",
}
# Adapted neuroimaging systems evaluated on the common scalar set.
SYSTEM_METHOD_ORDER = ("veritas_adapted", "neuroclaw_adapted_judge")
SYSTEM_METHOD_LABELS = {
    "veritas_adapted": "VERITAS-adapted",
    "neuroclaw_adapted_judge": "NeuroClaw-adapted",
}
# Recovery vs safety frontier: one restrained palette per family
# (neuroimaging systems = purple signal, simple baselines = neutral, CONFIRM = blue accent).
FRONTIER_ORDER = (
    "veritas_adapted",
    "neuroclaw_adapted_judge",
    "direct_llm_judge",
    "conventional_significance",
    "discovery",
    "replicated",
    "confirmed",
)
FRONTIER_LABELS = {
    **SYSTEM_METHOD_LABELS,
    "direct_llm_judge": "Direct LLM judge",
    "conventional_significance": "Significance filter",
    **TIER_LABELS,
}
FRONTIER_COLORS = {
    "veritas_adapted": "#8E6C9E",
    "neuroclaw_adapted_judge": "#B07AA1",
    "direct_llm_judge": "#69737D",
    "conventional_significance": "#9A8064",
    "discovery": "#4B8BBE",
    "replicated": "#2D6A9F",
    "confirmed": "#133C66",
}
FRONTIER_MARKERS = {
    "veritas_adapted": "s",
    "neuroclaw_adapted_judge": "s",
    "direct_llm_judge": "^",
    "conventional_significance": "^",
    "discovery": "o",
    "replicated": "o",
    "confirmed": "o",
}
GATE_ORDER = (
    "search_provenance",
    "multiplicity",
    "confounding",
    "power",
    "multiverse",
    "replication",
)
GATE_LABELS = {
    "search_provenance": "Search provenance",
    "multiplicity": "Multiplicity",
    "confounding": "Measured confounds",
    "power": "Power",
    "multiverse": "Analytic stability",
    "replication": "Replication",
}
ABLATION_STRATA = (
    ("scientific_literature", "confirm", "additional_supported_references"),
    ("scientific_literature", "abstain", "additional_unsafe_references"),
    ("synthetic_constructed", "abstain", "additional_synthetic_controls"),
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(
    rows: list[dict[str, str]],
    **criteria: str,
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if all(row.get(field) == value for field, value in criteria.items())
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {criteria}, found {len(selected)}"
        )
    return selected[0]


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric(count: str | int, denominator: str | int) -> dict[str, Any]:
    numerator = int(count)
    total = int(denominator)
    return {
        "count": numerator,
        "denominator": total,
        "rate": numerator / total,
    }


def _baseline_metric(
    rows: list[dict[str, str]],
    *,
    method: str,
    stratum: str,
    disposition: str,
) -> dict[str, Any]:
    row = _one(
        rows,
        method=method,
        stratum=stratum,
        reference_disposition=disposition,
    )
    # Denominator is every eligible case, not just those a method could score.
    # An unscoreable case counts as unsupported, so all methods share one
    # denominator (see _common_scalar_metrics).
    return _metric(row["supported_count"], row["eligible_count"])


def _tier_metric(
    rows: list[dict[str, str]],
    *,
    tier: str,
    metric: str,
) -> dict[str, Any]:
    row = _one(rows, minimum_evidence_tier=tier, metric=metric)
    return _metric(row["supported_count"], row["denominator"])


def _common_scalar_metrics(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    methods = set(SCALAR_METHOD_ORDER)
    by_task: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["method"] in methods:
            by_task.setdefault(row["task_id"], {})[row["method"]] = row
    # Every method must have scored the task. A method that cannot evaluate a
    # case counts as not supporting it, the same fail-closed convention CONFIRM
    # applies to missing gate information, so all methods share one denominator.
    common = {
        task_id: method_rows
        for task_id, method_rows in by_task.items()
        if methods == set(method_rows)
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for method in SCALAR_METHOD_ORDER:
        result[method] = {}
        for stratum, disposition, metric_name in (
            ("internal_scientific", "confirm", "recovery"),
            ("internal_scientific", "abstain", "unsafe"),
            ("synthetic_control", "abstain", "synthetic"),
        ):
            selected = [
                method_rows[method]
                for method_rows in common.values()
                if method_rows[method]["stratum"] == stratum
                and method_rows[method]["reference_disposition"] == disposition
            ]
            result[method][metric_name] = _metric(
                sum(row["supported"].lower() == "true" for row in selected),
                len(selected),
            )
    return result


def _comparison_data(
    baseline_rows: list[dict[str, str]],
    common_rows: list[dict[str, str]],
    tier_rows: list[dict[str, str]],
    tier_decisions: list[dict[str, str]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    full: dict[str, dict[str, dict[str, Any]]] = {
        "direct_llm_judge": {
            "recovery": _baseline_metric(
                baseline_rows,
                method="direct_llm_judge",
                stratum="internal_scientific",
                disposition="confirm",
            ),
            "unsafe": _baseline_metric(
                baseline_rows,
                method="direct_llm_judge",
                stratum="internal_scientific",
                disposition="abstain",
            ),
        }
    }
    for tier in TIER_LABELS:
        full[tier] = {
            "recovery": _tier_metric(
                tier_rows,
                tier=tier,
                metric="confirmable_reference_recall",
            ),
            "unsafe": _tier_metric(
                tier_rows,
                tier=tier,
                metric="literature_abstention_unsafe_support",
            ),
        }
    common = _common_scalar_metrics(common_rows)
    # Score the evidence policies on exactly the tasks that define the common
    # scalar set, so every panel of the figure covers the same method set.
    common_task_sets = _common_task_ids(common_rows)
    for tier in TIER_LABELS:
        common[tier] = {
            metric: _tier_counts_for_tasks(
                tier_decisions,
                tier=tier,
                task_ids=common_task_sets[key],
            )
            for metric, key in (
                ("recovery", ("internal_scientific", "confirm")),
                ("unsafe", ("internal_scientific", "abstain")),
                ("synthetic", ("synthetic_control", "abstain")),
            )
        }
    synthetic = {
        "direct_llm_judge": _baseline_metric(
            baseline_rows,
            method="direct_llm_judge",
            stratum="synthetic_control",
            disposition="abstain",
        ),
        "conventional_significance": _baseline_metric(
            baseline_rows,
            method="conventional_significance",
            stratum="synthetic_control",
            disposition="abstain",
        ),
    }
    for tier in TIER_LABELS:
        synthetic[tier] = _tier_metric(
            tier_rows,
            tier=tier,
            metric="synthetic_control_support",
        )
    return full, common, synthetic


def _annotate_bar(
    axis: Any,
    *,
    value: float,
    y: float,
    count: int,
    denominator: int,
) -> None:
    label = f"{count}/{denominator}"
    if value >= 62:
        axis.text(
            value - 1.5,
            y,
            label,
            ha="right",
            va="center",
            fontsize=6.1,
            color="white",
            fontweight="bold",
        )
    else:
        axis.text(
            value + 1.3,
            y,
            label,
            ha="left",
            va="center",
            fontsize=6.1,
            color="#202020",
        )


def _plot_pair_panel(
    axis: Any,
    *,
    data: dict[str, dict[str, dict[str, Any]]],
    order: tuple[str, ...],
    labels: dict[str, str],
    title: str,
    panel: str,
) -> None:
    y = np.arange(len(order))
    height = 0.26
    for index, method in enumerate(order):
        recovery = data[method]["recovery"]
        unsafe = data[method]["unsafe"]
        recovery_pct = 100.0 * recovery["rate"]
        unsafe_pct = 100.0 * unsafe["rate"]
        axis.barh(
            y[index] - height / 1.8,
            recovery_pct,
            height=height,
            color=RECOVERY_COLOR,
        )
        axis.barh(
            y[index] + height / 1.8,
            unsafe_pct,
            height=height,
            color=UNSAFE_COLOR,
        )
        _annotate_bar(
            axis,
            value=recovery_pct,
            y=y[index] - height / 1.8,
            count=recovery["count"],
            denominator=recovery["denominator"],
        )
        _annotate_bar(
            axis,
            value=unsafe_pct,
            y=y[index] + height / 1.8,
            count=unsafe["count"],
            denominator=unsafe["denominator"],
        )
    axis.set_yticks(y, [labels[method] for method in order])
    axis.invert_yaxis()
    axis.set_xlim(0, 72)
    axis.set_xticks((0, 25, 50))
    axis.set_xlabel("Claims supported (%)", fontsize=6.7)
    axis.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=8)
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.55, zorder=0)
    axis.tick_params(axis="y", labelsize=6.25, length=0, pad=4)
    axis.tick_params(axis="x", labelsize=6.2)
    axis.text(
        -0.17,
        1.10,
        panel,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
    )


def _plot_policy_comparison(
    baseline_rows: list[dict[str, str]],
    common_rows: list[dict[str, str]],
    tier_rows: list[dict[str, str]],
    tier_decisions: list[dict[str, str]],
    output_prefix: Path,
) -> list[dict[str, object]]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    full, common, synthetic = _comparison_data(
        baseline_rows,
        common_rows,
        tier_rows,
        tier_decisions,
    )
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

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.35, 3.25),
        gridspec_kw={"width_ratios": (1.12, 1.0, 1.10)},
    )
    _plot_pair_panel(
        axes[0],
        data=full,
        order=FULL_METHOD_ORDER,
        labels=FULL_METHOD_LABELS,
        title="All scientific modalities\n51 positive, 19 abstention references",
        panel="a",
    )
    _plot_pair_panel(
        axes[1],
        data=common,
        order=SCALAR_PANEL_ORDER,
        labels=SCALAR_METHOD_LABELS,
        title="Common scalar claims\n31 positive, 15 abstention references",
        panel="b",
    )
    axes[0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=RECOVERY_COLOR),
            plt.Rectangle((0, 0), 1, 1, color=UNSAFE_COLOR),
        ],
        labels=("Recovery (positive references)", "False support (literature)"),
        loc="lower left",
        bbox_to_anchor=(0.0, 1.18),
        ncol=2,
        fontsize=6.4,
        columnspacing=1.1,
        handlelength=1.2,
    )

    y = np.arange(len(SYNTHETIC_METHOD_ORDER))
    for index, method in enumerate(SYNTHETIC_METHOD_ORDER):
        metric = synthetic[method]
        value = 100.0 * metric["rate"]
        axes[2].barh(
            y[index],
            value,
            height=0.45,
            color=METHOD_COLORS[method],
        )
        _annotate_bar(
            axes[2],
            value=value,
            y=y[index],
            count=metric["count"],
            denominator=metric["denominator"],
        )
    axes[2].set_yticks(
        y,
        [SYNTHETIC_METHOD_LABELS[method] for method in SYNTHETIC_METHOD_ORDER],
    )
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 36)
    axes[2].set_xticks((0, 15, 30))
    axes[2].set_xlabel("Controls supported (%)", fontsize=6.7)
    axes[2].set_title(
        "False confirmations on controls\n(lower is better)",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=8,
    )
    axes[2].grid(axis="x", color="#E5E5E5", linewidth=0.55, zorder=0)
    axes[2].tick_params(axis="y", labelsize=6.25, length=0, pad=4)
    axes[2].tick_params(axis="x", labelsize=6.2)
    axes[2].text(
        -0.17,
        1.10,
        "c",
        transform=axes[2].transAxes,
        fontsize=9,
        fontweight="bold",
    )

    source_rows: list[dict[str, object]] = []
    for panel, data, order, labels in (
        ("a", full, FULL_METHOD_ORDER, FULL_METHOD_LABELS),
        ("b", common, SCALAR_PANEL_ORDER, SCALAR_METHOD_LABELS),
    ):
        for method in order:
            for metric_name, direction in (
                ("recovery", "higher"),
                ("unsafe", "lower"),
            ):
                metric = data[method][metric_name]
                source_rows.append(
                    {
                        "panel": panel,
                        "scope": (
                            "all_scientific_modalities"
                            if panel == "a"
                            else "common_scalar_scientific"
                        ),
                        "metric": metric_name,
                        "direction": direction,
                        "method": method,
                        "method_label": labels[method],
                        "supported_count": metric["count"],
                        "denominator": metric["denominator"],
                        "supported_rate": metric["rate"],
                    }
                )
    for method in SYNTHETIC_METHOD_ORDER:
        metric = synthetic[method]
        source_rows.append(
            {
                "panel": "c",
                "scope": "synthetic_controls",
                "metric": "synthetic_false_support",
                "direction": "lower",
                "method": method,
                "method_label": SYNTHETIC_METHOD_LABELS[method],
                "supported_count": metric["count"],
                "denominator": metric["denominator"],
                "supported_rate": metric["rate"],
            }
        )

    figure.subplots_adjust(
        left=0.155,
        right=0.995,
        bottom=0.17,
        top=0.77,
        wspace=0.54,
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
    return source_rows


def _ablation_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    indexed = {
        (
            row["removed_gate"],
            row["stratum"],
            row["reference_disposition"],
        ): row
        for row in rows
    }
    result: list[dict[str, object]] = []
    for gate in GATE_ORDER:
        output: dict[str, object] = {
            "removed_gate": gate,
            "removed_gate_label": GATE_LABELS[gate],
        }
        for stratum, disposition, field in ABLATION_STRATA:
            output[field] = int(
                indexed[(gate, stratum, disposition)]["added_confirmation_count"]
            )
        result.append(output)
    return result


def _write_ablation_table(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Leave-one-gate-out ablation of CONFIRM.} Each row",
        r"removes one gate from the otherwise complete pipeline and reports the",
        r"additional claims that would be confirmed. The first result column is",
        r"a sensitivity gain; the other two are safety losses. A zero does not",
        r"imply that a gate is unnecessary because overlapping gate failures can",
        r"mask its isolated contribution.}",
        r"\label{tab:confirm_gate_ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Removed gate & Supported literature & Literature abstentions & Synthetic negatives \\",
        r" & additionally recovered & additionally confirmed & additionally confirmed \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['removed_gate_label']} & "
            f"+{row['additional_supported_references']} & "
            f"+{row['additional_unsafe_references']} & "
            f"+{row['additional_synthetic_controls']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_feedback_table(
    method_rows: list[dict[str, str]],
    evidence_rows: list[dict[str, str]],
    path: Path,
) -> None:
    order = ("failure_blind", "self_refine", "failure_specific")
    labels = {
        "failure_blind": "Failure-blind retry",
        "self_refine": "Self-Refine",
        "failure_specific": "CONFIRM diagnosis",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Follow-up search under a matched three-round,",
        r"five-candidate budget.} All methods use GPT-5.5 and the same",
        r"contracts, validator, and multiplicity policy. Candidates are the",
        r"contracts evaluated on source data; supported candidates are",
        r"same-data exploratory passes after final multiplicity adjustment.",
        r"Parents counts abstained parents with at least one supported",
        r"candidate. Holdout results are retrospective.}",
        r"\label{tab:feedback_control_main}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Method & Calls & Cand. & Supported & Parents & Holdout \\",
        r"\midrule",
    ]
    for method in order:
        summary = _one(
            method_rows,
            method=method,
            track="scientific",
        )
        evidence = _one(
            evidence_rows,
            method=method,
            track="scientific",
            evidence_kind="holdout",
            evidence_set_id="internal_holdout",
        )
        lines.append(
            f"{labels[method]} & "
            f"{int(summary['llm_call_count']):,} & "
            f"{int(summary['source_evaluated_candidate_count']):,} & "
            f"{int(summary['final_source_supported_candidate_count']):,} & "
            f"{int(summary['parents_with_source_support_count'])}/"
            f"{int(summary['parent_count'])} & "
            f"{int(evidence['supported_candidate_count'])}/"
            f"{int(evidence['evaluated_candidate_count'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_fraction(
    rows: list[dict[str, str]],
    *,
    tier: str,
    stratum: str,
    unit: str,
    disposition: str,
) -> str:
    row = _one(
        rows,
        minimum_evidence_tier=tier,
        stratum=stratum,
        unit=unit,
        reference_disposition=disposition,
    )
    return f"{int(row['supported_count'])}/{int(row['available_count'])}"


def _write_tier_modality_table(
    tier_summary: list[dict[str, str]],
    path: Path,
) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{CONFIRM evidence policies by claim modality.} Recovery",
        r"uses positive references; unsafe support uses literature",
        r"abstention references. Synthetic controls are reported separately. The",
        r"adapted neuroimaging systems and the significance filter are scalar-only",
        r"and omitted here.}",
        r"\label{tab:confirm_tiers_modality}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Policy & Scalar recovery & Scalar unsafe & Brain-wide recovery & Brain-wide unsafe & Synthetic \\",
        r"\midrule",
    ]
    for tier in ("discovery", "replicated", "confirmed"):
        lines.append(
            f"{TIER_LABELS[tier]} & "
            f"{_summary_fraction(tier_summary, tier=tier, stratum='internal_scientific', unit='scalar', disposition='confirm')} & "
            f"{_summary_fraction(tier_summary, tier=tier, stratum='internal_scientific', unit='scalar', disposition='abstain')} & "
            f"{_summary_fraction(tier_summary, tier=tier, stratum='internal_scientific', unit='brainwide', disposition='confirm')} & "
            f"{_summary_fraction(tier_summary, tier=tier, stratum='internal_scientific', unit='brainwide', disposition='abstain')} & "
            f"{_summary_fraction(tier_summary, tier=tier, stratum='synthetic_control', unit='scalar', disposition='abstain')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _common_task_ids(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], set[str]]:
    methods = set(SCALAR_METHOD_ORDER)
    by_task: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row["method"] in methods:
            by_task.setdefault(row["task_id"], {})[row["method"]] = row
    output: dict[tuple[str, str], set[str]] = {}
    for task_id, method_rows in by_task.items():
        if methods != set(method_rows):
            continue
        # Unavailable counts as unsupported (see _common_scalar_metrics), so the
        # task stays in the shared denominator.
        example = method_rows["confirm"]
        output.setdefault(
            (example["stratum"], example["reference_disposition"]),
            set(),
        ).add(task_id)
    return output


def _tier_fraction_for_tasks(
    tier_decisions: list[dict[str, str]],
    *,
    tier: str,
    task_ids: set[str],
) -> str:
    selected = [
        row
        for row in tier_decisions
        if row["minimum_evidence_tier"] == tier
        and row["task_id"] in task_ids
    ]
    if len(selected) != len(task_ids):
        raise ValueError(
            f"Tier decision coverage mismatch for {tier}: "
            f"{len(selected)} != {len(task_ids)}"
        )
    supported = sum(row["supported"].lower() == "true" for row in selected)
    return f"{supported}/{len(selected)}"


def _write_common_scalar_table(
    common_rows: list[dict[str, str]],
    tier_decisions: list[dict[str, str]],
    systems_rows: list[dict[str, str]],
    path: Path,
) -> None:
    common = _common_scalar_metrics(common_rows)
    task_ids = _common_task_ids(common_rows)
    positive = task_ids[("internal_scientific", "confirm")]
    abstention = task_ids[("internal_scientific", "abstain")]
    synthetic = task_ids[("synthetic_control", "abstain")]
    systems = _systems_common_metrics(
        systems_rows,
        {"recovery": positive, "unsafe": abstention, "synthetic": synthetic},
    )

    def cell(metrics: dict[str, dict[str, Any]]) -> str:
        return (
            f"{metrics['recovery']['count']}/{metrics['recovery']['denominator']} & "
            f"{metrics['unsafe']['count']}/{metrics['unsafe']['denominator']} & "
            f"{metrics['synthetic']['count']}/{metrics['synthetic']['denominator']}"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Claim-evaluation methods on common scalar",
        r"coverage.} All methods score the same scientific claims and synthetic",
        r"controls. Recovery is better when higher; false and synthetic support",
        r"are better when lower. CONFIRM is the only family that reaches zero",
        r"false support on the constructed controls, and no adapted system is",
        r"Pareto-superior to it.}",
        r"\label{tab:claim_evaluation_common_scalar}",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Method & Recovery & False support & Synthetic support \\",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\textit{Adapted neuroimaging systems}}\\",
    ]
    for method in SYSTEM_METHOD_ORDER:
        lines.append(f"{SYSTEM_METHOD_LABELS[method]} & {cell(systems[method])} \\\\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{4}{@{}l}{\textit{Simple reporting baselines}}\\")
    for method in ("direct_llm_judge", "conventional_significance"):
        lines.append(f"{SCALAR_METHOD_LABELS[method]} & {cell(common[method])} \\\\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{4}{@{}l}{\textit{CONFIRM (this work)}}\\")
    for tier in ("discovery", "replicated", "confirmed"):
        lines.append(
            f"{TIER_LABELS[tier]} & "
            f"{_tier_fraction_for_tasks(tier_decisions, tier=tier, task_ids=positive)} & "
            f"{_tier_fraction_for_tasks(tier_decisions, tier=tier, task_ids=abstention)} & "
            f"{_tier_fraction_for_tasks(tier_decisions, tier=tier, task_ids=synthetic)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_external_tier_table(
    tier_summary: list[dict[str, str]],
    path: Path,
) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Evidence policies on retrospective external-transfer",
        r"tasks.} Literature-positive, literature-abstention, and constructed",
        r"control results are kept separate. A dash indicates that the",
        r"external set has no score-eligible abstention reference.}",
        r"\label{tab:confirm_tiers_external}",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"External set & Policy & Positive recovery & False support & Controls \\",
        r"\midrule",
    ]
    for dataset_label, literature_stratum, control_stratum in (
        (
            "AD/aging (NACC)",
            "external_literature_NACC",
            "external_constructed_control_NACC",
        ),
        (
            "Psychosis (CNP)",
            "external_literature_ds000030",
            "external_constructed_control_ds000030",
        ),
    ):
        for tier in ("discovery", "replicated", "confirmed"):
            unsafe = (
                "--"
                if dataset_label == "AD/aging (NACC)"
                else _summary_fraction(
                    tier_summary,
                    tier=tier,
                    stratum=literature_stratum,
                    unit="scalar",
                    disposition="abstain",
                )
            )
            lines.append(
                f"{dataset_label} & {TIER_LABELS[tier]} & "
                f"{_summary_fraction(tier_summary, tier=tier, stratum=literature_stratum, unit='scalar', disposition='confirm')} & "
                f"{unsafe} & "
                f"{_summary_fraction(tier_summary, tier=tier, stratum=control_stratum, unit='scalar', disposition='abstain')} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _systems_common_metrics(
    systems_rows: list[dict[str, str]],
    task_sets: dict[str, set[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Count adapted-system support on the exact common task sets."""

    index = {
        (row["task_id"], row["method"]): row
        for row in systems_rows
        if row["method"] in SYSTEM_METHOD_ORDER
    }
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for method in SYSTEM_METHOD_ORDER:
        result[method] = {}
        for metric_name, task_ids in task_sets.items():
            supported = 0
            for task_id in task_ids:
                row = index.get((task_id, method))
                if row is None or row["available"].lower() != "true":
                    raise ValueError(
                        f"Missing available {method} decision for {task_id}"
                    )
                supported += int(row["supported"].lower() == "true")
            result[method][metric_name] = _metric(supported, len(task_ids))
    return result


def _tier_counts_for_tasks(
    tier_decisions: list[dict[str, str]],
    *,
    tier: str,
    task_ids: set[str],
) -> dict[str, Any]:
    selected = [
        row
        for row in tier_decisions
        if row["minimum_evidence_tier"] == tier and row["task_id"] in task_ids
    ]
    if len(selected) != len(task_ids):
        raise ValueError(
            f"Tier decision coverage mismatch for {tier}: "
            f"{len(selected)} != {len(task_ids)}"
        )
    supported = sum(row["supported"].lower() == "true" for row in selected)
    return _metric(supported, len(task_ids))


def _frontier_metrics(
    common_rows: list[dict[str, str]],
    tier_decisions: list[dict[str, str]],
    systems_rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Recovery / unsafe / synthetic for all seven methods on the common set."""

    common = _common_scalar_metrics(common_rows)
    task_id_sets = _common_task_ids(common_rows)
    task_sets = {
        "recovery": task_id_sets[("internal_scientific", "confirm")],
        "unsafe": task_id_sets[("internal_scientific", "abstain")],
        "synthetic": task_id_sets[("synthetic_control", "abstain")],
    }
    systems = _systems_common_metrics(systems_rows, task_sets)
    data: dict[str, dict[str, dict[str, Any]]] = {}
    for method in ("direct_llm_judge", "conventional_significance"):
        data[method] = common[method]
    for method in SYSTEM_METHOD_ORDER:
        data[method] = systems[method]
    for tier in TIER_LABELS:
        data[tier] = {
            metric: _tier_counts_for_tasks(
                tier_decisions, tier=tier, task_ids=task_sets[metric]
            )
            for metric in ("recovery", "unsafe", "synthetic")
        }
    return data


def _plot_frontier(
    data: dict[str, dict[str, dict[str, Any]]],
    output_prefix: Path,
) -> list[dict[str, object]]:
    """Recovery vs. false support: CONFIRM is non-dominated at low false support."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    recovery_total = data["confirmed"]["recovery"]["denominator"]
    unsafe_total = data["confirmed"]["unsafe"]["denominator"]
    synthetic_total = data["confirmed"]["synthetic"]["denominator"]

    figure, axes = plt.subplots(
        1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": (1.0, 1.15)}
    )

    # Panel a keeps the frontier geometry, which holds on this plane. Counts rather
    # than rates, because one abstention claim is nearly seven percentage points.
    scatter = axes[0]
    confirm_points = sorted(
        (data[tier]["synthetic"]["count"], data[tier]["recovery"]["count"])
        for tier in TIER_LABELS
    )
    scatter.plot(
        [point[0] for point in confirm_points],
        [point[1] for point in confirm_points],
        color="#133C66",
        linewidth=0.9,
        linestyle="--",
        zorder=1,
    )
    scatter.annotate(
        "CONFIRM frontier",
        confirm_points[-1],
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=5.8,
        color="#133C66",
        fontstyle="italic",
    )
    for method in FRONTIER_ORDER:
        scatter.scatter(
            data[method]["synthetic"]["count"],
            data[method]["recovery"]["count"],
            marker=FRONTIER_MARKERS[method],
            s=52 if method in TIER_LABELS else 42,
            color=FRONTIER_COLORS[method],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    scatter.set_xlabel(
        f"False confirmations on controls (of {synthetic_total})", fontsize=6.7
    )
    scatter.set_ylabel(f"Recovery (of {recovery_total} positive refs)", fontsize=6.7)
    scatter.set_xlim(-3, 62)
    scatter.set_ylim(13, 22)
    scatter.grid(color="#E9E9E9", linewidth=0.55, zorder=0)
    scatter.tick_params(labelsize=6.2)
    scatter.set_title(
        "a  Recovery vs. false support on controls",
        loc="left",
        fontsize=7.5,
        fontweight="bold",
        pad=6,
    )

    # Panel b uses bars because three methods share one literature coordinate, which
    # a scatter collapses into a single marker.
    bars = axes[1]
    height = 0.36
    for index, method in enumerate(FRONTIER_ORDER):
        for offset, metric, color, total in (
            (-height / 2, "recovery", RECOVERY_COLOR, recovery_total),
            (height / 2, "unsafe", UNSAFE_COLOR, unsafe_total),
        ):
            count = data[method][metric]["count"]
            bars.barh(
                index + offset,
                count,
                height=height,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            bars.text(
                count + 0.4,
                index + offset,
                f"{count}/{total}",
                va="center",
                fontsize=6,
            )
    bars.set_yticks(
        range(len(FRONTIER_ORDER)),
        [FRONTIER_LABELS[method] for method in FRONTIER_ORDER],
        fontsize=6.4,
    )
    bars.invert_yaxis()
    bars.set_xlim(0, 27)
    bars.set_xlabel("Claims", fontsize=6.7)
    bars.grid(axis="x", color="#E9E9E9", linewidth=0.55, zorder=0)
    bars.tick_params(labelsize=6.2, length=0)
    bars.legend(
        handles=[
            Patch(facecolor=RECOVERY_COLOR, label="Recovery (positive references)"),
            Patch(facecolor=UNSAFE_COLOR, label="False support (literature)"),
        ],
        fontsize=6.2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.2,
    )
    bars.set_title(
        "b  Literature outcomes", loc="left", fontsize=7.5, fontweight="bold", pad=18
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=FRONTIER_MARKERS[method],
            markerfacecolor=FRONTIER_COLORS[method],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=6,
            label=FRONTIER_LABELS[method],
        )
        for method in FRONTIER_ORDER
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=6.2,
        bbox_to_anchor=(0.5, -0.14),
        columnspacing=1.2,
        handletextpad=0.3,
    )

    source_rows: list[dict[str, object]] = []
    for method in FRONTIER_ORDER:
        for metric_name in ("recovery", "unsafe", "synthetic"):
            metric = data[method][metric_name]
            source_rows.append(
                {
                    "method": method,
                    "method_label": FRONTIER_LABELS[method],
                    "family": (
                        "neuroimaging_system"
                        if method in SYSTEM_METHOD_ORDER
                        else "confirm"
                        if method in TIER_LABELS
                        else "simple_baseline"
                    ),
                    "metric": metric_name,
                    "supported_count": metric["count"],
                    "denominator": metric["denominator"],
                    "supported_rate": metric["rate"],
                }
            )

    figure.tight_layout()
    figure.savefig(
        output_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02
    )
    figure.savefig(
        output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
    )
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)
    return source_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-summary",
        default=(
            "review-stage/neuroclaimbench-v2.1/"
            "claim-evaluation-baselines-v1/method_summary.csv"
        ),
    )
    parser.add_argument(
        "--common-coverage",
        default=(
            "review-stage/neuroclaimbench-v2.1/"
            "claim-evaluation-baselines-v1/primary_common_coverage.csv"
        ),
    )
    parser.add_argument(
        "--tier-primary",
        default=(
            "review-stage/neuroclaimbench-v2.1/"
            "evidence-tiers/primary_metrics.csv"
        ),
    )
    parser.add_argument(
        "--tier-summary",
        default=(
            "review-stage/neuroclaimbench-v2.1/"
            "evidence-tiers/evidence_tier_summary.csv"
        ),
    )
    parser.add_argument(
        "--tier-decisions",
        default=(
            "review-stage/neuroclaimbench-v2.1/"
            "evidence-tiers/evidence_tier_decisions.csv"
        ),
    )
    parser.add_argument(
        "--feedback-summary",
        default=(
            "review-stage/claim-search-gpt55-feedback-baselines-v1/"
            "feedback_method_summary.csv"
        ),
    )
    parser.add_argument(
        "--feedback-evidence",
        default=(
            "review-stage/claim-search-gpt55-feedback-baselines-v1/"
            "feedback_evidence_summary.csv"
        ),
    )
    parser.add_argument(
        "--leave-one-out",
        default=(
            "review-stage/neuroclaimbench-v2.1/gate-attribution/"
            "gate_leave_one_out.csv"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="paper/figures/fig_confirm_policy_comparison",
    )
    parser.add_argument(
        "--ablation-table",
        default="paper/figures/_archive_20260730_inactive_results/tab_confirm_gate_ablation.tex",
        help="Archived supplementary gate-ablation table; not an active paper input.",
    )
    parser.add_argument(
        "--ablation-source",
        default="paper/figures/_archive_20260730_inactive_results/tab_confirm_gate_ablation_source.csv",
    )
    parser.add_argument(
        "--feedback-table",
        default="paper/figures/tab_feedback_control_main.tex",
    )
    parser.add_argument(
        "--tier-modality-table",
        default="paper/figures/_archive_20260730_inactive_results/tab_confirm_tiers_modality.tex",
    )
    parser.add_argument(
        "--common-scalar-table",
        default="paper/figures/_archive_20260730_inactive_results/tab_claim_evaluation_common_scalar.tex",
    )
    parser.add_argument(
        "--external-tier-table",
        default="paper/figures/_archive_20260730_inactive_results/tab_confirm_tiers_external.tex",
    )
    parser.add_argument(
        "--systems-coverage",
        default=(
            "review-stage/neuroclaimbench-v2.1/"
            "claim-evaluation-baselines-v2/joined_decisions.csv"
        ),
        help="Per-task decisions including VERITAS-adapted and NeuroClaw-adapted.",
    )
    parser.add_argument(
        "--frontier-output-prefix",
        default="paper/figures/fig_confirm_frontier",
    )
    args = parser.parse_args()
    baseline_rows = _read_rows(Path(args.baseline_summary))
    common_rows = _read_rows(Path(args.common_coverage))
    tier_primary_rows = _read_rows(Path(args.tier_primary))
    tier_summary_rows = _read_rows(Path(args.tier_summary))
    tier_decision_rows = _read_rows(Path(args.tier_decisions))
    feedback_rows = _read_rows(Path(args.feedback_summary))
    feedback_evidence_rows = _read_rows(Path(args.feedback_evidence))
    leave_one_out_rows = _read_rows(Path(args.leave_one_out))
    systems_rows = _read_rows(Path(args.systems_coverage))
    output_prefix = Path(args.output_prefix)
    source_rows = _plot_policy_comparison(
        baseline_rows,
        common_rows,
        tier_primary_rows,
        tier_decision_rows,
        output_prefix,
    )
    _write_rows(
        source_rows,
        output_prefix.with_name(f"{output_prefix.name}_source.csv"),
    )
    frontier_prefix = Path(args.frontier_output_prefix)
    frontier_data = _frontier_metrics(common_rows, tier_decision_rows, systems_rows)
    frontier_source = _plot_frontier(frontier_data, frontier_prefix)
    _write_rows(
        frontier_source,
        frontier_prefix.with_name(f"{frontier_prefix.name}_source.csv"),
    )
    ablation_rows = _ablation_rows(leave_one_out_rows)
    _write_rows(
        ablation_rows,
        Path(args.ablation_source),
    )
    _write_ablation_table(ablation_rows, Path(args.ablation_table))
    _write_feedback_table(
        feedback_rows,
        feedback_evidence_rows,
        Path(args.feedback_table),
    )
    _write_tier_modality_table(
        tier_summary_rows,
        Path(args.tier_modality_table),
    )
    _write_common_scalar_table(
        common_rows,
        tier_decision_rows,
        systems_rows,
        Path(args.common_scalar_table),
    )
    _write_external_tier_table(
        tier_summary_rows,
        Path(args.external_tier_table),
    )
    print(f"Wrote {output_prefix}.svg/.pdf/.png, active paper tables, and archived supplementary tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
