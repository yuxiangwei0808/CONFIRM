"""Attribute NeuroClaimBench v2.1 verdicts to the frozen CONFIRM gate vector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nbs.claim_search_analysis_common import (
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from bench.benchmark import BenchmarkDataset


GATE_ORDER = (
    "search_provenance",
    "multiplicity",
    "confounding",
    "power",
    "multiverse",
    "replication",
)
GATE_LABELS = {
    "search_provenance": "Search\nprovenance",
    "multiplicity": "Multiplicity",
    "confounding": "Measured\nconfounds",
    "power": "Power",
    "multiverse": "Stability",
    "replication": "Replication",
}
REPORTING_POLICIES = (
    ("execution_only", "Execution only", ()),
    (
        "multiplicity_corrected",
        "Multiplicity-corrected discovery",
        ("search_provenance", "multiplicity"),
    ),
    (
        "multiplicity_replication",
        "Multiplicity + replication",
        ("search_provenance", "multiplicity", "replication"),
    ),
    ("full_confirm", "Full CONFIRM", GATE_ORDER),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _gate_vector(outcome: dict[str, Any]) -> dict[str, bool]:
    gates = outcome["gate_verdict"]["gates"]
    return {
        "search_provenance": bool(gates.get("search_provenance")),
        "multiplicity": bool(gates.get("multiplicity")),
        "confounding": bool(gates.get("confound"))
        and bool(gates.get("confound_completeness")),
        "power": bool(gates.get("power")),
        "multiverse": bool(gates.get("multiverse")),
        "replication": bool(gates.get("replication")),
    }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
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


def _cluster_interval(
    records: list[dict[str, Any]],
    values: list[bool],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not records:
        return math.nan, math.nan
    grouped: dict[str, list[bool]] = defaultdict(list)
    for record, value in zip(records, values):
        grouped[str(record["semantic_cluster_id"])].append(value)
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sampled = [value for cluster in selected for value in grouped[str(cluster)]]
        estimates[index] = float(np.mean(sampled))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256(":".join((str(base), *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _stratum(
    item: dict[str, Any],
    profile: dict[str, Any],
    task: dict[str, Any],
) -> str:
    track = item["benchmark_track"]
    basis = profile.get("reference_basis") or "unresolved"
    dataset = str(task.get("dataset_id") or "unknown")
    if track == "scientific":
        return "scientific_literature" if basis == "literature" else "scientific_unresolved"
    if track == "external_transfer":
        kind = "literature" if basis == "literature" else (
            "constructed" if basis == "constructed_control" else "unresolved"
        )
        return f"external_{kind}_{dataset}"
    if track == "synthetic_stress":
        return "synthetic_constructed"
    return f"{track}_{basis}"


def build_records(package_dir: Path) -> list[dict[str, Any]]:
    dataset = BenchmarkDataset(package_dir)
    items = {
        row.benchmark_case_id: row for row in dataset.cases
    }
    profiles = {
        row.benchmark_case_id: row for row in dataset.references
    }
    tasks = {row.task_id: row for row in dataset.tasks}
    outcomes = [
        row.model_dump(mode="json") for row in dataset.outcomes
    ]
    records: list[dict[str, Any]] = []
    for outcome in outcomes:
        item = items[outcome["benchmark_case_id"]]
        profile = profiles[outcome["benchmark_case_id"]]
        task = tasks[outcome["task_id"]]
        gate_vector = _gate_vector(outcome)
        confirmed_from_gates = all(gate_vector.values())
        confirmed_from_outcome = outcome["confirm_outcome"] == "confirmed"
        if confirmed_from_gates != confirmed_from_outcome:
            raise ValueError(
                "Frozen verdict does not reconcile with its gate vector: "
                f"{outcome['benchmark_case_id']}"
            )
        records.append(
            {
                "benchmark_item_id": outcome["benchmark_case_id"],
                "semantic_cluster_id": item.semantic_cluster_id,
                "benchmark_track": item.benchmark_track,
                "target_family": item.target_family,
                "reference_basis": profile.basis,
                "reference_strength": profile.strength,
                "reference_disposition": profile.disposition,
                "dataset_id": task.dataset_id,
                "stratum": _stratum(
                    item.model_dump(mode="json"),
                    {
                        "reference_basis": profile.basis,
                    },
                    task.model_dump(mode="json"),
                ),
                "confirmed": confirmed_from_outcome,
                "gates": gate_vector,
            }
        )
    return records


def analyze_records(
    records: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["stratum"], str(record["reference_disposition"]))].append(record)
    failure_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    ladder_rows: list[dict[str, Any]] = []
    for (stratum, disposition), group in sorted(groups.items()):
        total = len(group)
        baseline_values = [bool(row["confirmed"]) for row in group]
        baseline_count = sum(baseline_values)
        for gate in GATE_ORDER:
            failed_values = [not row["gates"][gate] for row in group]
            exclusive_values = [
                not row["gates"][gate]
                and all(row["gates"][other] for other in GATE_ORDER if other != gate)
                for row in group
            ]
            failure_rows.append(
                {
                    "stratum": stratum,
                    "reference_disposition": disposition,
                    "gate": gate,
                    "claim_count": total,
                    "failed_count": sum(failed_values),
                    "failed_rate": float(np.mean(failed_values)),
                    "exclusive_failure_count": sum(exclusive_values),
                    "exclusive_failure_rate": float(np.mean(exclusive_values)),
                }
            )
            counterfactual_values = [
                all(row["gates"][other] for other in GATE_ORDER if other != gate)
                for row in group
            ]
            counterfactual_count = sum(counterfactual_values)
            wilson_low, wilson_high = _wilson(counterfactual_count, total)
            cluster_low, cluster_high = _cluster_interval(
                group,
                counterfactual_values,
                resamples=resamples,
                seed=_seed(seed, stratum, disposition, gate),
            )
            leave_one_out_rows.append(
                {
                    "stratum": stratum,
                    "reference_disposition": disposition,
                    "removed_gate": gate,
                    "claim_count": total,
                    "baseline_confirmed_count": baseline_count,
                    "baseline_confirmed_rate": baseline_count / total,
                    "counterfactual_confirmed_count": counterfactual_count,
                    "counterfactual_confirmed_rate": counterfactual_count / total,
                    "added_confirmation_count": counterfactual_count - baseline_count,
                    "wilson_low": wilson_low,
                    "wilson_high": wilson_high,
                    "cluster_bootstrap_low": cluster_low,
                    "cluster_bootstrap_high": cluster_high,
                    "interpretation": "descriptive_gate_removal_not_causal",
                }
            )
        cumulative: list[str] = []
        for stage_index, gate in enumerate(GATE_ORDER, start=1):
            cumulative.append(gate)
            pass_values = [
                all(row["gates"][active_gate] for active_gate in cumulative)
                for row in group
            ]
            pass_count = sum(pass_values)
            ladder_rows.append(
                {
                    "stratum": stratum,
                    "reference_disposition": disposition,
                    "stage_index": stage_index,
                    "added_gate": gate,
                    "active_gates": json.dumps(cumulative),
                    "claim_count": total,
                    "pass_count": pass_count,
                    "pass_rate": pass_count / total,
                }
            )
    return failure_rows, leave_one_out_rows, ladder_rows


def analyze_reporting_policies(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["stratum"], str(record["reference_disposition"]))].append(record)

    rows: list[dict[str, Any]] = []
    for policy_index, (policy, label, required_gates) in enumerate(
        REPORTING_POLICIES,
        start=1,
    ):
        for (stratum, disposition), group in sorted(groups.items()):
            pass_count = sum(
                all(record["gates"][gate] for gate in required_gates)
                for record in group
            )
            rows.append(
                {
                    "policy_index": policy_index,
                    "policy": policy,
                    "policy_label": label,
                    "required_gates": json.dumps(required_gates),
                    "stratum": stratum,
                    "reference_disposition": disposition,
                    "claim_count": len(group),
                    "pass_count": pass_count,
                    "pass_rate": pass_count / len(group),
                }
            )
    return rows


def _plot(
    rows: Iterable[dict[str, Any]],
    path_pdf: Path,
    path_png: Path,
) -> None:
    selected = [
        "scientific_literature",
        "external_literature_NACC",
        "external_literature_ds000030",
        "external_constructed_NACC",
        "external_constructed_ds000030",
        "synthetic_constructed",
    ]
    combined: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        if row["stratum"] in selected:
            combined[(row["stratum"], row["removed_gate"])] += int(
                row["added_confirmation_count"]
            )
    # A stratum whose counts are zero under every gate removal carries no signal,
    # so it is dropped rather than drawn as an empty heatmap row.
    present = [
        stratum
        for stratum in selected
        if any(combined.get((stratum, gate), 0) for gate in GATE_ORDER)
    ]
    values = np.asarray(
        [[combined[(stratum, gate)] for gate in GATE_ORDER] for stratum in present],
        dtype=float,
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.5,
            "axes.linewidth": 0.8,
        }
    )
    # Removing a gate is a magnitude question, so grouped bars read faster than a
    # colour grid whose cells are mostly zero.
    stratum_labels = {
        "scientific_literature": "Scientific literature",
        "external_literature_NACC": "External literature (NACC)",
        "external_literature_ds000030": "External literature (CNP)",
        "external_constructed_NACC": "External controls (NACC)",
        "external_constructed_ds000030": "External controls (CNP)",
        "synthetic_constructed": "Synthetic controls",
    }
    stratum_colors = {
        "scientific_literature": "#133C66",
        "external_literature_NACC": "#69737D",
        "external_literature_ds000030": "#9A8064",
        "external_constructed_NACC": "#B07AA1",
        "external_constructed_ds000030": "#4B8BBE",
        "synthetic_constructed": "#C1666B",
    }
    fig, axis = plt.subplots(figsize=(6.6, 2.85))
    gate_positions = np.arange(len(GATE_ORDER))
    span = 0.78
    height = span / len(present)
    for index, stratum in enumerate(present):
        offsets = gate_positions - span / 2 + height * (index + 0.5)
        counts = [combined[(stratum, gate)] for gate in GATE_ORDER]
        axis.barh(
            offsets,
            counts,
            height=height * 0.86,
            color=stratum_colors.get(stratum, "#69737D"),
            label=stratum_labels.get(stratum, stratum.replace("_", " ")),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for offset, count in zip(offsets, counts):
            if count:
                axis.text(
                    count + 0.7,
                    offset,
                    f"{int(count)}",
                    va="center",
                    ha="left",
                    fontsize=7,
                    zorder=4,
                )
    axis.set_yticks(gate_positions, [GATE_LABELS[gate] for gate in GATE_ORDER], fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0, float(values.max()) * 1.13)
    axis.grid(axis="x", color="#E9E9E9", linewidth=0.55, zorder=0)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.legend(fontsize=7, loc="upper right", frameon=False)
    axis.set_title("Additional confirmations when one gate is removed", fontsize=8.5)
    axis.set_xlabel("Additional confirmations", fontsize=7.5)
    axis.set_ylabel("Removed gate", fontsize=7.5)
    axis.tick_params(labelsize=7, length=0)
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = Path(args.package_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(package_dir)
    failure_rows, leave_one_out_rows, ladder_rows = analyze_records(
        records,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    policy_rows = analyze_reporting_policies(records)
    failure_path = out_dir / "gate_failure_summary.csv"
    leave_path = out_dir / "gate_leave_one_out.csv"
    ladder_path = out_dir / "gate_ladder.csv"
    policy_path = out_dir / "reporting_policy_comparison.csv"
    figure_pdf = out_dir / "fig_gate_attribution.pdf"
    figure_png = out_dir / "fig_gate_attribution.png"
    write_csv_atomic(failure_path, failure_rows)
    write_csv_atomic(leave_path, leave_one_out_rows)
    write_csv_atomic(ladder_path, ladder_rows)
    write_csv_atomic(policy_path, policy_rows)
    _plot(leave_one_out_rows, figure_pdf, figure_png)
    summary = {
        "version": "neuroclaimbench-v2.1-gate-attribution-v1",
        "claim_count": len(records),
        "confirmed_count": sum(row["confirmed"] for row in records),
        "gate_order": list(GATE_ORDER),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "verdict_gate_reconciliation_errors": 0,
        "interpretation_restrictions": [
            "Gate failures are correlated.",
            "Leave-one-gate-out results are descriptive counterfactual verdicts, not causal effects.",
            "No gate threshold was changed or optimized.",
            "Literature references and constructed controls remain separate.",
            "NACC and ds000030 remain separate.",
        ],
    }
    summary_path = out_dir / "gate_attribution_summary.json"
    write_json_atomic(summary_path, summary)
    inputs = [
        package_dir / "cases.jsonl",
        package_dir / "tasks.jsonl",
        package_dir / "references.jsonl",
        package_dir / "outcomes.jsonl",
    ]
    outputs = [
        failure_path,
        leave_path,
        ladder_path,
        policy_path,
        figure_pdf,
        figure_png,
        summary_path,
    ]
    manifest = {
        "version": summary["version"],
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in inputs],
        "outputs": [{"path": str(path), "sha256": sha256_file(path)} for path in outputs],
        "parameters": {
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "interpretation_restrictions": summary["interpretation_restrictions"],
    }
    write_json_atomic(out_dir / "gate_attribution_manifest.json", manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        default="benchmark/neuroclaimbench-v2.1/compact",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/gate-attribution",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
