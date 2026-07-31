"""Sweep the power gate's reference effect and re-derive CONFIRM verdicts.

Every executable claim in the benchmark falls back to the default reference
effect, so each power-gate decision rests on one constant. This replay varies
that constant, recomputes the power gate from the frozen sample size and alpha,
and re-scores the evidence policies with the other six gates held fixed. No
analysis is re-executed and no model is called.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from confirm.power import DEFAULT_MDE, _achieved_power
from confirm.verdict import classify_support

PROTOCOL_VERSION = "neuroclaimbench-v2.1-power-reference-sensitivity-v1"

POLICIES: tuple[tuple[str, str], ...] = (
    ("discovery", "CONFIRM-Screen"),
    ("replicated", "CONFIRM-Replicate"),
    ("confirmed", "CONFIRM-Full"),
)

#: Scored strata mapped to the denominators the paper reports.
METRIC_STRATA: dict[str, tuple[str, str]] = {
    "recovery": ("internal_scientific", "confirm"),
    "unsafe_support": ("internal_scientific", "abstain"),
    "synthetic_support": ("synthetic_control", "abstain"),
}

GATE_KEYS = (
    "search_provenance",
    "confound",
    "confound_completeness",
    "multiplicity",
    "power",
    "multiverse",
    "replication",
)


@dataclass(frozen=True)
class ScoredTask:
    task_id: str
    stratum: str
    reference_disposition: str
    gates: dict[str, bool]
    sample_size: int
    alpha: float
    min_power: float


def _load_tasks(outcomes_path: Path, decisions_path: Path) -> list[ScoredTask]:
    records: dict[str, dict[str, Any]] = {}
    with outcomes_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            records[record["task_id"]] = record

    tasks: list[ScoredTask] = []
    with decisions_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "confirm":
                continue
            record = records.get(row["task_id"])
            if record is None:
                raise KeyError(f"No execution record for scored task {row['task_id']}")
            gate_results = record.get("gate_results") or {}
            vector = (gate_results.get("verdict") or {}).get("gates") or {}
            contract = gate_results.get("contract") or {}
            gates_spec = contract.get("gates") or {}
            primary = gate_results.get("primary") or {}
            sample_size = primary.get("n")
            if sample_size is None:
                # Brain-wide claims carry regional tables rather than one effect;
                # their power gate is left at its frozen value.
                sample_size = -1
            tasks.append(
                ScoredTask(
                    task_id=row["task_id"],
                    stratum=row["stratum"],
                    reference_disposition=row["reference_disposition"],
                    gates={key: bool(vector.get(key)) for key in GATE_KEYS},
                    sample_size=int(sample_size),
                    alpha=float((gates_spec.get("multiplicity") or {}).get("alpha", 0.05)),
                    min_power=float((gates_spec.get("power") or {}).get("min_power", 0.8)),
                )
            )
    return tasks


def _power_gate(task: ScoredTask, reference_effect: float) -> bool:
    """Recompute the power gate at a candidate reference effect."""

    if task.sample_size < 0:
        return task.gates["power"]
    achieved = _achieved_power(reference_effect, task.sample_size, task.alpha)
    return achieved >= task.min_power


def _metric_rows(tasks: list[ScoredTask], grid: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_effect in grid:
        for policy, policy_label in POLICIES:
            counts: Counter[str] = Counter()
            denominators: Counter[str] = Counter()
            for task in tasks:
                gates = dict(task.gates)
                gates["power"] = _power_gate(task, reference_effect)
                supported = classify_support(gates, policy).supported
                for metric, (stratum, disposition) in METRIC_STRATA.items():
                    if task.stratum != stratum or task.reference_disposition != disposition:
                        continue
                    denominators[metric] += 1
                    if supported:
                        counts[metric] += 1
            row: dict[str, Any] = {
                "reference_effect": reference_effect,
                "is_published_default": reference_effect == DEFAULT_MDE,
                "policy": policy,
                "policy_label": policy_label,
            }
            for metric in METRIC_STRATA:
                row[f"{metric}_count"] = counts[metric]
                row[f"{metric}_denominator"] = denominators[metric]
            rows.append(row)
    return rows


def _power_pass_rows(tasks: list[ScoredTask], grid: tuple[float, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_effect in grid:
        scalar = [task for task in tasks if task.sample_size >= 0]
        passing = sum(_power_gate(task, reference_effect) for task in scalar)
        rows.append(
            {
                "reference_effect": reference_effect,
                "scalar_claim_count": len(scalar),
                "power_gate_pass_count": passing,
                "power_gate_pass_rate": passing / len(scalar) if scalar else 0.0,
            }
        )
    return rows


def _validate(rows: list[dict[str, Any]], decisions_path: Path) -> dict[str, Any]:
    """The published default must reproduce the frozen CONFIRM-Full numbers."""

    published = {
        row["policy"]: row
        for row in rows
        if row["reference_effect"] == DEFAULT_MDE
    }
    native: Counter[str] = Counter()
    denominators: Counter[str] = Counter()
    with decisions_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "confirm":
                continue
            for metric, (stratum, disposition) in METRIC_STRATA.items():
                if row["stratum"] != stratum or row["reference_disposition"] != disposition:
                    continue
                denominators[metric] += 1
                if row["supported"].strip().lower() == "true":
                    native[metric] += 1
    full = published["confirmed"]
    mismatches = {
        metric: {
            "replay": full[f"{metric}_count"],
            "frozen": native[metric],
        }
        for metric in METRIC_STRATA
        if full[f"{metric}_count"] != native[metric]
        or full[f"{metric}_denominator"] != denominators[metric]
    }
    return {
        "default_reproduces_frozen_verdicts": not mismatches,
        "mismatches": mismatches,
    }


def _latex_table(rows: list[dict[str, Any]], pass_rows: list[dict[str, Any]]) -> str:
    by_effect: dict[float, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_effect.setdefault(row["reference_effect"], {})[row["policy"]] = row
    passes = {row["reference_effect"]: row for row in pass_rows}

    # Screen and Replicate do not require the power gate, so only CONFIRM-Full can
    # move with the reference effect. Reporting the invariance is clearer than
    # printing three identical columns.
    invariant = {
        label: all(
            by_effect[effect][policy][f"{metric}_count"]
            == by_effect[min(by_effect)][policy][f"{metric}_count"]
            for effect in by_effect
            for metric in METRIC_STRATA
        )
        for policy, label in POLICIES
        if policy != "confirmed"
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Sensitivity of CONFIRM-Full to the power gate's",
        r"reference effect.} Every benchmark claim falls back to the default",
        r"$d=0.3$, so the power gate rests on one constant. Each row recomputes",
        r"that gate from the frozen sample size and $\alpha$, holds the other six",
        r"gates fixed, and re-scores the policy. The pass column counts scalar",
        r"claims clearing the gate. CONFIRM-Screen and CONFIRM-Replicate do not",
        r"require the power gate, so they are unchanged at every value.}",
        r"\label{tab:power_reference_sensitivity}",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"$\delta_{\mathrm{ref}}$ & Power gate pass & Recovery & False support & Controls \\",
        r"\midrule",
    ]
    for effect in sorted(by_effect):
        row = by_effect[effect]["confirmed"]
        pass_row = passes[effect]
        marker = r"$^{\ast}$" if effect == DEFAULT_MDE else ""
        lines.append(
            f"{effect:.1f}{marker} & "
            f"{pass_row['power_gate_pass_count']}/{pass_row['scalar_claim_count']} & "
            f"{row['recovery_count']}/{row['recovery_denominator']} & "
            f"{row['unsafe_support_count']}/{row['unsafe_support_denominator']} & "
            f"{row['synthetic_support_count']}/{row['synthetic_support_denominator']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{2pt}",
        r"\footnotesize $^{\ast}$ published configuration.",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    outcomes_path = Path(args.task_outcomes)
    decisions_path = Path(args.joined_decisions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = tuple(float(value) for value in args.reference_effects)

    tasks = _load_tasks(outcomes_path, decisions_path)
    rows = _metric_rows(tasks, grid)
    pass_rows = _power_pass_rows(tasks, grid)
    validation = _validate(rows, decisions_path)

    _write_csv(output_dir / "power_reference_sensitivity.csv", rows)
    _write_csv(output_dir / "power_gate_pass_rates.csv", pass_rows)
    table = _latex_table(rows, pass_rows)
    (output_dir / "tab_power_reference_sensitivity.tex").write_text(table)
    if args.paper_table:
        Path(args.paper_table).write_text(table)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "task_outcomes": str(outcomes_path),
        "task_outcomes_sha256": hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
        "joined_decisions": str(decisions_path),
        "scored_task_count": len(tasks),
        "reference_effect_grid": list(grid),
        "published_default": DEFAULT_MDE,
        "validation": validation,
        "interpretation_restrictions": [
            "Deterministic replay of frozen records; no analysis is re-executed.",
            "Only the power gate varies; the other six gates keep their frozen results.",
            "Brain-wide claims keep their frozen power gate because they carry no single sample size.",
        ],
    }
    (output_dir / "power_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    print(f"scored tasks: {len(tasks)}")
    print(f"validation: {validation}")
    for pass_row in pass_rows:
        print(
            f"  d={pass_row['reference_effect']:.1f}  power gate passes "
            f"{pass_row['power_gate_pass_count']}/{pass_row['scalar_claim_count']}"
        )
    print()
    for row in rows:
        if row["policy"] == "confirmed":
            print(
                f"  d={row['reference_effect']:.1f}  Full recovery "
                f"{row['recovery_count']}/{row['recovery_denominator']}  "
                f"false support {row['unsafe_support_count']}/{row['unsafe_support_denominator']}  "
                f"controls {row['synthetic_support_count']}/{row['synthetic_support_denominator']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-outcomes",
        default="review-stage/neuroclaimbench-v2.1/results/task_outcomes.jsonl",
    )
    parser.add_argument(
        "--joined-decisions",
        default="review-stage/neuroclaimbench-v2.1/claim-evaluation-baselines-v1/joined_decisions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="review-stage/neuroclaimbench-v2.1/power-reference-sensitivity-v1",
    )
    parser.add_argument(
        "--reference-effects",
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        type=float,
    )
    parser.add_argument("--paper-table", default="paper/figures/tab_power_reference_sensitivity.tex")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
