"""Split the common-scalar comparison by literature reference strength.

The recovery and unsafe-support metrics rest on multi-model adjudicated labels,
about half of which are provisional. This script re-scores every compared method
separately on strict and provisional references so the comparison can be checked
against label quality. Constructed controls are excluded because their
dispositions follow from the construction rather than from adjudication.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "neuroclaimbench-v2.1-reference-strength-sensitivity-v1"

#: Methods that must all be available for a task to enter the common scalar set.
COVERAGE_METHODS = ("direct_llm_judge", "conventional_significance", "confirm")
#: The adapted systems were scored in a later run, exactly as in the published
#: common-scalar table, so they are read from a separate decisions file.
SYSTEM_METHODS = ("veritas_adapted", "neuroclaw_adapted_judge")

TIER_LABELS = {
    "discovery": "CONFIRM-Screen",
    "replicated": "CONFIRM-Replicate",
    "confirmed": "CONFIRM-Full",
}
GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Adapted neuroimaging systems",
        (("veritas_adapted", "VERITAS-adapted"), ("neuroclaw_adapted_judge", "NeuroClaw-adapted")),
    ),
    (
        "Simple reporting baselines",
        (("direct_llm_judge", "Direct LLM judge"), ("conventional_significance", "Significance filter")),
    ),
    (
        "CONFIRM (this work)",
        tuple((tier, label) for tier, label in TIER_LABELS.items()),
    ),
)
STRENGTHS = ("strict", "provisional")
METRICS = (("recovery", "confirm"), ("unsafe", "abstain"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _reference_strength(path: Path) -> dict[str, str]:
    strength: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            strength[record["benchmark_case_id"]] = record["strength"]
    return strength


def _common_tasks(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    by_task: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["stratum"] == "internal_scientific":
            by_task[row["task_id"]].setdefault(row["method"], row)
    return {
        task_id: methods
        for task_id, methods in by_task.items()
        if all(
            name in methods and methods[name]["available"].strip().lower() == "true"
            for name in COVERAGE_METHODS
        )
    }


def run(args: argparse.Namespace) -> None:
    decisions = _read_rows(Path(args.joined_decisions))
    system_rows = _read_rows(Path(args.systems_decisions))
    tiers = _read_rows(Path(args.tier_decisions))
    strength = _reference_strength(Path(args.references))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = _common_tasks(decisions)
    tier_by_task: dict[tuple[str, str], dict[str, str]] = {
        (row["task_id"], row["minimum_evidence_tier"]): row for row in tiers
    }
    system_by_task: dict[tuple[str, str], dict[str, str]] = {
        (row["task_id"], row["method"]): row
        for row in system_rows
        if row["method"] in SYSTEM_METHODS
    }

    denominators: dict[tuple[str, str], int] = Counter()
    supported: dict[tuple[str, str, str], int] = Counter()
    for task_id, methods in common.items():
        anchor = methods["confirm"]
        case_strength = strength.get(anchor["benchmark_case_id"])
        if case_strength not in STRENGTHS:
            continue
        for metric, disposition in METRICS:
            if anchor["reference_disposition"] != disposition:
                continue
            denominators[(metric, case_strength)] += 1
            for _group, entries in GROUPS:
                for name, _label in entries:
                    if name in TIER_LABELS:
                        row = tier_by_task.get((task_id, name))
                    elif name in SYSTEM_METHODS:
                        row = system_by_task.get((task_id, name))
                    else:
                        row = methods.get(name)
                    if row is None:
                        raise KeyError(f"No decision for {name} on {task_id}")
                    if row["supported"].strip().lower() == "true":
                        supported[(metric, case_strength, name)] += 1

    source_rows: list[dict[str, Any]] = []
    for _group, entries in GROUPS:
        for name, label in entries:
            for metric, _disposition in METRICS:
                for case_strength in STRENGTHS:
                    source_rows.append(
                        {
                            "method": name,
                            "method_label": label,
                            "metric": metric,
                            "reference_strength": case_strength,
                            "supported_count": supported[(metric, case_strength, name)],
                            "denominator": denominators[(metric, case_strength)],
                        }
                    )
    with (output_dir / "reference_strength_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    def cell(metric: str, case_strength: str, name: str) -> str:
        return (
            f"{supported[(metric, case_strength, name)]}/"
            f"{denominators[(metric, case_strength)]}"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Common-scalar comparison split by literature reference",
        r"strength.} Recovery uses positive references and unsafe support uses",
        r"abstention references, scored on the same common scalar tasks as",
        r"Table~\ref{tab:claim_evaluation_common_scalar}, whose columns are the row",
        r"sums here. Method ordering is stable across the two tiers: no method",
        r"changes rank by more than one position on either metric. Only two",
        r"abstention references are strict, so that column cannot separate methods.",
        r"Constructed controls are excluded because their dispositions follow from",
        r"the construction rather than from adjudication.}",
        r"\label{tab:reference_strength_sensitivity}",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Recovery} & \multicolumn{2}{c}{Unsafe support} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Method & Strict & Provisional & Strict & Provisional \\",
        r"\midrule",
    ]
    for index, (group, entries) in enumerate(GROUPS):
        if index:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{group}}}}}\\")
        for name, label in entries:
            lines.append(
                f"{label} & "
                + " & ".join(
                    cell(metric, case_strength, name)
                    for metric, _ in METRICS
                    for case_strength in STRENGTHS
                )
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    table = "\n".join(lines)
    (output_dir / "tab_reference_strength_sensitivity.tex").write_text(table)
    if args.paper_table:
        Path(args.paper_table).write_text(table)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "joined_decisions": str(args.joined_decisions),
        "systems_decisions": str(args.systems_decisions),
        "tier_decisions": str(args.tier_decisions),
        "references": str(args.references),
        "common_scalar_task_count": len(common),
        "denominators": {f"{m}:{s}": denominators[(m, s)] for m, _ in METRICS for s in STRENGTHS},
        "interpretation_restrictions": [
            "Strict abstention references are too few to separate methods on safety.",
            "Constructed controls are excluded; their labels are not adjudicated.",
        ],
    }
    (output_dir / "reference_strength_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    print(f"common scalar tasks: {len(common)}")
    for metric, _ in METRICS:
        for case_strength in STRENGTHS:
            print(f"  {metric} {case_strength}: n={denominators[(metric, case_strength)]}")
    print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joined-decisions",
        default="review-stage/neuroclaimbench-v2.1/claim-evaluation-baselines-v1/joined_decisions.csv",
    )
    parser.add_argument(
        "--systems-decisions",
        default="review-stage/neuroclaimbench-v2.1/claim-evaluation-baselines-v2/joined_decisions.csv",
    )
    parser.add_argument(
        "--tier-decisions",
        default="review-stage/neuroclaimbench-v2.1/evidence-tiers/evidence_tier_decisions.csv",
    )
    parser.add_argument(
        "--references",
        default="review-stage/neuroclaimbench-v2.1/compact/references.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="review-stage/neuroclaimbench-v2.1/reference-strength-v1",
    )
    parser.add_argument(
        "--paper-table",
        default="paper/figures/tab_reference_strength_sensitivity.tex",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
