"""Analyze retrospective holdout/external evidence for frozen claim-search candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from nbs.claim_search_analysis_common import (
    arm_sort_key,
    merge_analysis_manifest,
    output_manifest,
    read_jsonl,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)


RESTRICTIONS = (
    "Excluded evidence was previously queried and is retrospective.",
    "Candidate-only support is not a causal feedback-loop improvement estimate.",
    "Holdout and external evidence are reported separately.",
    "External evidence sets and datasets are never pooled.",
    "No positive survival threshold was required or used for redesign.",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: Any) -> int:
    if value in (None, "", False):
        return 0
    return int(float(value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _matched_rows(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        if row.get("evidence_kind") == "holdout" and _as_int(row.get("matched_parent_holdout")):
            grouped[(str(row["arm_id"]), str(row["parent_claim_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (arm_id, parent_claim_id), rows in sorted(grouped.items(), key=lambda item: (arm_sort_key(item[0][0]), item[0][1])):
        evaluated = [row for row in rows if _as_int(row.get("evaluated"))]
        parent_evaluated = any(_as_int(row.get("parent_holdout_evaluated")) for row in rows)
        parent_supported = any(_as_int(row.get("parent_holdout_supported")) for row in rows)
        candidate_supported = any(_as_int(row.get("supported")) for row in evaluated)
        if not parent_evaluated or not evaluated:
            cell = "not_matched_evaluated"
        elif parent_supported and candidate_supported:
            cell = "both"
        elif parent_supported:
            cell = "parent_only"
        elif candidate_supported:
            cell = "candidate_only"
        else:
            cell = "neither"
        output.append(
            {
                "arm_id": arm_id,
                "parent_claim_id": parent_claim_id,
                "target_family": rows[0].get("target_family"),
                "source_mode": rows[0].get("source_mode"),
                "matched_candidate_count": len(rows),
                "matched_evaluated_candidate_count": len(evaluated),
                "parent_holdout_evaluated": int(parent_evaluated),
                "parent_holdout_supported": int(parent_supported),
                "candidate_holdout_supported": int(candidate_supported),
                "matched_outcome_cell": cell,
                "candidate_ids": json.dumps(sorted({str(row["candidate_id"]) for row in rows})),
            }
        )
    return output


def _summary_rows(
    evidence_rows: list[dict[str, Any]],
    lineages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lineage_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lineage in lineages:
        lineage_by_arm[str(lineage["arm_id"])].append(lineage)
    rows: list[dict[str, Any]] = []
    dimensions = (
        ("arm", lambda row: (str(row["arm_id"]), "all", "all")),
        ("target_family", lambda row: (str(row["arm_id"]), str(row.get("target_family") or "unknown"), "all")),
        ("source_mode", lambda row: (str(row["arm_id"]), str(row.get("source_mode") or "unknown"), "all")),
        ("evidence_set", lambda row: (str(row["arm_id"]), str(row.get("evidence_set_id") or "internal_holdout"), str(row.get("target_family") or "unknown"))),
    )
    for evidence_kind in ("holdout", "external"):
        kind_rows = [row for row in evidence_rows if row.get("evidence_kind") == evidence_kind]
        for dimension, key_fn in dimensions:
            grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in kind_rows:
                grouped[key_fn(row)].append(row)
            for (arm_id, value, target), group in sorted(grouped.items(), key=lambda item: (arm_sort_key(item[0][0]), item[0][1], item[0][2])):
                if evidence_kind == "external" and dimension in {"arm", "target_family", "source_mode"}:
                    continue
                compatible = [row for row in group if _as_int(row.get("compatible"))]
                evaluated = [row for row in group if _as_int(row.get("evaluated"))]
                supported = [row for row in evaluated if _as_int(row.get("supported"))]
                parent_count = len(lineage_by_arm[arm_id])
                supported_parents = {str(row["parent_claim_id"]) for row in supported}
                rows.append(
                    {
                        "arm_id": arm_id,
                        "evidence_kind": evidence_kind,
                        "dimension": dimension,
                        "dimension_value": value,
                        "target_family": target,
                        "parent_lineage_count": parent_count,
                        "internally_supported_parent_count": sum(
                            bool(row.get("internally_supported_candidate_ids"))
                            for row in lineage_by_arm[arm_id]
                        ),
                        "candidate_record_count": len(group),
                        "compatible_candidate_count": len(compatible),
                        "evaluated_candidate_count": len(evaluated),
                        "supported_candidate_count": len(supported),
                        "supported_parent_count": len(supported_parents),
                        "conditional_survival_rate": len(supported) / len(evaluated) if evaluated else None,
                        "candidate_system_yield": len(supported) / parent_count if parent_count else None,
                        "parent_system_yield": len(supported_parents) / parent_count if parent_count else None,
                        "unavailable_evidence_count": sum(row.get("preflight_status") != "eligible" for row in group),
                        "execution_error_count": sum(_as_int(row.get("execution_error")) for row in group),
                        "evidence_freshness": "previously_queried",
                        "final_confirmation_eligible": False,
                    }
                )
    return rows


def _case_cell(
    evidence_rows: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    lineages: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any] | None]]:
    holdout_evaluated = [
        row for row in evidence_rows
        if row.get("evidence_kind") == "holdout" and _as_int(row.get("evaluated"))
    ]
    exposure_by_key = {
        (row["lineage_event_id"], row["candidate_id"]): row
        for row in exposures
    }
    cases: list[tuple[str, dict[str, Any] | None]] = []
    selectors = (
        ("candidate_only_holdout_support", lambda row: _as_int(row.get("candidate_only_holdout_support")) == 1),
        ("parent_and_candidate_holdout_support", lambda row: _as_int(row.get("supported")) == 1 and _as_int(row.get("parent_holdout_supported")) == 1 and _as_int(row.get("matched_parent_holdout")) == 1),
        ("internal_support_then_holdout_failure", lambda row: _as_int(row.get("supported")) == 0),
    )
    for name, predicate in selectors:
        matches = sorted((row for row in holdout_evaluated if predicate(row)), key=lambda row: (str(row["parent_claim_id"]), str(row["candidate_id"]), str(row["arm_id"])))
        cases.append((name, matches[0] if matches else None))
    retracted = sorted(
        (row for row in exposures if row.get("multiplicity_retracted")),
        key=lambda row: (str(row["parent_claim_id"]), str(row["candidate_id"]), str(row["arm_id"])),
    )
    cases.append(("multiplicity_retracted", retracted[0] if retracted else None))
    asd = sorted(
        (
            row for row in lineages
            if row.get("arm_id") == "r10_c10"
            and row.get("target_family") == "asd"
            and not row.get("internally_supported_candidate_ids")
        ),
        key=lambda row: str(row["parent_claim_id"]),
    )
    cases.append(("maximum_budget_asd_no_support", asd[0] if asd else None))
    external = sorted(
        (
            row for row in evidence_rows
            if row.get("evidence_kind") == "external"
            and _as_int(row.get("supported"))
        ),
        key=lambda row: (str(row.get("evidence_set_id")), str(row["parent_claim_id"]), str(row["candidate_id"]), str(row["arm_id"])),
    )
    cases.append(("external_supported", external[0] if external else None))
    return cases


def _write_cases(
    path: Path,
    evidence_rows: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    lineages: list[dict[str, Any]],
) -> None:
    exposure_by_key = {(row["arm_id"], row["parent_claim_id"], row["candidate_id"]): row for row in exposures}
    lineage_by_key = {(row["arm_id"], row["parent_claim_id"]): row for row in lineages}
    lines = ["# Deterministic Feedback-Search Case Studies", "", "Cases are selected lexicographically within predeclared outcome cells.", ""]
    for name, selected in _case_cell(evidence_rows, exposures, lineages):
        lines.extend([f"## {name.replace('_', ' ').title()}", ""])
        if selected is None:
            lines.extend(["No eligible case exists in this outcome cell.", ""])
            continue
        arm_id = str(selected["arm_id"])
        parent_claim_id = str(selected["parent_claim_id"])
        candidate_id = selected.get("candidate_id")
        exposure = exposure_by_key.get((arm_id, parent_claim_id, str(candidate_id))) if candidate_id else None
        lineage = lineage_by_key.get((arm_id, parent_claim_id))
        if exposure is None and selected.get("effective_contract") is not None:
            exposure = selected
        lines.extend(
            [
                f"- Arm: `{arm_id}`",
                f"- Parent: `{parent_claim_id}`",
                f"- Candidate: `{candidate_id or 'none'}`",
                f"- Target: `{selected.get('target_family') or (lineage or {}).get('target_family') or 'unknown'}`",
                f"- Evidence: `{selected.get('evidence_kind') or 'source'}` / `{selected.get('evidence_set_id') or 'none'}`",
                f"- Evidence status: `{selected.get('interpretation_label') or selected.get('preflight_status') or 'not_evaluated'}`",
                "",
                "### Original Contract",
                "```json",
                json.dumps((lineage or {}).get("parent_contract") or (exposure or {}).get("parent_contract") or {}, indent=2, sort_keys=True),
                "```",
                "",
                "### Failure Diagnosis",
                "```json",
                json.dumps((lineage or {}).get("failure_localization") or {}, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
        if exposure:
            lines.extend(
                [
                    "### Candidate Contract And Delta",
                    "```json",
                    json.dumps(
                        {
                            "effective_contract": exposure.get("effective_contract"),
                            "executable_contract_delta": exposure.get("executable_contract_delta"),
                            "declared_transform": exposure.get("declared_transform"),
                            "inferred_transform": exposure.get("inferred_transform"),
                            "effective_family_size": exposure.get("effective_family_size"),
                            "source_label": exposure.get("current_data_label"),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                    "",
                ]
            )
        rounds = sorted(
            (
                row for row in exposures
                if row.get("arm_id") == arm_id
                and row.get("parent_claim_id") == parent_claim_id
                and row.get("generation_status") == "retained"
            ),
            key=lambda row: (int(row.get("round_index") or 0), int(row.get("candidate_index") or 0)),
        )
        lines.extend(["### Round Trace", ""])
        for row in rounds:
            lines.append(
                f"- Round {row.get('round_index')}, `{row.get('candidate_id')}`: "
                f"validation={row.get('validation_ok')}, source_label={row.get('current_data_label')}, "
                f"provisional={row.get('provisional_internal_supported')}, final={row.get('final_internal_supported')}, "
                f"retracted={row.get('multiplicity_retracted')}"
            )
        lines.append("")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence_dir = Path(args.evidence_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "candidate_evidence.jsonl"
    if not evidence_path.exists():
        evidence_path = evidence_dir / "candidate_evidence.csv"
    inventory_path = evidence_dir / "frozen_search_inventory.jsonl"
    lineage_path = evidence_dir / "frozen_lineages.jsonl"
    summary_path = evidence_dir / "summary.json"
    freeze_manifest_path = evidence_dir / "freeze_manifest.json"
    for path in (evidence_path, inventory_path, lineage_path, summary_path, freeze_manifest_path):
        if not path.exists():
            raise ValueError(f"Required evidence-analysis input is missing: {path}")
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    for artifact_name, record in (freeze_manifest.get("artifacts") or {}).items():
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ValueError(f"Freeze manifest has an invalid artifact record: {artifact_name}")
        local_path = evidence_dir / Path(str(record.get("path") or artifact_name)).name
        if not local_path.exists() or sha256_file(local_path) != record["sha256"]:
            raise ValueError(f"Frozen evidence artifact hash mismatch: {local_path}")
    evidence_rows = read_jsonl(evidence_path) if evidence_path.suffix == ".jsonl" else _read_csv(evidence_path)
    exposures = read_jsonl(inventory_path)
    lineages = read_jsonl(lineage_path)
    if any(str(row.get("evidence_freshness")) != "previously_queried" for row in evidence_rows):
        raise ValueError("Evidence rows must be explicitly labeled previously_queried.")
    if any(_as_bool(row.get("final_confirmation_eligible")) for row in evidence_rows):
        raise ValueError("Retrospective evidence rows cannot be final-confirmation eligible.")

    matched = _matched_rows(evidence_rows)
    summary_rows = _summary_rows(evidence_rows, lineages)
    matched_path = out_dir / "matched_parent_candidate_evidence.csv"
    excluded_path = out_dir / "excluded_evidence_summary.csv"
    cases_path = out_dir / "case_studies.md"
    write_csv_atomic(matched_path, matched)
    write_csv_atomic(excluded_path, summary_rows)
    _write_cases(cases_path, evidence_rows, exposures, lineages)

    funnel_path = out_dir / "search_funnel.csv"
    if funnel_path.exists():
        funnel = _read_csv(funnel_path)
        funnel = [row for row in funnel if row.get("branch") not in {"holdout", "external"}]
        branch_rows = []
        for row in summary_rows:
            if row["dimension"] != "evidence_set":
                continue
            branch_rows.extend(
                [
                    {
                        "arm_id": row["arm_id"],
                        "stage_order": 8,
                        "stage": f"{row['evidence_kind']}_evaluated",
                        "count": row["evaluated_candidate_count"],
                        "branch": row["evidence_kind"],
                        "evidence_set_id": row["dimension_value"],
                    },
                    {
                        "arm_id": row["arm_id"],
                        "stage_order": 9,
                        "stage": f"{row['evidence_kind']}_supported",
                        "count": row["supported_candidate_count"],
                        "branch": row["evidence_kind"],
                        "evidence_set_id": row["dimension_value"],
                    },
                ]
            )
        write_csv_atomic(funnel_path, [*funnel, *branch_rows])

    manifest_section = output_manifest(
        inputs=[evidence_path, inventory_path, lineage_path, summary_path, freeze_manifest_path],
        outputs=[matched_path, excluded_path, cases_path],
        restrictions=RESTRICTIONS,
        parameters={"case_selection": "lexicographic within predeclared cells"},
    )
    merge_analysis_manifest(
        out_dir / "analysis_manifest.json",
        section_name="retrospective_evidence_analysis",
        section_payload=manifest_section,
        inputs=[evidence_path, inventory_path, lineage_path, summary_path, freeze_manifest_path],
        outputs=[matched_path, excluded_path, cases_path, funnel_path],
        restrictions=RESTRICTIONS,
    )
    return {
        "matched_parent_rows": len(matched),
        "excluded_summary_rows": len(summary_rows),
        "case_study_cells": 6,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--out-dir", default="review-stage/claim-search-gpt55-paper-analysis-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
