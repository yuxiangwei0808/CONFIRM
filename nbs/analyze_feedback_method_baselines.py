"""Compare failure-specific feedback, failure-blind retry, and Self-Refine."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from nbs.claim_search_analysis_common import (
    iter_jsonl,
    read_result_header,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)

METHOD_LABELS = {
    "failure_specific": "Failure-specific diagnosis",
    "failure_blind": "Failure-blind retry",
    "self_refine": "Self-Refine",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalized_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _summary_from_rows(
    *,
    method: str,
    track: str,
    rows: list[dict[str, Any]],
    llm_calls: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    reported_cost: float | None,
    budget: str,
) -> dict[str, Any]:
    parent_count = len(rows)
    supported_candidates = sum(
        _as_int(
            row.get("final_multiplicity_adjusted_internal_pass_count")
            or row.get("supported_candidate_count")
        )
        for row in rows
    )
    supported_parents = sum(
        _as_bool(
            row.get("parent_with_internal_support")
            or row.get("internally_supported_candidate_ids")
        )
        for row in rows
    )
    returned_attempts = sum(
        _as_int(row.get("generated_candidate_count")) for row in rows
    )
    valid = sum(_as_int(row.get("valid_candidate_count")) for row in rows)
    unique = sum(_as_int(row.get("unique_candidate_count")) for row in rows)
    executed = sum(_as_int(row.get("current_data_evaluated_count")) for row in rows)
    retractions = sum(_as_int(row.get("multiplicity_retraction_count")) for row in rows)
    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "track": track,
        "budget": budget,
        "parent_count": parent_count,
        # Includes proposals from wholly invalid responses that triggered a retry.
        "returned_proposal_attempt_count": returned_attempts,
        "retained_unique_candidate_count": unique,
        "valid_candidate_count": valid,
        "unique_candidate_count": unique,
        "source_evaluated_candidate_count": executed,
        "final_source_supported_candidate_count": supported_candidates,
        "parents_with_source_support_count": supported_parents,
        "supported_parent_rate": (
            supported_parents / parent_count if parent_count else math.nan
        ),
        "multiplicity_retraction_count": retractions,
        "llm_call_count": llm_calls,
        "supported_candidates_per_100_calls": (
            100.0 * supported_candidates / llm_calls if llm_calls else math.nan
        ),
        "supported_parents_per_100_calls": (
            100.0 * supported_parents / llm_calls if llm_calls else math.nan
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "reported_cost": reported_cost,
        "token_status": (
            "provider_reported" if total_tokens is not None else "not_recorded"
        ),
        "cost_status": (
            "provider_reported" if reported_cost is not None else "not_recorded"
        ),
    }


def _paired_cells(
    reference_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    *,
    comparison_method: str,
) -> dict[str, Any]:
    def indexed(rows: list[dict[str, Any]]) -> dict[str, bool]:
        return {
            str(row["claim_id"]): _as_bool(
                row.get("parent_with_internal_support")
                or row.get("internally_supported_candidate_ids")
            )
            for row in rows
        }

    reference = indexed(reference_rows)
    comparison = indexed(comparison_rows)
    common = sorted(set(reference) & set(comparison))
    if len(common) != len(reference) or len(common) != len(comparison):
        raise ValueError(
            f"Parent mismatch for {comparison_method}: "
            f"reference={len(reference)} comparison={len(comparison)} common={len(common)}"
        )
    cells = {
        "neither": 0,
        "failure_specific_only": 0,
        "comparison_only": 0,
        "both": 0,
    }
    for claim_id in common:
        left = reference[claim_id]
        right = comparison[claim_id]
        if left and right:
            cells["both"] += 1
        elif left:
            cells["failure_specific_only"] += 1
        elif right:
            cells["comparison_only"] += 1
        else:
            cells["neither"] += 1
    return {
        "comparison_method": comparison_method,
        "comparison_label": METHOD_LABELS[comparison_method],
        "paired_parent_count": len(common),
        **cells,
    }


def _legacy_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    header = read_result_header(path)
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return _csv_rows(csv_path), header


def _self_refine_rows(root: Path, track: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = root / track / "replay" / "iterative_candidate_replay.json"
    return _legacy_rows(artifact)


def _evidence_rows(
    root: Path,
    *,
    arm_id: str,
    method: str,
    track: str,
) -> list[dict[str, Any]]:
    path = root / "candidate_evidence.jsonl"
    rows = [
        row
        for row in iter_jsonl(path)
        if str(row.get("arm_id")) == arm_id
    ]
    for row in rows:
        row["method"] = method
        row["track"] = track
    return rows


def _evidence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        evidence_kind = str(row.get("evidence_kind") or "unknown")
        evidence_set = _evidence_set_id(row)
        key = (
            str(row["method"]),
            str(row["track"]),
            evidence_kind,
            evidence_set,
        )
        groups.setdefault(key, []).append(row)
    output = []
    for (method, track, evidence_kind, evidence_set), group in sorted(
        groups.items()
    ):
        compatible = [row for row in group if _as_bool(row.get("compatible"))]
        evaluated = [row for row in group if _as_bool(row.get("evaluated"))]
        supported = [row for row in evaluated if _as_bool(row.get("supported"))]
        supported_parents = {
            str(row["parent_claim_id"]) for row in supported
        }
        output.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "track": track,
                "evidence_kind": evidence_kind,
                "evidence_set_id": evidence_set,
                "compatible_candidate_count": len(compatible),
                "evaluated_candidate_count": len(evaluated),
                "supported_candidate_count": len(supported),
                "supported_parent_count": len(supported_parents),
                "conditional_survival_rate": (
                    len(supported) / len(evaluated)
                    if evaluated
                    else math.nan
                ),
                "unavailable_count": len(group) - len(compatible),
                "execution_error_count": sum(
                    _as_bool(row.get("execution_error")) for row in group
                ),
            }
        )
    return output


def _evidence_parent_cells(
    reference_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    *,
    comparison_method: str,
    evidence_kind: str,
    evidence_set_id: str,
    parent_ids: set[str],
) -> dict[str, Any]:
    def select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if str(row.get("evidence_kind")) == evidence_kind
            and _evidence_set_id(row) == evidence_set_id
        ]

    def maps(
        rows: list[dict[str, Any]],
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        supported = {parent_id: False for parent_id in parent_ids}
        evaluated = {parent_id: False for parent_id in parent_ids}
        for row in select(rows):
            parent_id = str(row["parent_claim_id"])
            if parent_id not in parent_ids:
                continue
            evaluated[parent_id] = (
                evaluated[parent_id] or _as_bool(row.get("evaluated"))
            )
            supported[parent_id] = (
                supported[parent_id] or _as_bool(row.get("supported"))
            )
        return supported, evaluated

    reference, reference_evaluated = maps(reference_rows)
    comparison, comparison_evaluated = maps(comparison_rows)
    cells = {
        "neither": 0,
        "failure_specific_only": 0,
        "comparison_only": 0,
        "both": 0,
    }
    for parent_id in sorted(parent_ids):
        left = reference[parent_id]
        right = comparison[parent_id]
        if left and right:
            cells["both"] += 1
        elif left:
            cells["failure_specific_only"] += 1
        elif right:
            cells["comparison_only"] += 1
        else:
            cells["neither"] += 1
    return {
        "comparison_method": comparison_method,
        "comparison_label": METHOD_LABELS[comparison_method],
        "evidence_kind": evidence_kind,
        "evidence_set_id": evidence_set_id,
        "paired_parent_count": len(parent_ids),
        "both_evaluated_parent_count": sum(
            reference_evaluated[parent_id]
            and comparison_evaluated[parent_id]
            for parent_id in parent_ids
        ),
        **cells,
    }


def _evidence_set_id(row: dict[str, Any]) -> str:
    value = row.get("evidence_set_id")
    if value not in (None, ""):
        return str(value)
    if str(row.get("evidence_kind")) == "holdout":
        return "internal_holdout"
    if str(row.get("evidence_kind")) == "external":
        return "external_unavailable"
    return "unknown"


def _usage(header: dict[str, Any]) -> tuple[int | None, int | None, int | None, float | None]:
    summary = header.get("summary") or {}
    if not _as_bool(summary.get("llm_usage_complete")):
        return None, None, None, None
    reported_cost = (
        float(summary["llm_reported_cost"])
        if _as_bool(summary.get("llm_cost_complete"))
        and summary.get("llm_reported_cost") is not None
        else None
    )
    return (
        _as_int(summary.get("llm_prompt_tokens")),
        _as_int(summary.get("llm_completion_tokens")),
        _as_int(summary.get("llm_total_tokens")),
        reported_cost,
    )


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    scientific = [row for row in rows if row["track"] == "scientific"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Matched feedback controls at a three-round, five-candidate budget.} Source support is multiplicity-adjusted; call counts include schema retries. Exact token accounting was unavailable for reused legacy arms.}",
        r"\label{tab:feedback_controls}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Method & Parents & Returned attempts & Unique & Executed & Supported & Parents supported & LLM calls \\",
        r"\midrule",
    ]
    for row in scientific:
        lines.append(
            f"{row['method_label']} & {row['parent_count']} & "
            f"{row['returned_proposal_attempt_count']} & "
            f"{row['retained_unique_candidate_count']} & "
            f"{row['source_evaluated_candidate_count']} & "
            f"{row['final_source_supported_candidate_count']} & "
            f"{row['parents_with_source_support_count']} & "
            f"{row['llm_call_count']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    structured_dir = Path(args.structured_dir)
    structured_rows = _normalized_rows(
        structured_dir / "parent_summaries.jsonl"
    )
    structured_calls = sum(
        1 for _ in iter_jsonl(structured_dir / "llm_calls.jsonl")
    )

    generic_rows, _ = _legacy_rows(Path(args.generic_artifact))
    control = json.loads(Path(args.control_summary).read_text(encoding="utf-8"))
    generic_calls = _as_int(
        control["arms"]["generic_retry"].get("total_llm_attempt_count")
        or control["arms"]["generic_retry"].get("llm_call_count")
    )

    self_scientific_rows, self_scientific_header = _self_refine_rows(
        Path(args.self_refine_root),
        "scientific",
    )
    prompt, completion, total, cost = _usage(self_scientific_header)
    self_scientific_calls = _as_int(
        (self_scientific_header.get("summary") or {}).get("llm_call_count")
    )

    method_rows = [
        _summary_from_rows(
            method="failure_specific",
            track="scientific",
            rows=structured_rows,
            llm_calls=structured_calls,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            reported_cost=None,
            budget="R3/C5",
        ),
        _summary_from_rows(
            method="failure_blind",
            track="scientific",
            rows=generic_rows,
            llm_calls=generic_calls,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            reported_cost=None,
            budget="R3/C5",
        ),
        _summary_from_rows(
            method="self_refine",
            track="scientific",
            rows=self_scientific_rows,
            llm_calls=self_scientific_calls,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reported_cost=cost,
            budget="R3/C5",
        ),
    ]

    self_safety_rows, self_safety_header = _self_refine_rows(
        Path(args.self_refine_root),
        "safety",
    )
    prompt, completion, total, cost = _usage(self_safety_header)
    method_rows.append(
        _summary_from_rows(
            method="self_refine",
            track="known_negative",
            rows=self_safety_rows,
            llm_calls=_as_int(
                (self_safety_header.get("summary") or {}).get("llm_call_count")
            ),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reported_cost=cost,
            budget="R3/C5",
        )
    )

    safety_rows = []
    for budget, path in (
        ("R1/C2", Path(args.safety_r1_artifact)),
        ("R10/C10", Path(args.safety_r10_artifact)),
    ):
        rows, header = _legacy_rows(path)
        safety_rows.append(
            _summary_from_rows(
                method="failure_specific",
                track="known_negative",
                rows=rows,
                llm_calls=_as_int(
                    (header.get("provenance") or {}).get(
                        "total_prompt_attempt_record_count"
                    )
                ),
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                reported_cost=None,
                budget=budget,
            )
        )
    method_rows.extend(safety_rows)

    paired_rows = [
        _paired_cells(
            structured_rows,
            generic_rows,
            comparison_method="failure_blind",
        ),
        _paired_cells(
            structured_rows,
            self_scientific_rows,
            comparison_method="self_refine",
        ),
    ]
    structured_evidence = _evidence_rows(
        Path(args.structured_evidence_root),
        arm_id="r3_c5",
        method="failure_specific",
        track="scientific",
    )
    blind_evidence = _evidence_rows(
        Path(args.failure_blind_evidence_root),
        arm_id="r3_c5",
        method="failure_blind",
        track="scientific",
    )
    self_evidence = _evidence_rows(
        Path(args.self_refine_evidence_root),
        arm_id="r3_c5",
        method="self_refine",
        track="scientific",
    )
    self_safety_evidence = _evidence_rows(
        Path(args.self_refine_safety_evidence_root),
        arm_id="r3_c5",
        method="self_refine",
        track="known_negative",
    )
    all_evidence = [
        *structured_evidence,
        *blind_evidence,
        *self_evidence,
        *self_safety_evidence,
    ]
    evidence_summary_rows = _evidence_summary(all_evidence)
    parent_ids = {str(row["claim_id"]) for row in structured_rows}
    evidence_cells = []
    for comparison_method, comparison in (
        ("failure_blind", blind_evidence),
        ("self_refine", self_evidence),
    ):
        evidence_keys = {
            (
                str(row.get("evidence_kind")),
                str(row.get("evidence_set_id") or "internal_holdout"),
            )
            for row in [*structured_evidence, *comparison]
        }
        for evidence_kind, evidence_set_id in sorted(evidence_keys):
            evidence_cells.append(
                _evidence_parent_cells(
                    structured_evidence,
                    comparison,
                    comparison_method=comparison_method,
                    evidence_kind=evidence_kind,
                    evidence_set_id=evidence_set_id,
                    parent_ids=parent_ids,
                )
            )
    method_path = out_dir / "feedback_method_summary.csv"
    paired_path = out_dir / "feedback_parent_pairs.csv"
    evidence_path = out_dir / "feedback_evidence_summary.csv"
    evidence_pairs_path = out_dir / "feedback_evidence_parent_pairs.csv"
    write_csv_atomic(method_path, method_rows)
    write_csv_atomic(paired_path, paired_rows)
    write_csv_atomic(evidence_path, evidence_summary_rows)
    write_csv_atomic(evidence_pairs_path, evidence_cells)
    table_path = out_dir / "tab_feedback_controls.tex"
    _write_table(method_rows, table_path)
    inputs = [
        structured_dir / "parent_summaries.jsonl",
        structured_dir / "llm_calls.jsonl",
        Path(args.generic_artifact),
        Path(args.control_summary),
        Path(args.self_refine_root)
        / "scientific/replay/iterative_candidate_replay.json",
        Path(args.self_refine_root)
        / "safety/replay/iterative_candidate_replay.json",
        Path(args.safety_r1_artifact),
        Path(args.safety_r10_artifact),
        Path(args.structured_evidence_root) / "candidate_evidence.jsonl",
        Path(args.failure_blind_evidence_root) / "candidate_evidence.jsonl",
        Path(args.self_refine_evidence_root) / "candidate_evidence.jsonl",
        Path(args.self_refine_safety_evidence_root)
        / "candidate_evidence.jsonl",
    ]
    manifest = {
        "version": "feedback-method-baselines-v1",
        "methods": METHOD_LABELS,
        "scientific_parent_count": len(structured_rows),
        "known_negative_parent_count": len(self_safety_rows),
        "interpretation_restrictions": [
            "one_gpt55_realization",
            "descriptive_not_causal",
            "same_data_support_is_exploratory",
            "safety_failure_specific_arms_have_unmatched_budgets",
            "legacy_token_and_cost_usage_not_recorded",
        ],
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                method_path,
                paired_path,
                evidence_path,
                evidence_pairs_path,
                table_path,
            )
        ],
    }
    write_json_atomic(out_dir / "feedback_comparison_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structured-dir",
        default=(
            "review-stage/claim-search-gpt55-sweep-v7/"
            "normalized/arms/r3_c5"
        ),
    )
    parser.add_argument(
        "--generic-artifact",
        default=(
            "review-stage/claim-search-gpt55-control-r3-c5-v7/"
            "generic_retry/iterative_candidate_replay.json"
        ),
    )
    parser.add_argument(
        "--control-summary",
        default="review-stage/claim-search-gpt55-control-r3-c5-v7/control_summary.json",
    )
    parser.add_argument(
        "--self-refine-root",
        default="review-stage/claim-search-gpt55-self-refine-r3-c5-v1",
    )
    parser.add_argument(
        "--safety-r1-artifact",
        default=(
            "review-stage/claim-search-safety-gpt55-r1-c2-v7/"
            "replay/iterative_candidate_replay.json"
        ),
    )
    parser.add_argument(
        "--safety-r10-artifact",
        default=(
            "review-stage/claim-search-safety-gpt55-r10-c10-v7/"
            "replay/iterative_candidate_replay.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/claim-search-gpt55-feedback-baselines-v1",
    )
    parser.add_argument(
        "--structured-evidence-root",
        default="review-stage/claim-search-gpt55-retrospective-evidence-v3",
    )
    parser.add_argument(
        "--failure-blind-evidence-root",
        default=(
            "review-stage/claim-search-gpt55-feedback-baselines-v1/"
            "evidence/failure_blind"
        ),
    )
    parser.add_argument(
        "--self-refine-evidence-root",
        default=(
            "review-stage/claim-search-gpt55-feedback-baselines-v1/"
            "evidence/self_refine"
        ),
    )
    parser.add_argument(
        "--self-refine-safety-evidence-root",
        default=(
            "review-stage/claim-search-gpt55-feedback-baselines-v1/"
            "evidence/self_refine_safety"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
