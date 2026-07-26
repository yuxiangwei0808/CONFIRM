"""Generate paper-analysis tables and figures from a frozen claim-search sweep."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nbs.claim_search_analysis_common import (
    arm_sort_key,
    clustered_binary_interval,
    configure_matplotlib,
    iter_jsonl,
    merge_analysis_manifest,
    read_jsonl,
    read_result_header,
    sha256_file,
    wilson_interval,
    write_csv_atomic,
    write_json_atomic,
)


RESTRICTIONS = (
    "One stochastic GPT-5.5 realization; budget comparisons are descriptive, not causal.",
    "R3/C5 is the predeclared reference because it matches the generic-retry control.",
    "Internal support is adaptive same-data support, not independent confirmation.",
    "Novelty is relative to the parent contract, not literature-wide novelty.",
    "Holdout and external evidence are retrospective and previously queried.",
)


def _clustered_candidate_interval(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("current_data_evaluated"):
            by_parent[str(row["parent_claim_id"])].append(row)
    if not by_parent:
        return math.nan, math.nan, math.nan
    clusters = list(by_parent.values())
    numerators = np.asarray(
        [sum(bool(row.get("final_internal_supported")) for row in cluster) for cluster in clusters],
        dtype=float,
    )
    denominators = np.asarray([len(cluster) for cluster in clusters], dtype=float)
    numerator = float(numerators.sum())
    denominator = float(denominators.sum())
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(clusters), size=(resamples, len(clusters)))
    draws = numerators[indexes].sum(axis=1) / denominators[indexes].sum(axis=1)
    return (
        numerator / denominator,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _project_lineage(row: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "arm_id", "lineage_event_id", "parent_claim_id", "target_family",
        "source_mode", "failure_localization", "internally_supported_candidate_ids",
        "llm_response_count",
    }
    projected = {key: value for key, value in row.items() if key in fields}
    projected["round_failure_contexts"] = [
        {
            "round_index": context.get("round_index"),
            "candidate_id": context.get("candidate_id"),
            "failed_gates": context.get("failed_gates") or [],
            "failed_candidates": [
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "failed_gates": candidate.get("failed_gates") or [],
                }
                for candidate in context.get("failed_candidates") or []
                if isinstance(candidate, dict)
            ],
        }
        for context in row.get("round_failure_contexts") or []
        if isinstance(context, dict)
    ]
    return projected


def _arm_rows(matrix_path: Path, expected_parents: int) -> list[dict[str, Any]]:
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload.get("rows") or []]
    expected = {(rounds, candidates) for rounds in (1, 3, 5, 10) for candidates in (2, 5, 10)}
    observed = {
        (int(row["max_rounds"]), int(row["max_candidates_per_round"]))
        for row in rows
    }
    if observed != expected:
        raise ValueError(f"Sweep grid is incomplete: missing={sorted(expected - observed)}")
    for row in rows:
        if row.get("status") != "completed":
            raise ValueError(f"Incomplete arm: {row.get('artifact')}")
        if int(row.get("searchable_claim_count") or 0) != expected_parents:
            raise ValueError(f"Arm does not contain {expected_parents} parents: {row.get('artifact')}")
        if int(row.get("excluded_evidence_query_count") or 0):
            raise ValueError(f"Search arm queried excluded evidence: {row.get('artifact')}")
        row["arm_id"] = f"r{int(row['max_rounds'])}_c{int(row['max_candidates_per_round'])}"
    return sorted(rows, key=lambda item: arm_sort_key(item["arm_id"]))


def _validate_frozen(
    arms: list[dict[str, Any]],
    lineages: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    expected_parents: int,
) -> None:
    arm_ids = {row["arm_id"] for row in arms}
    if {row["arm_id"] for row in lineages} != arm_ids:
        raise ValueError("Frozen lineage arms do not match the matrix summary.")
    lineage_counts = Counter(row["arm_id"] for row in lineages)
    if set(lineage_counts.values()) != {expected_parents}:
        raise ValueError(f"Frozen parent counts are not all {expected_parents}: {lineage_counts}")
    response_counts = Counter(row["arm_id"] for row in responses)
    for arm in arms:
        arm_id = arm["arm_id"]
        arm_exposures = [row for row in exposures if row["arm_id"] == arm_id]
        observed_counts = {
            "generated_candidate_count": len(arm_exposures),
            "candidate_count": sum(row.get("generation_status") == "retained" for row in arm_exposures),
            "duplicate_candidate_count": sum(row.get("generation_status") == "duplicate" for row in arm_exposures),
            "unretained_generated_candidate_count": sum(
                row.get("generation_status") in {"unretained", "superseded_retry"}
                for row in arm_exposures
            ),
            "policy_valid_candidate_count": sum(
                row.get("generation_status") == "retained" and row.get("validation_ok") is True
                for row in arm_exposures
            ),
            "current_data_evaluated_count": sum(bool(row.get("current_data_evaluated")) for row in arm_exposures),
            "provisional_internal_pass_count": sum(bool(row.get("provisional_internal_supported")) for row in arm_exposures),
            "final_multiplicity_adjusted_internal_pass_count": sum(bool(row.get("final_internal_supported")) for row in arm_exposures),
            "multiplicity_retraction_count": sum(bool(row.get("multiplicity_retracted")) for row in arm_exposures),
        }
        for field, observed in observed_counts.items():
            expected = int(arm.get(field) or 0)
            if observed != expected:
                raise ValueError(
                    f"Frozen funnel mismatch for {arm_id}/{field}: observed={observed} expected={expected}"
                )
        lineage_response_count = sum(
            int(row.get("llm_response_count") or 0)
            for row in lineages
            if row["arm_id"] == arm_id
        )
        if response_counts[arm_id] != lineage_response_count:
            raise ValueError(
                f"LLM response count mismatch for {arm_id}: "
                f"responses={response_counts[arm_id]} lineages={lineage_response_count}"
            )
    invalid_noops = [
        row["exposure_id"]
        for row in exposures
        if row.get("generation_status") == "retained"
        and row.get("validation_ok") is True
        and row.get("current_data_evaluated")
        and not (row.get("executable_contract_delta") or {})
    ]
    if invalid_noops:
        raise ValueError(f"Accepted no-op candidate contracts remain: {invalid_noops[:5]}")
    unexplained = [
        row["exposure_id"]
        for row in exposures
        if row.get("generation_status") == "retained"
        and row.get("validation_ok") is True
        and row.get("current_data_label") is None
        and row.get("current_data_evaluated")
    ]
    if unexplained:
        raise ValueError(f"Source-evaluated candidates have no final label: {unexplained[:5]}")


def _budget_rows(
    arms: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    safety_paths: Iterable[Path],
) -> list[dict[str, Any]]:
    response_counts = Counter(row["arm_id"] for row in responses)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        calls = response_counts[arm["arm_id"]]
        supported_parents = int(arm.get("parents_with_internal_support_count") or 0)
        unique_candidates = int(arm.get("unique_candidate_count") or 0)
        rows.append(
            {
                "analysis_scope": "scientific_sweep",
                "arm_id": arm["arm_id"],
                "max_rounds": int(arm["max_rounds"]),
                "max_candidates": int(arm["max_candidates_per_round"]),
                "parent_count": int(arm["searchable_claim_count"]),
                "supported_parent_count": supported_parents,
                "parent_yield": supported_parents / int(arm["searchable_claim_count"]),
                "current_data_evaluated_candidates": int(arm.get("current_data_evaluated_count") or 0),
                "unique_candidates": unique_candidates,
                "duplicate_candidates": int(arm.get("duplicate_candidate_count") or 0),
                "duplicate_rate": (
                    int(arm.get("duplicate_candidate_count") or 0)
                    / int(arm.get("generated_candidate_count") or 1)
                ),
                "llm_calls": calls,
                "unique_candidates_per_100_llm_calls": 100.0 * unique_candidates / calls if calls else math.nan,
                "supported_parents_per_100_llm_calls": 100.0 * supported_parents / calls if calls else math.nan,
                "excluded_evidence_query_count": int(arm.get("excluded_evidence_query_count") or 0),
            }
        )
    for path in safety_paths:
        if not path.exists():
            continue
        payload = read_result_header(path)
        summary = payload.get("summary") or {}
        config = payload.get("config") or {}
        total = int(payload.get("searchable_claim_count") or summary.get("n_searches") or 0)
        supported = int(summary.get("parents_with_internal_support_count") or 0)
        lower, upper = wilson_interval(supported, total)
        rows.append(
            {
                "analysis_scope": "known_negative_safety",
                "arm_id": f"r{int(config.get('max_rounds') or 0)}_c{int(config.get('max_candidates_per_round') or 0)}",
                "max_rounds": int(config.get("max_rounds") or 0),
                "max_candidates": int(config.get("max_candidates_per_round") or 0),
                "parent_count": total,
                "supported_parent_count": supported,
                "parent_yield": supported / total if total else math.nan,
                "parent_yield_wilson_low": lower,
                "parent_yield_wilson_high": upper,
                "synthetic_stress_qualification": True,
                "artifact": str(path),
            }
        )
    return rows


def _funnel_rows(
    arms: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    response_counts = Counter(row["arm_id"] for row in responses)
    exposure_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exposures:
        exposure_by_arm[row["arm_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        arm_id = arm["arm_id"]
        candidates = exposure_by_arm[arm_id]
        stages = (
            ("llm_calls", response_counts[arm_id]),
            ("parsed_proposals", len(candidates)),
            ("used_response_proposals", sum(row.get("generation_status") != "superseded_retry" for row in candidates)),
            ("unique_retained", sum(row.get("generation_status") == "retained" for row in candidates)),
            ("policy_valid", sum(row.get("generation_status") == "retained" and row.get("validation_ok") is True for row in candidates)),
            ("source_executed", sum(bool(row.get("current_data_evaluated")) for row in candidates)),
            ("provisional_pass", sum(bool(row.get("provisional_internal_supported")) for row in candidates)),
            ("final_multiplicity_adjusted_support", sum(bool(row.get("final_internal_supported")) for row in candidates)),
        )
        for order, (stage, count) in enumerate(stages):
            rows.append({"arm_id": arm_id, "stage_order": order, "stage": stage, "count": int(count)})
    return rows


def _parent_strata(
    lineages: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = {
        "target_family": lambda row: row.get("target_family") or "unknown",
        "source_mode": lambda row: row.get("source_mode") or "unknown",
        "target_x_source": lambda row: f"{row.get('target_family') or 'unknown'}|{row.get('source_mode') or 'unknown'}",
        "primary_failure_kind": lambda row: ((row.get("failure_localization") or {}).get("failure_kind") or "unknown"),
        "primary_failed_gate": lambda row: ((row.get("failure_localization") or {}).get("primary_failure") or "unknown"),
    }
    for arm_id in sorted({row["arm_id"] for row in lineages}, key=arm_sort_key):
        arm_rows = [row for row in lineages if row["arm_id"] == arm_id]
        for dimension, getter in dimensions.items():
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in arm_rows:
                grouped[str(getter(row))].append(row)
            for value, group in sorted(grouped.items()):
                supported = [float(bool(row.get("internally_supported_candidate_ids"))) for row in group]
                estimate, lower, upper = clustered_binary_interval(
                    supported,
                    resamples=resamples,
                    seed=seed,
                )
                rows.append(
                    {
                        "arm_id": arm_id,
                        "dimension": dimension,
                        "dimension_value": value,
                        "parent_count": len(group),
                        "supported_parent_count": int(sum(supported)),
                        "parent_yield": estimate,
                        "bootstrap_low": lower,
                        "bootstrap_high": upper,
                        "bootstrap_unit": "parent_lineage",
                    }
                )
        by_source_target: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in arm_rows:
            by_source_target[str(row.get("source_mode") or "unknown")][str(row.get("target_family") or "unknown")].append(
                float(bool(row.get("internally_supported_candidate_ids")))
            )
        all_targets = sorted({row.get("target_family") or "unknown" for row in arm_rows})
        for source_mode, target_values in sorted(by_source_target.items()):
            rates = [float(np.mean(target_values[target])) for target in all_targets if target_values.get(target)]
            rng = np.random.default_rng(seed)
            standardized_draws = []
            represented = [target_values[target] for target in all_targets if target_values.get(target)]
            for _ in range(resamples):
                standardized_draws.append(
                    float(
                        np.mean(
                            [
                                np.mean(rng.choice(values, size=len(values), replace=True))
                                for values in represented
                            ]
                        )
                    )
                )
            rows.append(
                {
                    "arm_id": arm_id,
                    "dimension": "target_standardized_source_mode",
                    "dimension_value": source_mode,
                    "parent_count": sum(len(values) for values in target_values.values()),
                    "supported_parent_count": sum(sum(values) for values in target_values.values()),
                    "parent_yield": float(np.mean(rates)) if rates else math.nan,
                    "bootstrap_low": float(np.quantile(standardized_draws, 0.025)),
                    "bootstrap_high": float(np.quantile(standardized_draws, 0.975)),
                    "bootstrap_unit": "parent_lineage_within_target",
                    "standardization": "equal weight across represented target families",
                    "descriptive_only": True,
                }
            )
    retained = [row for row in exposures if row.get("generation_status") == "retained"]
    candidate_dimensions = {
        "candidate_target_family": lambda row: row.get("target_family") or "unknown",
        "candidate_source_mode": lambda row: row.get("source_mode") or "unknown",
        "candidate_target_x_source": lambda row: f"{row.get('target_family') or 'unknown'}|{row.get('source_mode') or 'unknown'}",
    }
    for arm_id in sorted({row["arm_id"] for row in retained}, key=arm_sort_key):
        arm_candidates = [row for row in retained if row["arm_id"] == arm_id]
        for dimension, getter in candidate_dimensions.items():
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in arm_candidates:
                grouped[str(getter(row))].append(row)
            for value, group in sorted(grouped.items()):
                evaluated = [row for row in group if row.get("current_data_evaluated")]
                supported = [row for row in evaluated if row.get("final_internal_supported")]
                estimate, lower, upper = _clustered_candidate_interval(
                    group,
                    resamples=resamples,
                    seed=seed,
                )
                rows.append(
                    {
                        "arm_id": arm_id,
                        "dimension": dimension,
                        "dimension_value": value,
                        "candidate_count": len(group),
                        "policy_valid_candidate_count": sum(row.get("validation_ok") is True for row in group),
                        "evaluated_candidate_count": len(evaluated),
                        "supported_candidate_count": len(supported),
                        "candidate_support_rate": estimate,
                        "bootstrap_low": lower,
                        "bootstrap_high": upper,
                        "bootstrap_unit": "parent_lineage",
                    }
                )
    return rows


def _transform_failure_rows(
    exposures: list[dict[str, Any]],
    lineages: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    retained = [row for row in exposures if row.get("generation_status") == "retained"]
    rows: list[dict[str, Any]] = []
    dimensions = {
        "inferred_transform": lambda row: row.get("inferred_transform") or "unknown",
        "declared_transform": lambda row: row.get("declared_transform") or row.get("transform_type") or "unknown",
        "search_round": lambda row: f"round_{int(row.get('round_index') or 0)}",
    }
    for arm_id in sorted({row["arm_id"] for row in retained}, key=arm_sort_key):
        arm_rows = [row for row in retained if row["arm_id"] == arm_id]
        for dimension, getter in dimensions.items():
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in arm_rows:
                grouped[str(getter(row))].append(row)
            for value, group in sorted(grouped.items()):
                estimate, lower, upper = _clustered_candidate_interval(
                    group,
                    resamples=resamples,
                    seed=seed,
                )
                rows.append(
                    {
                        "arm_id": arm_id,
                        "dimension": dimension,
                        "dimension_value": value,
                        "candidate_count": len(group),
                        "policy_valid_count": sum(row.get("validation_ok") is True for row in group),
                        "source_evaluated_count": sum(bool(row.get("current_data_evaluated")) for row in group),
                        "provisional_pass_count": sum(bool(row.get("provisional_internal_supported")) for row in group),
                        "final_support_count": sum(bool(row.get("final_internal_supported")) for row in group),
                        "candidate_support_rate": estimate,
                        "bootstrap_low": lower,
                        "bootstrap_high": upper,
                        "bootstrap_unit": "parent_lineage",
                        "multiplicity_retracted_count": sum(bool(row.get("multiplicity_retracted")) for row in group),
                    }
                )
        confusion: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in arm_rows:
            confusion[(str(row.get("declared_transform") or "unknown"), str(row.get("inferred_transform") or "unknown"))].append(row)
        for (declared, inferred), group in sorted(confusion.items()):
            rows.append(
                {
                    "arm_id": arm_id,
                    "dimension": "declared_vs_inferred_transform",
                    "dimension_value": f"{declared}|{inferred}",
                    "declared_transform": declared,
                    "inferred_transform": inferred,
                    "candidate_count": len(group),
                    "transform_match_count": sum(row.get("transform_match") is True for row in group),
                    "final_support_count": sum(bool(row.get("final_internal_supported")) for row in group),
                }
            )

    exposure_by_key = {(row["lineage_event_id"], row["candidate_id"]): row for row in retained}
    retained_by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exposure in retained:
        retained_by_lineage[str(exposure["lineage_event_id"])].append(exposure)
    for lineage in lineages:
        localization = lineage.get("failure_localization") or {}
        parent_failed = set(localization.get("failed_gates") or [])
        transitioned_candidate_ids: set[str] = set()
        for round_context in lineage.get("round_failure_contexts") or []:
            failed_candidates = round_context.get("failed_candidates") or []
            if not failed_candidates and round_context.get("candidate_id"):
                failed_candidates = [round_context]
            for context in failed_candidates:
                candidate_id = str(context.get("candidate_id") or "")
                exposure = exposure_by_key.get((lineage["lineage_event_id"], candidate_id))
                if not exposure:
                    continue
                transitioned_candidate_ids.add(candidate_id)
                candidate_failed = set(context.get("failed_gates") or [])
                for gate in sorted(parent_failed):
                    rows.append(
                        {
                            "arm_id": lineage["arm_id"],
                            "dimension": "parent_failure_transition",
                            "dimension_value": gate,
                            "candidate_id": candidate_id,
                            "parent_failed_gate": gate,
                            "candidate_satisfied_parent_gate": int(gate not in candidate_failed),
                            "candidate_final_support": int(bool(exposure.get("final_internal_supported"))),
                            "candidate_failed_gates": json.dumps(sorted(candidate_failed)),
                        }
                    )
        for exposure in retained_by_lineage.get(str(lineage.get("lineage_event_id")), []):
            if (
                exposure.get("lineage_event_id") != lineage.get("lineage_event_id")
                or not exposure.get("final_internal_supported")
                or exposure.get("candidate_id") in transitioned_candidate_ids
            ):
                continue
            for gate in sorted(parent_failed):
                rows.append(
                    {
                        "arm_id": lineage["arm_id"],
                        "dimension": "parent_failure_transition",
                        "dimension_value": gate,
                        "candidate_id": exposure["candidate_id"],
                        "parent_failed_gate": gate,
                        "candidate_satisfied_parent_gate": 1,
                        "candidate_final_support": 1,
                        "candidate_failed_gates": "[]",
                    }
                )
    supported_by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for exposure in retained:
        if exposure.get("final_internal_supported"):
            supported_by_parent[(exposure["arm_id"], exposure["parent_claim_id"])].append(exposure)
    for (arm_id, parent_claim_id), group in sorted(supported_by_parent.items(), key=lambda item: (arm_sort_key(item[0][0]), item[0][1])):
        first_round = min(int(row.get("round_index") or 0) for row in group)
        rows.append(
            {
                "arm_id": arm_id,
                "dimension": "marginal_supported_parent_by_round",
                "dimension_value": f"round_{first_round}",
                "parent_claim_id": parent_claim_id,
                "new_supported_parent_count": 1,
            }
        )
    return rows


def _family_bin(value: Any) -> str:
    size = int(value or 0)
    if size <= 10:
        return "<=10"
    if size <= 25:
        return "11-25"
    if size <= 50:
        return "26-50"
    return ">50"


def _multiplicity_rows(exposures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for exposure in exposures:
        if not exposure.get("provisional_internal_supported") and not exposure.get("multiplicity_retracted"):
            continue
        rows.append(
            {
                "record_type": "candidate",
                "arm_id": exposure["arm_id"],
                "parent_claim_id": exposure["parent_claim_id"],
                "candidate_id": exposure["candidate_id"],
                "round_index": exposure.get("round_index"),
                "effective_family_size": exposure.get("effective_family_size"),
                "family_size_bin": _family_bin(exposure.get("effective_family_size")),
                "provisional_internal_supported": int(bool(exposure.get("provisional_internal_supported"))),
                "final_internal_supported": int(bool(exposure.get("final_internal_supported"))),
                "multiplicity_retracted": int(bool(exposure.get("multiplicity_retracted"))),
                "inferred_transform": exposure.get("inferred_transform") or "unknown",
            }
        )
    aggregate_rows: list[dict[str, Any]] = []
    for dimensions in (("arm_id",), ("arm_id", "round_index"), ("arm_id", "family_size_bin")):
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row[field] for field in dimensions)].append(row)
        for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
            aggregate = {
                "record_type": "summary",
                "summary_dimensions": "+".join(dimensions),
                "provisional_pass_count": len(group),
                "retracted_count": sum(row["multiplicity_retracted"] for row in group),
                "retraction_rate": sum(row["multiplicity_retracted"] for row in group) / len(group),
            }
            aggregate.update(dict(zip(dimensions, key)))
            aggregate_rows.append(aggregate)
    return [*rows, *aggregate_rows]


def _lineage_stability_rows(
    exposures: list[dict[str, Any]],
    lineages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    arm_ids = sorted({row["arm_id"] for row in lineages}, key=arm_sort_key)
    support_sets = {
        arm_id: {
            row["parent_claim_id"]
            for row in lineages
            if row["arm_id"] == arm_id and row.get("internally_supported_candidate_ids")
        }
        for arm_id in arm_ids
    }
    rows: list[dict[str, Any]] = []
    parents = sorted({row["parent_claim_id"] for row in lineages})
    for parent in parents:
        supported_arms = [arm for arm in arm_ids if parent in support_sets[arm]]
        rows.append(
            {
                "record_type": "parent_support_frequency",
                "parent_claim_id": parent,
                "supported_arm_count": len(supported_arms),
                "support_frequency": len(supported_arms) / len(arm_ids),
                "supported_arms": json.dumps(supported_arms),
            }
        )
    matrix = np.zeros((len(arm_ids), len(arm_ids)), dtype=float)
    for left_index, left in enumerate(arm_ids):
        for right_index, right in enumerate(arm_ids):
            union = support_sets[left] | support_sets[right]
            value = len(support_sets[left] & support_sets[right]) / len(union) if union else 1.0
            matrix[left_index, right_index] = value
            rows.append(
                {
                    "record_type": "arm_pair_jaccard",
                    "arm_id": left,
                    "other_arm_id": right,
                    "supported_parent_jaccard": value,
                }
            )
    retained = [row for row in exposures if row.get("generation_status") == "retained"]
    for field, record_type in (("exact_contract_id", "exact_contract_recurrence"), ("semantic_cluster_id", "semantic_cluster_recurrence")):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in retained:
            if row.get(field):
                grouped[str(row[field])].append(row)
        for cluster_id, group in sorted(grouped.items()):
            rows.append(
                {
                    "record_type": record_type,
                    "cluster_id": cluster_id,
                    "candidate_count": len(group),
                    "arm_count": len({row["arm_id"] for row in group}),
                    "parent_count": len({row["parent_claim_id"] for row in group}),
                    "final_support_count": sum(bool(row.get("final_internal_supported")) for row in group),
                }
            )
    return rows, arm_ids, matrix


def _plot_budget(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    configure_matplotlib(out_dir)
    import matplotlib.pyplot as plt

    scientific = [row for row in rows if row["analysis_scope"] == "scientific_sweep"]
    rounds = (1, 3, 5, 10)
    candidates = (2, 5, 10)
    # Budget is a cost-yield question, so plotting yield against spend shows the
    # scaling and the efficiency turn directly, which a grid of colour cells hides.
    colors = {1: "#A8C6E5", 3: "#6FA3CE", 5: "#3C74A8", 10: "#133C66"}
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )

    def _arm(max_rounds: int, max_candidates: int) -> dict[str, Any]:
        return next(
            row
            for row in scientific
            if int(row["max_rounds"]) == max_rounds
            and int(row["max_candidates"]) == max_candidates
        )

    # One structured response can carry every candidate for a round, so call volume
    # tracks rounds and is nearly flat in candidates. Rounds therefore carry the
    # cost, and the legend states it once per line.
    call_cost = {
        max_rounds: sum(float(_arm(max_rounds, k)["llm_calls"]) for k in candidates)
        / len(candidates)
        for max_rounds in rounds
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    panels = (
        (
            axes[0],
            "supported_parent_count",
            "Source-supported parents (of 215)",
            "a  Yield rises on both axes",
        ),
        (
            axes[1],
            "supported_parents_per_100_llm_calls",
            "Supported parents per 100 calls",
            "b  Efficiency falls with rounds",
        ),
    )
    for axis, field, ylabel, title in panels:
        for max_rounds in rounds:
            arms = [_arm(max_rounds, k) for k in candidates]
            axis.plot(
                candidates,
                [float(row[field]) for row in arms],
                marker="o",
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                linewidth=1.1,
                color=colors[max_rounds],
                label=(
                    f"{max_rounds} round"
                    + ("s" if max_rounds > 1 else "")
                    + f" ($\\approx${call_cost[max_rounds]:,.0f} calls)"
                ),
                zorder=3,
            )
        axis.set_xlabel("Candidates per round", fontsize=7)
        axis.set_ylabel(ylabel, fontsize=7)
        axis.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=5)
        axis.set_xticks(candidates, [str(value) for value in candidates])
        axis.set_xlim(1.2, 10.8)
        axis.grid(color="#E9E9E9", linewidth=0.55, zorder=0)
        axis.tick_params(labelsize=6.5)

    axes[0].legend(fontsize=6.3, loc="upper left", handlelength=1.4, labelspacing=0.28)
    fig.tight_layout()
    png = out_dir / "fig_budget_heatmaps.png"
    pdf = out_dir / "fig_budget_heatmaps.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _plot_funnel(rows: list[dict[str, Any]], reference_arm: str, out_dir: Path) -> list[Path]:
    configure_matplotlib(out_dir)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    stage_order = (
        "llm_calls",
        "parsed_proposals",
        "used_response_proposals",
        "unique_retained",
        "policy_valid",
        "source_executed",
        "provisional_pass",
        "final_multiplicity_adjusted_support",
    )
    selected = {
        str(row["stage"]): int(row["count"])
        for row in rows
        if row["arm_id"] == reference_arm
    }
    reference_label = reference_arm.upper().replace("_C", "/K")
    missing = [stage for stage in stage_order if stage not in selected]
    if missing:
        raise ValueError(f"Reference-arm funnel is missing stages: {missing}")
    if any(selected[stage] < 0 for stage in stage_order):
        raise ValueError("Reference-arm funnel counts must be non-negative.")

    generation_stages = stage_order[:6]
    generation_labels = (
        "LLM calls",
        "Parsed proposals",
        "Used-response proposals",
        "Unique retained",
        "Policy valid",
        "Source executed",
    )
    generation_values = [selected[stage] for stage in generation_stages]
    generation_colors = (
        "#767676",
        "#B4C0E4",
        "#B4C0E4",
        "#B4C0E4",
        "#7884B4",
        "#7884B4",
    )

    provisional = selected["provisional_pass"]
    final = selected["final_multiplicity_adjusted_support"]
    executed = selected["source_executed"]
    retracted = provisional - final
    if retracted < 0:
        raise ValueError("Final support cannot exceed provisional support.")
    provisional_rate = provisional / executed if executed else math.nan
    final_retention = final / provisional if provisional else math.nan

    fig, (axis_generation, axis_support) = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.65),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )

    generation_positions = np.arange(len(generation_stages))
    axis_generation.barh(
        generation_positions,
        generation_values,
        height=0.62,
        color=generation_colors,
        edgecolor="none",
        zorder=2,
    )
    axis_generation.set_yticks(generation_positions, labels=generation_labels)
    axis_generation.invert_yaxis()
    axis_generation.set_xlim(0, max(generation_values) * 1.13)
    axis_generation.set_xlabel("Count")
    axis_generation.set_title(
        "a  Generation and execution",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=23,
    )
    axis_generation.text(
        0.0,
        1.02,
        f"{reference_label}; one response can contain multiple proposals",
        transform=axis_generation.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#4D4D4D",
    )
    axis_generation.tick_params(axis="y", length=0)
    for position, value in zip(generation_positions, generation_values):
        axis_generation.text(
            value + max(generation_values) * 0.015,
            position,
            f"{value:,}",
            va="center",
            fontsize=7,
            color="#272727",
        )

    support_positions = np.arange(2)
    axis_support.barh(
        [support_positions[0]],
        [provisional],
        height=0.62,
        color="#B4C0E4",
        edgecolor="none",
        zorder=2,
    )
    axis_support.barh(
        [support_positions[1]],
        [final],
        height=0.62,
        color="#0F4D92",
        edgecolor="none",
        zorder=2,
    )
    if retracted:
        axis_support.barh(
            [support_positions[1]],
            [retracted],
            left=[final],
            height=0.62,
            color="#F6CFCB",
            edgecolor="#B64342",
            hatch="////",
            linewidth=0.45,
            zorder=2,
        )
    axis_support.set_yticks(support_positions, labels=("Provisional pass", "Final support"))
    axis_support.invert_yaxis()
    axis_support.set_xlim(0, max(100, math.ceil(provisional / 10) * 10))
    axis_support.set_xlabel("Count")
    axis_support.set_title(
        "b  Support adjudication",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        pad=23,
    )
    axis_support.text(
        0.0,
        1.02,
        (
            f"{provisional_rate:.1%} pass provisionally; "
            f"{final_retention:.1%} remain after final adjustment"
        ),
        transform=axis_support.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#4D4D4D",
    )
    axis_support.tick_params(axis="y", length=0)
    axis_support.text(
        provisional / 2,
        support_positions[0],
        f"{provisional:,}",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#272727",
    )
    axis_support.text(
        final / 2,
        support_positions[1],
        f"{final:,} retained",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="white",
    )
    if retracted:
        axis_support.text(
            final + retracted / 2,
            support_positions[1],
            f"{retracted:,}",
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#8A2D2B",
        )
        axis_support.text(
            final + retracted / 2,
            support_positions[1] - 0.45,
            "retracted",
            ha="center",
            va="center",
            fontsize=6.2,
            color="#8A2D2B",
        )

    png = out_dir / "fig_search_funnel.png"
    pdf = out_dir / "fig_search_funnel.pdf"
    svg = out_dir / "fig_search_funnel.svg"
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [svg, pdf, png]


def _plot_stability(arm_ids: list[str], matrix: np.ndarray, out_dir: Path) -> list[Path]:
    configure_matplotlib(out_dir)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    axis.set_xticks(range(len(arm_ids)), labels=arm_ids, rotation=45, ha="right")
    axis.set_yticks(range(len(arm_ids)), labels=arm_ids)
    axis.set_title("Supported-parent Jaccard overlap")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", fontsize=6, color="white" if matrix[row_index, column_index] < 0.55 else "black")
    fig.colorbar(image, ax=axis, label="Jaccard")
    png = out_dir / "fig_lineage_stability.png"
    pdf = out_dir / "fig_lineage_stability.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def run(args: argparse.Namespace) -> dict[str, Any]:
    sweep = Path(args.sweep)
    frozen = Path(args.frozen_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = sweep / "matrix_summary.json"
    freeze_manifest_path = frozen / "freeze_manifest.json"
    inventory_path = frozen / "frozen_search_inventory.jsonl"
    lineage_path = frozen / "frozen_lineages.jsonl"
    response_path = frozen / "frozen_llm_responses.jsonl"
    inputs = [matrix_path, freeze_manifest_path, inventory_path, lineage_path, response_path]
    for path in inputs:
        if not path.exists():
            raise ValueError(f"Required frozen analysis input is missing: {path}")

    arms = _arm_rows(matrix_path, args.expected_parent_count)
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    frozen_matrix = freeze_manifest.get("matrix_summary") or {}
    if frozen_matrix.get("sha256") != sha256_file(matrix_path):
        raise ValueError("Frozen inventory was not built from this exact finalized matrix summary.")
    matrix_source_hashes = {str(row.get("source_sha256") or "") for row in arms}
    if matrix_source_hashes != {str(freeze_manifest.get("source_payload_sha256") or "")}:
        raise ValueError("Frozen inventory and matrix summary do not share the exact source hash.")
    exposure_fields = {
        "exposure_id", "lineage_event_id", "arm_id", "parent_claim_id", "candidate_id",
        "target_family", "source_mode",
        "round_index", "declared_transform", "transform_type", "inferred_transform",
        "transform_match", "executable_contract_delta", "generation_status",
        "validation_ok", "validation_violations", "current_data_evaluated",
        "current_data_label", "provisional_internal_supported", "final_internal_supported",
        "multiplicity_retracted", "effective_family_size", "exact_contract_id",
        "semantic_cluster_id",
    }
    lineages = [_project_lineage(row) for row in iter_jsonl(lineage_path)]
    exposures = [
        {key: value for key, value in row.items() if key in exposure_fields}
        for row in iter_jsonl(inventory_path)
    ]
    responses = [{"arm_id": row.get("arm_id")} for row in iter_jsonl(response_path)]
    _validate_frozen(arms, lineages, exposures, responses, args.expected_parent_count)
    if args.reference_arm not in {row["arm_id"] for row in arms}:
        raise ValueError(f"Reference arm is absent: {args.reference_arm}")

    safety_paths = [Path(path) for path in args.safety_artifact]
    budget_rows = _budget_rows(arms, responses, safety_paths)
    funnel_rows = _funnel_rows(arms, exposures, responses)
    target_rows = _parent_strata(
        lineages,
        exposures,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    transform_rows = _transform_failure_rows(
        exposures,
        lineages,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    multiplicity_rows = _multiplicity_rows(exposures)
    stability_rows, arm_ids, stability_matrix = _lineage_stability_rows(exposures, lineages)

    output_paths = [
        out_dir / "budget_metrics.csv",
        out_dir / "search_funnel.csv",
        out_dir / "target_source_summary.csv",
        out_dir / "transform_failure_summary.csv",
        out_dir / "multiplicity_retractions.csv",
        out_dir / "lineage_stability.csv",
    ]
    for path, rows in zip(
        output_paths,
        (budget_rows, funnel_rows, target_rows, transform_rows, multiplicity_rows, stability_rows),
    ):
        write_csv_atomic(path, rows)
    figure_paths = [
        *_plot_budget(budget_rows, out_dir),
        *_plot_funnel(funnel_rows, args.reference_arm, out_dir),
        *_plot_stability(arm_ids, stability_matrix, out_dir),
    ]
    manifest_path = out_dir / "analysis_manifest.json"
    parameters = {
        "reference_arm": args.reference_arm,
        "expected_parent_count": args.expected_parent_count,
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
    }
    analysis_blocks = {
        "budget_row_count": len(budget_rows),
        "funnel_row_count": len(funnel_rows),
        "target_source_row_count": len(target_rows),
        "transform_failure_row_count": len(transform_rows),
        "multiplicity_row_count": len(multiplicity_rows),
        "lineage_stability_row_count": len(stability_rows),
    }
    merged = merge_analysis_manifest(
        manifest_path,
        section_name="sweep_analysis",
        section_payload={"parameters": parameters, "analysis_blocks": analysis_blocks},
        inputs=[*inputs, *[path for path in safety_paths if path.exists()]],
        outputs=[*output_paths, *figure_paths],
        restrictions=RESTRICTIONS,
    )
    # Preserve the original top-level fields for downstream readers.
    merged["parameters"] = parameters
    merged["analysis_blocks"] = analysis_blocks
    write_json_atomic(manifest_path, merged)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--out-dir", default="review-stage/claim-search-gpt55-paper-analysis-v1")
    parser.add_argument("--reference-arm", default="r3_c5")
    parser.add_argument("--expected-parent-count", type=int, default=215)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--safety-artifact", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    manifest = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", "analysis_blocks": manifest["analysis_blocks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
