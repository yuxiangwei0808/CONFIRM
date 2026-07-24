"""Build and summarize a blinded parent-relative candidate novelty review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from confirm.llm import LLMClient, make_llm
from nbs.claim_search_analysis_common import (
    iter_jsonl,
    merge_analysis_manifest,
    read_jsonl,
    sha256_file,
    sha256_json,
    write_csv_atomic,
    write_json_atomic,
)


BLINDED_COLUMNS = (
    "item_id",
    "pair_id",
    "target_family",
    "parent_question",
    "candidate_question",
    "parent_contract_summary",
    "candidate_contract_summary",
    "executable_change_summary",
)
LEAKAGE_TOKENS = ("arm", "model", "support", "p_value", "pvalue", "holdout", "external", "gate_result")
FORCED_CHOICE_POLICY_VERSION = "candidate-forced-choice-review-v2"
DEFAULT_REVIEW_MODELS = (
    "google:gemini-3.5-flash",
    "openrouter:anthropic/claude-opus-4.8",
    "openrouter:deepseek/deepseek-v4-pro",
)


class ForcedChoiceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    preferred_candidate: Literal["A", "B", "tie", "neither"]
    reason: str


class ForcedChoiceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[ForcedChoiceDecision] = Field(min_length=1, max_length=10)


def _analysis_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"claim_id", "question", "reporting_language_allowed"}
    }


def _semantic_contract(contract: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(_analysis_contract(contract)))
    provenance = value.get("search_provenance")
    if isinstance(provenance, dict):
        provenance.pop("family_size", None)
        provenance.pop("selection", None)
    gates = value.get("gates")
    if isinstance(gates, dict) and isinstance(gates.get("multiplicity"), dict):
        gates["multiplicity"].pop("family_size", None)
    return value


def _changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_changed_paths(left[key], right[key], path))
        return paths
    return [prefix or "$"] if left != right else []


def _outcome_set(contract: dict[str, Any]) -> set[str]:
    estimand = contract.get("estimand") or {}
    values: set[str] = set()
    outcome = estimand.get("outcome")
    if isinstance(outcome, str) and outcome:
        values.add(outcome)
    elif isinstance(outcome, list):
        values.update(str(item) for item in outcome)
    region_set = estimand.get("region_set")
    if isinstance(region_set, list):
        values.update(str(item) for item in region_set)
    elif isinstance(region_set, str) and region_set:
        values.add(region_set)
    return values


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _contract_summary(contract: dict[str, Any]) -> str:
    estimand = contract.get("estimand") or {}
    return json.dumps(
        {
            "estimand": estimand,
            "covariates": contract.get("covariates") or [],
            "inclusion": contract.get("inclusion"),
            "discovery_cohort": contract.get("discovery_cohort"),
            "replication_cohorts": contract.get("replication_cohorts") or [],
        },
        sort_keys=True,
    )


def novelty_metrics(
    exposures: Iterable[dict[str, Any]],
    *,
    exact_counts: Counter[str] | None = None,
    semantic_counts: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    retained: Iterable[dict[str, Any]] = (
        row
        for row in exposures
        if row.get("generation_status") == "retained"
        and isinstance(row.get("effective_contract"), dict)
    )
    if exact_counts is None or semantic_counts is None:
        retained = list(retained)
        exact_counts = Counter(
            str(row.get("exact_contract_id")) for row in retained if row.get("exact_contract_id")
        )
        semantic_counts = Counter(
            str(row.get("semantic_cluster_id")) for row in retained if row.get("semantic_cluster_id")
        )
    rows: list[dict[str, Any]] = []
    for exposure in retained:
        parent = exposure["parent_contract"]
        candidate = exposure["effective_contract"]
        paths = _changed_paths(_semantic_contract(parent), _semantic_contract(candidate))
        parent_outcomes = _outcome_set(parent)
        candidate_outcomes = _outcome_set(candidate)
        outcome_jaccard = _jaccard(parent_outcomes, candidate_outcomes)
        estimand_paths = [path for path in paths if path.startswith("estimand.")]
        inclusion_changed = "inclusion" in paths
        covariates_changed = any(path.startswith("covariates") for path in paths)
        outcome_changed = any(path.startswith("estimand.outcome") or path.startswith("estimand.region_set") for path in paths)
        unit_changed = "estimand.unit" in paths
        explicit_specification = any("specification" in path or "analysis_spec" in path for path in paths)
        if not paths:
            novelty_class = "no_op"
        elif outcome_changed or unit_changed or outcome_jaccard < 1.0:
            novelty_class = "materially_different_connected_hypothesis"
        else:
            novelty_class = "refinement"
        rows.append(
            {
                "exposure_id": exposure["exposure_id"],
                "arm_id": exposure["arm_id"],
                "parent_claim_id": exposure["parent_claim_id"],
                "candidate_id": exposure["candidate_id"],
                "target_family": exposure.get("target_family"),
                "source_mode": exposure.get("source_mode"),
                "round_index": exposure.get("round_index"),
                "inferred_transform": exposure.get("inferred_transform") or "unknown",
                "changed_executable_path_count": len(paths),
                "changed_executable_paths": json.dumps(paths),
                "outcome_set_jaccard": outcome_jaccard,
                "outcome_changed": int(outcome_changed),
                "scalar_to_brainwide_or_reverse": int(
                    (parent.get("estimand") or {}).get("unit") != (candidate.get("estimand") or {}).get("unit")
                ),
                "subgroup_or_inclusion_changed": int(inclusion_changed),
                "covariates_changed": int(covariates_changed),
                "estimand_changed": int(bool(estimand_paths)),
                "fixed_estimand_assessability": "assessable" if explicit_specification else "not_assessable",
                "novelty_class": novelty_class,
                "exact_contract_recurrence": exact_counts.get(str(exposure.get("exact_contract_id")), 0),
                "semantic_cluster_recurrence": semantic_counts.get(str(exposure.get("semantic_cluster_id")), 0),
                "final_internal_supported": int(bool(exposure.get("final_internal_supported"))),
                "current_data_evaluated": int(bool(exposure.get("current_data_evaluated"))),
            }
        )
    return rows


def novelty_metrics_from_path(path: Path) -> list[dict[str, Any]]:
    exact_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        if row.get("generation_status") != "retained" or not isinstance(row.get("effective_contract"), dict):
            continue
        if row.get("exact_contract_id"):
            exact_counts[str(row["exact_contract_id"])] += 1
        if row.get("semantic_cluster_id"):
            semantic_counts[str(row["semantic_cluster_id"])] += 1
    return novelty_metrics(
        iter_jsonl(path),
        exact_counts=exact_counts,
        semantic_counts=semantic_counts,
    )


def build_metrics(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = Path(args.frozen_dir) / "frozen_search_inventory.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = novelty_metrics_from_path(inventory_path)
    metrics_path = out_dir / "candidate_novelty_metrics.csv"
    write_csv_atomic(metrics_path, metrics)
    result = {
        "metric_row_count": len(metrics),
        "no_op_count": sum(row["novelty_class"] == "no_op" for row in metrics),
        "literature_novelty_claim_allowed": False,
    }
    merge_analysis_manifest(
        out_dir / "analysis_manifest.json",
        section_name="deterministic_parent_relative_novelty",
        section_payload=result,
        inputs=[inventory_path],
        outputs=[metrics_path],
        restrictions=("Novelty is parent-relative, not literature-wide.",),
    )
    return result


def _control_candidates(control_dir: Path) -> list[dict[str, Any]]:
    checkpoint_dir = control_dir / "generic_retry" / "checkpoints" / "parents"
    if not checkpoint_dir.exists():
        raise ValueError(f"Generic-control checkpoints are missing: {checkpoint_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("parent_*.json")):
        state = json.loads(path.read_text(encoding="utf-8")).get("state") or {}
        parent = state.get("original_claim") or {}
        metadata = state.get("source_metadata") or {}
        final_supported_ids = {
            str(item)
            for item in state.get("internally_supported_candidate_ids") or []
        }
        evaluations = {
            str(row.get("candidate_id")): row
            for row in state.get("evaluations") or []
            if isinstance(row, dict)
        }
        for proposal in state.get("candidate_history") or []:
            if not isinstance(proposal, dict):
                continue
            candidate_id = str(proposal.get("candidate_id") or "")
            evaluation = evaluations.get(candidate_id) or {}
            contract = proposal.get("proposed_contract")
            validation = evaluation.get("validation") or {}
            if not isinstance(contract, dict):
                continue
            rows.append(
                {
                    "arm_id": "generic_retry_r3_c5",
                    "parent_claim_id": parent.get("claim_id"),
                    "candidate_id": candidate_id,
                    "target_family": metadata.get("target_family") or "unknown",
                    "source_mode": metadata.get("source_mode") or "unknown",
                    "parent_contract": parent,
                    "effective_contract": contract,
                    "executable_contract_delta": proposal.get("executable_contract_delta") or {},
                    "semantic_cluster_id": sha256_json(_semantic_contract(contract)),
                    "validation_ok": validation.get("ok"),
                    "current_data_evaluated": bool(evaluation.get("evaluated")),
                    "final_internal_supported": candidate_id in final_supported_ids,
                    "inferred_transform": proposal.get("inferred_transform") or "unknown",
                }
            )
    return rows


def _eligible(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("validation_ok") is True and row.get("current_data_evaluated") and isinstance(row.get("effective_contract"), dict)
    ]


def _ordered(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row.get('parent_claim_id')}:{row.get('candidate_id')}".encode("utf-8")
        ).hexdigest(),
    )


def _blind_item(row: dict[str, Any], *, item_id: str, pair_id: str = "") -> dict[str, Any]:
    parent = row["parent_contract"]
    candidate = row["effective_contract"]
    delta = row.get("executable_contract_delta") or {
        path: True for path in _changed_paths(_semantic_contract(parent), _semantic_contract(candidate))
    }
    return {
        "item_id": item_id,
        "pair_id": pair_id,
        "target_family": row.get("target_family") or "unknown",
        "parent_question": parent.get("question") or "",
        "candidate_question": candidate.get("question") or "",
        "parent_contract_summary": _contract_summary(parent),
        "candidate_contract_summary": _contract_summary(candidate),
        "executable_change_summary": json.dumps(delta, sort_keys=True),
    }


def _assert_blinded(packet: list[dict[str, Any]]) -> None:
    if not packet or set(packet[0]) != set(BLINDED_COLUMNS):
        raise ValueError(f"Blinded packet has unexpected columns: {set(packet[0]) if packet else set()}")
    for column in packet[0]:
        lowered = column.lower()
        if any(token in lowered for token in LEAKAGE_TOKENS):
            raise ValueError(f"Blinded packet leaks outcome metadata through column {column!r}")
    if len({row["item_id"] for row in packet}) != len(packet):
        raise ValueError("Blinded packet contains duplicate item IDs.")


def _forced_choice_prompt(batch: list[dict[str, Any]]) -> tuple[str, str]:
    system = (
        "You are an independent reviewer comparing two follow-up neuroimaging claims generated "
        "from the same failed parent claim. Choose the better scientifically connected follow-up. "
        "Do not infer empirical support or reward a candidate for appearing easier to confirm. "
        "Return structured JSON only."
    )
    user = json.dumps(
        {
            "policy_version": FORCED_CHOICE_POLICY_VERSION,
            "instructions": [
                "Prefer the candidate that is more scientifically meaningful, remains connected "
                "to the parent question, introduces a genuine testable follow-up, and does not "
                "appear designed merely to obtain significance.",
                "Choose tie when A and B are comparably good.",
                "Choose neither when both are scientifically unsuitable follow-ups.",
                "Formal schema and data executability were checked deterministically and are not "
                "part of this comparison.",
                "Give one concise reason per pair.",
                "Return exactly one decision for every supplied pair_id and no other pair_id.",
            ],
            "pairs": batch,
        },
        indent=2,
        sort_keys=True,
    )
    return system, user


def _validate_forced_choice_batch(
    parsed: ForcedChoiceBatch,
    expected_pair_ids: set[str],
) -> None:
    observed = [decision.pair_id for decision in parsed.decisions]
    if len(observed) != len(set(observed)):
        raise ValueError("Forced-choice response contains duplicate pair IDs.")
    if set(observed) != expected_pair_ids:
        raise ValueError(
            "Forced-choice response has the wrong pair IDs: "
            f"missing={sorted(expected_pair_ids - set(observed))} "
            f"extra={sorted(set(observed) - expected_pair_ids)}"
        )


def _call_forced_choice_batch(
    llm: LLMClient,
    *,
    system: str,
    prompt: str,
    expected_pair_ids: set[str],
    retries: int,
) -> tuple[ForcedChoiceBatch, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    active_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        raw = ""
        try:
            complete_structured = getattr(llm, "complete_structured", None)
            if not callable(complete_structured):
                raise TypeError(f"Reviewer model {llm.model!r} lacks structured output support.")
            raw = str(complete_structured(system, active_prompt, ForcedChoiceBatch))
            parsed = ForcedChoiceBatch.model_validate_json(raw)
            _validate_forced_choice_batch(parsed, expected_pair_ids)
            attempts.append(
                {
                    "attempt": attempt,
                    "prompt_sha256": hashlib.sha256(active_prompt.encode()).hexdigest(),
                    "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "schema_valid": True,
                    "raw_response": raw,
                    "call_metadata": dict(getattr(llm, "last_call_metadata", {}) or {}),
                }
            )
            return parsed, attempts
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "attempt": attempt,
                    "prompt_sha256": hashlib.sha256(active_prompt.encode()).hexdigest(),
                    "response_sha256": hashlib.sha256(raw.encode()).hexdigest() if raw else None,
                    "schema_valid": False,
                    "raw_response": raw,
                    "error": str(exc),
                    "call_metadata": dict(getattr(llm, "last_call_metadata", {}) or {}),
                }
            )
            active_prompt = (
                f"{prompt}\n\nPrevious structured-output error: {exc}. "
                "Return a corrected response with exactly the requested pair IDs."
            )
    raise RuntimeError(
        f"Structured forced choice failed after {retries + 1} attempts: {last_error}"
    )


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    frozen = Path(args.frozen_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze_manifest_path = frozen / "freeze_manifest.json"
    control_summary_path = Path(args.control_dir) / "control_summary.json"
    if not freeze_manifest_path.exists() or not control_summary_path.exists():
        raise ValueError("Novelty review requires completed freeze_manifest.json and control_summary.json.")
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    control_summary = json.loads(control_summary_path.read_text(encoding="utf-8"))
    if control_summary.get("source_sha256") != freeze_manifest.get("source_payload_sha256"):
        raise ValueError("Novelty control and frozen sweep do not share the exact source hash.")
    budget = control_summary.get("budget") or {}
    if (int(budget.get("max_rounds") or 0), int(budget.get("max_candidates_per_round") or 0)) != (3, 5):
        raise ValueError("Novelty control must use the predeclared R3/C5 budget.")
    if int(control_summary.get("parent_count") or 0) != 215:
        raise ValueError("Novelty control must contain exactly 215 matched parents.")
    inventory_path = frozen / "frozen_search_inventory.jsonl"
    metrics = novelty_metrics_from_path(inventory_path)
    metrics_path = out_dir / "candidate_novelty_metrics.csv"
    write_csv_atomic(metrics_path, metrics)

    structured = _eligible(
        row for row in iter_jsonl(inventory_path)
        if row.get("arm_id") == "r3_c5" and row.get("generation_status") == "retained"
    )
    generic = _eligible(_control_candidates(Path(args.control_dir)))
    structured_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    generic_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in structured:
        structured_by_parent[str(row["parent_claim_id"])].append(row)
    for row in generic:
        generic_by_parent[str(row["parent_claim_id"])].append(row)
    common_parents = sorted(set(structured_by_parent) & set(generic_by_parent))
    ordered_parents = sorted(
        common_parents,
        key=lambda parent: hashlib.sha256(f"{args.seed}:{parent}".encode("utf-8")).hexdigest(),
    )

    packet: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    used_semantic: set[str] = set()
    item_index = 0

    def add(
        row: dict[str, Any],
        group: str,
        pair_id: str = "",
        *,
        semantic_reserved: bool = False,
    ) -> None:
        nonlocal item_index
        cluster = str(row.get("semantic_cluster_id") or "")
        if cluster in used_semantic and not semantic_reserved:
            raise ValueError(f"Review packet would repeat semantic cluster {cluster}.")
        used_semantic.add(cluster)
        item_index += 1
        item_id = f"item_{item_index:03d}"
        packet.append(_blind_item(row, item_id=item_id, pair_id=pair_id))
        key_rows.append(
            {
                "item_id": item_id,
                "review_group": group,
                "arm_id": row.get("arm_id"),
                "parent_claim_id": row.get("parent_claim_id"),
                "candidate_id": row.get("candidate_id"),
                "semantic_cluster_id": cluster,
                "final_internal_supported": int(bool(row.get("final_internal_supported"))),
                "inferred_transform": row.get("inferred_transform") or "unknown",
                "match_quality": row.get("_review_match_quality") or "not_applicable",
            }
        )

    control_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for parent in ordered_parents:
        structured_options = sorted(
            _ordered(structured_by_parent[parent], args.seed),
            key=lambda row: (bool(row.get("final_internal_supported")), str(row.get("candidate_id"))),
        )
        chosen: tuple[dict[str, Any], dict[str, Any]] | None = None
        for structured_row in structured_options:
            structured_cluster = str(structured_row.get("semantic_cluster_id") or "")
            if not structured_cluster or structured_cluster in used_semantic:
                continue
            for generic_row in _ordered(generic_by_parent[parent], args.seed):
                generic_cluster = str(generic_row.get("semantic_cluster_id") or "")
                if generic_cluster and generic_cluster not in used_semantic and generic_cluster != structured_cluster:
                    chosen = structured_row, generic_row
                    break
            if chosen:
                break
        if not chosen:
            continue
        pair_index = len(control_pairs) + 1
        add(chosen[0], "control_structured", f"pair_{pair_index:03d}")
        add(chosen[1], "control_generic", f"pair_{pair_index:03d}")
        control_pairs.append(chosen)
        if len(control_pairs) == 50:
            break
    if len(control_pairs) != 50:
        raise ValueError(f"Could construct only {len(control_pairs)} semantically unique control pairs; need 50.")

    if len(packet) != 100:
        raise ValueError(
            f"Forced-choice packet must contain 100 candidates, observed {len(packet)}."
        )
    _assert_blinded(packet)
    packet_path = out_dir / "forced_choice_candidate_packet.csv"
    key_path = out_dir / "forced_choice_candidate_key.jsonl"
    write_csv_atomic(packet_path, packet)
    _write_jsonl_atomic(key_path, key_rows)
    result = {
        "pair_count": len(control_pairs),
        "candidate_count": len(packet),
        "semantic_cluster_count": len(used_semantic),
        "comparison": "structured_diagnosis_vs_generic_retry",
        "forced_choice_options": ["A", "B", "tie", "neither"],
    }
    write_json_atomic(out_dir / "forced_choice_packet_manifest.json", result)
    manifest = merge_analysis_manifest(
        out_dir / "analysis_manifest.json",
        section_name="forced_choice_review_packet",
        section_payload=result,
        inputs=[
            frozen / "frozen_search_inventory.jsonl",
            freeze_manifest_path,
            control_summary_path,
            Path(args.control_dir) / "generic_retry" / "iterative_candidate_replay.json",
        ],
        outputs=[
            metrics_path,
            packet_path,
            key_path,
            out_dir / "forced_choice_packet_manifest.json",
        ],
        restrictions=(
            "The forced choice compares parent-relative follow-up quality, not literature-wide novelty.",
            "Reviewers are blinded to source arm, support, p-values, and excluded evidence.",
        ),
    )
    structured_arm = (control_summary.get("arms") or {}).get("structured_diagnosis") or {}
    generic_arm = (control_summary.get("arms") or {}).get("generic_retry") or {}
    structured_summary = structured_arm.get("summary") or {}
    generic_summary = generic_arm.get("summary") or {}
    manifest["matched_generic_retry_control"] = {
        "parent_count": int(control_summary.get("parent_count") or 0),
        "paired_parent_support_cells": control_summary.get("paired_parent_support_cells") or {},
        "structured_llm_call_count": int(structured_arm.get("llm_call_count") or 0),
        "generic_llm_call_count": int(generic_arm.get("llm_call_count") or 0),
        "structured_completed_trace_llm_call_count": int(
            structured_arm.get("completed_trace_llm_call_count")
            or structured_arm.get("llm_call_count")
            or 0
        ),
        "generic_completed_trace_llm_call_count": int(
            generic_arm.get("completed_trace_llm_call_count")
            or generic_arm.get("llm_call_count")
            or 0
        ),
        "structured_superseded_transient_attempt_count": int(
            structured_arm.get("superseded_transient_attempt_count") or 0
        ),
        "generic_superseded_transient_attempt_count": int(
            generic_arm.get("superseded_transient_attempt_count") or 0
        ),
        "structured_final_candidate_support_count": int(
            structured_summary.get("final_multiplicity_adjusted_internal_pass_count") or 0
        ),
        "generic_final_candidate_support_count": int(
            generic_summary.get("final_multiplicity_adjusted_internal_pass_count") or 0
        ),
        "structured_supported_parent_count": int(
            structured_summary.get("parents_with_internal_support_count") or 0
        ),
        "generic_supported_parent_count": int(
            generic_summary.get("parents_with_internal_support_count") or 0
        ),
        "causal_interpretation_allowed": False,
        "implementation_compatibility_status": control_summary.get("implementation_compatibility_status"),
    }
    audit = dict(manifest.get("analysis_audit") or {})
    audit.update(
        {
            "generic_retry_control_complete": True,
            "forced_choice_review_complete": False,
            "status": "passed_with_legacy_provenance_warning",
        }
    )
    manifest["analysis_audit"] = audit
    manifest["interpretation_restrictions"] = [
        item
        for item in manifest.get("interpretation_restrictions") or []
        if item != "Generic-retry comparison is pending."
    ]
    write_json_atomic(out_dir / "analysis_manifest.json", manifest)
    return result


def _choice_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row["candidate_question"],
        "contract": json.loads(row["candidate_contract_summary"]),
        "executable_change": json.loads(row["executable_change_summary"]),
    }


def run_forced_choice_review(
    args: argparse.Namespace,
    *,
    clients: dict[str, LLMClient] | None = None,
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    packet_path = out_dir / "forced_choice_candidate_packet.csv"
    key_path = out_dir / "forced_choice_candidate_key.jsonl"
    with packet_path.open(newline="", encoding="utf-8") as handle:
        packet = list(csv.DictReader(handle))
    _assert_blinded(packet)
    key_by_item = {row["item_id"]: row for row in read_jsonl(key_path)}
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in packet:
        group = str(key_by_item[row["item_id"]]["review_group"])
        if group not in {"control_structured", "control_generic"}:
            raise ValueError(f"Unexpected forced-choice group: {group}")
        pairs[row["pair_id"]][group] = row
    if len(pairs) != 50 or any(
        set(pair) != {"control_structured", "control_generic"}
        for pair in pairs.values()
    ):
        raise ValueError("Forced-choice review requires exactly 50 complete matched pairs.")
    models = tuple(args.reviewer_model or DEFAULT_REVIEW_MODELS)
    if len(models) != 3 or len(set(models)) != 3:
        raise ValueError("Forced-choice review requires exactly three distinct reviewer models.")
    if any("gpt-5.5" in model.lower() for model in models):
        raise ValueError("GPT-5.5 generated the candidates and cannot be a model reviewer.")
    packet_sha256 = sha256_file(packet_path)
    key_sha256 = sha256_file(key_path)
    policy_sha256 = hashlib.sha256(FORCED_CHOICE_POLICY_VERSION.encode()).hexdigest()
    decision_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    checkpoint_root = out_dir / "forced_choice_checkpoints"
    max_output_tokens = int(getattr(args, "max_output_tokens", 8192))
    pair_order = sorted(
        pairs,
        key=lambda pair_id: hashlib.sha256(
            f"{FORCED_CHOICE_POLICY_VERSION}:assignment:{pair_id}".encode()
        ).hexdigest(),
    )
    assignment_patterns = (
        [
            ("A", "A", "B"),
            ("A", "B", "A"),
            ("B", "A", "A"),
            ("A", "B", "B"),
            ("B", "A", "B"),
            ("B", "B", "A"),
        ]
        * 8
        + [("A", "A", "B"), ("B", "B", "A")]
    )
    structured_label_by_model = {
        model: {
            pair_id: assignment_patterns[pair_index][model_index]
            for pair_index, pair_id in enumerate(pair_order)
        }
        for model_index, model in enumerate(models)
    }
    for model in models:
        llm = (clients or {}).get(model) if clients is not None else None
        if llm is None:
            llm = make_llm(model)
        if hasattr(llm, "max_tokens"):
            llm.max_tokens = max_output_tokens
        presented: list[dict[str, Any]] = []
        assignment_by_pair: dict[str, dict[str, str]] = {}
        for pair_id, pair in pairs.items():
            structured_label = structured_label_by_model[model][pair_id]
            generic_label = "B" if structured_label == "A" else "A"
            assignment_by_pair[pair_id] = {
                structured_label: "structured",
                generic_label: "generic",
            }
            structured = pair["control_structured"]
            generic = pair["control_generic"]
            candidate_by_source = {
                "structured": _choice_candidate(structured),
                "generic": _choice_candidate(generic),
            }
            presented.append(
                {
                    "pair_id": pair_id,
                    "target_family": structured["target_family"],
                    "parent": {
                        "question": structured["parent_question"],
                        "contract": json.loads(structured["parent_contract_summary"]),
                    },
                    "candidate_A": candidate_by_source[
                        assignment_by_pair[pair_id]["A"]
                    ],
                    "candidate_B": candidate_by_source[
                        assignment_by_pair[pair_id]["B"]
                    ],
                }
            )
            assignment_rows.append(
                {
                    "model_spec": model,
                    "pair_id": pair_id,
                    "structured_label": structured_label,
                    "generic_label": generic_label,
                }
            )
        ordered = sorted(
            presented,
            key=lambda row: hashlib.sha256(
                f"{FORCED_CHOICE_POLICY_VERSION}:order:{model}:{row['pair_id']}".encode()
            ).hexdigest(),
        )
        batches = [
            ordered[index : index + args.batch_size]
            for index in range(0, len(ordered), args.batch_size)
        ]
        model_dir = checkpoint_root / _safe_model_name(model)
        for batch_index, batch in enumerate(batches):
            system, prompt = _forced_choice_prompt(batch)
            expected_ids = {row["pair_id"] for row in batch}
            prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            fingerprint = sha256_json(
                {
                    "policy_version": FORCED_CHOICE_POLICY_VERSION,
                    "policy_sha256": policy_sha256,
                    "packet_sha256": packet_sha256,
                    "key_sha256": key_sha256,
                    "model": model,
                    "batch_index": batch_index,
                    "pair_ids": sorted(expected_ids),
                    "prompt_sha256": prompt_sha256,
                    "max_output_tokens": max_output_tokens,
                }
            )
            checkpoint_path = model_dir / f"batch_{batch_index:03d}.json"
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if checkpoint.get("fingerprint") != fingerprint:
                    raise ValueError(
                        f"Forced-choice checkpoint fingerprint mismatch: {checkpoint_path}"
                    )
                parsed = ForcedChoiceBatch.model_validate(
                    {"decisions": checkpoint.get("decisions") or []}
                )
                _validate_forced_choice_batch(parsed, expected_ids)
                attempts = list(checkpoint.get("attempts") or [])
                reused = True
            else:
                parsed, attempts = _call_forced_choice_batch(
                    llm,
                    system=system,
                    prompt=prompt,
                    expected_pair_ids=expected_ids,
                    retries=args.schema_retries,
                )
                write_json_atomic(
                    checkpoint_path,
                    {
                        "fingerprint": fingerprint,
                        "model_spec": model,
                        "policy_version": FORCED_CHOICE_POLICY_VERSION,
                        "packet_sha256": packet_sha256,
                        "key_sha256": key_sha256,
                        "batch_index": batch_index,
                        "pair_ids": sorted(expected_ids),
                        "decisions": [
                            decision.model_dump(mode="json")
                            for decision in parsed.decisions
                        ],
                        "attempts": attempts,
                    },
                )
                reused = False
            successful_attempt = next(
                (attempt for attempt in reversed(attempts) if attempt.get("schema_valid")),
                {},
            )
            call_metadata = dict(successful_attempt.get("call_metadata") or {})
            for decision in parsed.decisions:
                source_choice = (
                    assignment_by_pair[decision.pair_id][decision.preferred_candidate]
                    if decision.preferred_candidate in {"A", "B"}
                    else decision.preferred_candidate
                )
                decision_rows.append(
                    {
                        "reviewer_id": model,
                        "pair_id": decision.pair_id,
                        "preferred_candidate": decision.preferred_candidate,
                        "preferred_source": source_choice,
                        "reason": decision.reason,
                        "provider": call_metadata.get("provider"),
                        "routed_provider": call_metadata.get("routed_provider"),
                        "response_model": call_metadata.get("model"),
                    }
                )
            prompt_rows.append(
                {
                    "model_spec": model,
                    "batch_index": batch_index,
                    "fingerprint": fingerprint,
                    "pair_ids": sorted(expected_ids),
                    "system": system,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "reused_checkpoint": reused,
                }
            )
            response_rows.append(
                {
                    "model_spec": model,
                    "batch_index": batch_index,
                    "fingerprint": fingerprint,
                    "pair_ids": sorted(expected_ids),
                    "attempts": attempts,
                    "reused_checkpoint": reused,
                }
            )
            print(
                f"forced-choice model={model} batch={batch_index + 1}/{len(batches)} "
                f"pairs={len(batch)} reused={str(reused).lower()}",
                flush=True,
            )
    decision_rows.sort(key=lambda row: (str(row["reviewer_id"]), str(row["pair_id"])))
    decisions_path = out_dir / "forced_choice_decisions.csv"
    prompts_path = out_dir / "forced_choice_prompts.jsonl"
    responses_path = out_dir / "forced_choice_responses.jsonl"
    assignment_path = out_dir / "forced_choice_assignment_key.jsonl"
    write_csv_atomic(decisions_path, decision_rows)
    _write_jsonl_atomic(prompts_path, prompt_rows)
    _write_jsonl_atomic(responses_path, response_rows)
    _write_jsonl_atomic(assignment_path, assignment_rows)
    summary_rows: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in decision_rows if row["reviewer_id"] == model]
        counts = Counter(str(row["preferred_source"]) for row in model_rows)
        summary_rows.append(
            {
                "record_type": "reviewer",
                "reviewer_id": model,
                "pair_count": len(model_rows),
                **{f"{choice}_count": counts[choice] for choice in (
                    "structured", "generic", "tie", "neither"
                )},
                **{f"{choice}_rate": counts[choice] / len(model_rows) for choice in (
                    "structured", "generic", "tie", "neither"
                )},
            }
        )
    majority_rows: list[dict[str, Any]] = []
    for pair_id in sorted(pairs):
        rows = [row for row in decision_rows if row["pair_id"] == pair_id]
        counts = Counter(str(row["preferred_source"]) for row in rows)
        majority = next(
            (
                choice
                for choice in ("structured", "generic", "tie", "neither")
                if counts[choice] >= 2
            ),
            "no_majority",
        )
        majority_rows.append(
            {
                "pair_id": pair_id,
                "majority_preference": majority,
                "structured_vote_count": counts["structured"],
                "generic_vote_count": counts["generic"],
                "tie_vote_count": counts["tie"],
                "neither_vote_count": counts["neither"],
                "unanimous": int(max(counts.values(), default=0) == 3),
            }
        )
    majority_counts = Counter(row["majority_preference"] for row in majority_rows)
    summary_rows.append(
        {
            "record_type": "majority",
            "reviewer_id": "two_of_three",
            "pair_count": len(majority_rows),
            **{f"{choice}_count": majority_counts[choice] for choice in (
                "structured", "generic", "tie", "neither", "no_majority"
            )},
            **{f"{choice}_rate": majority_counts[choice] / len(majority_rows) for choice in (
                "structured", "generic", "tie", "neither", "no_majority"
            )},
            "unanimous_count": sum(row["unanimous"] for row in majority_rows),
        }
    )
    summary_path = out_dir / "forced_choice_summary.csv"
    majority_path = out_dir / "forced_choice_majority.csv"
    write_csv_atomic(summary_path, summary_rows)
    write_csv_atomic(majority_path, majority_rows)
    attempts = [
        attempt
        for response in response_rows
        for attempt in response.get("attempts") or []
    ]
    execution = {
        "policy_version": FORCED_CHOICE_POLICY_VERSION,
        "reviewer_models": list(models),
        "pair_count": len(pairs),
        "decision_count": len(decision_rows),
        "batch_size": args.batch_size,
        "max_output_tokens": max_output_tokens,
        "batch_count": len(response_rows),
        "attempt_count": len(attempts),
        "schema_valid_attempt_count": sum(
            bool(attempt.get("schema_valid")) for attempt in attempts
        ),
        "checkpoint_reuse_count": sum(
            bool(response.get("reused_checkpoint")) for response in response_rows
        ),
        "generator_model_excluded": True,
        "majority_preference_counts": dict(sorted(majority_counts.items())),
        "unanimous_pair_count": sum(row["unanimous"] for row in majority_rows),
        "interpretation": "descriptive_blinded_forced_choice",
        "causal_interpretation_allowed": False,
        "literature_novelty_claim_allowed": False,
    }
    manifest_path = out_dir / "forced_choice_manifest.json"
    write_json_atomic(manifest_path, execution)
    manifest = merge_analysis_manifest(
        out_dir / "analysis_manifest.json",
        section_name="blinded_forced_choice_review",
        section_payload=execution,
        inputs=[packet_path, key_path],
        outputs=[
            decisions_path,
            prompts_path,
            responses_path,
            assignment_path,
            summary_path,
            majority_path,
            manifest_path,
        ],
        restrictions=(
            "Forced-choice model reviews are not human expert review.",
            "Reviewers never receive arm, support, p-values, gate outcomes, or excluded-evidence outcomes.",
            "The comparison is parent-relative follow-up quality, not literature-wide novelty.",
            "Majority preference is descriptive and is not a causal effect estimate.",
        ),
    )
    audit = dict(manifest.get("analysis_audit") or {})
    audit["forced_choice_review_complete"] = True
    manifest["analysis_audit"] = audit
    write_json_atomic(out_dir / "analysis_manifest.json", manifest)
    return execution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["metrics", "build", "forced-choice"],
        default="build",
    )
    parser.add_argument("--frozen-dir")
    parser.add_argument("--control-dir")
    parser.add_argument("--out-dir", default="review-stage/claim-search-gpt55-paper-analysis-v1")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--reviewer-model", action="append")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--schema-retries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "metrics":
        if not args.frozen_dir:
            raise ValueError("--frozen-dir is required for deterministic novelty metrics.")
        result = build_metrics(args)
    elif args.phase == "build":
        if not args.frozen_dir or not args.control_dir:
            raise ValueError("--frozen-dir and --control-dir are required for packet building.")
        result = build_packet(args)
    elif args.phase == "forced-choice":
        result = run_forced_choice_review(args)
    else:
        raise ValueError(f"Unsupported phase: {args.phase}")
    print(json.dumps({"status": "completed", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
