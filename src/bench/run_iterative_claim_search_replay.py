"""Replay existing CONFIRM failures through iterative candidate generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bench.progress import iter_progress
from confirm.candidate_preflight import CandidatePreflightContext
from confirm.claim_search import (
    CLAIM_CANDIDATE_SYSTEM_PROMPT,
    LLMCandidateGenerationResponse,
    CandidateClaimProposal,
    ClaimSearchConfig,
    ClaimSearchState,
    run_claim_search,
    summarize_claim_search,
)
from confirm.contract import ClaimContract
from confirm.evidence_partitions import (
    EvidencePartitionManifest,
    load_evidence_manifest,
)
from confirm.llm import get_llm, make_llm
from confirm.provenance import claim_search_implementation_hashes, mapping_sha256
from confirm.self_refine import (
    SELF_REFINE_FEEDBACK_SYSTEM_PROMPT,
    SELF_REFINE_REFINEMENT_SYSTEM_PROMPT,
    SelfRefineCandidateGenerator,
    SelfRefineFeedback,
)

DEFAULT_DATA_ROOTS = (
    "data/prepared_data/evidence_partitions/benchmark_ready/cohorts",
    "data/prepared_data/evidence_partitions/cohorts",
)
_FILE_HASH_CACHE: dict[str, str | None] = {}


def _iter_initial_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in payload.get("models", []):
        model_spec = str(model.get("model_spec") or "")
        for row in model.get("initial_claims", []):
            if isinstance(row, dict):
                rows.append({"model_spec": model_spec, **row})
    return rows


def _contract_from_row(row: dict[str, Any]) -> ClaimContract | None:
    payload = row.get("drafted_contract")
    if not isinstance(payload, dict):
        gate_results = row.get("gate_results")
        if isinstance(gate_results, dict):
            payload = gate_results.get("contract")
    if not isinstance(payload, dict):
        return None
    return ClaimContract.model_validate(payload)


def _needs_search(row: dict[str, Any]) -> bool:
    return bool(
        not row.get("draft_success")
        or not row.get("gate_success")
        or not row.get("estimand_match", True)
        or str(row.get("gate_verdict_label")) != "confirmed"
    )


def _execute_candidate_contract(
    contract: ClaimContract,
    data_roots: list[Path],
) -> dict[str, Any]:
    from confirm.excluded_evidence import execute_contract

    return execute_contract(contract, data_roots, evidence_scope="current")


def _candidate_evaluator(
    data_roots: list[Path],
):
    def evaluator(candidate: CandidateClaimProposal) -> dict[str, Any]:
        return _execute_candidate_contract(candidate.proposed_contract, data_roots)

    return evaluator


def _configured_roots(values: list[str] | None, defaults: tuple[str, ...] = ()) -> list[Path]:
    return [Path(item) for item in (values if values is not None else list(defaults))]


def _log(message: str) -> None:
    print(message, flush=True)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resume_identities_compatible(left: Any, right: Any) -> bool:
    """Compare experiment identity while ignoring implementation-only provenance."""

    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    ignored = {"implementation_hashes_sha256"}
    return (
        {key: value for key, value in left.items() if key not in ignored}
        == {key: value for key, value in right.items() if key not in ignored}
    )


def _is_retryable_transient_generation_failure(state: ClaimSearchState) -> bool:
    if state.stopped_reason != "candidate_generation_failed" or state.candidate_history:
        return False
    responses = state.llm_candidate_responses
    if not responses:
        return False
    transient_markers = (
        "connection error",
        "timed out",
        "timeout",
        "rate limit",
        "service unavailable",
        "temporarily unavailable",
        "server disconnected",
    )
    return all(
        int(record.get("candidate_count") or 0) == 0
        and any(
            marker in str(record.get("parse_error") or "").lower()
            for marker in transient_markers
        )
        for record in responses
    )


def _file_sha256(path: Path) -> str | None:
    cache_key = str(path.resolve()) if path.exists() else str(path)
    if cache_key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[cache_key]
    if not path.exists() or not path.is_file():
        _FILE_HASH_CACHE[cache_key] = None
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"sha": sha, "dirty": bool(status), "dirty_entry_count": len(status)}
    except Exception as exc:  # noqa: BLE001
        return {"sha": None, "dirty": None, "error": str(exc)}


def _run_provenance(
    source: Path,
    manifest_path: Path | None,
    manifest: EvidencePartitionManifest | None,
    llm_model_spec: str,
    candidate_strategy: str,
) -> dict[str, Any]:
    if candidate_strategy == "self_refine":
        schema: dict[str, Any] = {
            "feedback": SelfRefineFeedback.model_json_schema(),
            "refinement": LLMCandidateGenerationResponse.model_json_schema(),
        }
        prompt_text = (
            SELF_REFINE_FEEDBACK_SYSTEM_PROMPT
            + "\n"
            + SELF_REFINE_REFINEMENT_SYSTEM_PROMPT
        )
    else:
        schema = LLMCandidateGenerationResponse.model_json_schema()
        prompt_text = CLAIM_CANDIDATE_SYSTEM_PROMPT
    repository_root = Path(__file__).resolve().parents[2]
    implementation_paths = [
        Path(__file__).resolve(),
        *sorted((repository_root / "src/confirm").glob("*.py")),
    ]
    implementation_hashes = {
        str(path.relative_to(repository_root)): _file_sha256(path)
        for path in implementation_paths
    }
    search_implementation_hashes = claim_search_implementation_hashes(repository_root)
    partition_hashes = {}
    partition_id_counts: Counter[str] = Counter()
    if manifest is not None:
        for record in manifest.records:
            partition_id_counts[record.partition_id] += 1
            partition_record = {
                "subject_id_sha256": record.subject_id_sha256,
                "schema_sha256": record.schema_sha256,
                "content_sha256": record.content_sha256 or _file_sha256(Path(record.path)),
                "manifest_record_sha256": _sha256_json(record.model_dump(mode="json")),
            }
            existing = partition_hashes.get(record.partition_id)
            if existing is not None:
                comparable_fields = ("subject_id_sha256", "schema_sha256", "content_sha256")
                if any(existing.get(field) != partition_record.get(field) for field in comparable_fields):
                    raise ValueError(
                        f"Evidence manifest has conflicting records for partition_id={record.partition_id!r}."
                    )
            else:
                partition_hashes[record.partition_id] = partition_record
    return {
        "command": list(sys.argv),
        "git": _git_state(),
        "python": sys.version,
        "llm_model": llm_model_spec,
        "candidate_strategy": candidate_strategy,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "schema_sha256": _sha256_json(schema),
        "implementation_hashes": implementation_hashes,
        "search_implementation_hashes": search_implementation_hashes,
        "search_implementation_hashes_sha256": mapping_sha256(search_implementation_hashes),
        "source": {"path": str(source), "sha256": _file_sha256(source)},
        "evidence_manifest": {
            "path": str(manifest_path) if manifest_path is not None else None,
            "sha256": _file_sha256(manifest_path) if manifest_path is not None else None,
        },
        "partition_hashes": partition_hashes,
        "partition_hashes_sha256": _sha256_json(partition_hashes),
        "partition_inventory": {
            "logical_record_count": len(manifest.records) if manifest is not None else 0,
            "unique_partition_id_count": len(partition_hashes),
            "duplicate_partition_id_counts": {
                partition_id: count
                for partition_id, count in sorted(partition_id_counts.items())
                if count > 1
            },
        },
        "random_seeds": {
            "manifest_seed": manifest.seed if manifest is not None else None,
            "partitions": (
                {record.partition_id: record.seed for record in manifest.records}
                if manifest is not None
                else {}
            ),
        },
    }


def _row_for_state(source_row: dict[str, Any], state: Any, candidate_generator_model_spec: str) -> dict[str, Any]:
    evaluations = state.evaluations
    valid = [item for item in evaluations if item.validation.ok]
    blocked = [item for item in evaluations if item.blocked_reason]
    execution_errors = [item for item in evaluations if item.execution_error]
    final_labels = [str(item.final_label) for item in evaluations if item.final_label]
    effective_final_labels = [str(item.final_label) for item in evaluations if item.final_label and not item.execution_error]
    same_data_exploratory = [
        item
        for item in evaluations
        if not item.execution_error
        and item.final_label == "exploratory_confirmed"
        and (item.validation_split == "current_data_adaptive" or item.same_underlying_data)
    ]
    source_model_spec = source_row.get("model_spec")
    return {
        "model_spec": source_model_spec,
        "source_model_spec": source_model_spec,
        "candidate_generator_model_spec": candidate_generator_model_spec,
        "claim_id": source_row.get("claim_id"),
        "target_family": source_row.get("target_family") or state.source_metadata.get("target_family"),
        "source_mode": source_row.get("source_mode") or state.source_metadata.get("source_mode"),
        "source_citation": source_row.get("source_citation") or state.source_metadata.get("source_citation"),
        "label_basis": source_row.get("label_basis") or state.source_metadata.get("label_basis"),
        "source_verdict": source_row.get("gate_verdict_label"),
        "source_scoring_label": source_row.get("source_scoring_label") or source_row.get("scoring_label") or source_row.get("label_class"),
        "source_label_authority": source_row.get("source_label_authority") or source_row.get("label_authority"),
        "source_result_path": source_row.get("source_result_path"),
        "synthetic_failure_family": (
            source_row.get("synthetic_failure_family")
            or source_row.get("failure_family")
            or source_row.get("negative_family")
            or state.source_metadata.get("synthetic_failure_family")
        ),
        "known_negative_or_fragile_source": _known_negative_or_fragile_source(source_row),
        "generated_candidate_count": state.generated_candidate_count,
        "proposals_returned_count": state.generated_candidate_count,
        "schema_valid_candidate_count": state.schema_valid_candidate_count,
        "candidate_count": len(state.candidate_history),
        "unique_candidate_count": state.unique_candidate_count,
        "duplicate_candidate_count": len(state.duplicate_candidates),
        "valid_candidate_count": len(valid),
        "current_data_evaluated_count": state.current_data_evaluated_count,
        "unique_source_tested_count": state.current_data_evaluated_count,
        "unique_hypotheses_tested_count": state.unique_hypotheses_tested_count,
        "execution_complete_candidate_count": sum(
            item.evaluated and not item.execution_error for item in evaluations
        ),
        "provisional_internal_pass_count": sum(
            item.provisional_supported and not item.execution_error for item in evaluations
        ),
        "final_multiplicity_adjusted_internal_pass_count": len(state.internally_supported_candidate_ids),
        "parent_with_internal_support": bool(state.internally_supported_candidate_ids),
        "multiplicity_retraction_count": sum(item.multiplicity_retracted for item in evaluations),
        "final_search_family_size": state.final_search_family_size,
        "blocked_candidate_count": len(blocked),
        "supported_candidate_count": len(state.internally_supported_candidate_ids),
        "confirmed_candidate_count": 0,
        "exploratory_confirmed_count": sum(
            1 for item in evaluations if not item.execution_error and item.final_label == "exploratory_confirmed"
        ),
        "same_data_exploratory_confirmed_count": len(same_data_exploratory),
        "final_confirmed_count": 0,
        "contract_repair_supported_count": sum(
            1 for item in evaluations if not item.execution_error and item.final_label == "contract_repair_supported"
        ),
        "holdout_confirmed_count": 0,
        "any_supported_candidate_count": sum(
            1
            for item in evaluations
            if not item.execution_error and item.current_data_supported
        ),
        "external_confirmed_count": 0,
        "same_underlying_data": any(item.same_underlying_data is True for item in evaluations),
        "excluded_evidence_used": False,
        "external_evidence_used": False,
        "excluded_evidence_query_count": 0,
        "excluded_evidence_status": "not_requested",
        "evidence_freshness": "not_applicable_source_search",
        "analysis_non_identifiable_count": sum(
            1
            for item in evaluations
            if item.blocked_reason == "analysis_non_identifiable"
            or str(item.execution_error or "").startswith("analysis_non_identifiable:")
        ),
        "excluded_evidence_unavailable_count": 0,
        "execution_error_count": len(execution_errors),
        "excluded_evidence_error_count": 0,
        "stopped_reason": state.stopped_reason,
        "transform_types": ";".join(candidate.transform_type for candidate in state.candidate_history),
        "declared_transforms": ";".join(
            str(candidate.declared_transform or candidate.transform_type) for candidate in state.candidate_history
        ),
        "inferred_transforms": ";".join(
            str(candidate.inferred_transform or "unknown") for candidate in state.candidate_history
        ),
        "transform_match_count": sum(candidate.transform_match is True for candidate in state.candidate_history),
        "policy_adjusted_candidate_count": sum(
            bool(candidate.policy_adjustments) for candidate in state.candidate_history
        ),
        "no_executable_change_count": sum(
            candidate.inferred_transform == "no_executable_change" for candidate in state.candidate_history
        ),
        "source_evidence_ledger": json.dumps(
            state.source_metadata.get("source_evidence_ledger", []), sort_keys=True
        ),
        "blocked_reasons": ";".join(str(item.blocked_reason) for item in blocked),
        "execution_errors": " | ".join(str(item.execution_error) for item in execution_errors),
        "excluded_evidence_errors": "",
        "final_labels": ";".join(final_labels),
        "effective_final_labels": ";".join(effective_final_labels),
    }


def _known_negative_or_fragile_source(source_row: dict[str, Any]) -> bool:
    claim_id = str(source_row.get("claim_id") or "")
    labels = {
        str(source_row.get("source_scoring_label") or "").lower(),
        str(source_row.get("scoring_label") or "").lower(),
        str(source_row.get("label_class") or "").lower(),
        str(source_row.get("ground_truth") or "").lower(),
        str(source_row.get("source_ground_truth") or "").lower(),
        str(source_row.get("source_label_class") or "").lower(),
    }
    negative_terms = {
        "known_null",
        "null_expected",
        "random_null",
        "fragile",
        "underpowered",
        "under_powered",
        "non_replicated",
    }
    return claim_id.startswith("neg_") or bool(labels & negative_terms)


def _source_evidence_ledger(
    source_row: dict[str, Any],
    manifest: EvidencePartitionManifest | None,
    preflight_context: CandidatePreflightContext | None,
) -> list[dict[str, Any]]:
    gate_results = source_row.get("gate_results")
    data_paths = gate_results.get("data_paths") if isinstance(gate_results, dict) else None
    records_by_path = {
        str(Path(record.path).resolve()): record
        for record in (manifest.records if manifest is not None else [])
    }
    records_by_id = {
        record.partition_id: record
        for record in (manifest.records if manifest is not None else [])
    }
    contract = _contract_from_row(source_row)
    cohort_entries: list[tuple[str, str]] = []
    if contract is not None:
        cohort_entries.append(("discovery", contract.discovery_cohort))
        cohort_entries.extend(("replication", cohort) for cohort in contract.replication_cohorts)

    authoritative_discovery: str | None = None
    authoritative_replication: list[str] = []
    if isinstance(data_paths, dict):
        discovery = data_paths.get("discovery")
        if discovery:
            authoritative_discovery = str(discovery)
        replication = data_paths.get("replication")
        if isinstance(replication, (list, tuple)):
            authoritative_replication = [str(path) for path in replication]
        elif replication:
            authoritative_replication = [str(replication)]

    entries: list[tuple[str, str, str]] = []
    replication_index = 0
    for role, cohort in cohort_entries:
        if role == "discovery":
            raw_path = authoritative_discovery
        else:
            raw_path = (
                authoritative_replication[replication_index]
                if replication_index < len(authoritative_replication)
                else None
            )
            replication_index += 1
        if raw_path is None and preflight_context is not None:
            info = preflight_context.resolve(cohort)
            raw_path = info.path if info is not None else None
        if raw_path is None:
            record = records_by_id.get(cohort)
            raw_path = record.path if record is not None else None
        if raw_path is not None:
            entries.append((role, cohort, raw_path))

    ledger: list[dict[str, Any]] = []
    for role, cohort, raw_path in entries:
        path = Path(raw_path)
        resolved = str(path.resolve()) if path.exists() else raw_path
        record = records_by_path.get(resolved) or records_by_id.get(cohort)
        content_hash = (
            record.content_sha256
            if record is not None and record.content_sha256
            else _file_sha256(path)
        )
        partition_hash = content_hash or (record.subject_id_sha256 if record is not None else None)
        ledger.append(
            {
                "role": role,
                "cohort": cohort,
                "path": raw_path,
                "resolved_path": resolved,
                "partition_id": record.partition_id if record is not None else path.stem,
                "partition_hash": partition_hash,
                "partition_hash_kind": (
                    "content_sha256"
                    if content_hash
                    else "subject_id_sha256"
                    if partition_hash
                    else None
                ),
                "holdout_named_source": "_HOLDOUT" in cohort or "_HOLDOUT" in path.stem,
            }
        )
    return ledger


def _source_metadata(
    source_row: dict[str, Any],
    manifest: EvidencePartitionManifest | None = None,
    preflight_context: CandidatePreflightContext | None = None,
) -> dict[str, Any]:
    keys = (
        "claim_id",
        "target_family",
        "source_mode",
        "source_result_path",
        "source_scoring_label",
        "source_label_authority",
        "source_label_class",
        "source_ground_truth",
        "ground_truth",
        "scoring_label",
        "label_class",
        "synthetic_failure_family",
        "failure_family",
        "negative_family",
        "source_citation",
        "label_basis",
        "model_spec",
    )
    metadata = {key: source_row.get(key) for key in keys if source_row.get(key) is not None}
    metadata["source_evidence_ledger"] = _source_evidence_ledger(
        source_row,
        manifest,
        preflight_context,
    )
    return metadata


def _replay_specific_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    negative_rows = [row for row in rows if row.get("known_negative_or_fragile_source")]
    negative_exploratory = sum(int(row.get("exploratory_confirmed_count") or 0) for row in negative_rows)
    negative_same_data_exploratory = sum(
        int(row.get("same_data_exploratory_confirmed_count") or 0) for row in negative_rows
    )
    negative_repair = sum(bool(row.get("contract_repair_supported_count")) for row in negative_rows)
    negative_holdout = sum(bool(row.get("holdout_confirmed_count")) for row in negative_rows)
    negative_external = sum(bool(row.get("external_confirmed_count")) for row in negative_rows)
    negative_any_support = sum(bool(row.get("any_supported_candidate_count")) for row in negative_rows)
    negative_non_identifiable = sum(bool(row.get("analysis_non_identifiable_count")) for row in negative_rows)
    negative_unavailable = sum(bool(row.get("excluded_evidence_unavailable_count")) for row in negative_rows)
    negative_execution_errors = sum(bool(row.get("execution_error_count")) for row in negative_rows)
    return {
        "known_negative_or_fragile_search_count": len(negative_rows),
        "known_negative_or_fragile_exploratory_confirmed_count": negative_exploratory,
        "known_negative_same_data_exploratory_confirmed_count": negative_same_data_exploratory,
        "known_negative_exploratory_risk_rate": (
            negative_same_data_exploratory / len(negative_rows) if negative_rows else 0.0
        ),
        "known_negative_contract_repair_supported_search_count": negative_repair,
        "known_negative_holdout_confirmed_search_count": negative_holdout,
        "known_negative_external_confirmed_search_count": negative_external,
        "known_negative_any_supported_search_count": negative_any_support,
        "known_negative_any_support_risk_rate": (
            negative_any_support / len(negative_rows) if negative_rows else 0.0
        ),
        "known_negative_analysis_non_identifiable_search_count": negative_non_identifiable,
        "known_negative_excluded_evidence_unavailable_search_count": negative_unavailable,
        "known_negative_execution_error_search_count": negative_execution_errors,
    }


def _llm_usage_summary(states: list[ClaimSearchState]) -> dict[str, Any]:
    records = [
        record
        for state in states
        for record in state.llm_candidate_responses
        if isinstance(record, dict)
    ]
    usage_records = []
    for record in records:
        metadata = record.get("call_metadata")
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
        if isinstance(usage, dict) and usage:
            usage_records.append(usage)
    cost_records = [usage for usage in usage_records if usage.get("cost") is not None]
    return {
        "llm_call_count": len(records),
        "llm_calls_with_usage_count": len(usage_records),
        "llm_prompt_tokens": sum(
            int(usage.get("prompt_tokens") or 0) for usage in usage_records
        ),
        "llm_completion_tokens": sum(
            int(usage.get("completion_tokens") or 0) for usage in usage_records
        ),
        "llm_total_tokens": sum(
            int(usage.get("total_tokens") or 0) for usage in usage_records
        ),
        "llm_reported_cost": (
            sum(float(usage["cost"]) for usage in cost_records)
            if len(cost_records) == len(records)
            else None
        ),
        "llm_usage_complete": len(usage_records) == len(records),
        "llm_calls_with_cost_count": len(cost_records),
        "llm_cost_complete": len(cost_records) == len(records),
    }


def _run_single_claim(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    config: ClaimSearchConfig,
    data_roots: list[Path],
    candidate_evaluation: str,
    llm_model_spec: str,
    llm: Any,
    preflight_context: CandidatePreflightContext | None,
    evidence_manifest: EvidencePartitionManifest | None,
    candidate_strategy: str,
) -> dict[str, Any]:
    claim_id = str(row.get("claim_id"))
    try:
        contract = _contract_from_row(row)
        if contract is None or not isinstance(row.get("gate_verdict"), dict):
            raise ValueError("missing executable contract or gate verdict")
        results = row.get("gate_results") if isinstance(row.get("gate_results"), dict) else None
        evaluator = _candidate_evaluator(data_roots) if candidate_evaluation == "on" else None
        candidate_generator = (
            SelfRefineCandidateGenerator(
                llm,
                preflight_context=preflight_context,
            )
            if candidate_strategy == "self_refine"
            else None
        )
        state = run_claim_search(
            contract,
            row["gate_verdict"],
            results,
            config=config,
            candidate_generator=candidate_generator,
            llm=llm if candidate_generator is None else None,
            evaluator=evaluator,
            preflight_context=preflight_context,
        )
        state = state.model_copy(
            update={
                "source_metadata": _source_metadata(
                    row,
                    evidence_manifest,
                    preflight_context,
                )
            }
        )
    except Exception as exc:
        return {
            "index": index,
            "total": total,
            "claim_id": claim_id,
            "status": "skipped",
            "skip": {"claim_id": claim_id, "reason": str(exc)},
            "message": f"[claim {index}/{total}] skipped claim_id={claim_id} reason={exc}",
        }

    output_row = _row_for_state(row, state, llm_model_spec)
    return {
        "index": index,
        "total": total,
        "claim_id": claim_id,
        "status": "completed",
        "row": output_row,
        "state": state.model_dump(mode="json"),
        "message": (
            f"[claim {index}/{total}] done claim_id={claim_id} "
            f"candidates={output_row['candidate_count']} valid={output_row['valid_candidate_count']} "
            f"exploratory_confirmed={output_row['exploratory_confirmed_count']} "
            f"stopped={output_row['stopped_reason']}"
        ),
    }


def _run_single_claim_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config = ClaimSearchConfig.model_validate(payload["config"])
    llm_spec = payload.get("llm_spec")
    llm = make_llm(llm_spec) if llm_spec else get_llm()
    llm_model_spec = llm_spec or getattr(llm, "model", type(llm).__name__)
    data_roots = [Path(item) for item in payload["data_roots"]]
    evidence_manifest = load_evidence_manifest(payload.get("evidence_manifest"))
    preflight_context = CandidatePreflightContext.from_roots(data_roots)
    return _run_single_claim(
        index=int(payload["index"]),
        total=int(payload["total"]),
        row=payload["row"],
        config=config,
        data_roots=data_roots,
        candidate_evaluation=str(payload["candidate_evaluation"]),
        llm_model_spec=llm_model_spec,
        llm=llm,
        preflight_context=preflight_context,
        evidence_manifest=evidence_manifest,
        candidate_strategy=str(payload.get("candidate_strategy") or "standard"),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    config = ClaimSearchConfig(
        max_rounds=args.max_rounds,
        max_candidates_per_round=args.max_candidates,
        llm_schema_retries=args.schema_retries,
        feedback_mode=args.feedback_mode,
    )
    data_roots = _configured_roots(getattr(args, "data_root", None), DEFAULT_DATA_ROOTS)
    candidate_evaluation = str(getattr(args, "candidate_evaluation", "on"))
    candidate_strategy = str(getattr(args, "candidate_strategy", "standard"))
    show_progress = not bool(getattr(args, "no_progress", False))
    evidence_manifest = load_evidence_manifest(getattr(args, "evidence_manifest", None))
    evidence_manifest_path = Path(args.evidence_manifest) if getattr(args, "evidence_manifest", None) else None
    preflight_context = CandidatePreflightContext.from_roots(data_roots)
    llm = make_llm(args.llm) if args.llm else get_llm()
    llm_model_spec = args.llm or getattr(llm, "model", type(llm).__name__)
    max_workers = max(1, int(getattr(args, "max_workers", 1) or 1))
    parallel_backend = str(getattr(args, "parallel_backend", "process"))
    active_parallel_backend = parallel_backend
    rows: list[dict[str, Any]] = []
    states: list[ClaimSearchState] = []
    skipped: list[dict[str, str]] = []
    rows_by_index: dict[int, dict[str, Any]] = {}
    states_by_index: dict[int, ClaimSearchState] = {}
    skipped_by_index: dict[int, dict[str, str]] = {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = _run_provenance(
        source,
        evidence_manifest_path,
        evidence_manifest,
        llm_model_spec,
        candidate_strategy,
    )
    initial_rows = [row for row in _iter_initial_rows(payload) if _needs_search(row)]
    if getattr(args, "limit", None) is not None:
        initial_rows = initial_rows[: max(0, int(args.limit))]
    claim_ids = [str(row.get("claim_id") or "") for row in initial_rows]
    duplicate_claim_ids = sorted(
        claim_id for claim_id in set(claim_ids) if claim_ids.count(claim_id) > 1
    )
    if "" in claim_ids or duplicate_claim_ids:
        raise ValueError(
            "Claim-search source must have nonempty unique claim IDs; "
            f"duplicates={duplicate_claim_ids}."
        )
    expected_parent_count = getattr(args, "expected_parent_count", None)
    if expected_parent_count is not None and len(initial_rows) != int(expected_parent_count):
        raise ValueError(
            f"Expected {expected_parent_count} failed parent claims, found {len(initial_rows)} in {source}."
        )
    json_path = out_dir / "iterative_candidate_replay.json"
    csv_path = out_dir / "iterative_candidate_replay.csv"
    provenance_path = out_dir / "run_provenance.json"
    ledger_path = out_dir / "excluded_query_ledger.json"
    parent_checkpoint_dir = out_dir / "checkpoints" / "parents"
    parent_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(0, int(getattr(args, "checkpoint_every", 0) or 0))
    resume_identity = {
        "source_sha256": provenance["source"]["sha256"],
        "config": config.model_dump(mode="json"),
        "llm_model": llm_model_spec,
        "prompt_sha256": provenance["prompt_sha256"],
        "schema_sha256": provenance["schema_sha256"],
        "evidence_manifest_sha256": provenance["evidence_manifest"]["sha256"],
        "partition_hashes_sha256": _sha256_json(provenance["partition_hashes"]),
        "data_roots": [str(root.resolve()) for root in data_roots],
        "candidate_evaluation": candidate_evaluation,
        "candidate_strategy": candidate_strategy,
    }
    provenance["resume_identity"] = resume_identity
    provenance["resume_identity_sha256"] = _sha256_json(resume_identity)
    accepted_resume_identity_sha256s = {provenance["resume_identity_sha256"]}
    compatible_resume_identities: dict[str, dict[str, Any]] = {}
    prior_provenance: dict[str, Any] = {}
    if provenance_path.exists():
        prior_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        prior_identity = prior_provenance.get("resume_identity")
        if _resume_identities_compatible(prior_identity, resume_identity):
            prior_hash = prior_provenance.get("resume_identity_sha256")
            if isinstance(prior_hash, str) and prior_hash:
                accepted_resume_identity_sha256s.add(prior_hash)
                if prior_hash != provenance["resume_identity_sha256"]:
                    compatible_resume_identities[prior_hash] = {
                        "resume_identity_sha256": prior_hash,
                        "resume_identity": prior_identity,
                        "implementation_hashes": prior_provenance.get(
                            "implementation_hashes", {}
                        ),
                    }
            for compatible_hash in prior_provenance.get(
                "compatible_resume_identity_sha256s", []
            ):
                if isinstance(compatible_hash, str) and compatible_hash:
                    accepted_resume_identity_sha256s.add(compatible_hash)
            for record in prior_provenance.get("compatible_resume_identities", []):
                if not isinstance(record, dict):
                    continue
                compatible_hash = record.get("resume_identity_sha256")
                if isinstance(compatible_hash, str) and compatible_hash:
                    accepted_resume_identity_sha256s.add(compatible_hash)
                    compatible_resume_identities[compatible_hash] = record
    provenance["compatible_resume_identity_sha256s"] = sorted(
        accepted_resume_identity_sha256s - {provenance["resume_identity_sha256"]}
    )
    provenance["compatible_resume_identities"] = [
        compatible_resume_identities[key]
        for key in sorted(compatible_resume_identities)
        if key != provenance["resume_identity_sha256"]
    ]
    superseded_transient_failures = list(
        prior_provenance.get("superseded_transient_generation_failures") or []
    )
    superseded_attempt_keys = {
        (int(record.get("index") or 0), str(record.get("attempt_records_sha256")))
        for record in superseded_transient_failures
        if isinstance(record, dict) and record.get("attempt_records_sha256")
    }
    retry_transient_indices: set[int] = set()

    def record_transient_failure(index: int, state: ClaimSearchState) -> None:
        attempt_hash = _sha256_json(state.llm_candidate_responses)
        attempt_key = (index, attempt_hash)
        if attempt_key in superseded_attempt_keys:
            return
        superseded_attempt_keys.add(attempt_key)
        superseded_transient_failures.append(
            {
                "index": index,
                "claim_id": state.original_claim.claim_id,
                "prompt_record_count": len(state.llm_candidate_prompts),
                "response_record_count": len(state.llm_candidate_responses),
                "parse_error_counts": dict(
                    Counter(
                        str(record.get("parse_error") or "unknown")
                        for record in state.llm_candidate_responses
                    )
                ),
                "attempt_records_sha256": attempt_hash,
            }
        )

    def parent_checkpoint_path(index: int) -> Path:
        return parent_checkpoint_dir / f"parent_{index:04d}.json"

    def write_parent_checkpoint(index: int, row: dict[str, Any], state: ClaimSearchState) -> None:
        body = {
            "resume_identity_sha256": provenance["resume_identity_sha256"],
            "index": index,
            "claim_id": state.original_claim.claim_id,
            "row": row,
            "state": state.model_dump(mode="json"),
        }
        payload = {**body, "checkpoint_sha256": _sha256_json(body)}
        _atomic_write_text(parent_checkpoint_path(index), json.dumps(payload, indent=2))

    def restore_parent_checkpoint(path: Path) -> tuple[int, dict[str, Any], ClaimSearchState]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        checksum = payload.pop("checkpoint_sha256", None)
        if checksum != _sha256_json(payload):
            raise ValueError(f"Parent checkpoint hash is invalid: {path}")
        if payload.get("resume_identity_sha256") not in accepted_resume_identity_sha256s:
            raise ValueError(
                f"Parent checkpoint provenance does not match this run: {path}"
            )
        index = int(payload["index"])
        if index < 1 or index > len(initial_rows):
            raise ValueError(f"Parent checkpoint index is outside the source: {path}")
        state = ClaimSearchState.model_validate(payload["state"])
        expected_claim_id = str(initial_rows[index - 1].get("claim_id"))
        if state.original_claim.claim_id != expected_claim_id or payload.get("claim_id") != expected_claim_id:
            raise ValueError(f"Parent checkpoint claim ID does not match the source: {path}")
        row = payload.get("row")
        if not isinstance(row, dict) or str(row.get("claim_id")) != expected_claim_id:
            raise ValueError(f"Parent checkpoint row is invalid: {path}")
        return index, row, state

    def refresh_lists() -> None:
        nonlocal rows, states, skipped
        rows = [rows_by_index[index] for index in sorted(rows_by_index)]
        states = [states_by_index[index] for index in sorted(states_by_index)]
        skipped = [skipped_by_index[index] for index in sorted(skipped_by_index)]

    def write_current(status: str) -> dict[str, Any]:
        refresh_lists()
        rendered_prompts = [
            prompt
            for state in states
            for prompt in state.llm_candidate_prompts
        ]
        provenance["rendered_prompt_records_sha256"] = _sha256_json(rendered_prompts)
        provenance["rendered_prompt_record_count"] = len(rendered_prompts)
        provenance["superseded_transient_generation_failures"] = sorted(
            superseded_transient_failures,
            key=lambda record: (int(record.get("index") or 0), str(record.get("attempt_records_sha256") or "")),
        )
        provenance["superseded_transient_prompt_record_count"] = sum(
            int(record.get("prompt_record_count") or 0)
            for record in superseded_transient_failures
        )
        provenance["total_prompt_attempt_record_count"] = (
            len(rendered_prompts)
            + int(provenance["superseded_transient_prompt_record_count"])
        )
        summary = {
            **summarize_claim_search(states),
            **_replay_specific_summary(rows),
            **_llm_usage_summary(states),
        }
        summary.update(
            {
                "searchable_claim_count": len(initial_rows),
                "completed_search_count": len(states),
                "skipped_search_count": len(skipped),
                "valid_connected_executable_lineage_coverage": (
                    int(summary.get("valid_connected_lineage_count") or 0) / len(initial_rows)
                    if initial_rows
                    else 0.0
                ),
            }
        )
        result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": "LLM-driven iterative candidate replay.",
            "status": status,
            "input": str(source),
            "llm_model": llm_model_spec,
            "max_workers": max_workers,
            "parallel_backend": active_parallel_backend,
            "config": config.model_dump(mode="json"),
            "candidate_evaluation": candidate_evaluation,
            "candidate_strategy": candidate_strategy,
            "data_roots": [str(root) for root in data_roots],
            "evidence_manifest": str(args.evidence_manifest) if getattr(args, "evidence_manifest", None) else None,
            "provenance": provenance,
            "searchable_claim_count": len(initial_rows),
            "completed_search_count": len(states),
            "skipped_search_count": len(skipped),
            "summary": summary,
            "skipped": skipped,
            "rows": rows,
            "states": [state.model_dump(mode="json") for state in states],
        }
        ledger: list[dict[str, Any]] = []
        if ledger:
            raise AssertionError("Adaptive search attempted excluded-evidence queries.")
        _atomic_write_text(json_path, json.dumps(result, indent=2))
        _atomic_write_csv(csv_path, rows)
        _atomic_write_text(provenance_path, json.dumps(provenance, indent=2))
        _atomic_write_text(ledger_path, json.dumps(ledger, indent=2))
        _log(f"[checkpoint] status={status} completed={len(rows)} skipped={len(skipped)} -> {json_path}")
        return result

    if getattr(args, "resume", "on") == "on":
        for checkpoint_path in sorted(parent_checkpoint_dir.glob("parent_*.json")):
            index, _, state = restore_parent_checkpoint(checkpoint_path)
            if index in states_by_index:
                raise ValueError(f"Duplicate parent checkpoint index: {index}")
            if (
                getattr(args, "retry_transient_generation_failures", "on") == "on"
                and _is_retryable_transient_generation_failure(state)
            ):
                record_transient_failure(index, state)
                retry_transient_indices.add(index)
                _log(
                    f"[resume] retrying transient candidate-generation failure "
                    f"index={index} claim_id={state.original_claim.claim_id}"
                )
                continue
            source_row = initial_rows[index - 1]
            state = state.model_copy(
                update={
                    "source_metadata": _source_metadata(
                        source_row,
                        evidence_manifest,
                        preflight_context,
                    )
                }
            )
            row = _row_for_state(source_row, state, llm_model_spec)
            states_by_index[index] = state
            rows_by_index[index] = row
        if states_by_index:
            _log(
                f"[resume] restored {len(states_by_index)} atomic parent checkpoints "
                f"from {parent_checkpoint_dir}"
            )

    if getattr(args, "resume", "on") == "on" and json_path.exists() and not states_by_index:
        prior = json.loads(json_path.read_text(encoding="utf-8"))
        prior_identity = (prior.get("provenance") or {}).get("resume_identity_sha256")
        if prior_identity not in accepted_resume_identity_sha256s:
            raise ValueError(
                "Existing checkpoint provenance does not match source/config/model/prompt hashes; "
                "use a new output directory or --resume off."
            )
        index_by_claim_id = {
            str(row.get("claim_id")): index
            for index, row in enumerate(initial_rows, start=1)
        }
        prior_rows = {
            str(row.get("claim_id")): row
            for row in prior.get("rows", [])
            if isinstance(row, dict)
        }
        for state_payload in prior.get("states", []):
            state = ClaimSearchState.model_validate(state_payload)
            claim_id = state.original_claim.claim_id
            index = index_by_claim_id.get(claim_id)
            if index is None or prior_rows.get(claim_id) is None:
                raise ValueError(f"Checkpoint contains unknown completed claim {claim_id!r}.")
            if index in retry_transient_indices:
                continue
            if (
                getattr(args, "retry_transient_generation_failures", "on") == "on"
                and _is_retryable_transient_generation_failure(state)
            ):
                record_transient_failure(index, state)
                retry_transient_indices.add(index)
                _log(
                    f"[resume] retrying transient candidate-generation failure "
                    f"index={index} claim_id={state.original_claim.claim_id}"
                )
                continue
            source_row = initial_rows[index - 1]
            state = state.model_copy(
                update={
                    "source_metadata": _source_metadata(
                        source_row,
                        evidence_manifest,
                        preflight_context,
                    )
                }
            )
            row = _row_for_state(source_row, state, llm_model_spec)
            states_by_index[index] = state
            rows_by_index[index] = row
        if states_by_index:
            _log(f"[resume] restored {len(states_by_index)} completed parent states from {json_path}")

    _log(
        "[start] iterative claim search "
        f"input={source} out_dir={out_dir} llm={llm_model_spec} "
        f"searchable_claims={len(initial_rows)} max_rounds={config.max_rounds} "
        f"max_candidates={config.max_candidates_per_round} candidate_evaluation={candidate_evaluation} "
        f"candidate_strategy={candidate_strategy} "
        f"max_workers={max_workers} parallel_backend={parallel_backend}"
    )

    def record_result(result: dict[str, Any]) -> None:
        index = int(result["index"])
        if result["status"] == "completed":
            rows_by_index[index] = result["row"]
            states_by_index[index] = ClaimSearchState.model_validate(result["state"])
            write_parent_checkpoint(
                index,
                rows_by_index[index],
                states_by_index[index],
            )
        else:
            skipped_by_index[index] = result["skip"]
        _log(str(result.get("message") or ""))
        if checkpoint_every and (len(rows_by_index) + len(skipped_by_index)) % checkpoint_every == 0:
            write_current("running")

    if max_workers == 1:
        indexed_rows = [
            (index, row)
            for index, row in enumerate(initial_rows, start=1)
            if index not in states_by_index
        ]
        for index, row in iter_progress(
            indexed_rows,
            total=len(indexed_rows),
            desc="claim-search",
            enabled=show_progress,
            unit="claim",
        ):
            claim_id = str(row.get("claim_id"))
            _log(f"[claim {index}/{len(initial_rows)}] start claim_id={claim_id} source_label={row.get('gate_verdict_label')}")
            record_result(
                _run_single_claim(
                    index=index,
                    total=len(initial_rows),
                    row=row,
                    config=config,
                    data_roots=data_roots,
                    candidate_evaluation=candidate_evaluation,
                    llm_model_spec=llm_model_spec,
                    llm=llm,
                    preflight_context=preflight_context,
                    evidence_manifest=evidence_manifest,
                    candidate_strategy=candidate_strategy,
                )
            )
    else:
        worker_count = min(max_workers, len(initial_rows)) if initial_rows else 1
        tasks = [
            {
                "index": index,
                "total": len(initial_rows),
                "row": row,
                "config": config.model_dump(mode="json"),
                "data_roots": [str(root) for root in data_roots],
                "candidate_evaluation": candidate_evaluation,
                "llm_spec": args.llm,
                "evidence_manifest": str(args.evidence_manifest) if getattr(args, "evidence_manifest", None) else None,
                "candidate_strategy": candidate_strategy,
            }
            for index, row in enumerate(initial_rows, start=1)
            if index not in states_by_index
        ]
        for task in tasks:
            _log(
                f"[claim {task['index']}/{task['total']}] queued "
                f"claim_id={task['row'].get('claim_id')} source_label={task['row'].get('gate_verdict_label')}"
            )

        def run_parallel(backend: str) -> None:
            nonlocal active_parallel_backend
            active_parallel_backend = backend
            executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
            _log(f"[workers] launching {backend} pool workers={worker_count}")
            with executor_cls(max_workers=worker_count) as pool:
                future_to_task = {pool.submit(_run_single_claim_worker, task): task for task in tasks}
                completed = as_completed(future_to_task)
                for future in iter_progress(
                    completed,
                    total=len(future_to_task),
                    desc=f"claim-search/{backend}",
                    enabled=show_progress,
                    unit="claim",
                ):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        index = int(task["index"])
                        claim_id = str(task["row"].get("claim_id"))
                        result = {
                            "index": index,
                            "total": int(task["total"]),
                            "claim_id": claim_id,
                            "status": "skipped",
                            "skip": {"claim_id": claim_id, "reason": f"worker failed: {exc}"},
                            "message": f"[claim {index}/{task['total']}] skipped claim_id={claim_id} reason=worker failed: {exc}",
                        }
                    record_result(result)

        try:
            run_parallel(parallel_backend)
        except (OSError, PermissionError) as exc:
            if parallel_backend != "process":
                raise
            _log(f"[workers] process pool unavailable ({exc}); falling back to thread pool")
            run_parallel("thread")

    refresh_lists()
    if len(states) != len(initial_rows) or skipped:
        write_current("incomplete")
        raise RuntimeError(
            f"Claim search incomplete: completed={len(states)} expected={len(initial_rows)} skipped={len(skipped)}."
        )
    result = write_current("completed")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llm", default=None, help="LLM spec such as openai:gpt-4o; defaults to CONFIRM_LLM")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument(
        "--feedback-mode",
        choices=["structured_diagnosis", "generic_retry"],
        default="structured_diagnosis",
    )
    parser.add_argument(
        "--candidate-strategy",
        choices=["standard", "self_refine"],
        default="standard",
        help="Candidate generator; self_refine adds a separate critique call before each round's refinement.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Rewrite aggregate artifacts every N parents; each completed parent is always checkpointed atomically.",
    )
    parser.add_argument("--resume", choices=["on", "off"], default="on")
    parser.add_argument(
        "--retry-transient-generation-failures",
        choices=["on", "off"],
        default="on",
        help=(
            "On resume, rerun only parents whose candidate generation produced no candidates "
            "because every recorded LLM attempt ended in a transient transport error."
        ),
    )
    parser.add_argument("--expected-parent-count", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for smoke tests.",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker processes for claim-level replay parallelism.")
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars/log progress updates.")
    parser.add_argument("--candidate-evaluation", choices=["on", "off"], default="on")
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="Repeatable root containing canonical cohort parquet files. Defaults to the active partitioned benchmark root.",
    )
    parser.add_argument(
        "--evidence-manifest",
        default=None,
        help="Optional partition manifest used only for source-cohort resolution, hashes, and provenance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
