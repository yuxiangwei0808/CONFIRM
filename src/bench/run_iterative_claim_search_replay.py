"""Replay existing CONFIRM failures through iterative candidate generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
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
from confirm.agent import _jsonable, _load_canonical, _run_brainwide_contract, _run_scalar_contract
from confirm.contract import ClaimContract
from confirm.evidence_partitions import (
    EvidencePartitionManifest,
    canonical_base_cohort,
    infer_target_family,
    load_evidence_manifest,
)
from confirm.llm import get_llm, make_llm

DEFAULT_DATA_ROOTS = (
    "data/prepared_data/evidence_partitions/benchmark_ready/cohorts",
)


class ExcludedEvidenceUnavailableError(FileNotFoundError):
    """Excluded evidence could not be mapped before any gate was evaluated."""

    code = "excluded_evidence_unavailable"


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


def _cohort_path(data_roots: list[Path], cohort: str, *, allow_aliases: bool = True) -> Path:
    candidates = _cohort_aliases(cohort) if allow_aliases else [cohort]
    for candidate in candidates:
        for root in data_roots:
            path = root / f"{candidate}.parquet"
            if path.exists():
                return path
    roots = ", ".join(str(root) for root in data_roots)
    raise FileNotFoundError(f"Cohort {cohort!r} was not found in data roots: {roots}")


def _cohort_aliases(cohort: str) -> list[str]:
    aliases = [cohort]
    suffixes = (
        "_DISC_SITES",
        "_REP_SITES",
        "_HOLDOUT_SITES",
        "_HOLDOUT_DISC",
        "_HOLDOUT_REP",
        "_EXTERNAL_DISC",
        "_EXTERNAL_REP",
        "_HOLDOUT",
        "_DISC",
        "_REP",
        "_CN",
    )
    for suffix in suffixes:
        if cohort.endswith(suffix):
            aliases.append(cohort[: -len(suffix)])
    for split in ("_DISC_s", "_REP_s", "_HOLDOUT_s"):
        if split in cohort:
            aliases.append(cohort.split(split, 1)[0])
    return list(dict.fromkeys(item for item in aliases if item))


def _execute_candidate_contract(
    contract: ClaimContract,
    data_roots: list[Path],
    *,
    evidence_scope: str = "current",
    target_family: str | None = None,
    source_contract: ClaimContract | None = None,
    evidence_set_id: str | None = None,
) -> dict[str, Any]:
    allow_aliases = evidence_scope == "current"
    discovery_path = _cohort_path(data_roots, contract.discovery_cohort, allow_aliases=allow_aliases)
    replication_paths = [
        _cohort_path(data_roots, cohort, allow_aliases=allow_aliases)
        for cohort in contract.replication_cohorts
    ]
    discovery_df = _load_canonical(discovery_path)
    replication_dfs = [_load_canonical(path) for path in replication_paths]
    if contract.estimand.unit == "brainwide":
        verdict, results = _run_brainwide_contract(contract, discovery_df, replication_dfs)
    else:
        verdict, results = _run_scalar_contract(
            contract,
            discovery_df,
            replication_dfs,
            ref_effect=contract.gates.power.ref_effect,
        )
    return {
        "final_label": verdict.label,
        "gate_results": _jsonable(
            {
                "contract": contract.model_dump(mode="json"),
                "data_paths": {
                    "discovery": str(discovery_path),
                    "replication": [str(path) for path in replication_paths],
                },
                "evidence_scope": {
                    "scope": evidence_scope,
                    "target_family": target_family,
                    "evidence_set_id": evidence_set_id,
                    "source_contract": source_contract.model_dump(mode="json") if source_contract is not None else None,
                },
                **results,
            }
        ),
    }


def _candidate_evaluator(
    data_roots: list[Path],
    *,
    evidence_manifest: EvidencePartitionManifest | None = None,
    evidence_kind: str = "current",
):
    preflight_context = (
        CandidatePreflightContext.from_roots(data_roots)
        if evidence_manifest is not None and evidence_kind in {"holdout", "external"}
        else None
    )

    def evaluator(candidate: CandidateClaimProposal) -> dict[str, Any]:
        source_contract = candidate.proposed_contract
        target_family = infer_target_family(source_contract) if evidence_manifest is not None else None
        if evidence_manifest is None or evidence_kind not in {"holdout", "external"}:
            return _execute_candidate_contract(
                source_contract,
                data_roots,
                evidence_scope=evidence_kind,
                target_family=target_family,
            )

        errors: list[str] = []
        evidence_kinds = ["external", "holdout"] if evidence_kind == "external" else ["holdout"]
        for candidate_evidence_kind in evidence_kinds:
            if candidate_evidence_kind == "external":
                try:
                    return _evaluate_external_sets(
                        source_contract,
                        evidence_manifest,
                        data_roots,
                        preflight_context,
                        target_family,
                    )
                except ExcludedEvidenceUnavailableError as exc:
                    errors.append(f"external: {exc}")
                    continue
            try:
                mapped_contract = _mapped_contract_for_excluded_evidence(
                    source_contract,
                    evidence_manifest,
                    candidate_evidence_kind,
                )
                _require_exact_contract_paths(mapped_contract, data_roots)
                preflight = preflight_context.validate_contract(mapped_contract) if preflight_context is not None else None
                if preflight is not None and not preflight.ok:
                    raise ValueError("; ".join(preflight.violations))
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"{candidate_evidence_kind}: {exc}")
                continue
            # Do not catch execution failures here. Fallback is permitted only
            # after schema/preflight failure and before observing gate outcomes.
            return _execute_candidate_contract(
                mapped_contract,
                data_roots,
                evidence_scope=candidate_evidence_kind,
                target_family=target_family,
                source_contract=source_contract,
            )
        raise ExcludedEvidenceUnavailableError("; ".join(errors))

    return evaluator


def _evaluate_external_sets(
    source_contract: ClaimContract,
    manifest: EvidencePartitionManifest,
    data_roots: list[Path],
    preflight_context: CandidatePreflightContext | None,
    target_family: str | None,
) -> dict[str, Any]:
    compatible = manifest.external_sets_for_contract(source_contract)
    primary_sets = [item for item in compatible if item.confirmation_role == "primary"]
    errors: list[str] = []
    selected = None
    mapped_contract = None
    for evidence_set in primary_sets:
        try:
            candidate_contract = _mapped_contract_for_excluded_evidence(
                source_contract,
                manifest,
                "external",
                evidence_set_id=evidence_set.evidence_set_id,
            )
            _require_exact_contract_paths(candidate_contract, data_roots)
            preflight = preflight_context.validate_contract(candidate_contract) if preflight_context is not None else None
            if preflight is not None and not preflight.ok:
                raise ValueError("; ".join(preflight.violations))
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{evidence_set.evidence_set_id}: {exc}")
            continue
        selected = evidence_set
        mapped_contract = candidate_contract
        break
    if selected is None or mapped_contract is None:
        detail = "; ".join(errors) if errors else "no schema-compatible primary set"
        raise ExcludedEvidenceUnavailableError(detail)

    result = _execute_candidate_contract(
        mapped_contract,
        data_roots,
        evidence_scope="external",
        target_family=target_family,
        source_contract=source_contract,
    )
    result.setdefault("gate_results", {}).setdefault("evidence_scope", {})["evidence_set_id"] = selected.evidence_set_id
    secondary_results: list[dict[str, Any]] = []
    for evidence_set in compatible:
        if evidence_set.confirmation_role != "secondary":
            continue
        try:
            secondary_contract = _mapped_contract_for_excluded_evidence(
                source_contract,
                manifest,
                "external",
                evidence_set_id=evidence_set.evidence_set_id,
            )
            _require_exact_contract_paths(secondary_contract, data_roots)
            preflight = preflight_context.validate_contract(secondary_contract) if preflight_context is not None else None
            if preflight is not None and not preflight.ok:
                raise ValueError("; ".join(preflight.violations))
            secondary = _execute_candidate_contract(
                secondary_contract,
                data_roots,
                evidence_scope="external_robustness",
                target_family=target_family,
                source_contract=source_contract,
            )
            secondary.setdefault("gate_results", {}).setdefault("evidence_scope", {})[
                "evidence_set_id"
            ] = evidence_set.evidence_set_id
            secondary_results.append(
                {
                    "evidence_set_id": evidence_set.evidence_set_id,
                    "status": "evaluated",
                    "final_label": secondary.get("final_label"),
                    "gate_results": secondary.get("gate_results"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            secondary_results.append(
                {
                    "evidence_set_id": evidence_set.evidence_set_id,
                    "status": "unavailable",
                    "error": str(exc),
                }
            )
    result.setdefault("gate_results", {})["secondary_external_evaluations"] = secondary_results
    return result


def _mapped_contract_for_excluded_evidence(
    contract: ClaimContract,
    manifest: EvidencePartitionManifest,
    evidence_kind: str,
    *,
    evidence_set_id: str | None = None,
) -> ClaimContract:
    data = contract.model_dump(mode="json")
    target_family = infer_target_family(contract)
    if evidence_kind == "holdout":
        pair = manifest.holdout_evaluation_pair_for_contract(contract)
        if pair is None:
            bases = [canonical_base_cohort(contract.discovery_cohort), *[canonical_base_cohort(cohort) for cohort in contract.replication_cohorts]]
            raise FileNotFoundError(f"No holdout evaluation pair for bases={bases!r} target_family={target_family!r}")
        discovery_record, replication_records = pair
        data["discovery_cohort"] = discovery_record.partition_id
        data["replication_cohorts"] = [record.partition_id for record in replication_records]
        return ClaimContract.model_validate(data)
    if evidence_kind == "external":
        pair = manifest.external_pair_for_contract(contract, evidence_set_id=evidence_set_id)
        if pair is None:
            raise FileNotFoundError(
                f"No contract-compatible external evaluation pair for target_family={target_family!r}"
            )
        discovery_record, replication_records, _ = pair
        data["discovery_cohort"] = discovery_record.partition_id
        data["replication_cohorts"] = [record.partition_id for record in replication_records]
        return ClaimContract.model_validate(data)
    return contract


def _require_exact_contract_paths(contract: ClaimContract, data_roots: list[Path]) -> None:
    for cohort in [contract.discovery_cohort, *contract.replication_cohorts]:
        _cohort_path(data_roots, cohort, allow_aliases=False)


def _configured_roots(values: list[str] | None, defaults: tuple[str, ...] = ()) -> list[Path]:
    return [Path(item) for item in (values if values is not None else list(defaults))]


def _log(message: str) -> None:
    print(message, flush=True)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    evidence_freshness: str,
) -> dict[str, Any]:
    schema = LLMCandidateGenerationResponse.model_json_schema()
    partition_hashes = {}
    if manifest is not None:
        for record in manifest.records:
            partition_hashes[record.partition_id] = {
                "subject_id_sha256": record.subject_id_sha256,
                "schema_sha256": record.schema_sha256,
                "content_sha256": record.content_sha256 or _file_sha256(Path(record.path)),
                "manifest_record_sha256": _sha256_json(record.model_dump(mode="json")),
            }
    return {
        "command": list(sys.argv),
        "git": _git_state(),
        "python": sys.version,
        "llm_model": llm_model_spec,
        "evidence_freshness": evidence_freshness,
        "prompt_sha256": hashlib.sha256(CLAIM_CANDIDATE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema_sha256": _sha256_json(schema),
        "source": {"path": str(source), "sha256": _file_sha256(source)},
        "evidence_manifest": {
            "path": str(manifest_path) if manifest_path is not None else None,
            "sha256": _file_sha256(manifest_path) if manifest_path is not None else None,
        },
        "partition_hashes": partition_hashes,
        "random_seeds": {
            "manifest_seed": manifest.seed if manifest is not None else None,
            "partitions": (
                {record.partition_id: record.seed for record in manifest.records}
                if manifest is not None
                else {}
            ),
        },
    }


def _excluded_query_ledger(states: list[ClaimSearchState]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for state in states:
        if state.excluded_evidence_query_count <= 0:
            continue
        selected = next(
            (item for item in state.evaluations if item.candidate_id == state.selected_candidate_id),
            None,
        )
        selected_gate_results = None
        if selected is not None:
            selected_gate_results = selected.external_gate_results or selected.holdout_gate_results
        evidence_scope = (
            selected_gate_results.get("evidence_scope")
            if isinstance(selected_gate_results, dict)
            and isinstance(selected_gate_results.get("evidence_scope"), dict)
            else {}
        )
        secondary = (
            selected_gate_results.get("secondary_external_evaluations", [])
            if isinstance(selected_gate_results, dict)
            else []
        )
        ledger.append(
            {
                "claim_id": state.original_claim.claim_id,
                "target_family": state.source_metadata.get("target_family"),
                "source_mode": state.source_metadata.get("source_mode"),
                "selected_candidate_id": state.selected_candidate_id,
                "selection_reason": state.selection_reason,
                "query_count": state.excluded_evidence_query_count,
                "status": state.excluded_evidence_status,
                "evidence_kind": selected.excluded_evidence_kind if selected is not None else None,
                "primary_evidence_set_id": evidence_scope.get("evidence_set_id"),
                "evidence_freshness": state.evidence_freshness,
                "discovery_path": selected.resolved_excluded_discovery_path if selected is not None else None,
                "replication_paths": selected.resolved_excluded_replication_paths if selected is not None else [],
                "secondary_external_queries": [
                    {
                        "evidence_set_id": item.get("evidence_set_id"),
                        "status": item.get("status"),
                        "final_label": item.get("final_label"),
                    }
                    for item in secondary
                    if isinstance(item, dict)
                ],
            }
        )
    return ledger


def _row_for_state(source_row: dict[str, Any], state: Any, candidate_generator_model_spec: str) -> dict[str, Any]:
    evaluations = state.evaluations
    valid = [item for item in evaluations if item.validation.ok]
    blocked = [item for item in evaluations if item.blocked_reason]
    execution_errors = [item for item in evaluations if item.execution_error]
    excluded_evidence_errors = [item for item in evaluations if item.excluded_evidence_error]
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
        "candidate_count": len(state.candidate_history),
        "unique_candidate_count": state.unique_candidate_count,
        "duplicate_candidate_count": len(state.duplicate_candidates),
        "valid_candidate_count": len(valid),
        "current_data_evaluated_count": state.current_data_evaluated_count,
        "blocked_candidate_count": len(blocked),
        "supported_candidate_count": len(state.supported_candidates),
        "confirmed_candidate_count": len(state.confirmed_candidates),
        "selected_candidate_id": state.selected_candidate_id,
        "selection_reason": state.selection_reason,
        "exploratory_confirmed_count": sum(
            1 for item in evaluations if not item.execution_error and item.final_label == "exploratory_confirmed"
        ),
        "same_data_exploratory_confirmed_count": len(same_data_exploratory),
        "final_confirmed_count": sum(
            1
            for item in evaluations
            if not item.execution_error and item.final_label in {"holdout_confirmed", "external_confirmed"}
        ),
        "contract_repair_supported_count": sum(
            1 for item in evaluations if not item.execution_error and item.final_label == "contract_repair_supported"
        ),
        "holdout_confirmed_count": sum(1 for item in evaluations if not item.execution_error and item.holdout_confirmed),
        "any_supported_candidate_count": sum(
            1
            for item in evaluations
            if not item.execution_error
            and (
                item.current_data_supported
                or item.final_label in {"holdout_confirmed", "external_confirmed"}
                or item.external_confirmed
                or item.holdout_confirmed
            )
        ),
        "external_confirmed_count": sum(1 for item in evaluations if not item.execution_error and item.external_confirmed),
        "same_underlying_data": any(item.same_underlying_data is True for item in evaluations),
        "excluded_evidence_used": any(item.excluded_evidence_used for item in evaluations),
        "external_evidence_used": any(item.external_evidence_used for item in evaluations),
        "excluded_evidence_query_count": state.excluded_evidence_query_count,
        "excluded_evidence_status": state.excluded_evidence_status,
        "evidence_freshness": state.evidence_freshness,
        "analysis_non_identifiable_count": sum(
            1
            for item in evaluations
            if item.blocked_reason == "analysis_non_identifiable"
            or str(item.execution_error or "").startswith("analysis_non_identifiable:")
            or "analysis_non_identifiable" in str(item.excluded_evidence_error or "")
        ),
        "excluded_evidence_unavailable_count": sum(
            1 for item in evaluations if item.excluded_evidence_status == "unavailable"
        ),
        "execution_error_count": len(execution_errors),
        "excluded_evidence_error_count": len(excluded_evidence_errors),
        "stopped_reason": state.stopped_reason,
        "transform_types": ";".join(candidate.transform_type for candidate in state.candidate_history),
        "blocked_reasons": ";".join(str(item.blocked_reason) for item in blocked),
        "execution_errors": " | ".join(str(item.execution_error) for item in execution_errors),
        "excluded_evidence_errors": " | ".join(
            str(item.excluded_evidence_error) for item in excluded_evidence_errors
        ),
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


def _source_metadata(source_row: dict[str, Any]) -> dict[str, Any]:
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
    return {key: source_row.get(key) for key in keys if source_row.get(key) is not None}


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


def _run_single_claim(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    config: ClaimSearchConfig,
    data_roots: list[Path],
    holdout_data_roots: list[Path],
    external_data_roots: list[Path],
    candidate_evaluation: str,
    llm_model_spec: str,
    llm: Any,
    preflight_context: CandidatePreflightContext | None,
    evidence_manifest: EvidencePartitionManifest | None,
    evidence_freshness: str,
) -> dict[str, Any]:
    claim_id = str(row.get("claim_id"))
    try:
        contract = _contract_from_row(row)
        if contract is None or not isinstance(row.get("gate_verdict"), dict):
            raise ValueError("missing executable contract or gate verdict")
        results = row.get("gate_results") if isinstance(row.get("gate_results"), dict) else None
        evaluator = _candidate_evaluator(data_roots) if candidate_evaluation == "on" else None
        excluded_data_roots = list(dict.fromkeys([*external_data_roots, *holdout_data_roots]))
        excluded_kind = "external" if evidence_manifest is not None else (
            "external" if external_data_roots else "holdout"
        )
        external_evaluator = None
        if candidate_evaluation == "on" and excluded_data_roots and evidence_manifest is not None:
            external_evaluator = _candidate_evaluator(
                excluded_data_roots,
                evidence_manifest=evidence_manifest,
                evidence_kind=excluded_kind,
            )
        validation_catalog = evidence_manifest.validation_catalog_for_contract(contract) if evidence_manifest is not None else None
        state = run_claim_search(
            contract,
            row["gate_verdict"],
            results,
            config=config,
            llm=llm,
            evaluator=evaluator,
            external_evaluator=external_evaluator,
            excluded_evidence_kind=excluded_kind,
            excluded_validation_available=candidate_evaluation == "on" and external_evaluator is not None,
            preflight_context=preflight_context,
            validation_evidence_catalog=validation_catalog,
            evidence_freshness=evidence_freshness,
        )
        state = state.model_copy(update={"source_metadata": _source_metadata(row)})
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
    holdout_data_roots = [Path(item) for item in payload.get("holdout_data_roots", [])]
    external_data_roots = [Path(item) for item in payload["external_data_roots"]]
    evidence_manifest = load_evidence_manifest(payload.get("evidence_manifest"))
    preflight_context = CandidatePreflightContext.from_roots([*data_roots, *holdout_data_roots, *external_data_roots])
    return _run_single_claim(
        index=int(payload["index"]),
        total=int(payload["total"]),
        row=payload["row"],
        config=config,
        data_roots=data_roots,
        holdout_data_roots=holdout_data_roots,
        external_data_roots=external_data_roots,
        candidate_evaluation=str(payload["candidate_evaluation"]),
        llm_model_spec=llm_model_spec,
        llm=llm,
        preflight_context=preflight_context,
        evidence_manifest=evidence_manifest,
        evidence_freshness=str(payload.get("evidence_freshness", "unknown")),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    config = ClaimSearchConfig(
        max_rounds=args.max_rounds,
        max_candidates_per_round=args.max_candidates,
        llm_schema_retries=args.schema_retries,
    )
    data_roots = _configured_roots(getattr(args, "data_root", None), DEFAULT_DATA_ROOTS)
    holdout_data_roots = _configured_roots(getattr(args, "holdout_data_root", None))
    external_data_roots = _configured_roots(getattr(args, "external_data_root", None))
    candidate_evaluation = str(getattr(args, "candidate_evaluation", "on"))
    show_progress = not bool(getattr(args, "no_progress", False))
    evidence_manifest = load_evidence_manifest(getattr(args, "evidence_manifest", None))
    evidence_manifest_path = Path(args.evidence_manifest) if getattr(args, "evidence_manifest", None) else None
    evidence_freshness = str(getattr(args, "evidence_freshness", "unknown"))
    preflight_context = CandidatePreflightContext.from_roots([*data_roots, *holdout_data_roots, *external_data_roots])
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
        evidence_freshness,
    )

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
        summary = {**summarize_claim_search(states), **_replay_specific_summary(rows)}
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
            "data_roots": [str(root) for root in data_roots],
            "holdout_data_roots": [str(root) for root in holdout_data_roots],
            "external_data_roots": [str(root) for root in external_data_roots],
            "evidence_manifest": str(args.evidence_manifest) if getattr(args, "evidence_manifest", None) else None,
            "evidence_freshness": evidence_freshness,
            "provenance": provenance,
            "searchable_claim_count": len(initial_rows),
            "completed_search_count": len(states),
            "skipped_search_count": len(skipped),
            "summary": summary,
            "skipped": skipped,
            "rows": rows,
            "states": [state.model_dump(mode="json") for state in states],
        }
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        ledger_path.write_text(json.dumps(_excluded_query_ledger(states), indent=2), encoding="utf-8")
        _log(f"[checkpoint] status={status} completed={len(rows)} skipped={len(skipped)} -> {json_path}")
        return result

    json_path = out_dir / "iterative_candidate_replay.json"
    csv_path = out_dir / "iterative_candidate_replay.csv"
    provenance_path = out_dir / "run_provenance.json"
    ledger_path = out_dir / "excluded_query_ledger.json"
    checkpoint_every = max(0, int(getattr(args, "checkpoint_every", 0) or 0))
    initial_rows = [row for row in _iter_initial_rows(payload) if _needs_search(row)]

    _log(
        "[start] iterative claim search "
        f"input={source} out_dir={out_dir} llm={llm_model_spec} "
        f"searchable_claims={len(initial_rows)} max_rounds={config.max_rounds} "
        f"max_candidates={config.max_candidates_per_round} candidate_evaluation={candidate_evaluation} "
        f"max_workers={max_workers} parallel_backend={parallel_backend}"
    )

    def record_result(result: dict[str, Any]) -> None:
        index = int(result["index"])
        if result["status"] == "completed":
            rows_by_index[index] = result["row"]
            states_by_index[index] = ClaimSearchState.model_validate(result["state"])
        else:
            skipped_by_index[index] = result["skip"]
        _log(str(result.get("message") or ""))
        if checkpoint_every and (len(rows_by_index) + len(skipped_by_index)) % checkpoint_every == 0:
            write_current("running")

    if max_workers == 1:
        indexed_rows = list(enumerate(initial_rows, start=1))
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
                    holdout_data_roots=holdout_data_roots,
                    external_data_roots=external_data_roots,
                    candidate_evaluation=candidate_evaluation,
                    llm_model_spec=llm_model_spec,
                    llm=llm,
                    preflight_context=preflight_context,
                    evidence_manifest=evidence_manifest,
                    evidence_freshness=evidence_freshness,
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
                "holdout_data_roots": [str(root) for root in holdout_data_roots],
                "external_data_roots": [str(root) for root in external_data_roots],
                "candidate_evaluation": candidate_evaluation,
                "llm_spec": args.llm,
                "evidence_manifest": str(args.evidence_manifest) if getattr(args, "evidence_manifest", None) else None,
                "evidence_freshness": evidence_freshness,
            }
            for index, row in enumerate(initial_rows, start=1)
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
    parser.add_argument("--checkpoint-every", type=int, default=5)
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
        "--holdout-data-root",
        action="append",
        default=None,
        help="Repeatable root containing materialized holdout partition parquet files.",
    )
    parser.add_argument(
        "--external-data-root",
        action="append",
        default=None,
        help="Repeatable optional external/holdout cohort root. Passing this enables external confirmation attempts.",
    )
    parser.add_argument(
        "--evidence-manifest",
        default=None,
        help="Optional evidence-partition manifest used to map source cohorts to holdout/external evaluation cohorts.",
    )
    parser.add_argument(
        "--evidence-freshness",
        choices=["fresh", "previously_queried", "unknown"],
        default="unknown",
        help="Whether excluded evidence was untouched before this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
