"""Typed artifacts for retrospective evidence audits of frozen claim searches."""

from __future__ import annotations

import hashlib
import fnmatch
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.contract import ClaimContract
from confirm.excluded_evidence import (
    ExcludedEvidenceUnavailableError,
    cohort_path,
    external_evidence_set_ids,
    execute_contract,
    mapped_contract_for_evidence,
)
from confirm.evidence_partitions import (
    EvidencePartitionManifest,
    EvidencePartitionRecord,
    canonical_base_cohort,
    contract_feature_scope,
    infer_target_family,
    load_evidence_manifest,
)
from confirm.search_artifacts import (
    read_legacy_states_from_normalized_arm,
    read_v7_result_header,
)

GenerationStatus = Literal["retained", "duplicate", "unretained", "superseded_retry"]
EvidenceKind = Literal["holdout", "external"]
EvidenceRole = Literal["candidate", "parent"]
PreflightStatus = Literal["eligible", "unavailable", "blocked"]

AUDIT_IMPLEMENTATION_PATHS = (
    "scripts/launch_initial_claim_evidence.sh",
    "scripts/launch_claim_search_retrospective_evidence.sh",
    "src/bench/run_initial_claim_evidence.py",
    "src/bench/run_frozen_claim_evidence.py",
    "src/confirm/agent.py",
    "src/confirm/analysis.py",
    "src/confirm/brainwide.py",
    "src/confirm/contract.py",
    "src/confirm/multiverse.py",
    "src/confirm/power.py",
    "src/confirm/replication.py",
    "src/confirm/results.py",
    "src/confirm/schema.py",
    "src/confirm/verdict.py",
    "src/confirm/candidate_preflight.py",
    "src/confirm/derived_columns.py",
    "src/confirm/evidence_partitions.py",
    "src/confirm/excluded_evidence.py",
    "src/confirm/frozen_evidence.py",
)


class FrozenLineage(BaseModel):
    """One parent claim in one frozen sweep arm."""

    model_config = ConfigDict(extra="forbid")

    lineage_event_id: str
    arm_id: str
    max_rounds: int
    max_candidates_per_round: int
    parent_claim_id: str
    parent_contract: dict[str, Any]
    target_family: str
    source_mode: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    failure_localization: Optional[dict[str, Any]] = None
    internally_supported_candidate_ids: list[str] = Field(default_factory=list)
    final_search_family_size: Optional[int] = None
    llm_response_count: int = 0
    proposals_returned_count: int = 0
    schema_valid_candidate_count: int = 0
    policy_valid_candidate_count: int = 0
    retained_candidate_count: int = 0
    duplicate_candidate_count: int = 0
    unretained_generated_candidate_count: int = 0
    current_data_evaluated_count: int = 0
    execution_complete_candidate_count: int = 0
    provisional_internal_pass_count: int = 0
    execution_error_count: int = 0
    analysis_non_identifiable_count: int = 0
    round_failure_contexts: list[dict[str, Any]] = Field(default_factory=list)
    stopped_reason: str
    source_artifact: str
    source_artifact_sha256: str


class FrozenLLMResponse(BaseModel):
    """One unmodified LLM response linked to its parsed candidate exposures."""

    model_config = ConfigDict(extra="forbid")

    response_id: str
    lineage_event_id: str
    arm_id: str
    parent_claim_id: str
    round_index: int
    response_sequence: int
    attempt_index: int
    raw_response: str
    raw_response_sha256: str
    parse_error: Optional[str] = None
    parsed_candidate_count: int = 0
    used_for_execution: bool = False
    source_artifact: str
    source_artifact_sha256: str


class FrozenCandidateExposure(BaseModel):
    """One candidate emitted by GPT-5.5, including superseded retries."""

    model_config = ConfigDict(extra="forbid")

    exposure_id: str
    lineage_event_id: str
    arm_id: str
    max_rounds: int
    max_candidates_per_round: int
    parent_claim_id: str
    target_family: str
    source_mode: str
    round_index: int
    response_id: str
    response_sequence: int
    attempt_index: int
    candidate_index: int
    candidate_id: str
    proposal_type: str
    transform_type: str
    declared_transform: Optional[str] = None
    inferred_transform: Optional[str] = None
    transform_match: Optional[bool] = None
    executable_contract_delta: dict[str, Any] = Field(default_factory=dict)
    generation_status: GenerationStatus
    parsed_proposal: dict[str, Any]
    normalized_proposal: Optional[dict[str, Any]] = None
    parent_contract: dict[str, Any]
    effective_contract: Optional[dict[str, Any]] = None
    validation_ok: Optional[bool] = None
    validation_violations: list[str] = Field(default_factory=list)
    current_data_evaluated: bool = False
    current_data_label: Optional[str] = None
    current_data_supported: bool = False
    provisional_internal_supported: bool = False
    final_internal_supported: bool = False
    multiplicity_retracted: bool = False
    effective_family_size: Optional[int] = None
    contract_normalization_changes: list[str] = Field(default_factory=list)
    proposal_normalization_changes: list[str] = Field(default_factory=list)
    parsed_proposal_sha256: str
    effective_contract_sha256: Optional[str] = None
    exact_contract_id: Optional[str] = None
    semantic_cluster_id: Optional[str] = None
    source_artifact: str
    source_artifact_sha256: str


class EvidenceReference(BaseModel):
    """Arm-level interpretation attached to one deduplicated execution task."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str
    lineage_event_id: str
    arm_id: str
    parent_claim_id: str
    candidate_id: Optional[str] = None
    role: EvidenceRole
    target_family: str
    source_mode: str
    transform_type: Optional[str] = None


class EvidencePreflightRecord(BaseModel):
    """Outcome-blind compatibility result for one contract/evidence pair."""

    model_config = ConfigDict(extra="forbid")

    preflight_id: str
    reference: EvidenceReference
    evidence_kind: EvidenceKind
    evidence_set_id: Optional[str] = None
    status: PreflightStatus
    schedule_for_evaluation: bool = False
    reason: Optional[str] = None
    reason_code: Optional[str] = None
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_contract: dict[str, Any]
    mapped_contract: Optional[dict[str, Any]] = None
    discovery_partition_id: Optional[str] = None
    replication_partition_ids: list[str] = Field(default_factory=list)
    discovery_path: Optional[str] = None
    replication_paths: list[str] = Field(default_factory=list)
    partition_hashes: dict[str, str] = Field(default_factory=dict)
    resolved_outcome_columns: dict[str, list[str]] = Field(default_factory=dict)
    design_diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    overlap_diagnostics: dict[str, Any] = Field(default_factory=dict)
    unit_diagnostics: dict[str, Any] = Field(default_factory=dict)
    outcome_blind: bool = True
    interpretation_label: Optional[
        Literal["excluded_evidence_compatible", "excluded_evidence_unavailable"]
    ] = None
    evidence_freshness: Literal["previously_queried"] = "previously_queried"
    final_confirmation_eligible: bool = False


class EvidenceQueryTask(BaseModel):
    """One immutable, deduplicated gate execution in the frozen query plan."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    execution_signature: str
    implementation_sha256: str
    evidence_kind: EvidenceKind
    evidence_set_id: Optional[str] = None
    target_family: str
    source_contract: dict[str, Any]
    mapped_contract: dict[str, Any]
    discovery_partition_id: str
    replication_partition_ids: list[str]
    discovery_path: str
    replication_paths: list[str]
    partition_hashes: dict[str, str]
    references: list[EvidenceReference]
    outcome_blind: Literal[True] = True
    evidence_freshness: Literal["previously_queried"] = "previously_queried"
    final_confirmation_eligible: bool = False


class EvidenceEvaluation(BaseModel):
    """Result of executing one frozen query task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    query_plan_sha256: str
    status: Literal["completed", "error"]
    raw_gate_label: Optional[str] = None
    interpretation_label: Optional[str] = None
    gate_results: Optional[dict[str, Any]] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    evidence_kind: EvidenceKind
    evidence_set_id: Optional[str] = None
    evidence_freshness: Literal["previously_queried"] = "previously_queried"
    final_confirmation_eligible: bool = False
    result_sha256: str


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                yield value


def write_jsonl_atomic(path: str | Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(destination)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _read_result_header(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Compatibility wrapper for the public v7 header reader."""

    return read_v7_result_header(path, max_bytes=max_bytes)


_FROZEN_STATE_FIELDS = (
    "original_claim",
    "source_metadata",
    "failure_localization",
    "candidate_history",
    "duplicate_candidates",
    "evaluations",
    "internally_supported_candidate_ids",
    "supported_candidates",
    "final_search_family_size",
    "llm_candidate_responses",
    "generated_candidate_count",
    "schema_valid_candidate_count",
    "valid_candidate_count",
    "current_data_evaluated_count",
    "round_failure_contexts",
    "stopped_reason",
)
_FROZEN_UNUSED_EVALUATION_FIELDS = {
    "gate_results",
    "exploratory_gate_results",
    "design_diagnostics",
}


def _load_and_project_checkpoint_direct(path: Path) -> dict[str, Any]:
    """Validate one checkpoint and discard fields unused by the freezer."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.pop("checkpoint_sha256", None)
    if checksum != sha256_json(payload):
        raise ValueError(f"Parent checkpoint hash is invalid: {path}")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"Parent checkpoint state is invalid: {path}")
    projected_state = {key: state.get(key) for key in _FROZEN_STATE_FIELDS if key in state}
    projected_evaluations = []
    for evaluation in projected_state.get("evaluations") or []:
        if isinstance(evaluation, dict):
            projected_evaluations.append(
                {
                    key: value
                    for key, value in evaluation.items()
                    if key not in _FROZEN_UNUSED_EVALUATION_FIELDS
                }
            )
    projected_state["evaluations"] = projected_evaluations
    return {
        "resume_identity_sha256": payload.get("resume_identity_sha256"),
        "index": payload.get("index"),
        "claim_id": payload.get("claim_id"),
        "row": {"claim_id": (payload.get("row") or {}).get("claim_id")},
        "state": projected_state,
    }


def _load_and_project_checkpoint(path: Path) -> dict[str, Any]:
    if path.stat().st_size <= 32 * 1024 * 1024:
        return _load_and_project_checkpoint_direct(path)
    command = (
        "import json,sys; "
        "from pathlib import Path; "
        "from confirm.frozen_evidence import _load_and_project_checkpoint_direct; "
        "print(json.dumps(_load_and_project_checkpoint_direct(Path(sys.argv[1])),separators=(',',':')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Large parent checkpoint validation failed: {path}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def _load_parent_checkpoints(
    artifact: Path,
    header: dict[str, Any],
    *,
    require_checkpoints: bool,
) -> tuple[Iterable[dict[str, Any]], dict[str, Any] | None, int]:
    checkpoint_dir = artifact.parent / "checkpoints" / "parents"
    provenance_path = artifact.parent / "run_provenance.json"
    checkpoint_paths = sorted(checkpoint_dir.glob("parent_*.json"))
    if not checkpoint_paths:
        if require_checkpoints:
            raise ValueError(f"Completed sweep arm has no atomic parent checkpoints: {artifact.parent}")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        states = list(payload.get("states") or [])
        return states, None, len(states)
    if not provenance_path.exists():
        raise ValueError(f"Sweep arm is missing run_provenance.json: {artifact.parent}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    header_provenance = header.get("provenance") or {}
    for field in ("resume_identity_sha256", "prompt_sha256", "schema_sha256"):
        if provenance.get(field) != header_provenance.get(field):
            raise ValueError(f"Run provenance/header mismatch for {field}: {artifact.parent}")
    expected_count = int(
        header.get("completed_search_count")
        or (header.get("summary") or {}).get("n_searches")
        or 0
    )
    if len(checkpoint_paths) != expected_count:
        raise ValueError(
            f"Parent checkpoint count mismatch in {artifact.parent}: "
            f"observed={len(checkpoint_paths)} expected={expected_count}"
        )
    resume_hashes = {
        provenance.get("resume_identity_sha256"),
        *(provenance.get("compatible_resume_identity_sha256s") or []),
    }
    resume_hashes.discard(None)

    def iter_states() -> Iterable[dict[str, Any]]:
        for expected_index, checkpoint_path in enumerate(checkpoint_paths, start=1):
            payload = _load_and_project_checkpoint(checkpoint_path)
            if payload.get("resume_identity_sha256") not in resume_hashes:
                raise ValueError(f"Parent checkpoint provenance mismatch: {checkpoint_path}")
            index = int(payload.get("index") or 0)
            if index != expected_index:
                raise ValueError(
                    f"Parent checkpoint indexes are not contiguous at {checkpoint_path}: "
                    f"observed={index} expected={expected_index}"
                )
            state = payload.get("state")
            row = payload.get("row")
            if not isinstance(state, dict) or not isinstance(row, dict):
                raise ValueError(f"Parent checkpoint payload is invalid: {checkpoint_path}")
            claim_id = str(payload.get("claim_id") or "")
            if (
                str((state.get("original_claim") or {}).get("claim_id") or "") != claim_id
                or str(row.get("claim_id") or "") != claim_id
            ):
                raise ValueError(f"Parent checkpoint claim ID mismatch: {checkpoint_path}")
            yield state

    return iter_states(), provenance, expected_count


def git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        return {"sha": sha, "dirty": bool(status), "dirty_entries": status}
    except Exception as exc:  # noqa: BLE001
        return {"sha": None, "dirty": None, "error": str(exc)}


def runtime_provenance() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "pydantic", "pyarrow"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": list(sys.argv),
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "blas_thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }


def implementation_hashes() -> dict[str, str]:
    return {
        path: sha256_file(path)
        for path in AUDIT_IMPLEMENTATION_PATHS
        if Path(path).exists()
    }


def _json_payload(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1]
        value = value.rsplit("```", 1)[0]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Structured candidate response must be a JSON object")
    return parsed


def _changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changes: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                changes.append(path)
            else:
                changes.extend(_changed_paths(left[key], right[key], path))
        return changes
    if left != right:
        return [prefix or "$"]
    return []


def _analysis_contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if key not in {"claim_id", "question", "reporting_language_allowed"}
    }


def _semantic_contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(_analysis_contract_payload(contract)))
    provenance = payload.get("search_provenance")
    if isinstance(provenance, dict):
        provenance.pop("family_size", None)
        provenance.pop("selection", None)
    gates = payload.get("gates")
    if isinstance(gates, dict) and isinstance(gates.get("multiplicity"), dict):
        gates["multiplicity"].pop("family_size", None)
    return payload


def _arm_id(config: dict[str, Any]) -> str:
    return f"r{int(config['max_rounds'])}_c{int(config['max_candidates_per_round'])}"


def _candidate_id(parent_claim_id: str, round_index: int, candidate_index: int, raw: dict[str, Any]) -> str:
    transform = str(raw.get("transform_type") or "unknown")
    return f"{parent_claim_id}_r{round_index}_c{candidate_index + 1}_{transform}"


def _candidate_scientific_signature(contract_payload: dict[str, Any]) -> str:
    data = ClaimContract.model_validate(contract_payload).model_dump(mode="json")
    data.pop("claim_id", None)
    data.pop("question", None)
    data.pop("reporting_language_allowed", None)
    estimand = dict(data.get("estimand") or {})
    if isinstance(estimand.get("outcome"), list):
        estimand["outcome"] = sorted({str(item) for item in estimand["outcome"]})
    data["estimand"] = estimand
    data.pop("search_provenance", None)
    gates = dict(data.get("gates") or {})
    multiplicity = dict(gates.get("multiplicity") or {})
    multiplicity.pop("family_size", None)
    gates["multiplicity"] = multiplicity
    data["gates"] = gates
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _generation_status(
    *,
    is_used_response: bool,
    candidate_id: str,
    scientific_signature: str,
    history_by_id: dict[str, Any],
    duplicate_occurrences: Counter[tuple[str, str]],
) -> GenerationStatus:
    if is_used_response and candidate_id in history_by_id:
        return "retained"
    duplicate_key = (candidate_id, scientific_signature)
    if duplicate_occurrences[duplicate_key] > 0:
        duplicate_occurrences[duplicate_key] -= 1
        return "duplicate"
    return "unretained" if is_used_response else "superseded_retry"


def _freeze_arm_sources(root: Path) -> list[dict[str, Any]]:
    """Prefer normalized arms while retaining legacy artifact identities."""

    normalized_root = root / "normalized" / "arms"
    normalized_dirs = sorted(
        path
        for path in normalized_root.glob("r*_c*")
        if (path / "manifest.json").exists()
    )
    if normalized_dirs:
        sources: list[dict[str, Any]] = []
        for arm_dir in normalized_dirs:
            manifest = json.loads(
                (arm_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if (
                manifest.get("artifact_schema")
                != "claim-search-normalized-v1"
                or manifest.get("reconciliation", {}).get("status")
                != "exact"
            ):
                raise ValueError(
                    f"Normalized sweep arm is not exact: {arm_dir}"
                )
            for record in manifest.get("files", {}).values():
                path = Path(str(record["path"]))
                if not path.exists() or sha256_file(path) != record["sha256"]:
                    raise ValueError(
                        f"Normalized sweep file hash mismatch: {path}"
                    )
            header_path = arm_dir / "run_header.json"
            provenance_path = arm_dir / "run_provenance.json"
            if not header_path.exists() or not provenance_path.exists():
                raise ValueError(
                    f"Normalized sweep arm lacks frozen run metadata: {arm_dir}"
                )
            source = manifest["source"]
            sources.append(
                {
                    "artifact": Path(source["legacy_result_path"]),
                    "artifact_sha256": source["legacy_result_sha256"],
                    "normalized_dir": arm_dir,
                    "header": json.loads(
                        header_path.read_text(encoding="utf-8")
                    ),
                    "run_provenance_path": provenance_path,
                }
            )
        return sources

    artifacts = sorted(
        root.glob(
            "matrix/rounds_*/candidates_*/"
            "iterative_candidate_replay.json"
        )
    )
    if not artifacts:
        artifacts = sorted(
            root.glob("replay/iterative_candidate_replay.json")
        )
    return [
        {
            "artifact": artifact,
            "artifact_sha256": None,
            "normalized_dir": None,
            "header": None,
            "run_provenance_path": artifact.parent
            / "run_provenance.json",
        }
        for artifact in artifacts
    ]


def freeze_sweep(
    sweep_root: str | Path,
    out_dir: str | Path,
    *,
    enforce_reference_counts: bool = True,
) -> dict[str, Any]:
    """Freeze every generated exposure and lineage from a completed 12-arm sweep."""

    root = Path(sweep_root)
    output = Path(out_dir)
    arm_sources = _freeze_arm_sources(root)
    if not arm_sources:
        raise ValueError(f"No sweep artifacts found under {root}")
    if enforce_reference_counts and len(arm_sources) != 12:
        raise ValueError(
            "Expected a complete 12-arm sweep, found "
            f"{len(arm_sources)} artifacts."
        )

    matrix_summary_path = root / "matrix_summary.json"
    matrix_rows_by_artifact: dict[str, dict[str, Any]] = {}
    matrix_summary_sha256: str | None = None
    if matrix_summary_path.exists():
        matrix_summary = json.loads(matrix_summary_path.read_text(encoding="utf-8"))
        matrix_summary_sha256 = sha256_file(matrix_summary_path)
        for row in matrix_summary.get("rows") or []:
            if isinstance(row, dict) and row.get("artifact"):
                matrix_rows_by_artifact[str(Path(row["artifact"]).resolve())] = row
    elif enforce_reference_counts:
        raise ValueError("Complete sweep freezing requires a finalized matrix_summary.json.")
    if (
        enforce_reference_counts
        and len(matrix_rows_by_artifact) != len(arm_sources)
    ):
        raise ValueError(
            "Finalized matrix summary does not contain every sweep artifact: "
            f"matrix={len(matrix_rows_by_artifact)} "
            f"artifacts={len(arm_sources)}"
        )

    candidate_path = output / "frozen_search_inventory.jsonl"
    response_path = output / "frozen_llm_responses.jsonl"
    lineage_path = output / "frozen_lineages.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    candidate_tmp = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    response_tmp = response_path.with_suffix(response_path.suffix + ".tmp")
    lineage_tmp = lineage_path.with_suffix(lineage_path.suffix + ".tmp")
    counts: Counter[str] = Counter()
    arm_summaries: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    source_payload_hashes: set[str] = set()
    models: set[str] = set()
    sweep_git_states: dict[str, Any] = {}
    prompt_hashes: set[str] = set()
    schema_hashes: set[str] = set()
    sweep_random_seeds: list[dict[str, Any]] = []
    run_provenance_hashes: dict[str, str] = {}

    llm_response_count = 0
    with (
        candidate_tmp.open("w", encoding="utf-8") as candidate_handle,
        response_tmp.open("w", encoding="utf-8") as response_handle,
        lineage_tmp.open("w", encoding="utf-8") as lineage_handle,
    ):
        for arm_source in arm_sources:
            artifact = arm_source["artifact"]
            artifact_sha = (
                arm_source["artifact_sha256"]
                or sha256_file(artifact)
            )
            source_hashes[str(artifact)] = artifact_sha
            payload = (
                arm_source["header"]
                or _read_result_header(artifact)
            )
            if payload.get("status") != "completed":
                raise ValueError(f"Sweep artifact is not completed: {artifact}")
            summary = payload.get("summary") or {}
            config = payload.get("config") or {}
            arm_id = _arm_id(config)
            matrix_row = matrix_rows_by_artifact.get(str(artifact.resolve()))
            if enforce_reference_counts and matrix_row is None:
                raise ValueError(f"Sweep artifact is absent from finalized matrix summary: {artifact}")
            if matrix_row is not None:
                matrix_identity = (
                    int(matrix_row.get("max_rounds") or 0),
                    int(matrix_row.get("max_candidates_per_round") or 0),
                    str(matrix_row.get("status") or ""),
                    str(matrix_row.get("source_sha256") or ""),
                    str(matrix_row.get("llm_model") or ""),
                    str(matrix_row.get("prompt_sha256") or ""),
                    str(matrix_row.get("schema_sha256") or ""),
                    str(matrix_row.get("implementation_hashes_sha256") or ""),
                    str(matrix_row.get("evidence_manifest_sha256") or ""),
                    str(matrix_row.get("partition_hashes_sha256") or ""),
                )
                header_provenance = payload.get("provenance") or {}
                header_identity = (
                    int(config.get("max_rounds") or 0),
                    int(config.get("max_candidates_per_round") or 0),
                    str(payload.get("status") or ""),
                    str((((payload.get("provenance") or {}).get("source") or {}).get("sha256")) or ""),
                    str(payload.get("llm_model") or ""),
                    str(header_provenance.get("prompt_sha256") or ""),
                    str(header_provenance.get("schema_sha256") or ""),
                    sha256_json(header_provenance.get("implementation_hashes") or {}),
                    str((header_provenance.get("evidence_manifest") or {}).get("sha256") or ""),
                    str(header_provenance.get("partition_hashes_sha256") or ""),
                )
                if matrix_identity != header_identity:
                    raise ValueError(f"Matrix summary identity mismatch for {artifact}")
            if int(summary.get("excluded_evidence_query_count") or 0) != 0:
                raise ValueError(f"Sweep arm {arm_id} queried excluded evidence")
            model = str(payload.get("llm_model") or "")
            models.add(model)
            provenance = payload.get("provenance") or {}
            source_sha = str((provenance.get("source") or {}).get("sha256") or "")
            source_payload_hashes.add(source_sha)
            sweep_git_states[arm_id] = provenance.get("git")
            if provenance.get("prompt_sha256"):
                prompt_hashes.add(str(provenance["prompt_sha256"]))
            if provenance.get("schema_sha256"):
                schema_hashes.add(str(provenance["schema_sha256"]))
            if isinstance(provenance.get("random_seeds"), dict):
                sweep_random_seeds.append(dict(provenance["random_seeds"]))
            normalized_dir = arm_source["normalized_dir"]
            if normalized_dir is not None:
                states = read_legacy_states_from_normalized_arm(
                    normalized_dir
                )
                run_provenance = json.loads(
                    arm_source["run_provenance_path"].read_text(
                        encoding="utf-8"
                    )
                )
                state_count = len(states)
            else:
                states, run_provenance, state_count = (
                    _load_parent_checkpoints(
                        artifact,
                        payload,
                        require_checkpoints=enforce_reference_counts,
                    )
                )
            if run_provenance is not None:
                run_provenance_hashes[arm_id] = sha256_file(
                    arm_source["run_provenance_path"]
                )
            if state_count != int(payload.get("completed_search_count") or summary.get("n_searches") or 0):
                raise ValueError(f"State count mismatch in {artifact}")
            if enforce_reference_counts and state_count != 215:
                raise ValueError(f"Sweep arm {arm_id} has {state_count} parents; expected 215.")

            arm_observed: Counter[str] = Counter()
            for state in states:
                parent = ClaimContract.model_validate(state["original_claim"])
                parent_claim_id = parent.claim_id
                lineage_event_id = f"{arm_id}:{parent_claim_id}"
                source_metadata = dict(state.get("source_metadata") or {})
                target_family = str(source_metadata.get("target_family") or infer_target_family(parent))
                source_mode = str(source_metadata.get("source_mode") or "unknown")
                internally_supported_ids = [
                    str(item)
                    for item in state.get("internally_supported_candidate_ids", state.get("supported_candidates", []))
                ]
                candidate_history = state.get("candidate_history") or []
                duplicate_candidates = state.get("duplicate_candidates") or []
                evaluations = state.get("evaluations") or []
                proposals_returned = int(state.get("generated_candidate_count") or 0)
                retained_count = len(candidate_history)
                duplicate_count = len(duplicate_candidates)
                lineage = FrozenLineage(
                        lineage_event_id=lineage_event_id,
                        arm_id=arm_id,
                        max_rounds=int(config["max_rounds"]),
                        max_candidates_per_round=int(config["max_candidates_per_round"]),
                        parent_claim_id=parent_claim_id,
                        parent_contract=parent.model_dump(mode="json"),
                        target_family=target_family,
                        source_mode=source_mode,
                        source_metadata=source_metadata,
                        failure_localization=(
                            dict(state["failure_localization"])
                            if isinstance(state.get("failure_localization"), dict)
                            else None
                        ),
                        internally_supported_candidate_ids=internally_supported_ids,
                        final_search_family_size=state.get("final_search_family_size"),
                        llm_response_count=len(state.get("llm_candidate_responses") or []),
                        proposals_returned_count=proposals_returned,
                        schema_valid_candidate_count=int(
                            state.get("schema_valid_candidate_count") or proposals_returned
                        ),
                        policy_valid_candidate_count=int(state.get("valid_candidate_count") or 0),
                        retained_candidate_count=retained_count,
                        duplicate_candidate_count=duplicate_count,
                        unretained_generated_candidate_count=max(
                            proposals_returned - retained_count - duplicate_count,
                            0,
                        ),
                        current_data_evaluated_count=int(
                            state.get("current_data_evaluated_count") or 0
                        ),
                        execution_complete_candidate_count=sum(
                            bool(item.get("evaluated")) and not item.get("execution_error")
                            for item in evaluations
                            if isinstance(item, dict)
                        ),
                        provisional_internal_pass_count=sum(
                            bool(item.get("provisional_supported")) and not item.get("execution_error")
                            for item in evaluations
                            if isinstance(item, dict)
                        ),
                        execution_error_count=sum(
                            bool(item.get("execution_error"))
                            for item in evaluations
                            if isinstance(item, dict)
                        ),
                        analysis_non_identifiable_count=sum(
                            item.get("blocked_reason") == "analysis_non_identifiable"
                            or str(item.get("execution_error") or "").startswith(
                                "analysis_non_identifiable:"
                            )
                            for item in evaluations
                            if isinstance(item, dict)
                        ),
                        round_failure_contexts=[
                            dict(item)
                            for item in state.get("round_failure_contexts") or []
                            if isinstance(item, dict)
                        ],
                        stopped_reason=str(state.get("stopped_reason") or "unknown"),
                        source_artifact=str(artifact),
                        source_artifact_sha256=artifact_sha,
                    )
                lineage_handle.write(
                    json.dumps(lineage.model_dump(mode="json"), sort_keys=True) + "\n"
                )
                counts["searched_lineage_events"] += 1
                arm_observed["searched_lineage_events"] += 1

                history_by_id = {
                    str(item["candidate_id"]): item
                    for item in state.get("candidate_history") or []
                }
                duplicate_occurrences: Counter[tuple[str, str]] = Counter(
                    (
                        str(item.get("candidate_id") or ""),
                        str(item.get("scientific_signature") or ""),
                    )
                    for item in state.get("duplicate_candidates") or []
                    if isinstance(item, dict)
                )
                evaluation_by_id = {
                    str(item["candidate_id"]): item
                    for item in state.get("evaluations") or []
                }
                responses = list(state.get("llm_candidate_responses") or [])
                successful_by_round: dict[int, list[int]] = defaultdict(list)
                parsed_responses: dict[int, list[dict[str, Any]]] = {}
                response_records: dict[int, FrozenLLMResponse] = {}
                for sequence, response in enumerate(responses):
                    raw_response = str(response.get("raw_response") or "")
                    response_id = sha256_json(
                        {
                            "arm_id": arm_id,
                            "parent_claim_id": parent_claim_id,
                            "round_index": int(response["round_index"]),
                            "response_sequence": sequence,
                            "attempt_index": int(response.get("attempt_index") or 0),
                            "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                        }
                    )
                    parse_error = response.get("parse_error")
                    response_records[sequence] = FrozenLLMResponse(
                        response_id=response_id,
                        lineage_event_id=lineage_event_id,
                        arm_id=arm_id,
                        parent_claim_id=parent_claim_id,
                        round_index=int(response["round_index"]),
                        response_sequence=sequence,
                        attempt_index=int(response.get("attempt_index") or 0),
                        raw_response=raw_response,
                        raw_response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                        parse_error=(str(parse_error) if parse_error is not None else None),
                        parsed_candidate_count=0,
                        used_for_execution=False,
                        source_artifact=str(artifact),
                        source_artifact_sha256=artifact_sha,
                    )
                    if parse_error is not None:
                        continue
                    parsed = _json_payload(raw_response or "{}")
                    raw_candidates = parsed.get("candidates")
                    if not isinstance(raw_candidates, list):
                        raise ValueError(f"Missing candidates list in {arm_id}/{parent_claim_id} response {sequence}")
                    candidate_count = int(response.get("candidate_count") or 0)
                    raw_candidates = raw_candidates[:candidate_count]
                    if len(raw_candidates) != candidate_count:
                        raise ValueError(f"Candidate response count mismatch in {arm_id}/{parent_claim_id}")
                    parsed_responses[sequence] = raw_candidates
                    response_records[sequence].parsed_candidate_count = candidate_count
                    successful_by_round[int(response["round_index"])].append(sequence)

                used_response = {
                    round_index: sequences[-1]
                    for round_index, sequences in successful_by_round.items()
                }
                for sequence, response_record in sorted(response_records.items()):
                    response_record.used_for_execution = (
                        used_response.get(response_record.round_index) == sequence
                    )
                    response_handle.write(
                        json.dumps(response_record.model_dump(mode="json"), sort_keys=True) + "\n"
                    )
                    llm_response_count += 1
                    arm_observed["llm_response_count"] += 1
                parsed_count = 0
                for sequence, response in enumerate(responses):
                    raw_candidates = parsed_responses.get(sequence)
                    if raw_candidates is None:
                        continue
                    round_index = int(response["round_index"])
                    is_used_response = used_response.get(round_index) == sequence
                    for candidate_index, raw_candidate_value in enumerate(raw_candidates):
                        if not isinstance(raw_candidate_value, dict):
                            raise ValueError(f"Candidate is not an object in {arm_id}/{parent_claim_id}")
                        raw_candidate = dict(raw_candidate_value)
                        parsed_count += 1
                        candidate_id = _candidate_id(parent_claim_id, round_index, candidate_index, raw_candidate)
                        raw_contract = raw_candidate.get("proposed_contract")
                        scientific_signature = (
                            _candidate_scientific_signature(raw_contract)
                            if isinstance(raw_contract, dict)
                            else ""
                        )
                        generation_status = _generation_status(
                            is_used_response=is_used_response,
                            candidate_id=candidate_id,
                            scientific_signature=scientific_signature,
                            history_by_id=history_by_id,
                            duplicate_occurrences=duplicate_occurrences,
                        )

                        normalized = history_by_id.get(candidate_id) if is_used_response else None
                        evaluation = evaluation_by_id.get(candidate_id) if is_used_response else None
                        if evaluation is not None and isinstance(evaluation.get("proposal"), dict):
                            normalized = evaluation["proposal"]
                        effective_contract = (
                            normalized.get("proposed_contract")
                            if isinstance(normalized, dict)
                            else None
                        )
                        validation = evaluation.get("validation") if isinstance(evaluation, dict) else None
                        proposal_keys = (
                            "proposal_type",
                            "transform_type",
                            "provenance",
                            "requires_new_evidence",
                            "can_confirm_on_current_data",
                            "validation_split",
                            "evidence_policy",
                        )
                        normalized_policy = (
                            {key: normalized.get(key) for key in proposal_keys}
                            if isinstance(normalized, dict)
                            else None
                        )
                        raw_policy = {key: raw_candidate.get(key) for key in proposal_keys}
                        exposure_seed = {
                            "arm_id": arm_id,
                            "parent_claim_id": parent_claim_id,
                            "round_index": round_index,
                            "response_sequence": sequence,
                            "candidate_index": candidate_index,
                            "parsed_proposal": raw_candidate,
                        }
                        final_internal_supported = bool(
                            is_used_response
                            and generation_status == "retained"
                            and candidate_id in internally_supported_ids
                            and evaluation
                            and evaluation.get("current_data_supported")
                        )
                        exact_contract_id = (
                            sha256_json(_analysis_contract_payload(effective_contract))
                            if isinstance(effective_contract, dict)
                            else None
                        )
                        semantic_cluster_id = (
                            sha256_json(_semantic_contract_payload(effective_contract))
                            if isinstance(effective_contract, dict)
                            else None
                        )
                        record = FrozenCandidateExposure(
                            exposure_id=sha256_json(exposure_seed),
                            lineage_event_id=lineage_event_id,
                            arm_id=arm_id,
                            max_rounds=int(config["max_rounds"]),
                            max_candidates_per_round=int(config["max_candidates_per_round"]),
                            parent_claim_id=parent_claim_id,
                            target_family=target_family,
                            source_mode=source_mode,
                            round_index=round_index,
                            response_id=response_records[sequence].response_id,
                            response_sequence=sequence,
                            attempt_index=int(response.get("attempt_index") or 0),
                            candidate_index=candidate_index,
                            candidate_id=candidate_id,
                            proposal_type=str(raw_candidate.get("proposal_type") or "unknown"),
                            transform_type=str(raw_candidate.get("transform_type") or "unknown"),
                            declared_transform=(
                                str(normalized.get("declared_transform"))
                                if isinstance(normalized, dict) and normalized.get("declared_transform")
                                else str(raw_candidate.get("transform_type") or "unknown")
                            ),
                            inferred_transform=(
                                str(normalized.get("inferred_transform"))
                                if isinstance(normalized, dict) and normalized.get("inferred_transform")
                                else None
                            ),
                            transform_match=(
                                bool(normalized.get("transform_match"))
                                if isinstance(normalized, dict) and normalized.get("transform_match") is not None
                                else None
                            ),
                            executable_contract_delta=(
                                dict(normalized.get("executable_contract_delta") or {})
                                if isinstance(normalized, dict)
                                else {}
                            ),
                            generation_status=generation_status,
                            parsed_proposal=raw_candidate,
                            normalized_proposal=normalized,
                            parent_contract=parent.model_dump(mode="json"),
                            effective_contract=effective_contract,
                            validation_ok=(bool(validation.get("ok")) if isinstance(validation, dict) else None),
                            validation_violations=(
                                [str(item) for item in validation.get("violations", [])]
                                if isinstance(validation, dict)
                                else []
                            ),
                            current_data_evaluated=bool(
                                evaluation
                                and evaluation.get("eligible_for_confirmation")
                                and evaluation.get("evaluated")
                            ),
                            current_data_label=(str(evaluation.get("final_label")) if evaluation and evaluation.get("final_label") else None),
                            current_data_supported=bool(evaluation and evaluation.get("current_data_supported")),
                            provisional_internal_supported=bool(
                                evaluation and evaluation.get("provisional_supported")
                            ),
                            final_internal_supported=final_internal_supported,
                            multiplicity_retracted=bool(
                                evaluation and evaluation.get("multiplicity_retracted")
                            ),
                            effective_family_size=(
                                int(evaluation["effective_family_size"])
                                if evaluation and evaluation.get("effective_family_size") is not None
                                else None
                            ),
                            contract_normalization_changes=(
                                _changed_paths(raw_contract, effective_contract)
                                if isinstance(raw_contract, dict) and isinstance(effective_contract, dict)
                                else []
                            ),
                            proposal_normalization_changes=(
                                _changed_paths(raw_policy, normalized_policy)
                                if normalized_policy is not None
                                else []
                            ),
                            parsed_proposal_sha256=sha256_json(raw_candidate),
                            effective_contract_sha256=(sha256_json(effective_contract) if effective_contract else None),
                            exact_contract_id=exact_contract_id,
                            semantic_cluster_id=semantic_cluster_id,
                            source_artifact=str(artifact),
                            source_artifact_sha256=artifact_sha,
                        )
                        candidate_handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
                        counts["generated_candidate_count"] += 1
                        arm_observed["generated_candidate_count"] += 1
                        counts[f"generation_status:{generation_status}"] += 1
                        arm_observed[f"generation_status:{generation_status}"] += 1
                        if record.current_data_evaluated:
                            counts["current_data_evaluated_count"] += 1
                            arm_observed["current_data_evaluated_count"] += 1
                        if record.final_internal_supported:
                            counts["final_internal_supported_count"] += 1
                            arm_observed["final_internal_supported_count"] += 1

                if parsed_count != int(state.get("generated_candidate_count") or 0):
                    raise ValueError(
                        f"Generated count mismatch for {arm_id}/{parent_claim_id}: "
                        f"responses={parsed_count} state={state.get('generated_candidate_count')}"
                    )
                unmatched_duplicates = sum(duplicate_occurrences.values())
                if unmatched_duplicates:
                    raise ValueError(
                        f"Could not map {unmatched_duplicates} duplicate candidate records to raw responses "
                        f"for {arm_id}/{parent_claim_id}"
                    )

            expected_arm = {
                "searched_lineage_events": int(summary.get("n_searches") or 0),
                "generated_candidate_count": int(summary.get("generated_candidate_count") or 0),
                "retained_candidate_count": int(summary.get("candidate_count") or 0),
                "current_data_evaluated_count": int(summary.get("current_data_evaluated_count") or 0),
                "duplicate_candidate_count": int(summary.get("duplicate_candidate_count") or 0),
                "unretained_generated_candidate_count": int(summary.get("unretained_generated_candidate_count") or 0),
                "final_internal_supported_count": int(
                    summary.get("final_multiplicity_adjusted_internal_pass_count") or 0
                ),
            }
            observed_arm = {
                "searched_lineage_events": arm_observed["searched_lineage_events"],
                "generated_candidate_count": arm_observed["generated_candidate_count"],
                "retained_candidate_count": arm_observed["generation_status:retained"],
                "current_data_evaluated_count": arm_observed["current_data_evaluated_count"],
                "duplicate_candidate_count": arm_observed["generation_status:duplicate"],
                "unretained_generated_candidate_count": (
                    arm_observed["generation_status:unretained"]
                    + arm_observed["generation_status:superseded_retry"]
                ),
                "final_internal_supported_count": arm_observed["final_internal_supported_count"],
            }
            if observed_arm != expected_arm:
                raise ValueError(f"Frozen count mismatch for {arm_id}: observed={observed_arm} expected={expected_arm}")
            arm_summaries.append(
                {
                    "arm_id": arm_id,
                    "artifact": str(artifact),
                    "llm_response_count": arm_observed["llm_response_count"],
                    "proposals_returned_count": int(
                        summary.get("proposals_returned_count", summary.get("generated_candidate_count", 0))
                    ),
                    "schema_valid_candidate_count": int(
                        summary.get("schema_valid_candidate_count", summary.get("generated_candidate_count", 0))
                    ),
                    "policy_valid_candidate_count": int(
                        summary.get("policy_valid_candidate_count", summary.get("valid_candidate_count", 0))
                    ),
                    "execution_complete_candidate_count": int(
                        summary.get("execution_complete_candidate_count", 0)
                    ),
                    "provisional_internal_pass_count": int(
                        summary.get("provisional_internal_pass_count", 0)
                    ),
                    **observed_arm,
                }
            )

    candidate_tmp.replace(candidate_path)
    response_tmp.replace(response_path)
    lineage_tmp.replace(lineage_path)
    observed = {
        "artifact_count": len(arm_sources),
        "searched_lineage_events": counts["searched_lineage_events"],
        "generated_candidate_count": counts["generated_candidate_count"],
        "retained_candidate_count": counts["generation_status:retained"],
        "current_data_evaluated_count": counts["current_data_evaluated_count"],
        "duplicate_candidate_count": counts["generation_status:duplicate"],
        "unretained_generated_candidate_count": (
            counts["generation_status:unretained"] + counts["generation_status:superseded_retry"]
        ),
        "final_internal_supported_count": counts["final_internal_supported_count"],
    }
    if len(models) != 1 or source_payload_hashes == {""} or len(source_payload_hashes) != 1:
        raise ValueError(f"Sweep arms do not share one model/source hash: models={models}, sources={source_payload_hashes}")
    if enforce_reference_counts and (len(prompt_hashes) != 1 or len(schema_hashes) != 1):
        raise ValueError(
            f"Sweep arms do not share one prompt/schema hash: prompts={prompt_hashes}, schemas={schema_hashes}"
        )
    if enforce_reference_counts and observed["searched_lineage_events"] != 12 * 215:
        raise ValueError(
            "Every arm must contain all 215 failed parent lineages; "
            f"observed total={observed['searched_lineage_events']}."
        )

    manifest = {
        **runtime_provenance(),
        "phase": "freeze",
        "sweep_root": str(root),
        "llm_model_recorded": next(iter(models)),
        "llm_calls_performed": 0,
        "source_payload_sha256": next(iter(source_payload_hashes)),
        "source_artifact_hashes": source_hashes,
        "matrix_summary": (
            {"path": str(matrix_summary_path), "sha256": matrix_summary_sha256}
            if matrix_summary_sha256
            else None
        ),
        "run_provenance_hashes": dict(sorted(run_provenance_hashes.items())),
        "sweep_git_states": sweep_git_states,
        "prompt_sha256_values": sorted(prompt_hashes),
        "schema_sha256_values": sorted(schema_hashes),
        "sweep_random_seeds": sweep_random_seeds,
        "llm_response_count": llm_response_count,
        "observed_counts": observed,
        "complete_grid_invariants_enforced": enforce_reference_counts,
        "v6_complete_grid_invariants_enforced": enforce_reference_counts,
        "arm_summaries": sorted(arm_summaries, key=lambda row: row["arm_id"]),
        "artifacts": {
            "frozen_search_inventory": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "frozen_llm_responses": {"path": str(response_path), "sha256": sha256_file(response_path)},
            "frozen_lineages": {"path": str(lineage_path), "sha256": sha256_file(lineage_path)},
        },
        "interpretation": "Frozen retrospective inventory; no candidate regeneration or excluded-evidence evaluation.",
    }
    write_json_atomic(output / "freeze_manifest.json", manifest)
    return manifest


def freeze_initial_claims(
    initial_results_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Freeze every Stage 2 claim contract and its internal gate label."""

    source_path = Path(initial_results_path)
    output = Path(out_dir)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = payload.get("claims")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError(f"Initial result artifact has no claims: {source_path}")

    output.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(source_path)
    lineage_path = output / "frozen_lineages.jsonl"
    inventory_path = output / "frozen_search_inventory.jsonl"
    response_path = output / "frozen_llm_responses.jsonl"
    lineages: list[FrozenLineage] = []
    seen_claim_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    source_mode_counts: Counter[str] = Counter()

    for index, row_value in enumerate(source_rows):
        if not isinstance(row_value, dict):
            raise ValueError(f"Initial claim row {index} is not an object")
        row = dict(row_value)
        if row.get("gate_success") is not True:
            raise ValueError(f"Initial claim row {index} was not successfully gate-evaluated")
        contract_payload = row.get("contract")
        gate_contract = (row.get("gate_results") or {}).get("contract")
        drafted_contract = row.get("drafted_contract")
        if not isinstance(contract_payload, dict):
            raise ValueError(f"Initial claim row {index} lacks an embedded contract")
        if contract_payload != gate_contract or contract_payload != drafted_contract:
            raise ValueError(f"Initial claim row {index} has inconsistent frozen contracts")
        contract = ClaimContract.model_validate(contract_payload)
        if str(row.get("claim_id") or "") != contract.claim_id:
            raise ValueError(f"Initial claim row {index} has a claim ID mismatch")
        if contract.claim_id in seen_claim_ids:
            raise ValueError(f"Duplicate initial claim ID: {contract.claim_id}")
        seen_claim_ids.add(contract.claim_id)

        initial_label = str(row.get("final_label") or "")
        if not initial_label or initial_label != str(row.get("gate_verdict_label") or ""):
            raise ValueError(f"Initial claim {contract.claim_id} has inconsistent gate labels")
        target_family = str(row.get("target_family") or infer_target_family(contract))
        source_mode = str(row.get("source_mode") or "unknown")
        source_metadata = {
            "target_family": target_family,
            "source_mode": source_mode,
            "initial_label": initial_label,
            "ground_truth": row.get("ground_truth"),
            "label_class": row.get("label_class"),
            "scoring_label": row.get("scoring_label"),
            "source_citation": row.get("source_citation"),
            "model_spec": row.get("model_spec"),
            "source_result_path": str(source_path),
        }
        lineages.append(
            FrozenLineage(
                lineage_event_id=f"initial:{contract.claim_id}",
                arm_id="initial",
                max_rounds=0,
                max_candidates_per_round=0,
                parent_claim_id=contract.claim_id,
                parent_contract=contract.model_dump(mode="json"),
                target_family=target_family,
                source_mode=source_mode,
                source_metadata=source_metadata,
                failure_localization=None,
                stopped_reason=f"initial_{initial_label}",
                source_artifact=str(source_path),
                source_artifact_sha256=source_sha256,
            )
        )
        label_counts[initial_label] += 1
        target_counts[target_family] += 1
        source_mode_counts[source_mode] += 1

    write_jsonl_atomic(lineage_path, lineages)
    write_jsonl_atomic(inventory_path, [])
    write_jsonl_atomic(response_path, [])
    observed = {
        "initial_claim_count": len(lineages),
        "initial_confirmed_count": label_counts["confirmed"],
        "initial_label_counts": dict(sorted(label_counts.items())),
        "target_family_counts": dict(sorted(target_counts.items())),
        "source_mode_counts": dict(sorted(source_mode_counts.items())),
    }
    manifest = {
        **runtime_provenance(),
        "phase": "freeze",
        "audit_type": "initial_claims",
        "initial_results_path": str(source_path),
        "llm_calls_performed": 0,
        "source_payload_sha256": source_sha256,
        "source_artifact_hashes": {str(source_path): source_sha256},
        "observed_counts": observed,
        "artifacts": {
            "frozen_initial_claims": {
                "path": str(lineage_path),
                "sha256": sha256_file(lineage_path),
            },
            "frozen_search_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
            "frozen_llm_responses": {
                "path": str(response_path),
                "sha256": sha256_file(response_path),
            },
        },
        "interpretation": (
            "Frozen Stage 2 initial contracts and internal labels; no claim regeneration "
            "or excluded-evidence evaluation."
        ),
    }
    write_json_atomic(output / "freeze_manifest.json", manifest)
    return manifest


def verify_frozen_sources(out_dir: str | Path) -> dict[str, Any]:
    """Fail if any sweep or frozen-inventory input changed after freezing."""

    output = Path(out_dir)
    manifest_path = output / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path_value, expected in manifest["source_artifact_hashes"].items():
        path = Path(path_value)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Frozen sweep artifact changed: {path} expected={expected} observed={observed}")
    for item in manifest["artifacts"].values():
        path = Path(item["path"])
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise ValueError(f"Frozen inventory changed: {path} expected={item['sha256']} observed={observed}")
    return manifest


def _partition_hashes(records: list[EvidencePartitionRecord]) -> dict[str, str]:
    return {
        record.partition_id: record.content_sha256 or sha256_file(record.path)
        for record in records
    }


def _subject_ids(path: Path, cache: dict[str, set[str]]) -> set[str]:
    key = str(path)
    if key not in cache:
        frame = pd.read_parquet(path, columns=["subject_id"])
        cache[key] = set(frame["subject_id"].dropna().astype(str))
    return cache[key]


def _source_path(
    cohort: str,
    manifest: EvidencePartitionManifest,
    target_family: str,
    source_roots: list[Path],
) -> Path | None:
    record = manifest.record_for_partition(cohort, target_family)
    if record is not None and Path(record.path).exists():
        return Path(record.path)
    try:
        return cohort_path(source_roots, cohort, allow_aliases=False)
    except FileNotFoundError:
        return None


def _overlap_diagnostics(
    source_contract: ClaimContract,
    target_family: str,
    discovery: EvidencePartitionRecord,
    replications: list[EvidencePartitionRecord],
    manifest: EvidencePartitionManifest,
    source_roots: list[Path],
    subject_cache: dict[str, set[str]],
    evidence_kind: EvidenceKind,
) -> tuple[dict[str, Any], list[str]]:
    excluded = [discovery, *replications]
    violations: list[str] = []
    pair_overlaps: dict[str, int] = {}
    for index, left in enumerate(excluded):
        left_ids = _subject_ids(Path(left.path), subject_cache)
        for right in excluded[index + 1 :]:
            overlap = len(left_ids & _subject_ids(Path(right.path), subject_cache))
            pair_overlaps[f"{left.partition_id}|{right.partition_id}"] = overlap
            if overlap:
                violations.append(
                    f"Excluded partitions overlap: {left.partition_id} and {right.partition_id} share {overlap} subjects."
                )

    source_overlaps: dict[str, int] = {}
    for source_cohort in [source_contract.discovery_cohort, *source_contract.replication_cohorts]:
        source_base = canonical_base_cohort(source_cohort)
        source_path = _source_path(source_cohort, manifest, target_family, source_roots)
        if source_path is None:
            violations.append(f"Source partition path unavailable for overlap check: {source_cohort}")
            continue
        source_ids = _subject_ids(source_path, subject_cache)
        for record in excluded:
            if evidence_kind == "external" and source_base == canonical_base_cohort(record.base_dataset):
                violations.append(
                    f"External evidence {record.partition_id} reuses source base dataset {source_base}."
                )
            if source_base != canonical_base_cohort(record.base_dataset):
                continue
            overlap = len(source_ids & _subject_ids(Path(record.path), subject_cache))
            source_overlaps[f"{source_cohort}|{record.partition_id}"] = overlap
            if overlap:
                violations.append(
                    f"Excluded evidence overlaps source data: {source_cohort} and {record.partition_id} share {overlap} subjects."
                )
    return {
        "excluded_pair_subject_overlaps": pair_overlaps,
        "source_excluded_subject_overlaps": source_overlaps,
        "all_disjoint": not violations,
    }, violations


def _unit_for_outcome(units: dict[str, str], outcome: str) -> str | None:
    exact = units.get(outcome)
    if exact:
        return exact
    matches = [
        (pattern, unit)
        for pattern, unit in units.items()
        if fnmatch.fnmatch(outcome, pattern)
        or (outcome.endswith("_") and pattern.startswith(outcome))
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


def _unit_diagnostics(
    source_contract: ClaimContract,
    target_records: list[EvidencePartitionRecord],
    manifest: EvidencePartitionManifest,
    target_family: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    source_records = [
        manifest.record_for_partition(cohort, target_family)
        for cohort in [source_contract.discovery_cohort, *source_contract.replication_cohorts]
    ]
    source_records = [record for record in source_records if record is not None]
    outcomes = (
        list(source_contract.estimand.outcome)
        if isinstance(source_contract.estimand.outcome, list)
        else [source_contract.estimand.outcome]
    )
    diagnostics: dict[str, Any] = {}
    violations: list[str] = []
    warnings: list[str] = []
    for outcome in outcomes:
        source_units = {
            unit
            for record in source_records
            for unit in [_unit_for_outcome(record.units, str(outcome))]
            if unit
        }
        target_units = {
            unit
            for record in target_records
            for unit in [_unit_for_outcome(record.units, str(outcome))]
            if unit
        }
        status = "compatible"
        if not source_units or not target_units:
            status = "unknown"
            warnings.append(f"Unit compatibility is unknown for outcome {outcome!r}.")
        elif source_units.isdisjoint(target_units):
            status = "incompatible"
            violations.append(
                f"Unit mismatch for outcome {outcome!r}: source={sorted(source_units)} target={sorted(target_units)}."
            )
        diagnostics[str(outcome)] = {
            "source_units": sorted(source_units),
            "target_units": sorted(target_units),
            "status": status,
        }
    return diagnostics, violations, warnings


def _reference(
    lineage: FrozenLineage,
    role: EvidenceRole,
    *,
    candidate: FrozenCandidateExposure | None = None,
) -> EvidenceReference:
    candidate_id = candidate.candidate_id if candidate is not None else None
    return EvidenceReference(
        reference_id=f"{lineage.lineage_event_id}:{role}:{candidate_id or lineage.parent_claim_id}",
        lineage_event_id=lineage.lineage_event_id,
        arm_id=lineage.arm_id,
        parent_claim_id=lineage.parent_claim_id,
        candidate_id=candidate_id,
        role=role,
        target_family=lineage.target_family,
        source_mode=lineage.source_mode,
        transform_type=(candidate.transform_type if candidate is not None else None),
    )


def _preflight_record(
    *,
    reference: EvidenceReference,
    source_contract: ClaimContract,
    evidence_kind: EvidenceKind,
    evidence_set_id: str | None,
    manifest: EvidencePartitionManifest,
    context: CandidatePreflightContext,
    evidence_roots: list[Path],
    source_roots: list[Path],
    subject_cache: dict[str, set[str]],
    schedule_for_evaluation: bool,
) -> EvidencePreflightRecord:
    preflight_id = sha256_json(
        {
            "reference_id": reference.reference_id,
            "evidence_kind": evidence_kind,
            "evidence_set_id": evidence_set_id,
            "source_contract": source_contract.model_dump(mode="json"),
        }
    )
    try:
        mapped, discovery, replications, resolved_set = mapped_contract_for_evidence(
            source_contract,
            manifest,
            evidence_kind,
            evidence_set_id=evidence_set_id,
        )
        discovery_path = cohort_path(evidence_roots, discovery.partition_id, allow_aliases=False)
        replication_paths = [
            cohort_path(evidence_roots, record.partition_id, allow_aliases=False)
            for record in replications
        ]
    except (ExcludedEvidenceUnavailableError, FileNotFoundError, ValueError) as exc:
        return EvidencePreflightRecord(
            preflight_id=preflight_id,
            reference=reference,
            evidence_kind=evidence_kind,
            evidence_set_id=evidence_set_id,
            status="unavailable",
            schedule_for_evaluation=False,
            reason=str(exc),
            reason_code="no_compatible_evidence",
            interpretation_label="excluded_evidence_unavailable",
            source_contract=source_contract.model_dump(mode="json"),
        )

    result = context.validate_contract(mapped, min_complete_rows=20, min_group_rows=20)
    overlap, overlap_violations = _overlap_diagnostics(
        source_contract,
        reference.target_family,
        discovery,
        replications,
        manifest,
        source_roots,
        subject_cache,
        evidence_kind,
    )
    unit_diagnostics, unit_violations, unit_warnings = _unit_diagnostics(
        source_contract,
        [discovery, *replications],
        manifest,
        reference.target_family,
    )
    violations = [*result.violations, *overlap_violations, *unit_violations]
    status: PreflightStatus = "eligible" if not violations else "blocked"
    reason_code = None
    joined_violations = " ".join(violations).lower()
    if overlap_violations:
        reason_code = "non_independent_evidence"
    elif "analysis_non_identifiable" in joined_violations:
        reason_code = "analysis_non_identifiable"
    elif "missing outcome" in joined_violations or "missing analysis columns" in joined_violations:
        reason_code = "schema_incompatible"
    elif "too few complete rows" in joined_violations:
        reason_code = "insufficient_complete_cases"
    elif "inclusion query" in joined_violations:
        reason_code = "inclusion_incompatible"
    elif "unit mismatch" in joined_violations:
        reason_code = "unit_incompatible"
    elif violations:
        reason_code = "preflight_failed"
    records = [discovery, *replications]
    return EvidencePreflightRecord(
        preflight_id=preflight_id,
        reference=reference,
        evidence_kind=evidence_kind,
        evidence_set_id=resolved_set,
        status=status,
        schedule_for_evaluation=bool(schedule_for_evaluation and status == "eligible"),
        reason=None if status == "eligible" else "; ".join(violations),
        reason_code=reason_code,
        violations=violations,
        warnings=[*result.warnings, *unit_warnings],
        source_contract=source_contract.model_dump(mode="json"),
        mapped_contract=mapped.model_dump(mode="json"),
        discovery_partition_id=discovery.partition_id,
        replication_partition_ids=[record.partition_id for record in replications],
        discovery_path=str(discovery_path),
        replication_paths=[str(path) for path in replication_paths],
        partition_hashes=_partition_hashes(records),
        resolved_outcome_columns=result.resolved_outcome_columns,
        design_diagnostics=result.design_diagnostics,
        overlap_diagnostics=overlap,
        unit_diagnostics=unit_diagnostics,
        interpretation_label=(
            "excluded_evidence_compatible"
            if status == "eligible"
            else "excluded_evidence_unavailable"
        ),
    )


def _preflight_contract_evidence(
    *,
    reference: EvidenceReference,
    source_contract: ClaimContract,
    manifest: EvidencePartitionManifest,
    context: CandidatePreflightContext,
    evidence_roots: list[Path],
    source_roots: list[Path],
    subject_cache: dict[str, set[str]],
    schedule_for_evaluation: bool,
) -> list[EvidencePreflightRecord]:
    records = [
        _preflight_record(
            reference=reference,
            source_contract=source_contract,
            evidence_kind="holdout",
            evidence_set_id=None,
            manifest=manifest,
            context=context,
            evidence_roots=evidence_roots,
            source_roots=source_roots,
            subject_cache=subject_cache,
            schedule_for_evaluation=schedule_for_evaluation,
        )
    ]
    external_ids = external_evidence_set_ids(source_contract, manifest)
    if not external_ids:
        records.append(
            EvidencePreflightRecord(
                preflight_id=sha256_json(
                    {
                        "reference_id": reference.reference_id,
                        "evidence_kind": "external",
                        "evidence_set_id": None,
                        "source_contract": source_contract.model_dump(mode="json"),
                    }
                ),
                reference=reference,
                evidence_kind="external",
                status="unavailable",
                reason="No schema-compatible external evidence set is available.",
                reason_code="no_schema_compatible_external_set",
                interpretation_label="excluded_evidence_unavailable",
                source_contract=source_contract.model_dump(mode="json"),
            )
        )
    for evidence_set_id in external_ids:
        records.append(
            _preflight_record(
                reference=reference,
                source_contract=source_contract,
                evidence_kind="external",
                evidence_set_id=evidence_set_id,
                manifest=manifest,
                context=context,
                evidence_roots=evidence_roots,
                source_roots=source_roots,
                subject_cache=subject_cache,
                schedule_for_evaluation=schedule_for_evaluation,
            )
        )
    return records


def _execution_signature(record: EvidencePreflightRecord) -> str:
    if record.mapped_contract is None:
        raise ValueError("Cannot build an execution signature for an unmapped contract")
    payload = {
        "target_family": record.reference.target_family,
        "evidence_kind": record.evidence_kind,
        "evidence_set_id": record.evidence_set_id,
        "analysis_contract": _analysis_contract_payload(record.mapped_contract),
        "discovery_partition_id": record.discovery_partition_id,
        "replication_partition_ids": record.replication_partition_ids,
        "partition_hashes": record.partition_hashes,
    }
    return sha256_json(payload)


def build_evidence_preflight(
    out_dir: str | Path,
    evidence_manifest_path: str | Path,
    *,
    evidence_roots: Iterable[str | Path],
    source_roots: Iterable[str | Path],
    schedule_all_parents: bool = False,
) -> dict[str, Any]:
    """Preflight lineages and freeze the requested parent/candidate query tasks."""

    output = Path(out_dir)
    freeze_manifest = verify_frozen_sources(output)
    audit_implementation_hashes = implementation_hashes()
    audit_implementation_sha256 = sha256_json(audit_implementation_hashes)
    manifest_path = Path(evidence_manifest_path)
    manifest = load_evidence_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"Evidence manifest not found: {manifest_path}")
    evidence_root_paths = [Path(path) for path in evidence_roots]
    source_root_paths = [Path(path) for path in source_roots]
    context = CandidatePreflightContext.from_roots(evidence_root_paths)
    lineages = [FrozenLineage.model_validate(row) for row in read_jsonl(output / "frozen_lineages.jsonl")]
    supported_candidates: dict[str, list[FrozenCandidateExposure]] = defaultdict(list)
    for row in iter_jsonl(output / "frozen_search_inventory.jsonl"):
        if row.get("final_internal_supported"):
            exposure = FrozenCandidateExposure.model_validate(row)
            supported_candidates[exposure.lineage_event_id].append(exposure)
    subject_cache: dict[str, set[str]] = {}
    preflight_records: list[EvidencePreflightRecord] = []
    contract_preflight_cache: dict[str, list[EvidencePreflightRecord]] = {}

    def records_for_reference(
        reference: EvidenceReference,
        contract: ClaimContract,
        *,
        schedule: bool,
    ) -> list[EvidencePreflightRecord]:
        cache_key = sha256_json(
            {
                "target_family": reference.target_family,
                "contract": contract.model_dump(mode="json"),
            }
        )
        templates = contract_preflight_cache.get(cache_key)
        if templates is None:
            templates = _preflight_contract_evidence(
                reference=reference,
                source_contract=contract,
                manifest=manifest,
                context=context,
                evidence_roots=evidence_root_paths,
                source_roots=source_root_paths,
                subject_cache=subject_cache,
                schedule_for_evaluation=False,
            )
            contract_preflight_cache[cache_key] = templates
        cloned: list[EvidencePreflightRecord] = []
        for template in templates:
            preflight_id = sha256_json(
                {
                    "reference_id": reference.reference_id,
                    "evidence_kind": template.evidence_kind,
                    "evidence_set_id": template.evidence_set_id,
                    "source_contract": template.source_contract,
                }
            )
            cloned.append(
                template.model_copy(
                    update={
                        "preflight_id": preflight_id,
                        "reference": reference,
                        "schedule_for_evaluation": bool(schedule and template.status == "eligible"),
                    }
                )
            )
        return cloned

    for lineage in lineages:
        if schedule_all_parents:
            parent_reference = _reference(lineage, "parent")
            parent_contract = ClaimContract.model_validate(lineage.parent_contract)
            preflight_records.extend(records_for_reference(parent_reference, parent_contract, schedule=True))
        for candidate in sorted(
            supported_candidates.get(lineage.lineage_event_id, []),
            key=lambda item: (item.round_index, item.candidate_index, item.candidate_id),
        ):
            if candidate.effective_contract is None:
                raise ValueError(f"Supported candidate lacks effective contract: {candidate.candidate_id}")
            candidate_reference = _reference(lineage, "candidate", candidate=candidate)
            candidate_contract = ClaimContract.model_validate(candidate.effective_contract)
            preflight_records.extend(records_for_reference(candidate_reference, candidate_contract, schedule=True))

    task_groups: dict[str, list[EvidencePreflightRecord]] = defaultdict(list)
    for record in preflight_records:
        if record.schedule_for_evaluation and record.status == "eligible":
            task_groups[_execution_signature(record)].append(record)
    tasks: list[EvidenceQueryTask] = []
    for signature in sorted(task_groups):
        records = task_groups[signature]
        representative = records[0]
        references = sorted(
            (record.reference for record in records),
            key=lambda item: item.reference_id,
        )
        task_id = f"evidence_{signature[:24]}"
        tasks.append(
            EvidenceQueryTask(
                task_id=task_id,
                execution_signature=signature,
                implementation_sha256=audit_implementation_sha256,
                evidence_kind=representative.evidence_kind,
                evidence_set_id=representative.evidence_set_id,
                target_family=representative.reference.target_family,
                source_contract=representative.source_contract,
                mapped_contract=representative.mapped_contract or {},
                discovery_partition_id=str(representative.discovery_partition_id),
                replication_partition_ids=representative.replication_partition_ids,
                discovery_path=str(representative.discovery_path),
                replication_paths=representative.replication_paths,
                partition_hashes=representative.partition_hashes,
                references=references,
            )
        )

    expected_candidate_references = {
        f"{item.lineage_event_id}:candidate:{item.candidate_id}"
        for items in supported_candidates.values()
        for item in items
    }
    observed_candidate_references = {
        record.reference.reference_id
        for record in preflight_records
        if record.reference.role == "candidate"
    }
    if observed_candidate_references != expected_candidate_references:
        raise ValueError(
            "Every frozen internally supported candidate must have an excluded-evidence preflight record."
        )
    eligible_candidate_references = {
        record.reference.reference_id
        for record in preflight_records
        if record.reference.role == "candidate" and record.status == "eligible"
    }
    planned_candidate_references = {
        reference.reference_id
        for task in tasks
        for reference in task.references
        if reference.role == "candidate"
    }
    if planned_candidate_references != eligible_candidate_references:
        raise ValueError("The query plan does not exactly cover all compatible frozen candidates.")

    preflight_path = output / "evidence_preflight.jsonl"
    query_path = output / "evidence_query_plan.jsonl"
    write_jsonl_atomic(preflight_path, preflight_records)
    write_jsonl_atomic(query_path, tasks)
    status_counts = Counter(record.status for record in preflight_records)
    scheduled_reference_count = sum(record.schedule_for_evaluation for record in preflight_records)
    cnp_levels: dict[str, list[str]] = {}
    for partition_id in ("ds000030_EXTERNAL_DISC", "ds000030_EXTERNAL_REP"):
        info = context.resolve(partition_id)
        if info is not None:
            cnp_levels[partition_id] = context.levels(partition_id, "confirm_dx")
    cnp_configured = any(item.evidence_set_id == "psychosis_cnp_smri" for item in manifest.external_evidence_sets)
    if cnp_configured and (
        set(cnp_levels) != {"ds000030_EXTERNAL_DISC", "ds000030_EXTERNAL_REP"}
        or any(set(levels) != {"case", "control"} for levels in cnp_levels.values())
    ):
        raise ValueError(f"CNP virtual diagnosis mapping is incomplete: {cnp_levels}")

    reason_counts = Counter(record.reason_code or "none" for record in preflight_records)
    scheduled_counts = Counter(
        f"{record.reference.role}:{record.evidence_kind}"
        for record in preflight_records
        if record.schedule_for_evaluation
    )
    candidate_status_by_target = Counter(
        f"{record.reference.target_family}:{record.evidence_kind}:{record.status}"
        for record in preflight_records
        if record.reference.role == "candidate"
    )
    external_parent_eligible_by_target = Counter(
        record.reference.target_family
        for record in preflight_records
        if record.reference.role == "parent"
        and record.evidence_kind == "external"
        and record.status == "eligible"
    )

    summary = {
        **runtime_provenance(),
        "phase": "preflight",
        "outcome_blind": True,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"),
        "evidence_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "partition_hashes": {
            record.partition_id: record.content_sha256 or sha256_file(record.path)
            for record in manifest.records
        },
        "implementation_hashes": audit_implementation_hashes,
        "implementation_sha256": audit_implementation_sha256,
        "lineage_event_count": len(lineages),
        "final_internal_supported_candidate_count": sum(
            len(items) for items in supported_candidates.values()
        ),
        "parents_with_internal_support_count": len(supported_candidates),
        "schedule_all_parents": schedule_all_parents,
        "preflight_record_count": len(preflight_records),
        "preflight_status_counts": dict(sorted(status_counts.items())),
        "preflight_reason_counts": dict(sorted(reason_counts.items())),
        "scheduled_reference_count": scheduled_reference_count,
        "scheduled_reference_counts": dict(sorted(scheduled_counts.items())),
        "candidate_status_by_target": dict(sorted(candidate_status_by_target.items())),
        "external_parent_eligible_by_target": dict(sorted(external_parent_eligible_by_target.items())),
        "deduplicated_query_task_count": len(tasks),
        "cnp_confirm_dx_levels": cnp_levels,
        "evidence_preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
        "evidence_query_plan": {"path": str(query_path), "sha256": sha256_file(query_path)},
        "evidence_freshness": "previously_queried",
        "final_confirmation_eligible": False,
        "llm_calls_performed": 0,
        "freeze_observed_counts": freeze_manifest["observed_counts"],
    }
    write_json_atomic(output / "preflight_summary.json", summary)
    return summary


def _verify_query_plan(out_dir: Path) -> tuple[dict[str, Any], list[EvidenceQueryTask], str]:
    verify_frozen_sources(out_dir)
    summary_path = out_dir / "preflight_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = Path(summary["evidence_manifest"]["path"])
    if sha256_file(manifest_path) != summary["evidence_manifest"]["sha256"]:
        raise ValueError("Evidence manifest changed after outcome-blind preflight")
    observed_implementation = implementation_hashes()
    if observed_implementation != summary.get("implementation_hashes"):
        raise ValueError("Audit or CONFIRM implementation changed after outcome-blind preflight")
    observed_implementation_sha256 = sha256_json(observed_implementation)
    if observed_implementation_sha256 != summary.get("implementation_sha256"):
        raise ValueError("Audit implementation hash summary is inconsistent")
    manifest = load_evidence_manifest(manifest_path)
    if manifest is None:
        raise ValueError(f"Evidence manifest not found: {manifest_path}")
    record_by_partition = {record.partition_id: record for record in manifest.records}
    for partition_id, expected in summary.get("partition_hashes", {}).items():
        record = record_by_partition.get(partition_id)
        if record is None:
            raise ValueError(f"Evidence partition disappeared from manifest: {partition_id}")
        observed = sha256_file(record.path)
        if observed != expected:
            raise ValueError(
                f"Evidence partition changed after preflight: {partition_id} expected={expected} observed={observed}"
            )
    query_path = Path(summary["evidence_query_plan"]["path"])
    query_sha = sha256_file(query_path)
    if query_sha != summary["evidence_query_plan"]["sha256"]:
        raise ValueError("Evidence query plan changed after outcome-blind preflight")
    tasks = [EvidenceQueryTask.model_validate(row) for row in read_jsonl(query_path)]
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Evidence query plan contains duplicate task IDs")
    for task in tasks:
        if task.implementation_sha256 != observed_implementation_sha256:
            raise ValueError(f"Query task has a stale implementation hash: {task.task_id}")
        paths = [Path(task.discovery_path), *[Path(path) for path in task.replication_paths]]
        partition_ids = [task.discovery_partition_id, *task.replication_partition_ids]
        for partition_id, path in zip(partition_ids, paths):
            expected = task.partition_hashes.get(partition_id)
            observed = sha256_file(path)
            if expected != observed:
                raise ValueError(
                    f"Evidence partition changed after preflight: {partition_id} expected={expected} observed={observed}"
                )
    return summary, tasks, query_sha


def _evaluation_payload(task_payload: dict[str, Any], query_plan_sha256: str) -> dict[str, Any]:
    task = EvidenceQueryTask.model_validate(task_payload)
    contract = ClaimContract.model_validate(task.mapped_contract)
    source_contract = ClaimContract.model_validate(task.source_contract)
    roots = sorted(
        {
            str(Path(task.discovery_path).parent),
            *[str(Path(path).parent) for path in task.replication_paths],
        }
    )
    try:
        result = execute_contract(
            contract,
            roots,
            evidence_scope=task.evidence_kind,
            target_family=task.target_family,
            source_contract=source_contract,
            evidence_set_id=task.evidence_set_id,
        )
        raw_label = str(result.get("final_label") or "unknown")
        supported = raw_label == "confirmed"
        interpretation = (
            f"retrospective_{task.evidence_kind}_supported"
            if supported
            else f"retrospective_{task.evidence_kind}_not_supported"
        )
        body = {
            "task_id": task.task_id,
            "query_plan_sha256": query_plan_sha256,
            "status": "completed",
            "raw_gate_label": raw_label,
            "interpretation_label": interpretation,
            "gate_results": result.get("gate_results"),
            "error_type": None,
            "error": None,
            "evidence_kind": task.evidence_kind,
            "evidence_set_id": task.evidence_set_id,
            "evidence_freshness": "previously_queried",
            "final_confirmation_eligible": False,
        }
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        error_type = "analysis_non_identifiable" if "analysis_non_identifiable" in text else type(exc).__name__
        body = {
            "task_id": task.task_id,
            "query_plan_sha256": query_plan_sha256,
            "status": "error",
            "raw_gate_label": None,
            "interpretation_label": None,
            "gate_results": None,
            "error_type": error_type,
            "error": text,
            "evidence_kind": task.evidence_kind,
            "evidence_set_id": task.evidence_set_id,
            "evidence_freshness": "previously_queried",
            "final_confirmation_eligible": False,
        }
    body["result_sha256"] = sha256_json(body)
    return EvidenceEvaluation.model_validate(body).model_dump(mode="json")


def _checkpoint_path(out_dir: Path, task_id: str) -> Path:
    return out_dir / "checkpoints" / "evidence" / f"{task_id}.json"


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def _validated_evaluation(payload: dict[str, Any]) -> EvidenceEvaluation:
    expected = payload.get("result_sha256")
    body = {key: value for key, value in payload.items() if key != "result_sha256"}
    if not expected or sha256_json(body) != expected:
        raise ValueError("Evidence evaluation result hash is invalid")
    return EvidenceEvaluation.model_validate(payload)


def evaluate_query_plan(
    out_dir: str | Path,
    *,
    max_workers: int = 1,
    parallel_backend: Literal["process", "thread"] = "process",
    progress: bool = True,
) -> dict[str, Any]:
    """Execute every predeclared task once, with deterministic resumable output."""

    output = Path(out_dir)
    preflight_summary, tasks, query_sha = _verify_query_plan(output)
    completed: dict[str, EvidenceEvaluation] = {}
    pending: list[EvidenceQueryTask] = []
    for task in tasks:
        checkpoint = _checkpoint_path(output, task.task_id)
        if checkpoint.exists():
            evaluation = _validated_evaluation(json.loads(checkpoint.read_text(encoding="utf-8")))
            if evaluation.task_id != task.task_id:
                raise ValueError(f"Checkpoint task ID mismatch: {checkpoint}")
            if evaluation.query_plan_sha256 != query_sha:
                raise ValueError(f"Checkpoint belongs to a different query plan: {checkpoint}")
            completed[task.task_id] = evaluation
        else:
            pending.append(task)

    def record(payload: dict[str, Any], expected_task_id: str) -> None:
        evaluation = _validated_evaluation(payload)
        if evaluation.task_id != expected_task_id:
            raise ValueError(
                f"Evidence worker returned task {evaluation.task_id}, expected {expected_task_id}"
            )
        _write_checkpoint(_checkpoint_path(output, evaluation.task_id), evaluation.model_dump(mode="json"))
        completed[evaluation.task_id] = evaluation
        if progress:
            print(f"[evidence] {len(completed)}/{len(tasks)} task={evaluation.task_id} status={evaluation.status}", flush=True)

    if max_workers <= 1:
        for task in pending:
            record(_evaluation_payload(task.model_dump(mode="json"), query_sha), task.task_id)
    else:
        executor_class = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
        with executor_class(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluation_payload, task.model_dump(mode="json"), query_sha): task.task_id
                for task in pending
            }
            for future in as_completed(futures):
                record(future.result(), futures[future])

    if set(completed) != {task.task_id for task in tasks}:
        raise ValueError("Not every frozen query task produced exactly one checkpoint")
    ordered = [completed[task.task_id] for task in tasks]
    evaluation_path = output / "evidence_evaluations.jsonl"
    write_jsonl_atomic(evaluation_path, ordered)

    ledger_rows: list[dict[str, Any]] = []
    previous_hash = "0" * 64
    for index, evaluation in enumerate(ordered, start=1):
        body = {
            "query_index": index,
            "task_id": evaluation.task_id,
            "query_plan_sha256": query_sha,
            "result_sha256": evaluation.result_sha256,
            "previous_ledger_sha256": previous_hash,
        }
        ledger_hash = sha256_json(body)
        ledger_rows.append({**body, "ledger_sha256": ledger_hash})
        previous_hash = ledger_hash
    ledger_path = output / "excluded_query_ledger.jsonl"
    write_jsonl_atomic(ledger_path, ledger_rows)

    status_counts = Counter(item.status for item in ordered)
    label_counts = Counter(item.interpretation_label or "none" for item in ordered)
    summary = {
        **runtime_provenance(),
        "phase": "evaluate",
        "query_plan_sha256": query_sha,
        "preflight_summary_sha256": sha256_file(output / "preflight_summary.json"),
        "task_count": len(tasks),
        "resumed_checkpoint_count": len(tasks) - len(pending),
        "newly_executed_count": len(pending),
        "status_counts": dict(sorted(status_counts.items())),
        "interpretation_label_counts": dict(sorted(label_counts.items())),
        "evidence_evaluations": {"path": str(evaluation_path), "sha256": sha256_file(evaluation_path)},
        "excluded_query_ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
        "ledger_terminal_sha256": previous_hash,
        "evidence_freshness": "previously_queried",
        "final_confirmation_eligible": False,
        "llm_calls_performed": 0,
        "preflight_task_count": preflight_summary["deduplicated_query_task_count"],
    }
    write_json_atomic(output / "evaluation_summary.json", summary)
    write_json_atomic(output / "run_provenance.json", summary)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    import csv

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clustered_rate_interval(
    rows: list[dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    seed: int = 20260717,
    samples: int = 2000,
) -> tuple[float | None, float | None]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_parent[str(row["parent_claim_id"])].append(row)
    parents = sorted(by_parent)
    if not parents or sum(float(row.get(denominator, 0)) for row in rows) <= 0:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(parents) for _ in parents]
        numerator_total = sum(
            float(row.get(numerator, 0))
            for parent in sampled
            for row in by_parent[parent]
        )
        denominator_total = sum(
            float(row.get(denominator, 0))
            for parent in sampled
            for row in by_parent[parent]
        )
        if denominator_total > 0:
            estimates.append(numerator_total / denominator_total)
    if not estimates:
        return None, None
    estimates.sort()
    lower = estimates[max(0, math.floor(0.025 * (len(estimates) - 1)))]
    upper = estimates[min(len(estimates) - 1, math.ceil(0.975 * (len(estimates) - 1)))]
    return lower, upper


def _supported(result: EvidenceEvaluation | None) -> bool:
    return bool(result and result.status == "completed" and result.raw_gate_label == "confirmed")


def _evaluated(result: EvidenceEvaluation | None) -> bool:
    return bool(result and result.status == "completed")


def _transform_audit_row(
    exposure: FrozenCandidateExposure,
    lineage: FrozenLineage,
) -> dict[str, Any]:
    parent = ClaimContract.model_validate(exposure.parent_contract)
    raw_contract_payload = exposure.parsed_proposal.get("proposed_contract")
    raw_contract = (
        ClaimContract.model_validate(raw_contract_payload)
        if isinstance(raw_contract_payload, dict)
        else None
    )
    changed = (
        _changed_paths(
            _analysis_contract_payload(parent.model_dump(mode="json")),
            _analysis_contract_payload(raw_contract.model_dump(mode="json")),
        )
        if raw_contract is not None
        else []
    )
    transform = exposure.transform_type
    immutable_prefixes = (
        "estimand.predictor",
        "estimand.group",
        "estimand.direction",
        "covariates",
        "discovery_cohort",
        "replication_cohorts",
        "gates",
    )
    permitted_prefixes: tuple[str, ...]
    if transform == "narrower_outcome_family":
        permitted_prefixes = ("estimand.outcome", "estimand.region_set", "search_provenance.family_size")
    elif transform == "moderator_or_subgroup":
        permitted_prefixes = ("inclusion", "search_provenance.family_size")
    elif transform in {"stronger_design", "fixed_estimand"}:
        permitted_prefixes = ("search_provenance.family_size", "search_provenance.selection")
    elif transform == "contract_correction":
        permitted_prefixes = tuple()
    else:
        permitted_prefixes = tuple()
    forbidden_changes = [
        path
        for path in changed
        if any(path.startswith(prefix) for prefix in immutable_prefixes)
        and not any(path.startswith(prefix) for prefix in permitted_prefixes)
    ]
    failure_kind = str((lineage.failure_localization or {}).get("failure_kind") or "unknown")
    if transform == "contract_correction":
        structural = "adherent" if failure_kind == "contract_error" else "non_adherent"
    elif raw_contract is None:
        structural = "non_adherent"
    elif transform == "narrower_outcome_family":
        structural = "adherent" if parent.estimand.outcome != raw_contract.estimand.outcome and not forbidden_changes else "non_adherent"
    elif transform == "moderator_or_subgroup":
        structural = "adherent" if parent.inclusion != raw_contract.inclusion and not forbidden_changes else "non_adherent"
    else:
        structural = "adherent" if not forbidden_changes else "non_adherent"

    semantic: str
    if transform == "fixed_estimand":
        semantic = "not_assessable"
    elif raw_contract is None or structural == "non_adherent":
        semantic = "non_adherent"
    elif transform == "stronger_design":
        raw_split = str(exposure.parsed_proposal.get("validation_split") or "")
        semantic = "adherent" if raw_split in {"excluded_validation", "future_required"} else "semantic_unknown"
    elif transform == "narrower_outcome_family":
        semantic = (
            "semantic_unknown"
            if contract_feature_scope(parent)[0] == contract_feature_scope(raw_contract)[0]
            else "non_adherent"
        )
    elif transform in {"moderator_or_subgroup", "contract_correction"}:
        semantic = "semantic_unknown"
    else:
        semantic = "semantic_unknown"
    return {
        "exposure_id": exposure.exposure_id,
        "arm_id": exposure.arm_id,
        "parent_claim_id": exposure.parent_claim_id,
        "candidate_id": exposure.candidate_id,
        "target_family": exposure.target_family,
        "source_mode": exposure.source_mode,
        "round_index": exposure.round_index,
        "generation_status": exposure.generation_status,
        "transform_type": transform,
        "declared_transform": exposure.declared_transform or transform,
        "inferred_transform": exposure.inferred_transform,
        "transform_match": exposure.transform_match,
        "executable_contract_delta": json.dumps(exposure.executable_contract_delta, sort_keys=True),
        "failure_kind": failure_kind,
        "structural_adherence": structural,
        "semantic_adherence": semantic,
        "raw_changed_fields": json.dumps(changed, sort_keys=True),
        "forbidden_changed_fields": json.dumps(forbidden_changes, sort_keys=True),
        "pipeline_contract_normalization": json.dumps(exposure.contract_normalization_changes, sort_keys=True),
        "pipeline_proposal_normalization": json.dumps(exposure.proposal_normalization_changes, sort_keys=True),
        "provisional_internal_supported": exposure.provisional_internal_supported,
        "final_internal_supported": exposure.final_internal_supported,
        "multiplicity_retracted": exposure.multiplicity_retracted,
    }


def _load_verified_evidence_results(
    output: Path,
    tasks: list[EvidenceQueryTask],
    query_sha: str,
) -> tuple[dict[str, Any], dict[str, EvidenceEvaluation]]:
    summary_path = output / "evaluation_summary.json"
    evaluation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if evaluation_summary.get("query_plan_sha256") != query_sha:
        raise ValueError("Evaluation summary does not match the frozen query plan")
    if evaluation_summary.get("preflight_summary_sha256") != sha256_file(
        output / "preflight_summary.json"
    ):
        raise ValueError("Evaluation summary does not match the frozen preflight summary")
    if evaluation_summary.get("task_count") != len(tasks):
        raise ValueError("Evaluation summary task count does not match the query plan")

    evaluation_path = Path(evaluation_summary["evidence_evaluations"]["path"])
    if sha256_file(evaluation_path) != evaluation_summary["evidence_evaluations"]["sha256"]:
        raise ValueError("Evidence evaluations changed after execution")
    evaluation_rows = read_jsonl(evaluation_path)
    evaluations = [_validated_evaluation(row) for row in evaluation_rows]
    expected_task_ids = [task.task_id for task in tasks]
    observed_task_ids = [evaluation.task_id for evaluation in evaluations]
    if observed_task_ids != expected_task_ids:
        raise ValueError("Evidence evaluations are not in frozen query-plan order")
    if any(evaluation.query_plan_sha256 != query_sha for evaluation in evaluations):
        raise ValueError("Evidence evaluation belongs to a different query plan")

    ledger_path = Path(evaluation_summary["excluded_query_ledger"]["path"])
    if sha256_file(ledger_path) != evaluation_summary["excluded_query_ledger"]["sha256"]:
        raise ValueError("Excluded-evidence query ledger changed after execution")
    ledger_rows = read_jsonl(ledger_path)
    if len(ledger_rows) != len(tasks):
        raise ValueError("Excluded-evidence query ledger length does not match the query plan")
    previous_hash = "0" * 64
    for index, (ledger, evaluation) in enumerate(zip(ledger_rows, evaluations), start=1):
        body = {
            "query_index": index,
            "task_id": evaluation.task_id,
            "query_plan_sha256": query_sha,
            "result_sha256": evaluation.result_sha256,
            "previous_ledger_sha256": previous_hash,
        }
        if any(ledger.get(key) != value for key, value in body.items()):
            raise ValueError(f"Excluded-evidence ledger entry {index} is inconsistent")
        ledger_hash = sha256_json(body)
        if ledger.get("ledger_sha256") != ledger_hash:
            raise ValueError(f"Excluded-evidence ledger entry {index} has an invalid hash")
        previous_hash = ledger_hash
    if previous_hash != evaluation_summary.get("ledger_terminal_sha256"):
        raise ValueError("Excluded-evidence ledger terminal hash is inconsistent")

    status_counts = dict(sorted(Counter(item.status for item in evaluations).items()))
    label_counts = dict(
        sorted(Counter(item.interpretation_label or "none" for item in evaluations).items())
    )
    if status_counts != evaluation_summary.get("status_counts"):
        raise ValueError("Evaluation status counts are inconsistent")
    if label_counts != evaluation_summary.get("interpretation_label_counts"):
        raise ValueError("Evaluation interpretation counts are inconsistent")
    provenance_path = output / "run_provenance.json"
    if json.loads(provenance_path.read_text(encoding="utf-8")) != evaluation_summary:
        raise ValueError("Run provenance does not match the evaluation summary")
    return evaluation_summary, {item.task_id: item for item in evaluations}


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    """Return a two-sided 95% Wilson interval, including boundary cases."""

    if total <= 0:
        return None, None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + (z * z / total)
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((rate * (1.0 - rate) / total) + (z * z / (4.0 * total * total)))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_initial_claim_evidence(out_dir: str | Path) -> dict[str, Any]:
    """Summarize retrospective excluded-evidence results for all initial claims."""

    output = Path(out_dir)
    freeze_manifest = verify_frozen_sources(output)
    if freeze_manifest.get("audit_type") != "initial_claims":
        raise ValueError("Frozen artifacts are not an initial-claim evidence audit")
    _, tasks, query_sha = _verify_query_plan(output)
    evaluation_summary, evaluations = _load_verified_evidence_results(output, tasks, query_sha)
    lineages = [
        FrozenLineage.model_validate(row)
        for row in read_jsonl(output / "frozen_lineages.jsonl")
    ]
    preflight = [
        EvidencePreflightRecord.model_validate(row)
        for row in read_jsonl(output / "evidence_preflight.jsonl")
    ]

    result_by_reference: dict[tuple[str, str, str | None], EvidenceEvaluation] = {}
    for task in tasks:
        result = evaluations[task.task_id]
        for reference in task.references:
            result_by_reference[(reference.reference_id, task.evidence_kind, task.evidence_set_id)] = result
    preflight_by_reference: dict[str, list[EvidencePreflightRecord]] = defaultdict(list)
    for record in preflight:
        preflight_by_reference[record.reference.reference_id].append(record)

    claim_rows: list[dict[str, Any]] = []
    for lineage in lineages:
        reference_id = f"{lineage.lineage_event_id}:parent:{lineage.parent_claim_id}"
        records = preflight_by_reference.get(reference_id, [])
        holdout_records = [record for record in records if record.evidence_kind == "holdout"]
        if len(holdout_records) != 1:
            raise ValueError(
                f"Expected exactly one holdout preflight record for {lineage.parent_claim_id}"
            )
        holdout_preflight = holdout_records[0]
        holdout_result = result_by_reference.get((reference_id, "holdout", None))
        external_records = [record for record in records if record.evidence_kind == "external"]
        eligible_external_records = [record for record in external_records if record.status == "eligible"]
        external_results = [
            result_by_reference[(reference_id, "external", record.evidence_set_id)]
            for record in eligible_external_records
            if (reference_id, "external", record.evidence_set_id) in result_by_reference
        ]
        initial_label = str(lineage.source_metadata.get("initial_label") or "unknown")
        external_supported = any(_supported(result) for result in external_results)
        external_completed = [result for result in external_results if _evaluated(result)]
        row = {
            "claim_id": lineage.parent_claim_id,
            "target_family": lineage.target_family,
            "source_mode": lineage.source_mode,
            "initial_label": initial_label,
            "initial_confirmed": int(initial_label == "confirmed"),
            "holdout_preflight_status": holdout_preflight.status,
            "holdout_preflight_reason_code": holdout_preflight.reason_code,
            "holdout_preflight_reason": holdout_preflight.reason,
            "holdout_eligible": int(holdout_preflight.status == "eligible"),
            "holdout_evaluated": int(_evaluated(holdout_result)),
            "holdout_execution_error": int(
                bool(holdout_result and holdout_result.status == "error")
            ),
            "holdout_raw_gate_label": (
                holdout_result.raw_gate_label if holdout_result is not None else None
            ),
            "holdout_supported": int(_supported(holdout_result)),
            "holdout_discovery_partition": holdout_preflight.discovery_partition_id,
            "holdout_replication_partitions": json.dumps(
                holdout_preflight.replication_partition_ids,
                sort_keys=True,
            ),
            "external_preflight_statuses": json.dumps(
                Counter(record.status for record in external_records),
                sort_keys=True,
            ),
            "external_eligible": int(bool(eligible_external_records)),
            "external_eligible_pair_count": len(eligible_external_records),
            "external_evaluated": int(bool(external_completed)),
            "external_evaluated_pair_count": len(external_completed),
            "external_execution_error": int(
                any(result.status == "error" for result in external_results)
            ),
            "external_supported": int(external_supported),
            "external_supported_pair_count": sum(
                int(_supported(result)) for result in external_results
            ),
            "external_raw_gate_labels": json.dumps(
                [result.raw_gate_label for result in external_results],
                sort_keys=True,
            ),
            "external_evidence_set_ids": json.dumps(
                [record.evidence_set_id for record in eligible_external_records],
                sort_keys=True,
            ),
            "evidence_freshness": "previously_queried",
            "final_confirmation_eligible": False,
        }
        claim_rows.append(row)

    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "initial_claim_count": len(rows),
            "initial_confirmed_count": sum(row["initial_confirmed"] for row in rows),
            "holdout_eligible_count": sum(row["holdout_eligible"] for row in rows),
            "holdout_evaluated_count": sum(row["holdout_evaluated"] for row in rows),
            "holdout_execution_error_count": sum(
                row["holdout_execution_error"] for row in rows
            ),
            "holdout_supported_count": sum(row["holdout_supported"] for row in rows),
            "external_eligible_count": sum(row["external_eligible"] for row in rows),
            "external_evaluated_count": sum(row["external_evaluated"] for row in rows),
            "external_execution_error_count": sum(
                row["external_execution_error"] for row in rows
            ),
            "external_supported_count": sum(row["external_supported"] for row in rows),
        }

    overall = summarize_rows(claim_rows)
    holdout_interval = _wilson_interval(
        overall["holdout_supported_count"],
        overall["holdout_evaluated_count"],
    )
    external_interval = _wilson_interval(
        overall["external_supported_count"],
        overall["external_evaluated_count"],
    )
    overall.update(
        {
            "internal_confirmation_rate": (
                overall["initial_confirmed_count"] / overall["initial_claim_count"]
                if overall["initial_claim_count"]
                else None
            ),
            "holdout_support_rate_among_evaluated": (
                overall["holdout_supported_count"] / overall["holdout_evaluated_count"]
                if overall["holdout_evaluated_count"]
                else None
            ),
            "holdout_support_ci_low": holdout_interval[0],
            "holdout_support_ci_high": holdout_interval[1],
            "external_support_rate_among_evaluated": (
                overall["external_supported_count"] / overall["external_evaluated_count"]
                if overall["external_evaluated_count"]
                else None
            ),
            "external_support_ci_low": external_interval[0],
            "external_support_ci_high": external_interval[1],
        }
    )

    matched_holdout = Counter()
    matched_external = Counter()
    for row in claim_rows:
        if row["holdout_evaluated"]:
            matched_holdout[
                f"internal_{row['initial_confirmed']}_holdout_{row['holdout_supported']}"
            ] += 1
        if row["external_evaluated"]:
            matched_external[
                f"internal_{row['initial_confirmed']}_external_{row['external_supported']}"
            ] += 1

    stratified_rows: list[dict[str, Any]] = []
    for dimension in ("target_family", "source_mode", "initial_label"):
        for value in sorted({str(row[dimension]) for row in claim_rows}):
            rows = [row for row in claim_rows if str(row[dimension]) == value]
            stratified_rows.append(
                {
                    "dimension": dimension,
                    "dimension_value": value,
                    **summarize_rows(rows),
                }
            )

    evidence_rows_path = output / "initial_claim_evidence.jsonl"
    evidence_csv_path = output / "initial_claim_evidence.csv"
    stratified_path = output / "initial_claim_stratified_summary.csv"
    write_jsonl_atomic(evidence_rows_path, claim_rows)
    _write_csv(evidence_csv_path, claim_rows)
    _write_csv(stratified_path, stratified_rows)
    task_counts = Counter(task.evidence_kind for task in tasks)
    summary = {
        **runtime_provenance(),
        "phase": "summarize",
        "audit_type": "initial_claims",
        "query_plan_sha256": query_sha,
        "freeze_manifest_sha256": sha256_file(output / "freeze_manifest.json"),
        "preflight_summary_sha256": sha256_file(output / "preflight_summary.json"),
        "evaluation_summary_sha256": sha256_file(output / "evaluation_summary.json"),
        "overall": overall,
        "matched_internal_holdout_counts": dict(sorted(matched_holdout.items())),
        "matched_internal_external_counts": dict(sorted(matched_external.items())),
        "deduplicated_query_task_counts": dict(sorted(task_counts.items())),
        "stratified_summary": stratified_rows,
        "evidence_freshness": "previously_queried",
        "final_confirmation_eligible": False,
        "llm_calls_performed": 0,
        "evaluation_status_counts": evaluation_summary["status_counts"],
        "artifacts": {
            "initial_claim_evidence": {
                "path": str(evidence_rows_path),
                "sha256": sha256_file(evidence_rows_path),
            },
            "initial_claim_evidence_csv": {
                "path": str(evidence_csv_path),
                "sha256": sha256_file(evidence_csv_path),
            },
            "initial_claim_stratified_summary": {
                "path": str(stratified_path),
                "sha256": sha256_file(stratified_path),
            },
        },
        "allowed_conclusions": [
            "retrospective initial-claim holdout concordance",
            "retrospective initial-claim external concordance",
            "matched internal-versus-excluded evidence outcomes",
        ],
        "disallowed_conclusions": [
            "fresh confirmation",
            "prospective independent validation",
        ],
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def summarize_evidence_audit(
    out_dir: str | Path,
    *,
    initial_evidence_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize every final internal pass without selecting across arms."""

    output = Path(out_dir)
    freeze_manifest = verify_frozen_sources(output)
    _, tasks, query_sha = _verify_query_plan(output)
    _, evaluations = _load_verified_evidence_results(output, tasks, query_sha)
    lineages = [FrozenLineage.model_validate(row) for row in read_jsonl(output / "frozen_lineages.jsonl")]
    supported = [
        FrozenCandidateExposure.model_validate(row)
        for row in iter_jsonl(output / "frozen_search_inventory.jsonl")
        if row.get("final_internal_supported")
    ]
    lineage_by_id = {item.lineage_event_id: item for item in lineages}
    exposure_by_reference = {
        f"{item.lineage_event_id}:candidate:{item.candidate_id}": item for item in supported
    }
    preflight = [EvidencePreflightRecord.model_validate(row) for row in read_jsonl(output / "evidence_preflight.jsonl")]
    result_by_reference: dict[tuple[str, str, str | None], EvidenceEvaluation] = {}
    task_by_reference: dict[tuple[str, str, str | None], EvidenceQueryTask] = {}
    for task in tasks:
        result = evaluations[task.task_id]
        for reference in task.references:
            key = (reference.reference_id, task.evidence_kind, task.evidence_set_id)
            result_by_reference[key] = result
            task_by_reference[key] = task

    initial_rows, initial_provenance = _load_initial_evidence(initial_evidence_dir)
    evidence_rows: list[dict[str, Any]] = []
    for record in preflight:
        if record.reference.role != "candidate":
            continue
        exposure = exposure_by_reference.get(record.reference.reference_id)
        if exposure is None:
            raise ValueError(f"Preflight references a non-frozen supported candidate: {record.reference.reference_id}")
        key = (record.reference.reference_id, record.evidence_kind, record.evidence_set_id)
        result = result_by_reference.get(key)
        task = task_by_reference.get(key)
        parent_result = initial_rows.get(exposure.parent_claim_id)
        matched_parent_holdout = bool(
            record.evidence_kind == "holdout"
            and parent_result
            and task
            and parent_result.get("holdout_discovery_partition") == task.discovery_partition_id
            and json.loads(parent_result.get("holdout_replication_partitions") or "[]")
            == task.replication_partition_ids
        )
        evidence_rows.append(
            {
                "lineage_event_id": exposure.lineage_event_id,
                "arm_id": exposure.arm_id,
                "parent_claim_id": exposure.parent_claim_id,
                "candidate_id": exposure.candidate_id,
                "target_family": exposure.target_family,
                "source_mode": exposure.source_mode,
                "declared_transform": exposure.declared_transform or exposure.transform_type,
                "inferred_transform": exposure.inferred_transform,
                "transform_match": exposure.transform_match,
                "effective_family_size": exposure.effective_family_size,
                "exact_contract_id": exposure.exact_contract_id,
                "semantic_cluster_id": exposure.semantic_cluster_id,
                "evidence_kind": record.evidence_kind,
                "evidence_set_id": record.evidence_set_id,
                "preflight_status": record.status,
                "preflight_reason_code": record.reason_code,
                "compatible": int(record.status == "eligible"),
                "evaluation_attempted": int(result is not None),
                "evaluated": int(_evaluated(result)),
                "execution_error": int(bool(result and result.status == "error")),
                "raw_gate_label": result.raw_gate_label if result else None,
                "interpretation_label": result.interpretation_label if result else record.interpretation_label,
                "supported": int(_supported(result)),
                "evidence_freshness": "previously_queried",
                "final_confirmation_eligible": False,
                "parent_initial_label": parent_result.get("initial_label") if parent_result else None,
                "parent_holdout_evaluated": int(bool(parent_result and parent_result.get("holdout_evaluated"))),
                "parent_holdout_supported": int(bool(parent_result and parent_result.get("holdout_supported"))),
                "matched_parent_holdout": int(matched_parent_holdout),
                "candidate_only_holdout_support": int(
                    matched_parent_holdout
                    and _supported(result)
                    and bool(parent_result and parent_result.get("holdout_evaluated"))
                    and not bool(parent_result and parent_result.get("holdout_supported"))
                ),
            }
        )

    frozen_arm_by_id = {str(item["arm_id"]): item for item in freeze_manifest["arm_summaries"]}
    arm_rows = [
        _aggregate_evidence_group(
            arm_id,
            [item for item in lineages if item.arm_id == arm_id],
            [item for item in supported if item.arm_id == arm_id],
            [row for row in evidence_rows if row["arm_id"] == arm_id],
        )
        for arm_id in sorted(frozen_arm_by_id)
    ]
    for row in arm_rows:
        frozen_arm = frozen_arm_by_id[row["arm_id"]]
        for field in (
            "llm_response_count",
            "proposals_returned_count",
            "schema_valid_candidate_count",
            "policy_valid_candidate_count",
            "generated_candidate_count",
            "retained_candidate_count",
            "duplicate_candidate_count",
            "unretained_generated_candidate_count",
            "current_data_evaluated_count",
            "execution_complete_candidate_count",
            "provisional_internal_pass_count",
        ):
            if int(row[field] or 0) != int(frozen_arm.get(field) or 0):
                raise ValueError(
                    f"Frozen arm denominator mismatch for {row['arm_id']} field={field}: "
                    f"lineages={row[field]} freeze={frozen_arm.get(field)}"
                )

    stratified_rows: list[dict[str, Any]] = []
    dimensions = {
        "target_family": lambda item: item.target_family,
        "source_mode": lambda item: item.source_mode,
        "synthetic_failure_family": lambda item: str(
            item.source_metadata.get("synthetic_failure_family") or "not_applicable"
        ),
    }
    for dimension, value_for in dimensions.items():
        keys = sorted({(item.arm_id, str(value_for(item))) for item in lineages})
        for arm_id, value in keys:
            group_lineages = [
                item
                for item in lineages
                if item.arm_id == arm_id and str(value_for(item)) == value
            ]
            lineage_ids = {item.lineage_event_id for item in group_lineages}
            group_supported = [
                item for item in supported if item.lineage_event_id in lineage_ids
            ]
            group_evidence = [
                row for row in evidence_rows if row["lineage_event_id"] in lineage_ids
            ]
            row = _aggregate_evidence_group(
                arm_id,
                group_lineages,
                group_supported,
                group_evidence,
            )
            row.update({"dimension": dimension, "dimension_value": value})
            stratified_rows.append(row)

    external_dataset_rows: list[dict[str, Any]] = []
    external_keys = sorted(
        {
            (row["arm_id"], str(row.get("evidence_set_id") or "unavailable"), row["target_family"])
            for row in evidence_rows
            if row["evidence_kind"] == "external"
        }
    )
    for arm_id, evidence_set_id, target_family in external_keys:
        rows = [
            row for row in evidence_rows
            if row["arm_id"] == arm_id
            and row["evidence_kind"] == "external"
            and str(row.get("evidence_set_id") or "unavailable") == evidence_set_id
            and row["target_family"] == target_family
        ]
        evaluated = sum(row["evaluated"] for row in rows)
        external_dataset_rows.append(
            {
                "arm_id": arm_id,
                "evidence_set_id": evidence_set_id,
                "target_family": target_family,
                "compatible_candidate_count": sum(row["compatible"] for row in rows),
                "evaluated_candidate_count": evaluated,
                "supported_candidate_count": sum(row["supported"] for row in rows),
                "conditional_survival_rate": (
                    sum(row["supported"] for row in rows) / evaluated if evaluated else None
                ),
                "unavailable_count": sum(not row["compatible"] for row in rows),
                "execution_error_count": sum(row["execution_error"] for row in rows),
            }
        )

    audit_rows: list[dict[str, Any]] = []
    evidence_by_exposure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_exposure[f"{row['lineage_event_id']}:{row['candidate_id']}"].append(row)
    preference_groups: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    for exposure_row in iter_jsonl(output / "frozen_search_inventory.jsonl"):
        exposure = FrozenCandidateExposure.model_validate(exposure_row)
        audit_rows.append(_transform_audit_row(exposure, lineage_by_id[exposure.lineage_event_id]))
        dimensions = [
            ("overall", "all"),
            ("round", str(exposure.round_index)),
            ("target_family", exposure.target_family),
            ("source_mode", exposure.source_mode),
        ]
        failure_kind = str((lineage_by_id[exposure.lineage_event_id].failure_localization or {}).get("failure_kind") or "unknown")
        dimensions.append(("failure_kind", failure_kind))
        linked = evidence_by_exposure.get(f"{exposure.lineage_event_id}:{exposure.candidate_id}", [])
        for dimension, value in dimensions:
            key = (
                exposure.arm_id,
                dimension,
                value,
                str(exposure.declared_transform or exposure.transform_type),
                str(exposure.inferred_transform or "unknown"),
            )
            counter = preference_groups[key]
            counter["generated"] += 1
            counter["retained_unique"] += int(exposure.generation_status == "retained")
            counter["policy_valid"] += int(exposure.validation_ok is True)
            counter["provisional_internal_supported"] += int(exposure.provisional_internal_supported)
            counter["final_internal_supported"] += int(exposure.final_internal_supported)
            counter["multiplicity_retracted"] += int(exposure.multiplicity_retracted)
            counter["holdout_evaluated"] += sum(row["evaluated"] for row in linked if row["evidence_kind"] == "holdout")
            counter["holdout_supported"] += sum(row["supported"] for row in linked if row["evidence_kind"] == "holdout")
            counter["external_evaluated"] += sum(row["evaluated"] for row in linked if row["evidence_kind"] == "external")
            counter["external_supported"] += sum(row["supported"] for row in linked if row["evidence_kind"] == "external")
    preference_rows = [
        {
            "arm_id": key[0],
            "dimension": key[1],
            "dimension_value": key[2],
            "declared_transform": key[3],
            "inferred_transform": key[4],
            **dict(counts),
        }
        for key, counts in sorted(preference_groups.items())
    ]

    paths = {
        "candidate_evidence": output / "candidate_evidence.csv",
        "arm_summary": output / "arm_summary.csv",
        "stratified_summary": output / "stratified_summary.csv",
        "external_dataset_summary": output / "external_dataset_summary.csv",
        "transform_audit": output / "transform_audit.csv",
        "transform_preference": output / "transform_preference.csv",
        "case_studies": output / "case_studies.md",
    }
    write_jsonl_atomic(output / "candidate_evidence.jsonl", evidence_rows)
    _write_csv(paths["candidate_evidence"], evidence_rows)
    _write_csv(paths["arm_summary"], arm_rows)
    _write_csv(paths["stratified_summary"], stratified_rows)
    _write_csv(paths["external_dataset_summary"], external_dataset_rows)
    _write_csv(paths["transform_audit"], audit_rows)
    _write_csv(paths["transform_preference"], preference_rows)
    _write_case_studies_v2(paths["case_studies"], evidence_rows, supported)

    known_negative_lineage_count = sum(_known_negative_lineage(item) for item in lineages)
    summary = {
        **runtime_provenance(),
        "phase": "summarize",
        "primary_configuration": None,
        "arm_count": len(arm_rows),
        "final_internal_supported_candidate_count": len(supported),
        "parents_with_internal_support_count": len({(item.arm_id, item.parent_claim_id) for item in supported}),
        "query_plan_sha256": query_sha,
        "initial_claim_evidence": initial_provenance,
        "evidence_freshness": "previously_queried",
        "final_confirmation_eligible": False,
        "causal_interpretation_allowed": False,
        "known_negative_safety_audit": bool(lineages)
        and known_negative_lineage_count == len(lineages),
        "known_negative_safety_established": False,
        "known_negative_lineage_count": known_negative_lineage_count,
        "external_scope": "Compatible external evidence is reported independently by evidence_set_id.",
        "arm_summary": arm_rows,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()
        },
        "allowed_conclusions": [
            "retrospective evidence concordance",
            "per-arm internally supported candidate yield",
            "conditional holdout and per-dataset external survival",
            "candidate-only holdout support when linked to identical initial-claim evidence",
            "descriptive transform usage and arm differences",
        ],
        "disallowed_conclusions": [
            "independent validation",
            "fresh confirmation",
            "causal feedback-loop improvement",
            "optimal search budget",
            "pooled best-claim or cross-arm winner claims",
        ],
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def _load_initial_evidence(
    initial_evidence_dir: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if initial_evidence_dir is None:
        return {}, {"available": False}
    root = Path(initial_evidence_dir)
    path = root / "initial_claim_evidence.jsonl"
    if not path.exists():
        raise ValueError(f"Initial-claim evidence artifact not found: {path}")
    rows = read_jsonl(path)
    return (
        {str(row["claim_id"]): row for row in rows},
        {"available": True, "path": str(path), "sha256": sha256_file(path), "row_count": len(rows)},
    )


def _known_negative_lineage(lineage: FrozenLineage) -> bool:
    metadata = lineage.source_metadata
    labels = {
        str(metadata.get(key) or "").lower()
        for key in (
            "source_scoring_label",
            "source_label_class",
            "source_ground_truth",
            "scoring_label",
            "label_class",
            "ground_truth",
        )
    }
    return lineage.parent_claim_id.startswith("neg_") or bool(
        labels
        & {
            "known_null",
            "null_expected",
            "random_null",
            "fragile",
            "underpowered",
            "under_powered",
            "non_replicated",
        }
    )


def _aggregate_evidence_group(
    arm_id: str,
    lineages: list[FrozenLineage],
    supported: list[FrozenCandidateExposure],
    evidence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_count = len(lineages)
    supported_parents = {item.parent_claim_id for item in supported}
    known_negative_lineage_ids = {
        item.lineage_event_id for item in lineages if _known_negative_lineage(item)
    }
    known_negative_supported = [
        item for item in supported if item.lineage_event_id in known_negative_lineage_ids
    ]
    known_negative_evidence = [
        item
        for item in evidence_rows
        if item["lineage_event_id"] in known_negative_lineage_ids
    ]
    known_negative_any_supported = {
        (item.lineage_event_id, item.candidate_id)
        for item in known_negative_supported
    } | {
        (str(item["lineage_event_id"]), str(item["candidate_id"]))
        for item in known_negative_evidence
        if item["supported"]
    }
    known_negative_supported_parents = {
        lineage_event_id for lineage_event_id, _ in known_negative_any_supported
    }
    gate_failure_counts: Counter[str] = Counter()
    for lineage in lineages:
        for context in lineage.round_failure_contexts:
            for failure in context.get("failed_candidates") or []:
                if not isinstance(failure, dict):
                    continue
                gate_failure_counts.update(
                    str(gate) for gate in failure.get("failed_gates") or []
                )
    row: dict[str, Any] = {
        "arm_id": arm_id,
        "parent_lineage_count": parent_count,
        "llm_response_count": sum(item.llm_response_count for item in lineages),
        "proposals_returned_count": sum(item.proposals_returned_count for item in lineages),
        "schema_valid_candidate_count": sum(item.schema_valid_candidate_count for item in lineages),
        "policy_valid_candidate_count": sum(item.policy_valid_candidate_count for item in lineages),
        "generated_candidate_count": sum(item.proposals_returned_count for item in lineages),
        "retained_candidate_count": sum(item.retained_candidate_count for item in lineages),
        "duplicate_candidate_count": sum(item.duplicate_candidate_count for item in lineages),
        "unretained_generated_candidate_count": sum(
            item.unretained_generated_candidate_count for item in lineages
        ),
        "current_data_evaluated_count": sum(
            item.current_data_evaluated_count for item in lineages
        ),
        "unique_source_tested_count": sum(
            item.current_data_evaluated_count for item in lineages
        ),
        "execution_complete_candidate_count": sum(
            item.execution_complete_candidate_count for item in lineages
        ),
        "provisional_internal_pass_count": sum(
            item.provisional_internal_pass_count for item in lineages
        ),
        "execution_error_count": sum(item.execution_error_count for item in lineages),
        "analysis_non_identifiable_count": sum(
            item.analysis_non_identifiable_count for item in lineages
        ),
        "final_internal_supported_candidate_count": len(supported),
        "unique_internally_supported_contract_count": len({item.exact_contract_id for item in supported}),
        "semantic_cluster_count": len({item.semantic_cluster_id for item in supported}),
        "parents_with_internal_support_count": len(supported_parents),
        "internal_candidate_system_yield": len(supported) / parent_count if parent_count else 0.0,
        "internal_parent_system_yield": len(supported_parents) / parent_count if parent_count else 0.0,
        "gate_failure_counts": json.dumps(dict(sorted(gate_failure_counts.items())), sort_keys=True),
        "stopped_reason_counts": json.dumps(
            dict(sorted(Counter(item.stopped_reason for item in lineages).items())),
            sort_keys=True,
        ),
        "known_negative_parent_lineage_count": len(known_negative_lineage_ids),
        "known_negative_internal_supported_candidate_count": len(
            known_negative_supported
        ),
        "known_negative_exploratory_supported_candidate_count": sum(
            item.current_data_label == "exploratory_confirmed"
            for item in known_negative_supported
        ),
        "known_negative_contract_repair_supported_candidate_count": sum(
            item.current_data_label == "contract_repair_supported"
            for item in known_negative_supported
        ),
        "known_negative_any_supported_candidate_count": len(
            known_negative_any_supported
        ),
        "known_negative_any_supported_parent_count": len(
            known_negative_supported_parents
        ),
        "known_negative_any_support_risk_rate": (
            len(known_negative_supported_parents) / len(known_negative_lineage_ids)
            if known_negative_lineage_ids
            else 0.0
        ),
        "known_negative_analysis_non_identifiable_count": sum(
            item.analysis_non_identifiable_count
            for item in lineages
            if item.lineage_event_id in known_negative_lineage_ids
        ),
        "known_negative_source_execution_error_count": sum(
            item.execution_error_count
            for item in lineages
            if item.lineage_event_id in known_negative_lineage_ids
        ),
        "known_negative_holdout_supported_candidate_pair_count": sum(
            item["supported"]
            for item in known_negative_evidence
            if item["evidence_kind"] == "holdout"
        ),
        "known_negative_external_supported_candidate_pair_count": sum(
            item["supported"]
            for item in known_negative_evidence
            if item["evidence_kind"] == "external"
        ),
        "known_negative_excluded_evidence_unavailable_count": sum(
            not item["compatible"] for item in known_negative_evidence
        ),
        "known_negative_excluded_evidence_execution_error_count": sum(
            item["execution_error"] for item in known_negative_evidence
        ),
    }
    for kind in ("holdout", "external"):
        rows = [item for item in evidence_rows if item["evidence_kind"] == kind]
        compatible = sum(item["compatible"] for item in rows)
        evaluated = sum(item["evaluated"] for item in rows)
        supported_count = sum(item["supported"] for item in rows)
        row.update(
            {
                f"{kind}_compatible_candidate_pair_count": compatible,
                f"{kind}_evaluated_candidate_pair_count": evaluated,
                f"{kind}_supported_candidate_pair_count": supported_count,
                f"{kind}_unavailable_count": sum(not item["compatible"] for item in rows),
                f"{kind}_execution_error_count": sum(item["execution_error"] for item in rows),
                f"{kind}_conditional_survival_rate": supported_count / evaluated if evaluated else None,
                f"{kind}_supported_parent_count": len(
                    {item["parent_claim_id"] for item in rows if item["supported"]}
                ),
            }
        )
    matched = [item for item in evidence_rows if item["matched_parent_holdout"]]
    row["matched_parent_candidate_holdout_count"] = len(matched)
    row["candidate_only_holdout_support_count"] = sum(
        item["candidate_only_holdout_support"] for item in matched
    )
    return row


def _write_case_studies_v2(
    path: Path,
    evidence_rows: list[dict[str, Any]],
    exposures: list[FrozenCandidateExposure],
) -> None:
    exposure_by_key = {(item.lineage_event_id, item.candidate_id): item for item in exposures}
    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        cells[(row["evidence_kind"], str(row.get("evidence_set_id") or "none"), row["supported"])].append(row)
    lines = [
        "# Deterministic Retrospective Case Studies",
        "",
        "Cases use the lexicographically first claim ID in each evidence/outcome cell, never the strongest p-value.",
        "All excluded evidence was previously queried; these are descriptive, not fresh confirmations.",
        "",
    ]
    for cell, rows in sorted(cells.items()):
        row = min(rows, key=lambda item: (item["parent_claim_id"], item["candidate_id"], item["arm_id"]))
        exposure = exposure_by_key[(row["lineage_event_id"], row["candidate_id"])]
        parent = ClaimContract.model_validate(exposure.parent_contract)
        candidate = ClaimContract.model_validate(exposure.effective_contract)
        lines.extend(
            [
                f"## {cell[0]} / {cell[1]} / supported={cell[2]}",
                "",
                f"- Arm: `{exposure.arm_id}`",
                f"- Parent claim: {parent.question}",
                f"- Candidate claim: {candidate.question}",
                f"- Declared transform: `{exposure.declared_transform or exposure.transform_type}`",
                f"- Inferred transform: `{exposure.inferred_transform or 'unknown'}`",
                f"- Effective family size: `{exposure.effective_family_size}`",
                f"- Evidence label: `{row.get('raw_gate_label') or row.get('preflight_status')}`",
                f"- Interpretation: `{row.get('interpretation_label')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
