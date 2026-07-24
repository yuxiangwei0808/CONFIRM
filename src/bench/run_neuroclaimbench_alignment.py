"""Run the outcome-blind NeuroClaimBench v2.1 question/contract alignment audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from bench.neuroclaimbench_v21_compat import (
    BenchmarkItem,
    DeterministicContractRepair,
    EvaluationTask,
    FieldAlignment,
    GeminiAlignmentAssessment,
    QuestionContractAlignment,
    exact_contract_hash,
    scientific_question_hash,
    sha256_payload,
)
from bench.progress import iter_progress
from confirm.candidate_preflight import CandidatePreflightContext, CandidatePreflightResult
from confirm.contract import ClaimContract
from confirm.llm import LLMClient, make_llm

POLICY_VERSION = "neuroclaimbench-v2.1-alignment-20260723"
DEFAULT_DATA_ROOTS = [
    "data/prepared_data/evidence_partitions/benchmark_ready/cohorts",
    "data/prepared_data/evidence_partitions/cohorts",
    "review-stage/claim-search-safety-gpt55-r10-c10-v7/data/cohorts",
]

ALIGNMENT_POLICY = {
    "version": POLICY_VERSION,
    "canonical_question": "The original scientific question is canonical.",
    "immutable_fields": [
        "predictor",
        "group_contrast",
        "population",
        "outcome",
        "modality",
        "direction",
    ],
    "allowed_repairs": [
        "restore an explicitly stated ADHD subtype contrast using raw dx labels",
        "restore a question-stated executable outcome",
        "add required measured confounds as safety_covariate_augmentation",
    ],
    "non_executable_only_when": [
        "requested group is absent",
        "requested outcome is absent",
        "required columns are unavailable",
        "design is rank deficient",
        "residual degrees of freedom are nonpositive",
    ],
    "prohibited_inputs": [
        "CONFIRM verdicts",
        "p-values",
        "effect estimates",
        "benchmark labels",
        "adjudication votes",
        "feedback-search results",
    ],
    "low_power_policy": "Low sample size alone is not an exclusion; the unchanged power gate adjudicates it.",
    "gemini_role": "Advisory semantic audit only; deterministic rules authorize repairs.",
}


class GeminiAlignmentResponse(BaseModel):
    """Strict outcome-blind Gemini response before provenance is attached."""

    model_config = ConfigDict(extra="forbid")

    aligned: bool
    field_assessments: list[FieldAlignment] = Field(default_factory=list)
    recommended_disposition: Literal[
        "aligned",
        "aligned_with_safety_augmentation",
        "repairable_contract",
        "non_executable",
        "ambiguous_unresolved",
    ]
    rationale: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    os.replace(temp, path)


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(
                row.model_dump(mode="json") if hasattr(row, "model_dump") else row,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literature_derived(item: BenchmarkItem) -> bool:
    return item.adjudication_status != "construction_derived"


def _explicit_direction(question: str) -> str | None:
    text = question.lower()
    padded = f" {text} "
    negative = (
        " expected direction negative ",
        " show lower ",
        " show decreased ",
        " show reduced ",
        " lower than ",
        " negative for ",
    )
    positive = (
        " expected direction positive ",
        " show higher ",
        " show increased ",
        " show elevated ",
        " higher than ",
        " positive for ",
    )
    if any(token in padded for token in negative):
        return "negative"
    if any(token in padded for token in positive):
        return "positive"
    return None


def _adhd_requested_contrast(question: str) -> tuple[str, str] | None:
    """Map an explicitly worded ADHD subtype contrast to raw ADHD200 dx labels."""

    text = question.lower()
    labels = {
        "combined": "1",
        "inattentive": "2",
        "hyperactive": "3",
        "control": "0",
        "controls": "0",
    }
    mentioned = [(text.find(name), name, code) for name, code in labels.items() if name in text]
    deduplicated: list[tuple[int, str, str]] = []
    seen_codes: set[str] = set()
    for position, name, code in sorted(mentioned):
        if code not in seen_codes:
            deduplicated.append((position, name, code))
            seen_codes.add(code)
    if len(deduplicated) < 2:
        return None
    return deduplicated[0][2], deduplicated[1][2]


def _standard_confirm_dx_contrast(item: BenchmarkItem, contract: ClaimContract) -> bool:
    """Recognize only the documented disease/control virtual-label normalization."""

    group = contract.estimand.group
    if (
        contract.estimand.predictor != "confirm_dx"
        or group is None
        or group.var != "confirm_dx"
        or group.case != "case"
        or group.control != "control"
    ):
        return False
    text = f" {item.question.lower()} "
    has_control = any(token in text for token in (" hc ", " control", " cn "))
    if not has_control:
        return False
    if item.target_family == "asd":
        return " asd " in text and "cluster" not in text
    if item.target_family == "psychosis":
        return (
            (" sz " in text or "schizophren" in text)
            and not any(token in text for token in ("relative", "relapse", "breakthrough"))
        )
    if item.target_family == "adhd":
        return (
            " adhd" in text
            and not any(token in text for token in ("combined", "inattentive", "hyperactive", "subtype"))
        )
    if item.target_family == "ad_aging":
        return (
            ("dementia" in text or "alzheimer" in text or " ad " in text)
            and "mci" not in text
        )
    return False


def _replace_group_contrast(
    contract: ClaimContract,
    case: str,
    control: str,
) -> tuple[ClaimContract, list[DeterministicContractRepair]]:
    payload = contract.model_dump(mode="json")
    old_group = payload["estimand"].get("group")
    old_predictor = payload["estimand"].get("predictor")
    payload["estimand"]["type"] = "group_diff"
    payload["estimand"]["predictor"] = "dx"
    payload["estimand"]["group"] = {"var": "dx", "case": case, "control": control}
    inclusion = str(payload.get("inclusion") or "")
    if "confirm_dx" in inclusion or ("dx" in inclusion and case not in inclusion and control not in inclusion):
        payload["inclusion"] = None
    repaired = ClaimContract.model_validate(payload)
    repair = DeterministicContractRepair(
        field_path="estimand.group",
        old_value={"predictor": old_predictor, "group": old_group},
        new_value={"predictor": "dx", "group": payload["estimand"]["group"]},
        repair_type="restore_question_contrast",
        rationale="The canonical question explicitly names an ADHD subtype contrast represented by raw dx labels.",
    )
    return repaired, [repair]


def _field_alignments(item: BenchmarkItem, contract: ClaimContract) -> list[FieldAlignment]:
    estimand = contract.estimand
    rows: list[FieldAlignment] = []
    requested = _adhd_requested_contrast(item.question) if item.target_family == "adhd" else None
    normalized_confirm_dx = _standard_confirm_dx_contrast(item, contract)
    actual = None
    if estimand.group is not None:
        actual = {
            "var": estimand.group.var,
            "case": estimand.group.case,
            "control": estimand.group.control,
        }
    if requested is None and not normalized_confirm_dx:
        rows.append(
            FieldAlignment(
                field="group_contrast",
                status="not_assessable",
                question_value=None,
                contract_value=actual,
                reason="The deterministic parser found no explicit supported subtype contrast in the question.",
            )
        )
    elif requested is not None:
        expected = {"var": "dx", "case": requested[0], "control": requested[1]}
        rows.append(
            FieldAlignment(
                field="group_contrast",
                status="preserved" if actual == expected else "mismatch",
                question_value=expected,
                contract_value=actual,
                reason="Explicit ADHD subtype labels are deterministically mapped to raw dx codes.",
            )
        )
    else:
        rows.append(
            FieldAlignment(
                field="group_contrast",
                status="preserved",
                question_value="standard disease-versus-control dx contrast",
                contract_value=actual,
                reason=(
                    "confirm_dx is the documented virtual case/control normalization of the "
                    "question's standard diagnosis contrast."
                ),
            )
        )

    direction = (
        None
        if item.adjudication_status == "construction_derived"
        else _explicit_direction(item.question)
    )
    rows.append(
        FieldAlignment(
            field="direction",
            status=(
                "not_assessable"
                if direction is None
                else ("preserved" if direction == estimand.direction else "mismatch")
            ),
            question_value=direction,
            contract_value=estimand.direction,
            reason=(
                "No explicit directional phrase was found."
                if direction is None
                else "An explicit directional phrase was compared with estimand.direction."
            ),
        )
    )
    rows.extend(
        [
            FieldAlignment(
                field="predictor",
                status="preserved" if normalized_confirm_dx else "not_assessable",
                question_value=(
                    "standard disease-versus-control dx contrast"
                    if normalized_confirm_dx
                    else None
                ),
                contract_value=estimand.predictor,
                reason=(
                    "confirm_dx is the documented virtual normalization of the raw diagnosis contrast."
                    if normalized_confirm_dx
                    else "General predictor semantics require the blinded Gemini advisory audit."
                ),
            ),
            FieldAlignment(
                field="outcome",
                status="not_assessable",
                question_value=None,
                contract_value=estimand.outcome,
                reason="General outcome synonyms require the blinded Gemini advisory audit.",
            ),
            FieldAlignment(
                field="modality",
                status="preserved",
                question_value=item.modality,
                contract_value=item.modality,
                reason="The frozen item modality is derived from the executable outcome family.",
            ),
            FieldAlignment(
                field="population",
                status="not_assessable",
                question_value=None,
                contract_value=[contract.discovery_cohort, *contract.replication_cohorts],
                reason="Dataset split names are execution provenance; population semantics require advisory review.",
            ),
        ]
    )
    return rows


def _cohort_prompt_metadata(
    context: CandidatePreflightContext,
    contract: ClaimContract,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    requested_columns = list(
        dict.fromkeys(
            [
                contract.estimand.predictor,
                *contract.covariates,
                *(contract.gates.confound.require_covariates or []),
                "dx",
                "confirm_dx",
            ]
        )
    )
    if contract.estimand.group is not None:
        requested_columns.append(contract.estimand.group.var)
    for cohort in [contract.discovery_cohort, *contract.replication_cohorts]:
        info = context.resolve(cohort)
        if info is None:
            metadata[cohort] = {"available": False}
            continue
        columns = [column for column in requested_columns if column in info.columns]
        table = context._read_columns(info, columns) if columns else pd.DataFrame()  # noqa: SLF001
        categorical: dict[str, dict[str, int]] = {}
        for column in ("dx", "confirm_dx", contract.estimand.group.var if contract.estimand.group else ""):
            if column and column in table.columns:
                categorical[column] = {
                    str(level): int(count)
                    for level, count in table[column].astype(str).value_counts(dropna=False).items()
                }
        metadata[cohort] = {
            "available": True,
            "row_count": int(
                pd.read_parquet(
                    info.path,
                    columns=[info.source_columns[0]] if info.source_columns else None,
                ).shape[0]
            ),
            "columns": sorted(info.columns),
            "categorical_levels_and_counts": categorical,
            "nonmissing_counts": {
                column: int(table[column].notna().sum()) for column in table.columns
            },
        }
    return metadata


def _deterministic_record(
    item: BenchmarkItem,
    context: CandidatePreflightContext,
    task: EvaluationTask | None = None,
) -> QuestionContractAlignment:
    if item.contract is None:
        return QuestionContractAlignment(
            benchmark_item_id=item.benchmark_item_id,
            canonical_question=item.question,
            canonical_question_sha256=scientific_question_hash(item.question),
            executable_contract_sha256=None,
            mismatch_disposition="non_executable",
            final_outcome_blind_resolution="non_executable",
            resolution_reason="No executable contract is available.",
            policy_version=POLICY_VERSION,
        )

    original = item.contract
    repaired = original.model_copy(update={"question": item.question})
    repairs: list[DeterministicContractRepair] = []
    alignments = _field_alignments(item, repaired)
    requested = _adhd_requested_contrast(item.question) if item.target_family == "adhd" else None
    group_alignment = next(row for row in alignments if row.field == "group_contrast")
    if requested is not None and group_alignment.status == "mismatch":
        repaired, repairs = _replace_group_contrast(repaired, *requested)
        repaired = repaired.model_copy(update={"question": item.question})
        alignments = _field_alignments(item, repaired)

    question_text = item.question.lower()
    augmented_covariates = [
        covariate
        for covariate in repaired.covariates
        if covariate.lower() not in question_text
    ]
    safety_augmentations = [
        DeterministicContractRepair(
            field_path="covariates",
            old_value=[],
            new_value=augmented_covariates,
            repair_type="safety_covariate_augmentation",
            rationale="Measured covariates required by the frozen confound gate are transparent safety adjustments.",
        )
    ] if augmented_covariates else []

    preflight = (
        _generated_control_preflight(item, repaired, task, context)
        if task is not None and task.generator_spec
        else context.validate_contract(
            repaired,
            min_complete_rows=1,
            min_group_rows=1,
        )
    )
    unresolved_mismatches = [
        row.field for row in alignments if row.status == "mismatch"
    ]
    if not preflight.ok:
        disposition = "non_executable"
        reason = "Outcome-blind executable preflight failed: " + "; ".join(preflight.violations)
    elif unresolved_mismatches:
        disposition = "ambiguous_unresolved"
        reason = (
            "Deterministic checks found immutable mismatches without a policy-authorized repair: "
            + ", ".join(unresolved_mismatches)
        )
    elif repairs:
        disposition = "repairable_contract"
        reason = "A deterministic question-preserving contrast repair passed executable preflight."
    elif safety_augmentations:
        disposition = "aligned_with_safety_augmentation"
        reason = "The executable contract is aligned and includes transparent measured confound adjustments."
    else:
        disposition = "aligned"
        reason = "No deterministic question-contract mismatch was found."

    return QuestionContractAlignment(
        benchmark_item_id=item.benchmark_item_id,
        canonical_question=item.question,
        canonical_question_sha256=scientific_question_hash(item.question),
        executable_contract_sha256=exact_contract_hash(original),
        repaired_contract_sha256=exact_contract_hash(repaired),
        field_alignments=alignments,
        safety_augmentations=safety_augmentations,
        mismatch_disposition=disposition,
        deterministic_repairs=repairs,
        repaired_contract=repaired,
        deterministic_preflight=preflight.model_dump(mode="json"),
        final_outcome_blind_resolution=disposition,
        resolution_reason=reason,
        policy_version=POLICY_VERSION,
    )


def _generated_control_preflight(
    item: BenchmarkItem,
    contract: ClaimContract,
    task: EvaluationTask,
    context: CandidatePreflightContext,
) -> CandidatePreflightResult:
    spec = task.generator_spec or {}
    assignment_path = Path(str(spec.get("assignment_path") or ""))
    violations: list[str] = []
    resolved_paths: dict[str, str] = {}
    resolved_outcomes = context.resolved_outcomes(contract)
    if not assignment_path.exists():
        violations.append(f"Generated-control assignment artifact is missing: {assignment_path}")
        assignments = pd.DataFrame()
    else:
        assignments = pd.read_parquet(assignment_path)
        assignments = assignments[
            assignments["control_id"].astype(str) == item.benchmark_item_id
        ]
        if assignments.empty:
            violations.append(f"Generated-control assignments are absent for {item.benchmark_item_id}")
    group_column = str(spec.get("group_column") or "ncb_random_group")
    for cohort in [contract.discovery_cohort, *contract.replication_cohorts]:
        info = context.resolve(cohort)
        if info is None:
            violations.append(f"Preflight: cohort {cohort!r} was not found in configured data roots.")
            continue
        resolved_paths[cohort] = info.path
        if not resolved_outcomes.get(cohort):
            violations.append(f"Preflight: cohort {cohort!r} is missing the requested outcome.")
        missing = [
            column
            for column in contract.covariates
            if column not in info.columns
        ]
        if missing:
            violations.append(f"Preflight: cohort {cohort!r} is missing analysis columns: {missing}")
        cohort_assignments = assignments[
            assignments.get("cohort", pd.Series(dtype=str)).astype(str) == cohort
        ]
        levels = set(
            cohort_assignments.get(group_column, pd.Series(dtype=str)).astype(str)
        )
        expected = {
            contract.estimand.group.case,
            contract.estimand.group.control,
        } if contract.estimand.group is not None else set()
        if not expected.issubset(levels):
            violations.append(
                f"Generated-control assignments for {cohort!r} are missing group levels: "
                f"{sorted(expected - levels)}"
            )
    return CandidatePreflightResult(
        ok=not violations,
        violations=violations,
        resolved_data_paths=resolved_paths,
        resolved_outcome_columns=resolved_outcomes,
        design_diagnostics={
            "generator": {
                "assignment_path": str(assignment_path),
                "assignment_row_count": int(len(assignments)),
                "group_column": group_column,
            }
        },
    )


def _gemini_prompt(
    item: BenchmarkItem,
    record: QuestionContractAlignment,
    context: CandidatePreflightContext,
) -> str:
    assert record.repaired_contract is not None
    payload = {
        "task": (
            "Assess whether the executable contract preserves the canonical scientific question. "
            "Do not judge truth, statistical support, novelty, power, or likely gate outcome."
        ),
        "canonical_question": item.question,
        "contract": record.repaired_contract.model_dump(mode="json"),
        "cohort_schemas_and_counts": _cohort_prompt_metadata(context, record.repaired_contract),
        "policy": {
            "immutable": ALIGNMENT_POLICY["immutable_fields"],
            "low_power": ALIGNMENT_POLICY["low_power_policy"],
            "measured_confound_adjustment": (
                "Adding available required confounds is a safety augmentation, not scientific broadening."
            ),
            "allowed_dispositions": [
                "aligned",
                "aligned_with_safety_augmentation",
                "repairable_contract",
                "non_executable",
                "ambiguous_unresolved",
            ],
        },
        "instructions": [
            "Compare predictor, contrast, population, outcome, modality, and direction field by field.",
            "A split/cohort execution name is not itself a literature construct.",
            "Small but present groups remain executable if the design is identifiable.",
            "Return structured JSON only.",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _call_gemini(
    llm: LLMClient,
    prompt: str,
    *,
    retries: int,
) -> tuple[GeminiAlignmentAssessment, list[dict[str, Any]]]:
    system = (
        "You are an outcome-blind scientific question/contract alignment auditor. "
        "You receive no outcomes or benchmark labels and have no authority to repair contracts."
    )
    active_prompt = prompt
    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        raw = ""
        try:
            structured = getattr(llm, "complete_structured", None)
            raw = (
                str(structured(system, active_prompt, GeminiAlignmentResponse))
                if callable(structured)
                else llm.complete(system, active_prompt)
            )
            parsed = GeminiAlignmentResponse.model_validate_json(raw)
            attempts.append({"attempt": attempt, "prompt": active_prompt, "raw_response": raw, "error": ""})
            return (
                GeminiAlignmentAssessment(
                    **parsed.model_dump(mode="json"),
                    model_spec=f"google:{llm.model}",
                    prompt_sha256=sha256_payload({"system": system, "user": prompt}),
                    response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    schema_attempts=attempt,
                    call_metadata=dict(getattr(llm, "last_call_metadata", {}) or {}),
                ),
                attempts,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempts.append({"attempt": attempt, "prompt": active_prompt, "raw_response": raw, "error": str(exc)})
            active_prompt = (
                prompt
                + "\n\nThe previous response failed the strict schema. Return only one JSON object matching it. "
                + f"Validation error: {exc}"
            )
    assert last_error is not None
    raise last_error


def _finalize_with_gemini(
    record: QuestionContractAlignment,
    assessment: GeminiAlignmentAssessment | None,
) -> QuestionContractAlignment:
    if assessment is None:
        return record
    resolution = record.mismatch_disposition
    reason = record.resolution_reason
    if resolution != "non_executable" and not assessment.aligned:
        normalized_fields = {
            row.field
            for row in record.field_alignments
            if row.status == "preserved" and "confirm_dx" in row.reason
        }
        substantive_mismatches = [
            row
            for row in assessment.field_assessments
            if row.status == "mismatch"
            and not (
                row.field == "covariates"
                and bool(record.safety_augmentations)
                and any(
                    phrase in record.canonical_question.lower()
                    for phrase in (
                        "no additional covariates",
                        "without additional covariates",
                        "with no covariates",
                        "unadjusted",
                    )
                )
            )
            and row.field not in normalized_fields
        ]
        if substantive_mismatches or not assessment.field_assessments:
            resolution = "ambiguous_unresolved"
            reason = (
                "Deterministic checks did not authorize the semantic discrepancy reported by "
                "the advisory audit: " + assessment.rationale
            )
        else:
            reason = (
                record.resolution_reason
                + " The Gemini advisory treated required measured confounds as a mismatch, "
                "but the frozen policy classifies them as a safety augmentation."
            )
    return record.model_copy(
        update={
            "gemini_assessment": assessment,
            "final_outcome_blind_resolution": resolution,
            "resolution_reason": reason,
        }
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(args.package_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "alignment_policy.json"
    policy_payload = {**ALIGNMENT_POLICY, "policy_sha256": sha256_payload(ALIGNMENT_POLICY)}
    policy_text = json.dumps(policy_payload, indent=2, sort_keys=True) + "\n"
    if policy_path.exists() and policy_path.read_text(encoding="utf-8") != policy_text:
        raise ValueError("Existing alignment policy differs; use a new versioned output directory.")
    _atomic_text(policy_path, policy_text)

    items = [
        BenchmarkItem.model_validate(row)
        for row in _read_jsonl(package / "benchmark_items.jsonl")
    ]
    task_by_item = {
        task.benchmark_item_id: task
        for task in (
            EvaluationTask.model_validate(row)
            for row in _read_jsonl(package / "evaluation_tasks.jsonl")
        )
    }
    context = CandidatePreflightContext.from_roots(args.data_root)
    deterministic_path = out_dir / "deterministic_alignment.jsonl"
    if args.phase in {"deterministic", "all"}:
        deterministic = [
            _deterministic_record(item, context, task_by_item.get(item.benchmark_item_id))
            for item in iter_progress(
                items,
                total=len(items),
                desc="Outcome-blind alignment",
                enabled=not args.no_progress,
                unit="claim",
            )
        ]
        _write_jsonl(deterministic_path, deterministic)
    else:
        deterministic = [
            QuestionContractAlignment.model_validate(row)
            for row in _read_jsonl(deterministic_path)
        ]

    item_by_id = {item.benchmark_item_id: item for item in items}
    prompt_path = out_dir / "gemini_alignment_prompts.jsonl"
    response_path = out_dir / "gemini_alignment_responses.jsonl"
    existing_prompts = _read_jsonl(prompt_path) if prompt_path.exists() else []
    existing_responses = _read_jsonl(response_path) if response_path.exists() else []
    prompt_by_item = {str(row["benchmark_item_id"]): row for row in existing_prompts}
    response_by_item = {str(row["benchmark_item_id"]): row for row in existing_responses}
    checkpoint_dir = out_dir / "gemini_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    assessments: dict[str, GeminiAlignmentAssessment] = {}
    jobs: list[tuple[BenchmarkItem, str, Path, str]] = []
    for record in deterministic:
        item = item_by_id[record.benchmark_item_id]
        if _literature_derived(item) and record.repaired_contract is not None:
            prompt = _gemini_prompt(item, record, context)
            fingerprint = sha256_payload(
                {
                    "item": item.benchmark_item_id,
                    "question": record.canonical_question_sha256,
                    "contract": record.repaired_contract_sha256,
                    "policy": policy_payload["policy_sha256"],
                    "model": args.model,
                }
            )
            checkpoint = checkpoint_dir / f"{item.benchmark_item_id}.json"
            if checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("fingerprint") != fingerprint:
                    raise ValueError(f"Stale Gemini alignment checkpoint for {item.benchmark_item_id}")
                assessment = GeminiAlignmentAssessment.model_validate(
                    payload["assessment"]
                )
                assessments[item.benchmark_item_id] = assessment
                attempts = list(payload.get("attempts") or [])
                prompt_by_item[item.benchmark_item_id] = {
                    "benchmark_item_id": item.benchmark_item_id,
                    "fingerprint": fingerprint,
                    "prompt": payload.get("prompt") or prompt,
                    "prompt_sha256": assessment.prompt_sha256,
                    "trace_source": (
                        "checkpoint_full_trace"
                        if attempts
                        else "checkpoint_parsed_assessment"
                    ),
                }
                response_by_item[item.benchmark_item_id] = {
                    "benchmark_item_id": item.benchmark_item_id,
                    "fingerprint": fingerprint,
                    "attempts": attempts,
                    "assessment": assessment.model_dump(mode="json"),
                    "raw_response_available": bool(attempts),
                    "trace_source": (
                        "checkpoint_full_trace"
                        if attempts
                        else "checkpoint_parsed_assessment"
                    ),
                }
                if "prompt" not in payload or "attempts" not in payload:
                    payload.update(
                        {
                            "prompt": prompt,
                            "attempts": attempts,
                            "raw_response_available": bool(attempts),
                        }
                    )
                    _atomic_text(
                        checkpoint,
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    )
            elif args.phase in {"gemini", "all"}:
                jobs.append(
                    (
                        item,
                        fingerprint,
                        checkpoint,
                        prompt,
                    )
                )

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {
                executor.submit(
                    _call_gemini,
                    make_llm(args.model),
                    prompt,
                    retries=args.schema_retries,
                ): (item, fingerprint, checkpoint, prompt)
                for item, fingerprint, checkpoint, prompt in jobs
            }
            for future in iter_progress(
                as_completed(futures),
                total=len(futures),
                desc="Gemini alignment advisory",
                enabled=not args.no_progress,
                unit="claim",
            ):
                item, fingerprint, checkpoint, prompt = futures[future]
                assessment, attempts = future.result()
                assessments[item.benchmark_item_id] = assessment
                prompt_by_item[item.benchmark_item_id] = {
                    "benchmark_item_id": item.benchmark_item_id,
                    "fingerprint": fingerprint,
                    "prompt": prompt,
                    "prompt_sha256": assessment.prompt_sha256,
                }
                response_by_item[item.benchmark_item_id] = {
                    "benchmark_item_id": item.benchmark_item_id,
                    "fingerprint": fingerprint,
                    "attempts": attempts,
                    "assessment": assessment.model_dump(mode="json"),
                    "raw_response_available": True,
                    "trace_source": "checkpoint_full_trace",
                }
                _atomic_text(
                    checkpoint,
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "assessment": assessment.model_dump(mode="json"),
                            "prompt": prompt,
                            "attempts": attempts,
                            "raw_response_available": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )

    final_records = [
        _finalize_with_gemini(record, assessments.get(record.benchmark_item_id))
        for record in deterministic
    ]
    if prompt_by_item:
        _write_jsonl(prompt_path, [prompt_by_item[key] for key in sorted(prompt_by_item)])
    if response_by_item:
        _write_jsonl(response_path, [response_by_item[key] for key in sorted(response_by_item)])
    _write_jsonl(out_dir / "alignment_records.jsonl", final_records)

    resolution_counts = Counter(row.final_outcome_blind_resolution for row in final_records)
    manifest = {
        "version": "2.1.0",
        "policy_version": POLICY_VERSION,
        "policy_sha256": policy_payload["policy_sha256"],
        "source_package": str(package),
        "source_benchmark_items_sha256": _sha256_file(package / "benchmark_items.jsonl"),
        "record_count": len(final_records),
        "source_executable_count": sum(
            bool(item.contract is not None and item.evaluation_task_ids)
            for item in items
        ),
        "literature_advisory_count": sum(row.gemini_assessment is not None for row in final_records),
        "gemini_trace_count": len(response_by_item),
        "gemini_raw_response_trace_count": sum(
            bool(row.get("raw_response_available"))
            for row in response_by_item.values()
        ),
        "gemini_parsed_assessment_only_count": sum(
            not bool(row.get("raw_response_available"))
            for row in response_by_item.values()
        ),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "repair_count": sum(bool(row.deterministic_repairs) for row in final_records),
        "outcome_blind": True,
        "forbidden_inputs_used": [],
        "output_sha256": {
            "alignment_policy.json": _sha256_file(policy_path),
            "alignment_records.jsonl": _sha256_file(out_dir / "alignment_records.jsonl"),
            "gemini_alignment_prompts.jsonl": (
                _sha256_file(prompt_path) if prompt_path.exists() else ""
            ),
            "gemini_alignment_responses.jsonl": (
                _sha256_file(response_path) if response_path.exists() else ""
            ),
        },
    }
    _atomic_text(out_dir / "alignment_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["deterministic", "gemini", "all"], default="all")
    parser.add_argument("--package-dir", default="data/neuroclaimbench/v2.1-source")
    parser.add_argument("--out-dir", default="review-stage/neuroclaimbench-v2.1/alignment")
    parser.add_argument("--model", default="google:gemini-3.5-flash")
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--data-root", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is None:
        args.data_root = list(DEFAULT_DATA_ROOTS)
    manifest = run(args)
    print(json.dumps({"status": "completed", "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
