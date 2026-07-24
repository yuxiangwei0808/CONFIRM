"""Iterative failure-diagnosis claim generation with provenance controls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.contract import ClaimContract
from confirm.llm import LLMClient
from confirm.proposals import (
    FailureLocalization,
    NewClaimProposal,
    ProposalProvenance,
    ProposalValidation,
    localize_failure,
    validate_new_claim_proposal,
)

TransformType = Literal[
    "narrower_outcome_family",
    "alternative_same_modality_outcome",
    "multivariate_pattern",
    "moderator_or_subgroup",
    "stronger_design",
    "fixed_estimand",
    "contract_correction",
]

DEFAULT_BRAINWIDE_MIN_FEATURES = 3
DEFAULT_ADAPTIVE_PATTERN_CORR_FLOOR = 0.5
CandidateProposalType = Literal["corrected_contract", "exploratory_followup_claim"]
FeedbackMode = Literal["structured_diagnosis", "generic_retry"]
ValidationSplit = Literal[
    "none",
    "current_data_adaptive",
    "current_data_contract_repair",
    "excluded_validation",
    "future_required",
]
Stage3ValidationSplit = Literal["current_data_adaptive", "current_data_contract_repair"]
StoppedReason = Literal[
    "max_rounds_exhausted",
    "no_evaluator",
    "already_confirmed",
    "no_candidates",
    "llm_unavailable",
    "candidate_generation_failed",
    "all_candidates_supported",
    "no_valid_candidates",
]

CLAIM_CANDIDATE_SYSTEM_PROMPT = """You generate scientifically connected follow-up claim candidates after CONFIRM rejects a claim.
Rules:
- Generate creative, scientifically plausible, connected next claims that have a legitimate chance of passing the unchanged CONFIRM gates.
- Improve the hypothesis, estimand, or population scientifically; do not manipulate the contract merely to obtain a passing label.
- Failed-claim evidence may be used only for diagnosis and hypothesis generation.
- Post-hoc candidates may be evaluated on the same data, but same-data support must be labeled exploratory_confirmed, not confirmed.
- Generate executable follow-up claims for evaluation on the parent source evidence only. Excluded evidence is handled later by a frozen audit and is never routed by you.
- Preserve the original disease/cohort family, outcome modality, biological direction family, and scientific motivation.
- Return structured JSON only. Do not use markdown fences.
- Do not invent p-values, effect sizes, cohorts, or gate results.
- Do not weaken CONFIRM gates, drop confound covariates, reverse direction, switch to unrelated outcomes, or present same-data adaptive support as final confirmation.
- Preserve predictor, group contrast, direction, gates, search family size, and cohort family. Never remove covariates; a measured source-data confound may be added with an explicit scientific justification.
- You may propose creative connected outcomes within the original modality, including multivariate or brainwide patterns. Use only executable source columns and state why the pattern is scientifically connected.
- proposed_contract must keep the parent discovery and replication cohorts exactly, including when a parent already uses a holdout-named source partition.
"""


class ClaimSearchConfig(BaseModel):
    """Configurable budget and safety switches for iterative claim search."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=3, ge=1, le=20)
    max_candidates_per_round: int = Field(default=5, ge=1, le=20)
    candidate_timeout: float = Field(default=30.0, gt=0.0)
    allow_current_data_contract_repairs: bool = True
    llm_schema_retries: int = Field(default=2, ge=0, le=5)
    feedback_mode: FeedbackMode = "structured_diagnosis"
    brainwide_min_features: int = Field(default=DEFAULT_BRAINWIDE_MIN_FEATURES, ge=3)
    adaptive_pattern_corr_floor: float = Field(
        default=DEFAULT_ADAPTIVE_PATTERN_CORR_FLOOR,
        ge=-1.0,
        le=1.0,
    )


class CandidateDomainCore(BaseModel):
    """Structured relevance anchor a candidate must preserve from the parent claim."""

    model_config = ConfigDict(extra="forbid")

    population_or_disease: str
    cohort_family: str
    predictor_or_contrast: str
    outcome_modality: str
    outcome_family: str
    direction_family: str
    scientific_motivation: str


class CandidatePreservationCheck(BaseModel):
    """LLM-declared connection checks that are later verified deterministically."""

    model_config = ConfigDict(extra="forbid")

    preserves_population: bool
    preserves_cohort_family: bool
    preserves_predictor_or_contrast: bool
    preserves_outcome_modality: bool
    preserves_direction_family: bool
    preserves_scientific_motivation: bool
    changed_fields: list[str] = Field(default_factory=list)
    allowed_change_rationale: str


class CandidateEvidencePolicy(BaseModel):
    """Structured provenance and validation policy emitted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    provenance: ProposalProvenance
    requires_new_evidence: bool
    can_confirm_on_current_data: bool
    validation_split: Stage3ValidationSplit


class _LLMCandidateProposalBase(BaseModel):
    """Fields shared by all structured LLM candidate variants."""

    model_config = ConfigDict(extra="forbid")

    domain_core: CandidateDomainCore
    preservation_check: CandidatePreservationCheck
    proposed_question: str
    proposed_contract: ClaimContract
    rationale: str
    connection_rationale: str
    evidence_policy: CandidateEvidencePolicy
    supported_by_evidence: list[str] = Field(default_factory=list)
    disposition_label: Optional[str] = None
    responds_to_candidate_ids: list[str] = Field(default_factory=list)


class LLMCandidateProposal(_LLMCandidateProposalBase):
    """Structured executable proposal; transform intent is audited after parsing."""

    proposal_type: Literal["corrected_contract", "exploratory_followup_claim"]
    transform_type: TransformType


class LLMCandidateGenerationResponse(BaseModel):
    """Top-level strict schema for candidate-generation LLM output."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[LLMCandidateProposal]


# Frozen v7 wire-format name used by result-preserving readers and tests.
RawCandidateResponseV7 = LLMCandidateGenerationResponse


class CandidateClaimProposal(NewClaimProposal):
    """A follow-up proposal inside an iterative search lineage."""

    proposal_type: CandidateProposalType
    proposed_question: str
    proposed_contract: ClaimContract
    candidate_id: str
    parent_claim_id: str
    round_index: int = Field(ge=1)
    transform_type: TransformType
    domain_core: CandidateDomainCore
    preservation_check: CandidatePreservationCheck
    evidence_policy: CandidateEvidencePolicy
    connection_rationale: str
    validation_split: ValidationSplit
    responds_to_candidate_ids: list[str] = Field(default_factory=list)
    declared_transform: Optional[TransformType] = None
    inferred_transform: Optional[str] = None
    transform_match: Optional[bool] = None
    executable_contract_delta: dict[str, Any] = Field(default_factory=dict)
    policy_adjustments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_candidate_shape(self) -> "CandidateClaimProposal":
        if self.parent_claim_id == self.candidate_id:
            raise ValueError("candidate_id must differ from parent_claim_id")
        if self.evidence_policy.provenance != self.provenance:
            raise ValueError("evidence_policy.provenance must match provenance")
        if self.evidence_policy.requires_new_evidence != self.requires_new_evidence:
            raise ValueError("evidence_policy.requires_new_evidence must match requires_new_evidence")
        if self.evidence_policy.can_confirm_on_current_data != self.can_confirm_on_current_data:
            raise ValueError("evidence_policy.can_confirm_on_current_data must match can_confirm_on_current_data")
        if self.evidence_policy.validation_split != self.validation_split:
            raise ValueError("evidence_policy.validation_split must match validation_split")
        return self


class CandidateEvaluation(BaseModel):
    """Evaluation outcome for one candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    proposal: CandidateClaimProposal
    validation: ProposalValidation
    eligible_for_confirmation: bool
    evaluated: bool = False
    final_label: Optional[str] = None
    gate_results: Optional[dict[str, Any]] = None
    validation_split: ValidationSplit
    blocked_reason: Optional[str] = None
    execution_error: Optional[str] = None
    exploratory_label: Optional[str] = None
    exploratory_gate_results: Optional[dict[str, Any]] = None
    exploratory_confirmed: bool = False
    resolved_discovery_path: Optional[str] = None
    resolved_replication_paths: list[str] = Field(default_factory=list)
    same_underlying_data: Optional[bool] = None
    current_data_supported: bool = False
    provisional_supported: bool = False
    declared_family_size: Optional[int] = None
    effective_family_size: Optional[int] = None
    final_family_size: Optional[int] = None
    search_hypothesis_count: int = 0
    cumulative_search_hypothesis_count: int = 0
    multiplicity_retracted: bool = False
    multiplicity_recheck_count: int = 0
    design_diagnostics: dict[str, Any] = Field(default_factory=dict)


class UnretainedCandidateAttempt(BaseModel):
    """Parsed candidate discarded when a wholly invalid response is retried."""

    model_config = ConfigDict(extra="forbid")

    round_index: int
    validation_retry_index: int
    disposition: Literal["deterministic_validation_retry"]
    proposal: CandidateClaimProposal
    validation: ProposalValidation


class DuplicateCandidateRecord(BaseModel):
    """Candidate omitted because its scientific specification was already seen."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    duplicate_of: str
    parent_claim_id: str
    round_index: int
    scientific_signature: str


class CandidateFailureSummary(BaseModel):
    """Typed failure context supplied to the next generation round."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    failed_gates: list[str] = Field(default_factory=list)
    localized_cause: Optional[FailureLocalization] = None
    effective_contract: ClaimContract
    design_diagnostics: dict[str, Any] = Field(default_factory=dict)
    execution_status: str


class RoundFailureContext(BaseModel):
    """Failures from one round that later proposals must address."""

    model_config = ConfigDict(extra="forbid")

    round_index: int
    failed_candidates: list[CandidateFailureSummary] = Field(default_factory=list)


class RoundSearchSummary(BaseModel):
    """Auditable counts and multiplicity state for one search round."""

    model_config = ConfigDict(extra="forbid")

    round_index: int
    proposals_returned: int
    schema_valid: int
    policy_valid: int
    unique_source_tested: int
    execution_complete: int
    provisional_passes: int
    final_round_passes: int
    multiplicity_retractions: int
    cumulative_unique_tested: int
    cumulative_unique_hypotheses_tested: int = 0
    final_family_size: int


class ClaimSearchState(BaseModel):
    """Serializable trace state for an iterative claim-search run."""

    model_config = ConfigDict(extra="forbid")

    original_claim: ClaimContract
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    failure_localization: Optional[FailureLocalization] = None
    lineage_graph: dict[str, Any]
    used_evidence: list[str] = Field(default_factory=list)
    candidate_history: list[CandidateClaimProposal] = Field(default_factory=list)
    unretained_candidate_attempts: list[UnretainedCandidateAttempt] = Field(default_factory=list)
    duplicate_candidates: list[DuplicateCandidateRecord] = Field(default_factory=list)
    evaluations: list[CandidateEvaluation] = Field(default_factory=list)
    internally_supported_candidate_ids: list[str] = Field(default_factory=list)
    round_failure_contexts: list[RoundFailureContext] = Field(default_factory=list)
    round_summaries: list[RoundSearchSummary] = Field(default_factory=list)
    provisional_supported_candidate_ids: list[str] = Field(default_factory=list)
    final_search_family_size: Optional[int] = None
    generated_candidate_count: int = 0
    schema_valid_candidate_count: int = 0
    unique_candidate_count: int = 0
    valid_candidate_count: int = 0
    current_data_evaluated_count: int = 0
    unique_hypotheses_tested_count: int = 0
    llm_candidate_prompts: list[dict[str, Any]] = Field(default_factory=list)
    llm_candidate_responses: list[dict[str, Any]] = Field(default_factory=list)
    stopped_reason: StoppedReason


CandidateEvaluator = Callable[[CandidateClaimProposal], Mapping[str, Any]]
CandidateGenerator = Callable[[ClaimContract, FailureLocalization, ClaimSearchConfig, int, str], list[CandidateClaimProposal]]


class CandidateGenerationError(RuntimeError):
    """Raised when an LLM candidate-generation attempt cannot produce parseable candidates."""


class LLMClaimCandidateGenerator:
    """LLM-backed generator for connected follow-up candidates."""

    def __init__(
        self,
        llm: LLMClient,
        preflight_context: CandidatePreflightContext | None = None,
    ) -> None:
        self.llm = llm
        self.preflight_context = preflight_context
        self.candidate_history: list[CandidateClaimProposal] = []
        self.round_failure_context: RoundFailureContext | None = None
        self.validation_feedback: dict[str, Any] | None = None
        self.validation_retry_index = 0
        self.prompt_records: list[dict[str, Any]] = []
        self.response_records: list[dict[str, Any]] = []

    def __call__(
        self,
        contract: ClaimContract,
        localization: FailureLocalization,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
    ) -> list[CandidateClaimProposal]:
        schema_error: str | None = None
        previous_response: str | None = None
        max_attempts = config.llm_schema_retries + 1
        validation_retry_index = int(self.validation_retry_index)
        for attempt_index in range(max_attempts):
            is_validation_retry = validation_retry_index > 0
            is_schema_retry = attempt_index > 0
            retry_kind = (
                "validation_and_schema"
                if is_validation_retry and is_schema_retry
                else "deterministic_validation"
                if is_validation_retry
                else "schema"
                if is_schema_retry
                else "none"
            )
            system, user = build_candidate_generation_prompt(
                contract,
                localization,
                config,
                round_index,
                parent_claim_id,
                candidate_history=self.candidate_history,
                schema_error=schema_error,
                previous_response=previous_response,
                executable_catalog=(
                    self.preflight_context.prompt_catalog(contract)
                    if self.preflight_context is not None
                    else None
                ),
                validation_feedback=self.validation_feedback,
                round_failure_context=self.round_failure_context,
            )
            prompt_record = {
                "round_index": round_index,
                "parent_claim_id": parent_claim_id,
                "attempt_index": attempt_index,
                "schema_attempt_index": attempt_index,
                "validation_retry_index": validation_retry_index,
                "is_retry": is_validation_retry or is_schema_retry,
                "retry_kind": retry_kind,
                "model": getattr(self.llm, "model", type(self.llm).__name__),
                "system": system,
                "user": user,
                "prompt_hash": hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest(),
            }
            self.prompt_records.append(prompt_record)
            raw = ""
            try:
                raw = _complete_structured_candidate_response(self.llm, system, user)
                candidates = parse_candidate_generation_response(
                    raw,
                    contract,
                    localization,
                    config,
                    round_index,
                    parent_claim_id,
                )
            except Exception as exc:
                schema_error = str(exc)
                previous_response = raw
                self.response_records.append(
                    {
                        "round_index": round_index,
                        "parent_claim_id": parent_claim_id,
                        "attempt_index": attempt_index,
                        "schema_attempt_index": attempt_index,
                        "validation_retry_index": validation_retry_index,
                        "is_retry": is_validation_retry or is_schema_retry,
                        "retry_kind": retry_kind,
                        "model": prompt_record["model"],
                        "raw_response": raw,
                        "candidate_count": 0,
                        "parse_error": schema_error,
                    }
                )
                if attempt_index + 1 < max_attempts:
                    continue
                raise CandidateGenerationError(schema_error) from exc

            self.response_records.append(
                {
                    "round_index": round_index,
                    "parent_claim_id": parent_claim_id,
                    "attempt_index": attempt_index,
                    "schema_attempt_index": attempt_index,
                    "validation_retry_index": validation_retry_index,
                    "is_retry": is_validation_retry or is_schema_retry,
                    "retry_kind": retry_kind,
                    "model": prompt_record["model"],
                    "raw_response": raw,
                    "candidate_count": len(candidates),
                    "parse_error": None,
                }
            )
            self.validation_feedback = None
            return candidates
        raise CandidateGenerationError(schema_error or "LLM candidate generation failed schema validation.")


def build_candidate_generation_prompt(
    contract: ClaimContract,
    localization: FailureLocalization,
    config: ClaimSearchConfig,
    round_index: int,
    parent_claim_id: str,
    candidate_history: list[CandidateClaimProposal] | None = None,
    schema_error: str | None = None,
    previous_response: str | None = None,
    executable_catalog: dict[str, Any] | None = None,
    validation_feedback: dict[str, Any] | None = None,
    round_failure_context: RoundFailureContext | None = None,
) -> tuple[str, str]:
    """Build the auditable prompt for LLM follow-up claim generation."""

    failure_context: dict[str, Any]
    if config.feedback_mode == "structured_diagnosis":
        failure_context = localization.model_dump(mode="json")
    else:
        failure_context = {
            "failure_kind": "unspecified_failure",
            "failed_gates": [],
            "diagnosis": "The parent claim was not confirmed; gate-specific diagnosis is withheld in this control arm.",
            "evidence": [],
        }
    payload = {
        "task": "generate_connected_followup_claim_candidates",
        "max_candidates": config.max_candidates_per_round,
        "round_index": round_index,
        "parent_claim_id": parent_claim_id,
        "original_contract": contract.model_dump(mode="json"),
        "failure_localization": failure_context,
        "candidate_history": [
            item.model_dump(mode="json") for item in (candidate_history or [])
        ],
        "immutable_contract_fields": {
            "estimand_type": contract.estimand.type,
            "predictor": contract.estimand.predictor,
            "group": contract.estimand.group.model_dump(mode="json") if contract.estimand.group else None,
            "direction": contract.estimand.direction,
            "covariates": list(contract.covariates),
            "required_confound_covariates": list(contract.gates.confound.require_covariates),
            "discovery_cohort": contract.discovery_cohort,
            "replication_cohorts": list(contract.replication_cohorts),
            "multiplicity": contract.gates.multiplicity.model_dump(mode="json"),
            "power": contract.gates.power.model_dump(mode="json"),
            "multiverse": contract.gates.multiverse.model_dump(mode="json"),
            "replication": contract.gates.replication.model_dump(mode="json"),
            "search_provenance": contract.search_provenance.model_dump(mode="json"),
        },
        "allowed_proposal_types": (
            _active_proposal_types(localization, config)
            if config.feedback_mode == "structured_diagnosis"
            else [
                "exploratory_followup_claim",
                *(["corrected_contract"] if config.allow_current_data_contract_repairs else []),
            ]
        ),
        "allowed_transform_types": [
            "narrower_outcome_family",
            "alternative_same_modality_outcome",
            "multivariate_pattern",
            "moderator_or_subgroup",
            "stronger_design",
            "fixed_estimand",
            "contract_correction",
        ],
        "generation_policy": [
            (
                f"Generate up to {config.max_candidates_per_round} scientifically distinct candidates. "
                f"Aim for exactly {config.max_candidates_per_round} when that many valid candidates exist; "
                "otherwise return fewer candidates or an empty list."
            ),
            "Every candidate must be connected to the original disease/cohort family, outcome modality, direction family, and motivation.",
            "Every candidate must include domain_core, preservation_check, and evidence_policy.",
            "Preserve immutable_contract_fields exactly except that a source-measured confound may be added to covariates and required_confound_covariates with an explicit scientific justification.",
            "proposed_contract.discovery_cohort and proposed_contract.replication_cohorts must exactly equal the parent cohorts in immutable_contract_fields.",
            "Novelty may come from any scientifically connected executable outcome within the original modality, including an alternative regional biomarker, a narrower family, or a multivariate/brainwide pattern; it may also come from a justified subgroup/inclusion, estimand refinement, added measured confound, or true contract correction.",
            (
                "A multivariate_pattern may change a scalar parent into unit=brainwide and must resolve to at least "
                f"{config.brainwide_min_features} distinct source-data outcome columns. The pipeline will apply a fixed "
                f"pattern-correlation floor of {config.adaptive_pattern_corr_floor:g} and feature-aware multiplicity; "
                "do not alter the parent gates yourself."
            ),
            "Every candidate intended for current_data_adaptive or current_data_contract_repair evaluation must include an executable proposed_contract.",
            "Patch-like proposed_contracts must be executable against executable_data_catalog when that field is present.",
            "Use only outcomes, predictors, group variables, covariates, and inclusion terms present in executable_data_catalog.",
            "Set disposition_label to null for generated candidates; do not emit downgrade labels.",
            "If preserving the original predictor/group contrast is impossible with the executable catalog, return candidates=[] instead of substituting a new predictor.",
            "Do not emit placeholder variables such as bench_group, bench_predictor, low_motion_subset, or free-text inclusion descriptions unless they are actual catalog columns.",
            "String values in inclusion filters must be quoted, for example sex == \"F\" rather than sex == female.",
            "Post-hoc follow-up candidates may set validation_split=current_data_adaptive, requires_new_evidence=false, and can_confirm_on_current_data=true.",
            "Same-data adaptive support is labeled exploratory_confirmed by the pipeline, never plain confirmed.",
            "Use current_data_contract_repair only for true contract_correction proposals that preserve the original scientific question.",
            "Use only evidence strings supplied in failure_localization.evidence for supported_by_evidence.",
            "candidate_history may show stricter pipeline policy_adjustments on executed contracts; always copy gate fields from original_contract, not from those adjusted history entries.",
            "For round 2 or later, responds_to_candidate_ids must contain one or more IDs from the immediately preceding round_failure_context.",
            "Changing only claim_id, question wording, transform label, evidence policy, or family size is a no-op and will be rejected.",
        ],
        "forbidden_actions": [
            "Do not switch to an unrelated outcome modality. Creative alternative outcomes and multivariate patterns within the same modality are allowed when the connection rationale is explicit.",
            "Do not change the predictor or group contrast for exploratory follow-up candidates.",
            "Do not reverse biological direction after seeing results.",
            "Do not weaken gates or lower thresholds.",
            "Do not drop original covariates or required confound covariates.",
            "Do not shrink multiplicity or search-provenance family size.",
            "Do not label same-data adaptive support as final confirmation.",
            "Do not invent cohorts, p-values, effect sizes, or gate results.",
            "Do not invent columns, filters, group labels, or data subsets.",
            "Do not introduce any holdout or external cohort that is absent from immutable_contract_fields.",
            "Do not use disposition_label unless the output schema explicitly requires a downgrade disposition; for this task use null.",
        ],
        "output_model": "LLMCandidateGenerationResponse",
        "output_schema": LLMCandidateGenerationResponse.model_json_schema(),
    }
    if executable_catalog is not None:
        payload["executable_data_catalog"] = executable_catalog
    if round_failure_context is not None:
        if config.feedback_mode == "structured_diagnosis":
            payload["round_failure_context"] = round_failure_context.model_dump(mode="json")
        else:
            payload["round_failure_context"] = {
                "round_index": round_failure_context.round_index,
                "failed_candidate_ids": [item.candidate_id for item in round_failure_context.failed_candidates],
                "message": "The preceding candidates failed. Propose connected executable alternatives.",
            }
    if validation_feedback is not None:
        payload["candidate_validation_retry"] = validation_feedback
    if schema_error is not None:
        payload["schema_retry"] = {
            "instruction": "Your previous response failed schema validation. Return a corrected JSON object matching output_schema exactly.",
            "schema_validation_error": schema_error,
            "previous_response": previous_response or "",
        }
    return CLAIM_CANDIDATE_SYSTEM_PROMPT, json.dumps(payload, indent=2, sort_keys=True)


def parse_candidate_generation_response(
    text: str,
    contract: ClaimContract,
    localization: FailureLocalization,
    config: ClaimSearchConfig,
    round_index: int,
    parent_claim_id: str,
) -> list[CandidateClaimProposal]:
    """Parse structured LLM output into typed candidate proposals."""

    payload = _parse_json_payload(text)
    response_payload = payload if isinstance(payload, dict) else {"candidates": payload}
    response = LLMCandidateGenerationResponse.model_validate(response_payload)
    candidates: list[CandidateClaimProposal] = []
    for index, item in enumerate(response.candidates[: config.max_candidates_per_round]):
        candidates.append(_candidate_from_llm_payload(contract, localization, parent_claim_id, round_index, index, item))
    return candidates


def _complete_structured_candidate_response(llm: LLMClient, system: str, user: str) -> str:
    complete_structured = getattr(llm, "complete_structured", None)
    if callable(complete_structured):
        return str(complete_structured(system, user, LLMCandidateGenerationResponse))
    return llm.complete(system, user)


def validate_candidate_claim(
    original: ClaimContract,
    candidate: CandidateClaimProposal,
    localization: FailureLocalization,
    config: ClaimSearchConfig,
    *,
    preflight_context: CandidatePreflightContext | None = None,
    preceding_failed_candidate_ids: set[str] | None = None,
) -> ProposalValidation:
    """Validate anti-hacking and connection constraints for a candidate."""

    if candidate.inferred_transform is None:
        candidate = _annotate_candidate_delta(original, candidate)
    active_localization = localization.model_copy(
        update={"allowed_proposal_types": _active_proposal_types(localization, config)}
    )
    base_payload = _proposal_payload(candidate)
    if candidate.proposal_type != "corrected_contract":
        base_payload["proposed_contract"] = None
    base = validate_new_claim_proposal(
        original,
        NewClaimProposal.model_validate(base_payload),
        active_localization,
        allowed_numeric_context=_candidate_numeric_context(original, candidate),
    )
    violations = list(base.violations)
    warnings = list(base.warnings)
    design_diagnostics: dict[str, Any] = {}

    if candidate.round_index > config.max_rounds:
        violations.append("Candidate round_index exceeds configured max_rounds.")
    if candidate.proposal_type not in {"corrected_contract", "exploratory_followup_claim"}:
        violations.append("Candidate proposals must be executable new claims.")
    if candidate.validation_split == "current_data_adaptive":
        if candidate.proposal_type != "exploratory_followup_claim":
            violations.append("Only connected follow-up proposals may use adaptive same-data evaluation.")
        if not candidate.can_confirm_on_current_data:
            violations.append("Adaptive same-data candidates must declare can_confirm_on_current_data=true.")
    if candidate.validation_split == "current_data_contract_repair":
        if candidate.proposal_type != "corrected_contract" or not config.allow_current_data_contract_repairs:
            violations.append("Only allowed contract corrections may use current-data repair evaluation.")
    if candidate.validation_split in {"excluded_validation", "future_required", "none"}:
        violations.append("Stage 3 candidates must be executable on source data; excluded evidence is evaluated only after freezing.")
    if candidate.can_confirm_on_current_data and candidate.proposal_type == "downgraded_claim":
        violations.append("Downgraded claims cannot be confirmable on current data.")
    unsupported_evidence = [item for item in candidate.supported_by_evidence if item not in localization.evidence]
    if unsupported_evidence:
        violations.append("Candidate cites evidence that was not supplied by failure localization.")

    if candidate.round_index == 1 and candidate.responds_to_candidate_ids:
        warnings.append("Round-1 candidate supplied responds_to_candidate_ids; references were ignored.")
    if candidate.round_index > 1:
        allowed_responses = preceding_failed_candidate_ids or set()
        if not candidate.responds_to_candidate_ids:
            violations.append("Later-round candidate must respond to a failed candidate from the preceding round.")
        elif not set(candidate.responds_to_candidate_ids).issubset(allowed_responses):
            violations.append("responds_to_candidate_ids includes a candidate outside the preceding failed round.")

    if not candidate.executable_contract_delta:
        violations.append("no_executable_change: candidate changes only non-executable identity or search-routing fields.")
    if candidate.transform_match is False:
        warnings.append(
            f"Declared transform {candidate.declared_transform!r} does not match inferred transform {candidate.inferred_transform!r}."
        )

    violations.extend(_domain_core_violations(original, candidate))
    violations.extend(_followup_contract_connection_violations(original, candidate))
    candidate_outcome = candidate.proposed_contract.estimand.outcome
    if (
        candidate.proposed_contract.estimand.unit == "brainwide"
        and isinstance(candidate_outcome, list)
        and len(set(candidate_outcome)) < config.brainwide_min_features
    ):
        violations.append(
            "Brainwide candidate contracts require at least "
            f"{config.brainwide_min_features} distinct outcome columns for pattern replication."
        )
    if candidate.proposed_contract is not None and preflight_context is not None:
        violations.extend(_inclusion_policy_violations(original, candidate, preflight_context))
        preflight = preflight_context.validate_contract(candidate.proposed_contract)
        violations.extend(preflight.violations)
        if candidate.proposed_contract.estimand.unit == "brainwide":
            resolved_sets = [
                set(outcomes)
                for outcomes in preflight.resolved_outcome_columns.values()
            ]
            outcome_counts = {
                cohort: len(outcomes)
                for cohort, outcomes in preflight.resolved_outcome_columns.items()
            }
            shared_outcome_count = len(set.intersection(*resolved_sets)) if resolved_sets else 0
            undersized = {
                cohort: count
                for cohort, count in outcome_counts.items()
                if count < config.brainwide_min_features
            }
            if undersized:
                violations.append(
                    "Preflight: brainwide candidates require at least "
                    f"{config.brainwide_min_features} distinct outcome columns "
                    f"in every source cohort; observed {undersized}."
                )
            if shared_outcome_count < config.brainwide_min_features:
                violations.append(
                    "Preflight: brainwide candidates require at least "
                    f"{config.brainwide_min_features} outcome columns shared across every source cohort; "
                    f"observed {shared_outcome_count}."
                )
        warnings.extend(preflight.warnings)
        design_diagnostics = dict(preflight.design_diagnostics)
        if candidate.proposed_contract.estimand.unit == "brainwide":
            design_diagnostics["brainwide_pattern"] = {
                "minimum_feature_count": config.brainwide_min_features,
                "resolved_outcome_counts": outcome_counts,
                "shared_outcome_count": shared_outcome_count,
            }

    ok = not violations
    return ProposalValidation(
        ok=ok,
        violations=violations,
        warnings=warnings,
        provenance_compliant=ok,
        current_data_confirmability_ok=ok and not any("current data" in item for item in violations),
        checked_contract=candidate.proposed_contract is not None,
        useful=ok,
        accepted_proposal_type=base.accepted_proposal_type,
        design_diagnostics=design_diagnostics,
    )


def run_claim_search(
    contract: ClaimContract,
    verdict: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
    *,
    config: ClaimSearchConfig | None = None,
    candidate_generator: CandidateGenerator | None = None,
    llm: LLMClient | None = None,
    evaluator: CandidateEvaluator | None = None,
    preflight_context: CandidatePreflightContext | None = None,
) -> ClaimSearchState:
    """Run a bounded iterative candidate-generation loop."""

    cfg = config or ClaimSearchConfig()
    localization = localize_failure(contract, verdict, results)
    if localization.failure_kind == "none":
        return ClaimSearchState(
            original_claim=contract,
            failure_localization=localization,
            lineage_graph={"nodes": [contract.claim_id], "edges": []},
            used_evidence=localization.evidence,
            stopped_reason="already_confirmed",
        )

    if candidate_generator is None:
        if llm is None:
            return ClaimSearchState(
                original_claim=contract,
                failure_localization=localization,
                lineage_graph={"nodes": [contract.claim_id], "edges": []},
                used_evidence=localization.evidence,
                stopped_reason="llm_unavailable",
            )
        generator: CandidateGenerator = LLMClaimCandidateGenerator(
            llm,
            preflight_context=preflight_context,
        )
    else:
        generator = candidate_generator
    history: list[CandidateClaimProposal] = []
    unretained_candidate_attempts: list[UnretainedCandidateAttempt] = []
    duplicate_candidates: list[DuplicateCandidateRecord] = []
    evaluations: list[CandidateEvaluation] = []
    round_failure_contexts: list[RoundFailureContext] = []
    round_summaries: list[RoundSearchSummary] = []
    lineage = {"nodes": [contract.claim_id], "edges": []}
    stopped: StoppedReason = "max_rounds_exhausted"
    generated_candidate_count = 0
    schema_valid_candidate_count = 0
    valid_candidate_count = 0
    current_data_evaluated_count = 0
    tested_hypothesis_signatures: set[str] = set()
    parent_declared_family_size = max(
        int(contract.search_provenance.family_size),
        int(contract.gates.multiplicity.family_size),
    )

    def finish(reason: StoppedReason) -> ClaimSearchState:
        supported = [
            item.candidate_id
            for item in evaluations
            if item.current_data_supported and not item.execution_error
        ]
        provisional = [
            item.candidate_id
            for item in evaluations
            if item.provisional_supported and not item.execution_error
        ]
        return ClaimSearchState(
            original_claim=contract,
            failure_localization=localization,
            lineage_graph=lineage,
            used_evidence=localization.evidence,
            candidate_history=history,
            unretained_candidate_attempts=unretained_candidate_attempts,
            duplicate_candidates=duplicate_candidates,
            evaluations=evaluations,
            internally_supported_candidate_ids=supported,
            round_failure_contexts=round_failure_contexts,
            round_summaries=round_summaries,
            provisional_supported_candidate_ids=provisional,
            final_search_family_size=max(
                [parent_declared_family_size + len(tested_hypothesis_signatures)]
                + [int(item.final_family_size or 0) for item in evaluations]
            ),
            generated_candidate_count=generated_candidate_count,
            schema_valid_candidate_count=schema_valid_candidate_count,
            unique_candidate_count=len(history),
            valid_candidate_count=valid_candidate_count,
            current_data_evaluated_count=current_data_evaluated_count,
            unique_hypotheses_tested_count=len(tested_hypothesis_signatures),
            llm_candidate_prompts=_generator_prompt_records(generator),
            llm_candidate_responses=_generator_response_records(generator),
            stopped_reason=reason,
        )

    for round_index in range(1, cfg.max_rounds + 1):
        candidate_validations: list[tuple[CandidateClaimProposal, ProposalValidation]] = []
        candidates: list[CandidateClaimProposal] = []
        round_duplicates: list[DuplicateCandidateRecord] = []
        round_proposals_returned = 0
        preceding_failed_ids = (
            {item.candidate_id for item in round_failure_contexts[-1].failed_candidates}
            if round_failure_contexts
            else set()
        )
        for validation_attempt in range(cfg.llm_schema_retries + 1):
            try:
                if hasattr(generator, "candidate_history"):
                    setattr(generator, "candidate_history", list(history))
                if hasattr(generator, "round_failure_context"):
                    setattr(
                        generator,
                        "round_failure_context",
                        round_failure_contexts[-1] if round_failure_contexts else None,
                    )
                if hasattr(generator, "validation_retry_index"):
                    setattr(generator, "validation_retry_index", validation_attempt)
                candidates = generator(contract, localization, cfg, round_index, contract.claim_id)
            except CandidateGenerationError:
                stopped = "candidate_generation_failed"
                break
            generated_candidate_count += len(candidates)
            schema_valid_candidate_count += len(candidates)
            round_proposals_returned += len(candidates)
            candidates = [
                _annotate_candidate_delta(contract, candidate)
                for candidate in candidates
            ]
            candidates = candidates[: cfg.max_candidates_per_round]
            candidates, attempt_duplicates = _deduplicate_candidates(candidates, history)
            round_duplicates.extend(attempt_duplicates)
            candidate_validations = [
                (
                    candidate,
                    validate_candidate_claim(
                        contract,
                        candidate,
                        localization,
                        cfg,
                        preflight_context=preflight_context,
                        preceding_failed_candidate_ids=preceding_failed_ids,
                    ),
                )
                for candidate in candidates
            ]
            if not _should_retry_after_validation(candidate_validations, validation_attempt, cfg, generator):
                break
            unretained_candidate_attempts.extend(
                UnretainedCandidateAttempt(
                    round_index=round_index,
                    validation_retry_index=validation_attempt,
                    disposition="deterministic_validation_retry",
                    proposal=candidate,
                    validation=validation,
                )
                for candidate, validation in candidate_validations
            )
            setattr(generator, "validation_feedback", _validation_retry_feedback(candidate_validations))
        duplicate_candidates.extend(round_duplicates)
        if stopped == "candidate_generation_failed" or not candidates:
            round_summaries.append(
                RoundSearchSummary(
                    round_index=round_index,
                    proposals_returned=round_proposals_returned,
                    schema_valid=round_proposals_returned,
                    policy_valid=sum(validation.ok for _, validation in candidate_validations),
                    unique_source_tested=0,
                    execution_complete=0,
                    provisional_passes=0,
                    final_round_passes=0,
                    multiplicity_retractions=0,
                    cumulative_unique_tested=current_data_evaluated_count,
                    cumulative_unique_hypotheses_tested=len(tested_hypothesis_signatures),
                    final_family_size=max(
                        [parent_declared_family_size + len(tested_hypothesis_signatures)]
                        + [int(item.final_family_size or 0) for item in evaluations]
                    ),
                )
            )
            if stopped != "candidate_generation_failed":
                stopped = "no_candidates"
            break
        round_evaluations: list[CandidateEvaluation] = []
        for candidate, validation in candidate_validations:
            valid_candidate_count += int(validation.ok)
            eligible = _eligible_for_evaluation(candidate, validation, cfg, evaluator)
            if eligible:
                current_data_evaluated_count += 1
                declared_family_size = max(
                    int(candidate.proposed_contract.search_provenance.family_size),
                    int(candidate.proposed_contract.gates.multiplicity.family_size),
                )
                candidate_hypothesis_signatures = _candidate_hypothesis_signatures(
                    candidate,
                    preflight_context,
                )
                tested_hypothesis_signatures.update(candidate_hypothesis_signatures)
                candidate = _candidate_with_effective_family_size(
                    candidate,
                    contract,
                    len(tested_hypothesis_signatures),
                    cfg,
                )
            history.append(candidate)
            lineage["nodes"].append(candidate.candidate_id)
            lineage["edges"].append(
                {
                    "source": candidate.parent_claim_id,
                    "target": candidate.candidate_id,
                    "transform_type": candidate.transform_type,
                    "round_index": candidate.round_index,
                }
            )
            evaluation = CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                proposal=candidate,
                validation=validation,
                eligible_for_confirmation=eligible,
                validation_split=candidate.validation_split,
                declared_family_size=(declared_family_size if eligible else None),
                effective_family_size=(
                    candidate.proposed_contract.search_provenance.family_size if eligible else None
                ),
                search_hypothesis_count=(len(candidate_hypothesis_signatures) if eligible else 0),
                cumulative_search_hypothesis_count=(
                    len(tested_hypothesis_signatures) if eligible else 0
                ),
                design_diagnostics=(
                    {"preflight": validation.design_diagnostics}
                    if validation.design_diagnostics
                    else {}
                ),
                blocked_reason=None if eligible else _blocked_reason(candidate, validation, evaluator),
            )
            if eligible and evaluator is not None:
                try:
                    _evaluate_candidate(candidate, evaluation, evaluator)
                except Exception as exc:
                    evaluation.execution_error = _analysis_execution_error(exc)
                    evaluation.design_diagnostics = _exception_design_diagnostics(exc)
                    if str(evaluation.execution_error).startswith("analysis_non_identifiable:"):
                        evaluation.blocked_reason = "analysis_non_identifiable"
                    evaluation.evaluated = True
            evaluation.provisional_supported = evaluation.current_data_supported
            evaluations.append(evaluation)
            round_evaluations.append(evaluation)
        if evaluator is None:
            stopped = "no_evaluator"
            break

        retracted_ids = _readjudicate_provisional_passes(
            contract,
            history,
            evaluations,
            evaluator,
            len(tested_hypothesis_signatures),
        )
        failure_context = _build_round_failure_context(
            round_index,
            round_evaluations,
            evaluations,
            retracted_ids,
        )
        round_failure_contexts.append(failure_context)
        round_summaries.append(
            RoundSearchSummary(
                round_index=round_index,
                proposals_returned=round_proposals_returned,
                schema_valid=round_proposals_returned,
                policy_valid=sum(validation.ok for _, validation in candidate_validations),
                unique_source_tested=sum(item.evaluated for item in round_evaluations),
                execution_complete=sum(item.evaluated and not item.execution_error for item in round_evaluations),
                provisional_passes=sum(item.provisional_supported for item in round_evaluations),
                final_round_passes=sum(item.current_data_supported for item in round_evaluations),
                multiplicity_retractions=len(retracted_ids),
                cumulative_unique_tested=current_data_evaluated_count,
                cumulative_unique_hypotheses_tested=len(tested_hypothesis_signatures),
                final_family_size=max(
                    [parent_declared_family_size + len(tested_hypothesis_signatures)]
                    + [int(item.final_family_size or 0) for item in evaluations]
                ),
            )
        )
        if not any(validation.ok for _, validation in candidate_validations):
            stopped = "no_valid_candidates"
            break
        if not any(item.evaluated for item in round_evaluations):
            stopped = "no_valid_candidates"
            break
        if not failure_context.failed_candidates:
            stopped = "all_candidates_supported"
            break

    return finish(stopped)


def build_claim_search_artifacts(
    contract: ClaimContract,
    verdict: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
    *,
    config: ClaimSearchConfig | None = None,
    llm: LLMClient | None = None,
    candidate_generator: CandidateGenerator | None = None,
    evaluator: CandidateEvaluator | None = None,
    preflight_context: CandidatePreflightContext | None = None,
) -> dict[str, Any]:
    """Build serializable claim-search artifacts with optional candidate execution."""

    cfg = config or ClaimSearchConfig()
    state = run_claim_search(
        contract,
        verdict,
        results,
        config=cfg,
        candidate_generator=candidate_generator,
        llm=llm,
        evaluator=evaluator,
        preflight_context=preflight_context,
    )
    return {
        "claim_search_config": cfg.model_dump(mode="json"),
        "failure_localization": state.failure_localization.model_dump(mode="json") if state.failure_localization else None,
        "claim_search_trace": state.model_dump(mode="json"),
        "candidate_claims": [item.model_dump(mode="json") for item in state.candidate_history],
        "unretained_candidate_attempts": [
            item.model_dump(mode="json") for item in state.unretained_candidate_attempts
        ],
        "duplicate_candidates": [item.model_dump(mode="json") for item in state.duplicate_candidates],
        "proposal_validation": [item.validation.model_dump(mode="json") for item in state.evaluations],
        "candidate_evaluations": [item.model_dump(mode="json") for item in state.evaluations],
        "claim_lineage": state.lineage_graph,
        "llm_candidate_prompts": state.llm_candidate_prompts,
        "llm_candidate_responses": state.llm_candidate_responses,
    }


def summarize_claim_search(states: list[ClaimSearchState]) -> dict[str, Any]:
    """Summarize iterative search traces."""

    evaluations = [evaluation for state in states for evaluation in state.evaluations]
    effective_evaluations = [evaluation for evaluation in evaluations if not evaluation.execution_error]
    candidates = [candidate for state in states for candidate in state.candidate_history]
    unretained_attempts = [
        attempt
        for state in states
        for attempt in state.unretained_candidate_attempts
    ]
    duplicates = [duplicate for state in states for duplicate in state.duplicate_candidates]
    valid = [evaluation for evaluation in evaluations if evaluation.validation.ok]
    connected = [evaluation for evaluation in valid if not any("connection" in item.lower() for item in evaluation.validation.violations)]
    connected_executable = [
        evaluation
        for evaluation in connected
        if evaluation.evaluated and not evaluation.execution_error
    ]
    gaming = [
        evaluation
        for evaluation in evaluations
        if any(_is_hacking_violation(item) for item in evaluation.validation.violations)
    ]
    unretained_gaming = [
        attempt
        for attempt in unretained_attempts
        if any(_is_hacking_violation(item) for item in attempt.validation.violations)
    ]
    any_supported = [
        evaluation for evaluation in effective_evaluations if evaluation.current_data_supported
    ]
    exploratory_confirmed = [
        evaluation for evaluation in evaluations if not evaluation.execution_error and evaluation.final_label == "exploratory_confirmed"
    ]
    same_data_exploratory = [
        evaluation
        for evaluation in exploratory_confirmed
        if evaluation.validation_split == "current_data_adaptive" or evaluation.same_underlying_data is True
    ]
    contract_repair_supported = [
        evaluation
        for evaluation in effective_evaluations
        if evaluation.final_label == "contract_repair_supported"
    ]
    execution_errors = [evaluation for evaluation in evaluations if evaluation.execution_error]
    non_identifiable = [
        evaluation
        for evaluation in evaluations
        if evaluation.blocked_reason == "analysis_non_identifiable"
        or any("analysis_non_identifiable" in str(item) for item in evaluation.validation.violations)
        or str(evaluation.execution_error or "").startswith("analysis_non_identifiable:")
    ]
    execution_error_types = Counter(
        str(evaluation.execution_error).split(":", 1)[0]
        for evaluation in execution_errors
    )
    preflight_blocked = [
        evaluation
        for evaluation in evaluations
        if any(str(violation).startswith("Preflight:") for violation in evaluation.validation.violations)
    ]
    preflight_pass_count = len(evaluations) - len(preflight_blocked)
    searches_by_target = Counter(str(state.source_metadata.get("target_family") or "unknown") for state in states)
    searches_by_source_mode = Counter(str(state.source_metadata.get("source_mode") or "unknown") for state in states)
    generated_candidate_count = sum(
        state.generated_candidate_count or len(state.candidate_history) + len(state.duplicate_candidates) for state in states
    )
    unretained_generated_candidate_count = max(
        generated_candidate_count - len(candidates) - len(duplicates),
        0,
    )
    valid_connected_lineage_count = sum(
        1
        for state in states
        if any(
            evaluation.validation.ok
            and evaluation.evaluated
            and not evaluation.execution_error
            and not any("connection" in item.lower() for item in evaluation.validation.violations)
            for evaluation in state.evaluations
        )
    )
    supported_search_count = sum(
        1 for state in states if state.internally_supported_candidate_ids
    )
    confirmed_search_count = 0
    raw_final_label_counts = Counter(str(evaluation.final_label or "none") for evaluation in evaluations)
    effective_final_label_counts = Counter(str(evaluation.final_label or "none") for evaluation in effective_evaluations)
    raw_final_labels_by_target: dict[str, dict[str, int]] = {}
    effective_final_labels_by_target: dict[str, dict[str, int]] = {}
    for state in states:
        target = str(state.source_metadata.get("target_family") or "unknown")
        for evaluation in state.evaluations:
            label = str(evaluation.final_label or "none")
            raw_final_labels_by_target.setdefault(target, {})
            raw_final_labels_by_target[target][label] = raw_final_labels_by_target[target].get(label, 0) + 1
            if not evaluation.execution_error:
                effective_final_labels_by_target.setdefault(target, {})
                effective_final_labels_by_target[target][label] = effective_final_labels_by_target[target].get(label, 0) + 1
    metrics_by_target_family = _stratified_state_metrics(states, "target_family")
    metrics_by_source_mode = _stratified_state_metrics(states, "source_mode")
    metrics_by_synthetic_failure_family = _stratified_state_metrics(states, "synthetic_failure_family")
    metrics_by_transform = _stratified_evaluation_metrics(evaluations)
    metrics_by_inferred_transform = _stratified_evaluation_metrics(evaluations, inferred=True)
    provisional_passes = [
        evaluation
        for evaluation in effective_evaluations
        if evaluation.provisional_supported
    ]
    final_internal_passes = [
        evaluation
        for evaluation in effective_evaluations
        if evaluation.current_data_supported
    ]
    unique_supported_contracts = {
        _candidate_exact_contract_signature(evaluation.proposal)
        for evaluation in final_internal_passes
    }
    gate_failure_counts = Counter(
        gate
        for state in states
        for context in state.round_failure_contexts
        for failure in context.failed_candidates
        for gate in failure.failed_gates
    )
    declared_transform_counts = Counter(
        str(candidate.declared_transform or candidate.transform_type) for candidate in candidates
    )
    inferred_transform_counts = Counter(str(candidate.inferred_transform or "unknown") for candidate in candidates)
    return {
        "n_searches": len(states),
        "parent_lineage_count": len(states),
        "proposals_returned_count": generated_candidate_count,
        "schema_valid_candidate_count": sum(
            state.schema_valid_candidate_count for state in states
        ),
        "generated_candidate_count": generated_candidate_count,
        "unretained_generated_candidate_count": unretained_generated_candidate_count,
        "unretained_validation_candidate_count": len(unretained_attempts),
        "unretained_validation_reason_counts": dict(
            Counter(
                violation
                for attempt in unretained_attempts
                for violation in attempt.validation.violations
            )
        ),
        "candidate_count": len(candidates),
        "unique_candidate_count": len(candidates),
        "duplicate_candidate_count": len(duplicates),
        "valid_candidate_count": len(valid),
        "policy_valid_candidate_count": len(valid),
        "valid_connected_candidate_count": len(connected),
        "valid_connected_candidate_rate": len(connected) / len(candidates) if candidates else 0.0,
        "valid_connected_executable_candidate_count": len(connected_executable),
        "valid_connected_executable_candidate_rate": (
            len(connected_executable) / len(candidates) if candidates else 0.0
        ),
        "valid_connected_lineage_count": valid_connected_lineage_count,
        "valid_connected_lineage_rate": valid_connected_lineage_count / len(states) if states else 0.0,
        "preflight_pass_candidate_count": preflight_pass_count,
        "preflight_pass_candidate_rate": preflight_pass_count / len(candidates) if candidates else 0.0,
        "preflight_block_count": len(preflight_blocked),
        "admissible_evaluation_count": sum(1 for item in evaluations if item.eligible_for_confirmation),
        "current_data_evaluated_count": sum(state.current_data_evaluated_count for state in states),
        "unique_source_tested_count": sum(state.current_data_evaluated_count for state in states),
        "unique_hypotheses_tested_count": sum(
            state.unique_hypotheses_tested_count for state in states
        ),
        "execution_complete_candidate_count": sum(
            evaluation.evaluated and not evaluation.execution_error for evaluation in evaluations
        ),
        "provisional_internal_pass_count": len(provisional_passes),
        "final_multiplicity_adjusted_internal_pass_count": len(final_internal_passes),
        "unique_internally_supported_contract_count": len(unique_supported_contracts),
        "parents_with_internal_support_count": supported_search_count,
        "system_yield_supported_candidates_per_parent": (
            len(final_internal_passes) / len(states) if states else 0.0
        ),
        "system_yield_supported_parents": (
            supported_search_count / len(states) if states else 0.0
        ),
        "multiplicity_retraction_count": sum(
            evaluation.multiplicity_retracted for evaluation in evaluations
        ),
        "exploratory_confirmed_count": len(exploratory_confirmed),
        "same_data_exploratory_confirmed_count": len(same_data_exploratory),
        "confirmed_count": 0,
        "final_confirmed_count": 0,
        "supported_candidate_count": len(any_supported),
        "any_supported_candidate_count": len(any_supported),
        "supported_search_count": supported_search_count,
        "any_supported_search_count": supported_search_count,
        "confirmed_search_count": confirmed_search_count,
        "contract_repair_supported_count": len(contract_repair_supported),
        "contract_repair_confirmed_count": len(contract_repair_supported),
        "holdout_confirmed_count": 0,
        "external_confirmed_count": 0,
        "confirmed_on_external_evidence_count": 0,
        "confirmed_on_excluded_evidence_count": 0,
        "false_current_data_confirmation_count": sum(1 for item in evaluations if _false_current_data_confirmation(item)),
        "hacking_block_count": len(gaming) + len(unretained_gaming),
        "no_holdout_abstention_count": 0,
        "execution_error_count": len(execution_errors),
        "execution_error_type_counts": dict(execution_error_types),
        "excluded_evidence_error_count": 0,
        "excluded_evidence_unavailable_count": 0,
        "analysis_non_identifiable_count": len(non_identifiable),
        "excluded_evidence_query_count": 0,
        "excluded_evidence_error_type_counts": {},
        "raw_final_label_counts": dict(raw_final_label_counts),
        "effective_final_label_counts": dict(effective_final_label_counts),
        "final_label_counts": dict(effective_final_label_counts),
        "stopped_reason_counts": dict(Counter(state.stopped_reason for state in states)),
        "gate_failure_counts": dict(gate_failure_counts),
        "declared_transform_counts": dict(declared_transform_counts),
        "inferred_transform_counts": dict(inferred_transform_counts),
        "transform_match_count": sum(candidate.transform_match is True for candidate in candidates),
        "transform_mismatch_count": sum(candidate.transform_match is False for candidate in candidates),
        "policy_adjusted_candidate_count": sum(
            bool(candidate.policy_adjustments) for candidate in candidates
        ),
        "no_executable_change_count": (
            sum(candidate.inferred_transform == "no_executable_change" for candidate in candidates)
            + sum(
                attempt.proposal.inferred_transform == "no_executable_change"
                for attempt in unretained_attempts
            )
        ),
        "searches_by_target_family": dict(searches_by_target),
        "searches_by_source_mode": dict(searches_by_source_mode),
        "raw_candidate_final_label_counts_by_target_family": raw_final_labels_by_target,
        "effective_candidate_final_label_counts_by_target_family": effective_final_labels_by_target,
        "candidate_final_label_counts_by_target_family": effective_final_labels_by_target,
        "metrics_by_target_family": metrics_by_target_family,
        "metrics_by_source_mode": metrics_by_source_mode,
        "metrics_by_transform": metrics_by_transform,
        "metrics_by_inferred_transform": metrics_by_inferred_transform,
        "metrics_by_synthetic_failure_family": metrics_by_synthetic_failure_family,
        "deprecated_metric_aliases": {
            "confirmed_count": "final_confirmed_count",
            "contract_repair_confirmed_count": "contract_repair_supported_count",
            "any_supported_candidate_count": "supported_candidate_count",
        },
    }


def _stratified_state_metrics(
    states: list[ClaimSearchState],
    metadata_key: str,
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[ClaimSearchState]] = {}
    for state in states:
        value = str(state.source_metadata.get(metadata_key) or "unknown")
        grouped.setdefault(value, []).append(state)
    out: dict[str, dict[str, int | float]] = {}
    for value, group in sorted(grouped.items()):
        evaluations = [item for state in group for item in state.evaluations]
        supported = sum(bool(state.internally_supported_candidate_ids) for state in group)
        out[value] = {
            "n_searches": len(group),
            "generated_candidate_count": sum(state.generated_candidate_count for state in group),
            "unique_candidate_count": sum(state.unique_candidate_count for state in group),
            "valid_candidate_count": sum(state.valid_candidate_count for state in group),
            "current_data_evaluated_count": sum(state.current_data_evaluated_count for state in group),
            "unique_hypotheses_tested_count": sum(
                state.unique_hypotheses_tested_count for state in group
            ),
            "execution_complete_candidate_count": sum(
                item.evaluated and not item.execution_error for item in evaluations
            ),
            "provisional_internal_pass_count": sum(
                item.provisional_supported and not item.execution_error for item in evaluations
            ),
            "final_multiplicity_adjusted_internal_pass_count": sum(
                item.current_data_supported and not item.execution_error for item in evaluations
            ),
            "multiplicity_retraction_count": sum(item.multiplicity_retracted for item in evaluations),
            "supported_search_count": supported,
            "supported_search_rate": supported / len(group) if group else 0.0,
            "contract_repair_supported_count": sum(
                item.final_label == "contract_repair_supported" and not item.execution_error
                for item in evaluations
            ),
            "holdout_confirmed_count": 0,
            "external_confirmed_count": 0,
            "analysis_non_identifiable_count": sum(
                item.blocked_reason == "analysis_non_identifiable"
                or str(item.execution_error or "").startswith("analysis_non_identifiable:")
                for item in evaluations
            ),
            "excluded_evidence_unavailable_count": 0,
            "execution_error_count": sum(bool(item.execution_error) for item in evaluations),
        }
    return out


def _stratified_evaluation_metrics(
    evaluations: list[CandidateEvaluation],
    *,
    inferred: bool = False,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[CandidateEvaluation]] = {}
    for evaluation in evaluations:
        key = (
            str(evaluation.proposal.inferred_transform or "unknown")
            if inferred
            else str(evaluation.proposal.declared_transform or evaluation.proposal.transform_type)
        )
        grouped.setdefault(key, []).append(evaluation)
    return {
        transform: {
            "candidate_count": len(group),
            "valid_count": sum(item.validation.ok for item in group),
            "current_data_evaluated_count": sum(item.evaluated for item in group),
            "current_data_supported_count": sum(item.current_data_supported and not item.execution_error for item in group),
            "provisional_internal_pass_count": sum(item.provisional_supported and not item.execution_error for item in group),
            "multiplicity_retraction_count": sum(item.multiplicity_retracted for item in group),
            "holdout_confirmed_count": 0,
            "external_confirmed_count": 0,
            "analysis_non_identifiable_count": sum(
                item.blocked_reason == "analysis_non_identifiable"
                or str(item.execution_error or "").startswith("analysis_non_identifiable:")
                for item in group
            ),
            "excluded_evidence_unavailable_count": 0,
            "execution_error_count": sum(bool(item.execution_error) for item in group),
        }
        for transform, group in sorted(grouped.items())
    }


def _parse_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _candidate_from_llm_payload(
    contract: ClaimContract,
    localization: FailureLocalization,
    parent_claim_id: str,
    round_index: int,
    index: int,
    payload: LLMCandidateProposal,
) -> CandidateClaimProposal:
    transform_type = payload.transform_type
    if not transform_type:
        raise ValueError("LLM candidate missing transform_type.")
    candidate_id = f"{contract.claim_id}_r{round_index}_c{index + 1}_{transform_type}"
    evidence_policy = payload.evidence_policy
    data = {
        "candidate_id": candidate_id,
        "parent_claim_id": parent_claim_id,
        "round_index": round_index,
        "transform_type": transform_type,
        "domain_core": payload.domain_core.model_dump(mode="json"),
        "preservation_check": payload.preservation_check.model_dump(mode="json"),
        "evidence_policy": evidence_policy.model_dump(mode="json"),
        "connection_rationale": payload.connection_rationale,
        "validation_split": evidence_policy.validation_split,
        "source_claim_id": contract.claim_id,
        "proposal_type": payload.proposal_type,
        "rationale": payload.rationale,
        "proposed_question": payload.proposed_question,
        "proposed_contract": payload.proposed_contract.model_dump(mode="json"),
        "disposition_label": payload.disposition_label,
        "provenance": evidence_policy.provenance,
        "requires_new_evidence": evidence_policy.requires_new_evidence,
        "can_confirm_on_current_data": evidence_policy.can_confirm_on_current_data,
        "supported_by_evidence": payload.supported_by_evidence,
        "responds_to_candidate_ids": payload.responds_to_candidate_ids if round_index > 1 else [],
        "declared_transform": transform_type,
    }
    return CandidateClaimProposal.model_validate(data)


def _candidate_scientific_signature(candidate: CandidateClaimProposal) -> str:
    payload = _contract_identity_payload(candidate.proposed_contract, include_family_size=False)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _candidate_exact_contract_signature(candidate: CandidateClaimProposal) -> str:
    payload = _contract_identity_payload(candidate.proposed_contract, include_family_size=True)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _deduplicate_candidates(
    candidates: list[CandidateClaimProposal],
    history: list[CandidateClaimProposal],
) -> tuple[list[CandidateClaimProposal], list[DuplicateCandidateRecord]]:
    seen = {_candidate_scientific_signature(candidate): candidate.candidate_id for candidate in history}
    unique: list[CandidateClaimProposal] = []
    duplicates: list[DuplicateCandidateRecord] = []
    for candidate in candidates:
        signature = _candidate_scientific_signature(candidate)
        duplicate_of = seen.get(signature)
        if duplicate_of is not None:
            duplicates.append(
                DuplicateCandidateRecord(
                    candidate_id=candidate.candidate_id,
                    duplicate_of=duplicate_of,
                    parent_claim_id=candidate.parent_claim_id,
                    round_index=candidate.round_index,
                    scientific_signature=signature,
                )
            )
            continue
        seen[signature] = candidate.candidate_id
        unique.append(candidate)
    return unique, duplicates


def _proposal_payload(candidate: CandidateClaimProposal) -> dict[str, Any]:
    return {
        "source_claim_id": candidate.source_claim_id,
        "proposal_type": candidate.proposal_type,
        "rationale": candidate.rationale,
        "proposed_question": candidate.proposed_question,
        "proposed_contract": (
            candidate.proposed_contract.model_dump(mode="json") if candidate.proposed_contract is not None else None
        ),
        "disposition_label": candidate.disposition_label,
        "provenance": candidate.provenance,
        "requires_new_evidence": candidate.requires_new_evidence,
        "can_confirm_on_current_data": candidate.can_confirm_on_current_data,
        "supported_by_evidence": list(candidate.supported_by_evidence),
    }


def _active_proposal_types(
    localization: FailureLocalization,
    config: ClaimSearchConfig,
) -> list[str]:
    active = ["exploratory_followup_claim"]
    if localization.current_data_repair_allowed and config.allow_current_data_contract_repairs:
        active.append("corrected_contract")
    return active


def _contract_identity_payload(
    contract: ClaimContract,
    *,
    include_family_size: bool,
) -> dict[str, Any]:
    """Return executable contract identity without wording or claim identifiers."""

    data = contract.model_dump(mode="json")
    data.pop("claim_id", None)
    data.pop("question", None)
    data.pop("reporting_language_allowed", None)
    estimand = dict(data.get("estimand") or {})
    if isinstance(estimand.get("outcome"), list):
        estimand["outcome"] = sorted({str(item) for item in estimand["outcome"]})
    data["estimand"] = estimand
    provenance = dict(data.pop("search_provenance", {}) or {})
    if include_family_size:
        data["search_family_size"] = provenance.get("family_size")
    gates = dict(data.get("gates") or {})
    multiplicity = dict(gates.get("multiplicity") or {})
    if not include_family_size:
        multiplicity.pop("family_size", None)
    gates["multiplicity"] = multiplicity
    data["gates"] = gates
    return data


def _flatten_contract_delta(
    original: Any,
    revised: Any,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    if isinstance(original, dict) and isinstance(revised, dict):
        out: dict[str, Any] = {}
        for key in sorted(set(original) | set(revised)):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_contract_delta(original.get(key), revised.get(key), prefix=path))
        return out
    if original == revised:
        return {}
    return {prefix: {"parent": original, "candidate": revised}}


def _executable_contract_delta(
    original: ClaimContract,
    revised: ClaimContract,
) -> dict[str, Any]:
    original_payload = _contract_identity_payload(original, include_family_size=False)
    revised_payload = _contract_identity_payload(revised, include_family_size=False)
    return _flatten_contract_delta(original_payload, revised_payload)


def _infer_transform(
    delta: Mapping[str, Any],
    original: ClaimContract,
    revised: ClaimContract,
) -> str:
    paths = set(delta)
    if not paths:
        return "no_executable_change"
    if original.estimand.unit == "scalar" and revised.estimand.unit == "brainwide":
        inferred = "multivariate_pattern"
    elif (
        original.estimand.unit == "scalar"
        and revised.estimand.unit == "scalar"
        and original.estimand.outcome != revised.estimand.outcome
    ):
        inferred = "alternative_same_modality_outcome"
    elif any(path.startswith("estimand.outcome") or path.startswith("estimand.unit") or path.startswith("estimand.region_set") for path in paths):
        inferred = "narrower_outcome_family"
    elif any(path == "inclusion" or path.startswith("inclusion.") for path in paths):
        inferred = "moderator_or_subgroup"
    else:
        inferred = "contract_correction"
    families = {
        "outcome" if path.startswith("estimand.outcome") or path.startswith("estimand.unit") or path.startswith("estimand.region_set") else
        "inclusion" if path == "inclusion" or path.startswith("inclusion.") else
        "other"
        for path in paths
    }
    return inferred if len(families) == 1 else "compound_change"


def _annotate_candidate_delta(
    original: ClaimContract,
    candidate: CandidateClaimProposal,
) -> CandidateClaimProposal:
    delta = _executable_contract_delta(original, candidate.proposed_contract)
    inferred = _infer_transform(delta, original, candidate.proposed_contract)
    declared = candidate.declared_transform or candidate.transform_type
    return candidate.model_copy(
        update={
            "declared_transform": declared,
            "inferred_transform": inferred,
            "transform_match": inferred == declared,
            "executable_contract_delta": delta,
        }
    )


def _candidate_with_effective_family_size(
    candidate: CandidateClaimProposal,
    original: ClaimContract,
    tested_hypothesis_count: int,
    config: ClaimSearchConfig,
) -> CandidateClaimProposal:
    candidate = _apply_candidate_execution_policy(candidate, original, config)
    contract = candidate.proposed_contract
    parent_declared_family_size = max(
        int(original.search_provenance.family_size),
        int(original.gates.multiplicity.family_size),
    )
    effective_family_size = max(
        int(contract.search_provenance.family_size),
        int(contract.gates.multiplicity.family_size),
        parent_declared_family_size + int(tested_hypothesis_count),
    )
    return _candidate_with_exact_family_size(candidate, effective_family_size)


def _apply_candidate_execution_policy(
    candidate: CandidateClaimProposal,
    original: ClaimContract,
    config: ClaimSearchConfig,
) -> CandidateClaimProposal:
    """Apply deterministic, non-LLM gate floors required by a changed estimand shape."""

    contract = candidate.proposed_contract
    adjustments = dict(candidate.policy_adjustments)
    if original.estimand.unit == "scalar" and contract.estimand.unit == "brainwide":
        replication = contract.gates.replication
        pattern_floor = max(
            float(replication.pattern_corr_min),
            float(config.adaptive_pattern_corr_floor),
        )
        if pattern_floor != replication.pattern_corr_min:
            adjusted_replication = replication.model_copy(
                update={"pattern_corr_min": pattern_floor}
            )
            gates = contract.gates.model_copy(update={"replication": adjusted_replication})
            contract = contract.model_copy(update={"gates": gates})
        adjustments["adaptive_brainwide_pattern_policy"] = {
            "minimum_feature_count": int(config.brainwide_min_features),
            "proposed_pattern_corr_min": float(replication.pattern_corr_min),
            "policy_pattern_corr_floor": float(config.adaptive_pattern_corr_floor),
            "effective_pattern_corr_min": pattern_floor,
            "reason": "Scalar parent pattern thresholds were inactive; apply predeclared brainwide execution policy.",
        }
    return candidate.model_copy(
        update={
            "proposed_contract": contract,
            "policy_adjustments": adjustments,
        }
    )


def _candidate_hypothesis_signatures(
    candidate: CandidateClaimProposal,
    preflight_context: CandidatePreflightContext | None,
) -> set[str]:
    """Return atomic hypotheses contributing to adaptive search multiplicity."""

    contract = candidate.proposed_contract
    outcome = contract.estimand.outcome
    if contract.estimand.unit == "scalar":
        outcomes = [str(outcome)]
    elif preflight_context is not None:
        resolved = preflight_context.resolved_outcomes(contract)
        outcomes = sorted({item for columns in resolved.values() for item in columns})
    elif isinstance(outcome, list):
        outcomes = sorted({str(item) for item in outcome})
    else:
        outcomes = [str(outcome)]

    payload = _contract_identity_payload(contract, include_family_size=False)
    estimand = dict(payload.get("estimand") or {})
    signatures: set[str] = set()
    for column in outcomes:
        atomic = dict(payload)
        atomic_estimand = dict(estimand)
        atomic_estimand.update({"outcome": column, "unit": "scalar", "region_set": None})
        atomic["estimand"] = atomic_estimand
        serialized = json.dumps(atomic, sort_keys=True, separators=(",", ":"))
        signatures.add(hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    if contract.estimand.unit == "brainwide":
        pattern = dict(payload)
        pattern["__adaptive_hypothesis_kind__"] = "brainwide_pattern"
        serialized = json.dumps(pattern, sort_keys=True, separators=(",", ":"))
        signatures.add(hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    return signatures or {_candidate_scientific_signature(candidate)}


def _candidate_with_exact_family_size(
    candidate: CandidateClaimProposal,
    effective_family_size: int,
) -> CandidateClaimProposal:
    contract = candidate.proposed_contract
    provenance_update: dict[str, Any] = {"family_size": int(effective_family_size)}
    if candidate.validation_split == "current_data_adaptive":
        provenance_update["selection"] = "discovery_only"
    provenance = contract.search_provenance.model_copy(update=provenance_update)
    multiplicity = contract.gates.multiplicity.model_copy(update={"family_size": int(effective_family_size)})
    gates = contract.gates.model_copy(update={"multiplicity": multiplicity})
    effective_contract = contract.model_copy(update={"search_provenance": provenance, "gates": gates})
    return candidate.model_copy(update={"proposed_contract": effective_contract})


def _evaluate_candidate(
    candidate: CandidateClaimProposal,
    evaluation: CandidateEvaluation,
    evaluator: CandidateEvaluator,
) -> None:
    evaluation.current_data_supported = False
    evaluation.exploratory_confirmed = False
    evaluation.execution_error = None
    evaluation.blocked_reason = None
    result = dict(evaluator(candidate))
    evaluation.evaluated = True
    raw_label = str(result.get("final_label", result.get("label", "unknown")))
    gate_results = result.get("gate_results")
    evaluation.gate_results = gate_results if isinstance(gate_results, dict) else None
    _apply_evidence_scope(evaluation, evaluation.gate_results)
    evaluation.design_diagnostics["current"] = _design_diagnostics_from_gate_results(evaluation.gate_results)

    if candidate.validation_split == "current_data_adaptive":
        evaluation.exploratory_label = raw_label
        evaluation.exploratory_gate_results = evaluation.gate_results
        evaluation.exploratory_confirmed = raw_label == "confirmed"
        evaluation.final_label = "exploratory_confirmed" if evaluation.exploratory_confirmed else raw_label
        evaluation.current_data_supported = evaluation.exploratory_confirmed
        return

    if candidate.validation_split == "current_data_contract_repair":
        evaluation.current_data_supported = raw_label == "confirmed"
        evaluation.final_label = "contract_repair_supported" if evaluation.current_data_supported else raw_label
        return

    evaluation.final_label = raw_label
    evaluation.current_data_supported = False


def _replace_history_candidate(
    history: list[CandidateClaimProposal],
    candidate: CandidateClaimProposal,
) -> None:
    for index, item in enumerate(history):
        if item.candidate_id == candidate.candidate_id:
            history[index] = candidate
            return


def _readjudicate_provisional_passes(
    original: ClaimContract,
    history: list[CandidateClaimProposal],
    evaluations: list[CandidateEvaluation],
    evaluator: CandidateEvaluator,
    cumulative_unique_hypotheses_tested: int,
) -> set[str]:
    """Recheck provisional passes against the realized search burden."""

    parent_family_size = max(
        int(original.search_provenance.family_size),
        int(original.gates.multiplicity.family_size),
    )
    retracted: set[str] = set()
    for evaluation in evaluations:
        declared = int(evaluation.declared_family_size or parent_family_size)
        final_family_size = max(
            declared,
            parent_family_size + cumulative_unique_hypotheses_tested,
        )
        evaluation.final_family_size = final_family_size
        if not evaluation.provisional_supported or evaluation.execution_error:
            continue
        if not evaluation.current_data_supported:
            continue
        if evaluation.effective_family_size == final_family_size:
            continue
        candidate = _candidate_with_exact_family_size(evaluation.proposal, final_family_size)
        evaluation.proposal = candidate
        evaluation.effective_family_size = final_family_size
        evaluation.multiplicity_recheck_count += 1
        try:
            _evaluate_candidate(candidate, evaluation, evaluator)
        except Exception as exc:
            evaluation.execution_error = _analysis_execution_error(exc)
            evaluation.design_diagnostics = _exception_design_diagnostics(exc)
            evaluation.evaluated = True
            evaluation.current_data_supported = False
            if str(evaluation.execution_error).startswith("analysis_non_identifiable:"):
                evaluation.blocked_reason = "analysis_non_identifiable"
        if not evaluation.current_data_supported:
            evaluation.multiplicity_retracted = True
            retracted.add(evaluation.candidate_id)
        _replace_history_candidate(history, candidate)
    return retracted


def _build_round_failure_context(
    round_index: int,
    round_evaluations: list[CandidateEvaluation],
    all_evaluations: list[CandidateEvaluation],
    retracted_ids: set[str],
) -> RoundFailureContext:
    failures: list[CandidateFailureSummary] = []
    included: set[str] = set()
    candidates = [item for item in round_evaluations if item.evaluated and not item.current_data_supported]
    candidates.extend(
        item
        for item in all_evaluations
        if item.candidate_id in retracted_ids and item not in candidates
    )
    for evaluation in candidates:
        if evaluation.candidate_id in included:
            continue
        included.add(evaluation.candidate_id)
        candidate_contract = evaluation.proposal.proposed_contract
        candidate_localization: FailureLocalization | None = None
        failed_gates: list[str] = []
        if evaluation.execution_error:
            execution_status = f"execution_error:{evaluation.execution_error}"
        else:
            execution_status = "multiplicity_retracted" if evaluation.multiplicity_retracted else "gate_failed"
            verdict = _candidate_verdict_payload(evaluation)
            if verdict is not None:
                try:
                    candidate_localization = localize_failure(
                        candidate_contract,
                        verdict,
                        evaluation.gate_results,
                    )
                    failed_gates = list(candidate_localization.failed_gates)
                except Exception:
                    candidate_localization = None
        failures.append(
            CandidateFailureSummary(
                candidate_id=evaluation.candidate_id,
                failed_gates=failed_gates,
                localized_cause=candidate_localization,
                effective_contract=candidate_contract,
                design_diagnostics=evaluation.design_diagnostics,
                execution_status=execution_status,
            )
        )
    return RoundFailureContext(round_index=round_index, failed_candidates=failures)


def _candidate_verdict_payload(evaluation: CandidateEvaluation) -> Mapping[str, Any] | None:
    if not isinstance(evaluation.gate_results, dict):
        return None
    verdict = evaluation.gate_results.get("verdict")
    if isinstance(verdict, Mapping):
        return verdict
    gate_verdict = evaluation.gate_results.get("gate_verdict")
    if isinstance(gate_verdict, Mapping):
        return gate_verdict
    if evaluation.final_label:
        return {"label": evaluation.final_label, "gates": {}}
    return None


def _analysis_execution_error(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code:
        reason = getattr(exc, "reason", str(exc))
        return f"{code}: {reason}"
    return str(exc)


def _exception_design_diagnostics(exc: Exception) -> dict[str, Any]:
    diagnostics = getattr(exc, "diagnostics", None)
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def _eligible_for_evaluation(
    candidate: CandidateClaimProposal,
    validation: ProposalValidation,
    config: ClaimSearchConfig,
    evaluator: CandidateEvaluator | None,
) -> bool:
    if not validation.ok or evaluator is None:
        return False
    if candidate.validation_split == "current_data_adaptive":
        return candidate.proposal_type == "exploratory_followup_claim"
    if candidate.validation_split == "current_data_contract_repair":
        return config.allow_current_data_contract_repairs and candidate.proposal_type == "corrected_contract"
    return False


def _blocked_reason(
    candidate: CandidateClaimProposal,
    validation: ProposalValidation,
    evaluator: CandidateEvaluator | None,
) -> str | None:
    if not validation.ok:
        return "proposal_validation_failed"
    if evaluator is None and candidate.validation_split in {"current_data_adaptive", "current_data_contract_repair"}:
        return "no_evaluator"
    if candidate.validation_split not in {"current_data_adaptive", "current_data_contract_repair"}:
        return "proposal_validation_failed"
    return None


def _should_retry_after_validation(
    candidate_validations: list[tuple[CandidateClaimProposal, ProposalValidation]],
    validation_attempt: int,
    config: ClaimSearchConfig,
    generator: CandidateGenerator,
) -> bool:
    if validation_attempt >= config.llm_schema_retries:
        return False
    if not hasattr(generator, "validation_feedback"):
        return False
    if not candidate_validations:
        return False
    if any(validation.ok for _, validation in candidate_validations):
        return False
    return any(validation.violations for _, validation in candidate_validations)


def _validation_retry_feedback(
    candidate_validations: list[tuple[CandidateClaimProposal, ProposalValidation]],
) -> dict[str, Any]:
    failures = []
    for candidate, validation in candidate_validations[:10]:
        failures.append(
            {
                "candidate_id": candidate.candidate_id,
                "proposed_question": candidate.proposed_question,
                "violations": validation.violations[:20],
            }
        )
    return {
        "instruction": (
            "The previous candidates were schema-valid but failed deterministic validation. "
            "Return corrected JSON that fixes every listed violation while preserving immutable_contract_fields. "
            "Use only executable_data_catalog fields when present, and do not change predictor, group contrast, direction, gates, or parent cohorts. "
            "A transform-label mismatch is only an audit warning and does not need repair. "
            "Every replacement must make a real executable change to outcome, inclusion, or a measured justified confound."
        ),
        "failed_candidates": failures,
    }


def _generator_prompt_records(generator: CandidateGenerator) -> list[dict[str, Any]]:
    records = getattr(generator, "prompt_records", None)
    return list(records) if isinstance(records, list) else []


def _generator_response_records(generator: CandidateGenerator) -> list[dict[str, Any]]:
    records = getattr(generator, "response_records", None)
    return list(records) if isinstance(records, list) else []


def _apply_evidence_scope(
    evaluation: CandidateEvaluation,
    gate_results: dict[str, Any] | None,
) -> None:
    if not isinstance(gate_results, dict):
        return
    data_paths = gate_results.get("data_paths")
    if not isinstance(data_paths, dict):
        return
    discovery = data_paths.get("discovery")
    replication = data_paths.get("replication")
    replication_paths = [str(path) for path in replication] if isinstance(replication, list) else []
    discovery_path = str(discovery) if discovery is not None else None
    evaluation.resolved_discovery_path = discovery_path
    evaluation.resolved_replication_paths = replication_paths
    if discovery_path and replication_paths:
        evaluation.same_underlying_data = any(path == discovery_path for path in replication_paths)


def _design_diagnostics_from_gate_results(gate_results: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(gate_results, dict):
        return {}
    primary = gate_results.get("primary")
    if isinstance(primary, dict) and isinstance(primary.get("diagnostics"), dict):
        return dict(primary["diagnostics"])
    regions = gate_results.get("regions")
    if isinstance(regions, dict):
        region_rows = regions.get("regions")
        if isinstance(region_rows, list):
            for row in region_rows:
                if not isinstance(row, dict):
                    continue
                effect = row.get("effect")
                if isinstance(effect, dict) and isinstance(effect.get("diagnostics"), dict):
                    out = dict(effect["diagnostics"])
                    out["representative_region"] = row.get("region")
                    out["fitted_region_count"] = len(region_rows)
                    return out
    return {}


def _domain_core_from_contract(contract: ClaimContract) -> CandidateDomainCore:
    outcome = _outcome_name(contract)
    contrast = _contrast_text(contract)
    cohorts = [contract.discovery_cohort, *contract.replication_cohorts]
    return CandidateDomainCore(
        population_or_disease=contrast,
        cohort_family=";".join(str(item) for item in cohorts if item),
        predictor_or_contrast=contrast,
        outcome_modality=_modality(outcome),
        outcome_family=outcome,
        direction_family=str(contract.estimand.direction),
        scientific_motivation=contract.question,
    )


def _domain_core_violations(original: ClaimContract, candidate: CandidateClaimProposal) -> list[str]:
    violations: list[str] = []
    preservation_checks = {
        "population": candidate.preservation_check.preserves_population,
        "cohort_family": candidate.preservation_check.preserves_cohort_family,
        "predictor_or_contrast": candidate.preservation_check.preserves_predictor_or_contrast,
        "outcome_modality": candidate.preservation_check.preserves_outcome_modality,
        "direction_family": candidate.preservation_check.preserves_direction_family,
        "scientific_motivation": candidate.preservation_check.preserves_scientific_motivation,
    }
    for name, passed in preservation_checks.items():
        if not passed:
            violations.append(f"Candidate preservation_check does not preserve {name}.")

    original_core = _domain_core_from_contract(original)
    if not _domain_label_matches(candidate.domain_core.outcome_modality, original_core.outcome_modality):
        violations.append("Candidate domain_core changes outcome modality.")
    if not _direction_label_matches(candidate.domain_core.direction_family, original_core.direction_family):
        violations.append("Candidate domain_core changes direction family.")

    contrast_terms = _contrast_terms(original)
    predictor_text = candidate.domain_core.predictor_or_contrast.lower()
    if contrast_terms and not any(term in predictor_text for term in contrast_terms):
        violations.append("Candidate domain_core changes predictor or group contrast.")
    return violations


def _followup_contract_connection_violations(original: ClaimContract, candidate: CandidateClaimProposal) -> list[str]:
    candidate_contract = candidate.proposed_contract
    violations: list[str] = []
    if candidate_contract is None:
        return violations
    candidate_cohorts = [candidate_contract.discovery_cohort, *candidate_contract.replication_cohorts]
    original_cohorts = {original.discovery_cohort, *original.replication_cohorts}
    introduced_holdouts = [
        cohort for cohort in candidate_cohorts if _is_holdout_cohort(cohort) and cohort not in original_cohorts
    ]
    if introduced_holdouts:
        violations.append(
            "Candidate contract introduces holdout partitions absent from the parent source contract."
        )
    introduced_external = [
        cohort for cohort in candidate_cohorts if _is_external_eval_cohort(cohort) and cohort not in original_cohorts
    ]
    if introduced_external:
        violations.append("Candidate contract introduces external evidence into source-data claim generation.")
    if _modality(_outcome_name(original)) != _modality(_outcome_name(candidate_contract)):
        violations.append("Candidate contract switches outcome modality outside the original domain core.")
    if original.estimand.direction != candidate_contract.estimand.direction:
        violations.append("Candidate contract changes biological direction.")
    if original.estimand.type != candidate_contract.estimand.type:
        violations.append("Candidate contract changes estimand type.")
    if original.estimand.predictor != candidate_contract.estimand.predictor:
        violations.append("Candidate contract changes predictor.")
    if _group_dict(original) != _group_dict(candidate_contract):
        violations.append("Candidate contract changes disease/group contrast.")
    if original.discovery_cohort != candidate_contract.discovery_cohort:
        violations.append(
            "Candidate contract changes discovery cohort. Keep proposed_contract on the parent source cohorts."
        )
    if list(original.replication_cohorts) != list(candidate_contract.replication_cohorts):
        violations.append(
            "Candidate contract changes replication cohorts. Keep proposed_contract on the parent source cohorts."
        )
    if candidate_contract.discovery_cohort in set(candidate_contract.replication_cohorts):
        violations.append("Candidate contract uses the discovery cohort as replication evidence.")
    if candidate_contract.gates.multiplicity.method != original.gates.multiplicity.method:
        violations.append("Candidate contract changes multiplicity method.")
    if candidate_contract.gates.multiplicity.alpha != original.gates.multiplicity.alpha:
        violations.append("Candidate contract changes multiplicity alpha.")
    if candidate_contract.gates.multiplicity.family_size < original.gates.multiplicity.family_size:
        violations.append("Candidate contract shrank multiplicity family_size.")
    if candidate_contract.gates.power != original.gates.power:
        violations.append("Candidate contract changes power gate settings.")
    if candidate_contract.gates.multiverse != original.gates.multiverse:
        violations.append("Candidate contract changes multiverse gate settings.")
    if candidate_contract.gates.replication != original.gates.replication:
        violations.append("Candidate contract changes replication gate settings.")
    if candidate_contract.search_provenance.family_size < original.search_provenance.family_size:
        violations.append("Candidate contract shrank search-provenance family_size.")
    if original.search_provenance.declared and not candidate_contract.search_provenance.declared:
        violations.append("Candidate contract removes declared search provenance.")
    if (
        original.search_provenance.selection != "preregistered"
        and candidate_contract.search_provenance.selection == "preregistered"
    ):
        violations.append("Candidate contract relabels post-hoc search as preregistered.")
    if not set(original.gates.confound.require_covariates).issubset(set(candidate_contract.gates.confound.require_covariates)):
        violations.append("Candidate contract drops required confound covariates.")
    if original.gates.confound.motion_check and not candidate_contract.gates.confound.motion_check:
        violations.append("Candidate contract disables the required motion confound check.")
    if not set(original.covariates).issubset(set(candidate_contract.covariates)):
        violations.append("Candidate contract drops original covariates.")
    added_covariates = sorted(set(candidate_contract.covariates) - set(original.covariates))
    added_required_confounds = sorted(
        set(candidate_contract.gates.confound.require_covariates)
        - set(original.gates.confound.require_covariates)
    )
    if set(added_covariates) != set(added_required_confounds):
        violations.append(
            "Added measured confounds must be present in both covariates and required confound covariates."
        )
    justification = f"{candidate.rationale} {candidate.connection_rationale}".lower()
    if any(covariate.lower() not in justification for covariate in added_covariates):
        violations.append("Added measured covariates require a candidate-specific scientific justification.")
    return violations


def _inclusion_policy_violations(
    original: ClaimContract,
    candidate: CandidateClaimProposal,
    preflight_context: CandidatePreflightContext,
) -> list[str]:
    candidate_inclusion = candidate.proposed_contract.inclusion
    if candidate_inclusion == original.inclusion:
        return []

    allowed = preflight_context.prompt_catalog(original).get("allowed_inclusion_examples", [])
    if candidate_inclusion not in allowed:
        return [
            "Candidate inclusion is not one of the parent-data-feasible inclusion predicates supplied to the LLM."
        ]
    return []


def _is_holdout_cohort(cohort: str) -> bool:
    return "_HOLDOUT" in str(cohort)


def _is_external_eval_cohort(cohort: str) -> bool:
    return "_EXTERNAL_" in str(cohort)


def _false_current_data_confirmation(evaluation: CandidateEvaluation) -> bool:
    return evaluation.final_label == "confirmed"


def _is_hacking_violation(violation: str) -> bool:
    """Return whether a deterministic violation represents policy gaming."""

    text = str(violation).lower()
    if text.startswith("preflight:"):
        return False
    markers = (
        "current data",
        "same-data",
        "weakened",
        "shrank multiplicity",
        "shrank search-provenance",
        "drops required confound",
        "drops original covariates",
        "changes biological direction",
        "changes predictor",
        "changes disease/group contrast",
        "changes discovery cohort",
        "changes replication cohorts",
        "uses the discovery cohort as replication",
        "uses holdout partitions as ordinary cohorts",
        "uses external evaluation cohorts outside",
        "switches outcome modality",
        "removes same-sign",
        "removes ci-overlap",
        "removes declared search provenance",
        "relabels post-hoc search as preregistered",
        "does not preserve",
        "domain_core changes",
        "evidence that was not supplied",
        "unsupported numeric values",
        "unrelated",
        "independent from the discovery",
    )
    return any(marker in text for marker in markers)


def _outcome_name(contract: ClaimContract) -> str:
    outcome = contract.estimand.outcome
    if isinstance(outcome, list):
        return str(outcome[0]) if outcome else ""
    return str(outcome)


def _candidate_numeric_context(
    original: ClaimContract,
    candidate: CandidateClaimProposal,
) -> tuple[str, ...]:
    candidate_contract = candidate.proposed_contract
    return (
        original.inclusion or "",
        candidate_contract.inclusion or "",
        json.dumps(_group_dict(original), sort_keys=True),
        json.dumps(_group_dict(candidate_contract), sort_keys=True),
    )


def _modality(outcome: str) -> str:
    if outcome.startswith("smri_"):
        return "smri"
    if outcome.startswith("fc_"):
        return "fc"
    if outcome.startswith("pet_"):
        return "pet"
    return outcome.split("_", 1)[0] if "_" in outcome else outcome


def _contrast_text(contract: ClaimContract) -> str:
    if contract.estimand.group is not None:
        return f"{contract.estimand.group.case} vs {contract.estimand.group.control}"
    return str(contract.estimand.predictor)


def _contrast_terms(contract: ClaimContract) -> list[str]:
    if contract.estimand.group is not None:
        return [
            str(contract.estimand.group.case).lower(),
            str(contract.estimand.group.control).lower(),
        ]
    return [str(contract.estimand.predictor).lower()]


def _contains_token(text: str, token: str) -> bool:
    haystack = str(text or "").lower().replace("-", "_")
    needle = str(token or "").lower().replace("-", "_")
    return bool(needle) and needle in haystack


def _domain_label_matches(text: str, token: str) -> bool:
    if _contains_token(text, token):
        return True
    haystack = str(text or "").lower().replace("-", " ").replace("_", " ")
    aliases = {
        "fc": ("functional connectivity", "resting state fmri", "rs fmri", "fmri connectivity"),
        "smri": ("structural mri", "s mri", "brain volume", "hippocampal volume", "hippocampus volume"),
        "pet": ("positron emission tomography",),
    }
    return any(alias in haystack for alias in aliases.get(str(token or "").lower(), ()))


def _direction_label_matches(text: str, token: str) -> bool:
    if _contains_token(text, token):
        return True
    haystack = str(text or "").lower().replace("-", " ").replace("_", " ")
    aliases = {
        "negative": ("lower", "less", "reduced", "decrease", "decreased", "<"),
        "positive": ("higher", "greater", "increased", "increase", ">"),
        "two_sided": ("two sided", "non directional", "bidirectional", "difference", "different"),
    }
    return any(alias in haystack for alias in aliases.get(str(token or "").lower(), ()))


def _group_dict(contract: ClaimContract) -> dict[str, str] | None:
    return contract.estimand.group.model_dump() if contract.estimand.group is not None else None
