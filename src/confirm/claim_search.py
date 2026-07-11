"""Iterative failure-diagnosis claim generation with provenance controls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, Literal, Optional, Union

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
    "moderator_or_subgroup",
    "stronger_design",
    "fixed_estimand",
    "contract_correction",
]
CandidateProposalType = Literal["corrected_contract", "exploratory_followup_claim", "independent_replication_claim"]
ValidationSplit = Literal[
    "none",
    "current_data_adaptive",
    "current_data_contract_repair",
    "excluded_validation",
    "future_required",
]
StoppedReason = Literal[
    "confirmed",
    "exploratory_confirmed",
    "holdout_confirmed",
    "external_confirmed",
    "max_rounds_exhausted",
    "no_excluded_validation_evidence",
    "no_evaluator",
    "already_confirmed",
    "no_candidates",
    "llm_unavailable",
    "candidate_generation_failed",
]

CLAIM_CANDIDATE_SYSTEM_PROMPT = """You generate scientifically connected follow-up claim candidates after CONFIRM rejects a claim.
Rules:
- The goal is not to make the failed claim pass.
- The goal is to generate scientifically connected next claims.
- Failed-claim evidence may be used only for diagnosis and hypothesis generation.
- Post-hoc candidates may be evaluated on the same data, but same-data support must be labeled exploratory_confirmed, not confirmed.
- External or holdout evidence is optional; a same-data supported candidate can be upgraded only to holdout_confirmed or external_confirmed if it also passes excluded evaluation.
- Preserve the original disease/cohort family, outcome modality, biological direction family, and scientific motivation.
- Return structured JSON only. Do not use markdown fences.
- Do not invent p-values, effect sizes, cohorts, or gate results.
- Do not weaken CONFIRM gates, drop confound covariates, reverse direction, switch to unrelated outcomes, or present same-data adaptive support as final confirmation.
- Preserve immutable contract fields unless the proposal is a true contract correction: predictor, group contrast, direction, covariates, gates, search family size, and cohort family.
- For patch-like follow-ups, proposed_contract must keep the parent discovery and replication cohorts; holdout/external cohorts are evaluation evidence, not ordinary contract cohorts.
"""


def _proposal_type_shape_errors(proposal_type: str, transform_type: str) -> list[str]:
    """Return schema-level errors for incompatible proposal/transform pairs."""

    if transform_type == "contract_correction" and proposal_type != "corrected_contract":
        return ["contract_correction transform must use corrected_contract proposal type."]
    if transform_type in {"narrower_outcome_family", "moderator_or_subgroup", "fixed_estimand"}:
        if proposal_type != "exploratory_followup_claim":
            return ["Exploratory transforms must use exploratory_followup_claim proposal type."]
    if transform_type == "stronger_design" and proposal_type != "independent_replication_claim":
        return ["stronger_design transform must use independent_replication_claim proposal type."]
    return []


class ClaimSearchConfig(BaseModel):
    """Configurable budget and safety switches for iterative claim search."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=3, ge=1, le=20)
    max_candidates_per_round: int = Field(default=5, ge=1, le=20)
    candidate_timeout: float = Field(default=30.0, gt=0.0)
    stop_on_first_confirmed: bool = True
    allow_current_data_contract_repairs: bool = True
    llm_schema_retries: int = Field(default=2, ge=0, le=5)


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
    validation_split: ValidationSplit


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


class LLMExploratoryCandidateProposal(_LLMCandidateProposalBase):
    """Structured exploratory candidate with only exploratory transforms."""

    proposal_type: Literal["exploratory_followup_claim"]
    transform_type: Literal["narrower_outcome_family", "moderator_or_subgroup", "fixed_estimand"]


class LLMIndependentReplicationCandidateProposal(_LLMCandidateProposalBase):
    """Structured independent-replication candidate."""

    proposal_type: Literal["independent_replication_claim"]
    transform_type: Literal["stronger_design"]


class LLMCorrectedContractCandidateProposal(_LLMCandidateProposalBase):
    """Structured true contract-correction candidate."""

    proposal_type: Literal["corrected_contract"]
    transform_type: Literal["contract_correction"]


LLMCandidateProposal = Union[
    LLMExploratoryCandidateProposal,
    LLMIndependentReplicationCandidateProposal,
    LLMCorrectedContractCandidateProposal,
]


class LLMCandidateGenerationResponse(BaseModel):
    """Top-level strict schema for candidate-generation LLM output."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[LLMCandidateProposal]


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

    @model_validator(mode="after")
    def validate_candidate_shape(self) -> "CandidateClaimProposal":
        errors = _proposal_type_shape_errors(self.proposal_type, self.transform_type)
        if errors:
            raise ValueError(" ".join(errors))
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
    excluded_evidence_error: Optional[str] = None
    exploratory_label: Optional[str] = None
    exploratory_gate_results: Optional[dict[str, Any]] = None
    exploratory_confirmed: bool = False
    holdout_label: Optional[str] = None
    holdout_gate_results: Optional[dict[str, Any]] = None
    holdout_confirmed: bool = False
    external_label: Optional[str] = None
    external_gate_results: Optional[dict[str, Any]] = None
    external_confirmed: bool = False
    resolved_discovery_path: Optional[str] = None
    resolved_replication_paths: list[str] = Field(default_factory=list)
    same_underlying_data: Optional[bool] = None
    excluded_evidence_kind: Optional[Literal["holdout", "external"]] = None
    excluded_evidence_used: bool = False
    external_evidence_used: bool = False
    confirmed: bool = False


class DuplicateCandidateRecord(BaseModel):
    """Candidate omitted because its scientific specification was already seen."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    duplicate_of: str
    parent_claim_id: str
    round_index: int
    scientific_signature: str


class ClaimSearchState(BaseModel):
    """Serializable trace state for an iterative claim-search run."""

    model_config = ConfigDict(extra="forbid")

    original_claim: ClaimContract
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    failure_localization: Optional[FailureLocalization] = None
    lineage_graph: dict[str, Any]
    used_evidence: list[str] = Field(default_factory=list)
    candidate_history: list[CandidateClaimProposal] = Field(default_factory=list)
    duplicate_candidates: list[DuplicateCandidateRecord] = Field(default_factory=list)
    evaluations: list[CandidateEvaluation] = Field(default_factory=list)
    confirmed_candidates: list[str] = Field(default_factory=list)
    llm_candidate_prompts: list[dict[str, Any]] = Field(default_factory=list)
    llm_candidate_responses: list[dict[str, Any]] = Field(default_factory=list)
    stopped_reason: StoppedReason


CandidateEvaluator = Callable[[CandidateClaimProposal], Mapping[str, Any]]
CandidateGenerator = Callable[[ClaimContract, FailureLocalization, ClaimSearchConfig, int, str], list[CandidateClaimProposal]]


class CandidateGenerationError(RuntimeError):
    """Raised when an LLM candidate-generation attempt cannot produce parseable candidates."""


def generate_connected_candidates(
    contract: ClaimContract,
    localization: FailureLocalization,
    config: ClaimSearchConfig,
    round_index: int,
    parent_claim_id: str,
) -> list[CandidateClaimProposal]:
    """Generate deterministic connected follow-up candidates for tests and controlled baselines."""

    if localization.failure_kind == "none":
        return []

    candidates: list[CandidateClaimProposal] = []
    evidence = localization.evidence
    outcome = _outcome_name(contract)
    modality = _modality(outcome)
    contrast = _contrast_text(contract)

    if localization.current_data_repair_allowed and config.allow_current_data_contract_repairs:
        repaired = _contract_repair_candidate(contract, localization)
        if repaired is not None:
            candidates.append(
                _candidate(
                    contract,
                    localization,
                    parent_claim_id,
                    round_index,
                    len(candidates),
                    proposal_type="corrected_contract",
                    transform_type="contract_correction",
                    provenance="contract_correction",
                    validation_split="current_data_contract_repair",
                    question=repaired.question,
                    rationale="Repair an auditable contract/design issue while preserving the original scientific question.",
                    connection="This candidate keeps the original question and only repairs allowed contract fields.",
                    proposed_contract=repaired,
                    requires_new_evidence=False,
                    can_confirm_on_current_data=True,
                    evidence=evidence,
                )
            )

    if localization.failure_kind in {"evidence_failure", "search_lineage_failure", "confound_failure"}:
        narrower_contract = _followup_contract(
            contract,
            suffix="narrower_outcome",
            outcome=_narrower_outcome_name(contract),
            family_size_increment=len(candidates) + 1,
        )
        candidates.append(
            _candidate(
                contract,
                localization,
                parent_claim_id,
                round_index,
                len(candidates),
                proposal_type="exploratory_followup_claim",
                transform_type="narrower_outcome_family",
                provenance="post_hoc_followup",
                validation_split="current_data_adaptive",
                question=(
                    f"Under adaptive same-data evaluation, does the {contrast} effect appear in a narrower, "
                    f"predeclared {modality} outcome family related to {outcome}?"
                ),
                rationale="The failed claim may be too coarse; propose a narrower same-modality follow-up for adaptive evaluation.",
                connection=f"Preserves the original contrast and {modality} modality while narrowing the outcome family.",
                proposed_contract=narrower_contract,
                requires_new_evidence=False,
                can_confirm_on_current_data=True,
                evidence=evidence,
            )
        )
        fixed_contract = _followup_contract(contract, suffix="fixed_estimand", family_size_increment=len(candidates) + 1)
        candidates.append(
            _candidate(
                contract,
                localization,
                parent_claim_id,
                round_index,
                len(candidates),
                proposal_type="exploratory_followup_claim",
                transform_type="fixed_estimand",
                provenance="future_design",
                validation_split="current_data_adaptive",
                question=(
                    f"Using a fixed estimand and analysis specification, does the original {contrast} "
                    f"claim show adaptive support for {outcome} on the current evidence?"
                ),
                rationale="The follow-up fixes the analysis specification before adaptive same-data evaluation.",
                connection="Preserves the original claim family while making the next analysis specification explicit.",
                proposed_contract=fixed_contract,
                requires_new_evidence=False,
                can_confirm_on_current_data=True,
                evidence=evidence,
            )
        )

    if localization.failure_kind in {"evidence_failure", "design_limitation", "confound_failure"}:
        stronger_contract = _followup_contract(contract, suffix="stronger_design", family_size_increment=len(candidates) + 1)
        candidates.append(
            _candidate(
                contract,
                localization,
                parent_claim_id,
                round_index,
                len(candidates),
                proposal_type="independent_replication_claim",
                transform_type="stronger_design",
                provenance="independent_replication",
                validation_split="current_data_adaptive",
                question=(
                    f"Evaluate whether a replication-ready version of the original {contrast} claim for {outcome} "
                    "has adaptive same-data support before optional external validation."
                ),
                rationale="The next claim keeps the stronger-design target but allows same-data adaptive screening before optional external validation.",
                connection="Preserves the original contrast, outcome family, direction, and gate stack.",
                proposed_contract=stronger_contract,
                requires_new_evidence=False,
                can_confirm_on_current_data=True,
                evidence=evidence,
            )
        )

    if localization.failure_kind in {"evidence_failure", "confound_failure"}:
        moderator_contract = _followup_contract(contract, suffix="moderator_or_subgroup", family_size_increment=len(candidates) + 1)
        candidates.append(
            _candidate(
                contract,
                localization,
                parent_claim_id,
                round_index,
                len(candidates),
                proposal_type="exploratory_followup_claim",
                transform_type="moderator_or_subgroup",
                provenance="post_hoc_followup",
                validation_split="current_data_adaptive",
                question=(
                    f"Under adaptive same-data evaluation, is the original {contrast} association for {outcome} moderated by "
                    "predeclared cohort/site or demographic strata?"
                ),
                rationale="A failed aggregate effect may reflect heterogeneity; this candidate can be adaptively screened before optional external validation.",
                connection="Keeps the original domain core while testing a connected heterogeneity explanation.",
                proposed_contract=moderator_contract,
                requires_new_evidence=False,
                can_confirm_on_current_data=True,
                evidence=evidence,
            )
        )

    allowed_types = set(localization.allowed_proposal_types) - {"downgraded_claim", "abandon_claim"}
    return [candidate for candidate in candidates if candidate.proposal_type in allowed_types][: config.max_candidates_per_round]


class LLMClaimCandidateGenerator:
    """LLM-backed generator for connected follow-up candidates."""

    def __init__(
        self,
        llm: LLMClient,
        preflight_context: CandidatePreflightContext | None = None,
        validation_evidence_catalog: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.preflight_context = preflight_context
        self.validation_evidence_catalog = validation_evidence_catalog
        self.candidate_history: list[CandidateClaimProposal] = []
        self.validation_feedback: dict[str, Any] | None = None
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
        for attempt_index in range(max_attempts):
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
                validation_evidence_catalog=self.validation_evidence_catalog,
                validation_feedback=self.validation_feedback,
            )
            prompt_record = {
                "round_index": round_index,
                "parent_claim_id": parent_claim_id,
                "attempt_index": attempt_index,
                "is_retry": attempt_index > 0,
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
                        "is_retry": attempt_index > 0,
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
                    "is_retry": attempt_index > 0,
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
    validation_evidence_catalog: dict[str, Any] | None = None,
    validation_feedback: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Build the auditable prompt for LLM follow-up claim generation."""

    payload = {
        "task": "generate_connected_followup_claim_candidates",
        "max_candidates": config.max_candidates_per_round,
        "round_index": round_index,
        "parent_claim_id": parent_claim_id,
        "original_contract": contract.model_dump(mode="json"),
        "failure_localization": localization.model_dump(mode="json"),
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
        "allowed_proposal_types": _executable_proposal_types(localization.allowed_proposal_types),
        "allowed_transform_types": [
            "narrower_outcome_family",
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
            "For patch-like follow-up candidates, preserve immutable_contract_fields exactly.",
            "For patch-like current_data_adaptive or excluded_validation candidates, proposed_contract.discovery_cohort and proposed_contract.replication_cohorts must remain the parent cohorts from immutable_contract_fields.",
            "For every non-external-only proposal, proposed_contract.discovery_cohort must exactly equal immutable_contract_fields.discovery_cohort and proposed_contract.replication_cohorts must exactly equal immutable_contract_fields.replication_cohorts.",
            "It is acceptable for proposed_question to mention holdout/external evaluation evidence, but proposed_contract must still use the parent discovery/replication cohorts unless this is a true external-only independent_replication_claim.",
            "Novelty should come only from a narrower same-modality outcome, a justified subgroup/inclusion, a fixed analysis specification, a stronger validation design, or a true contract correction.",
            "Every candidate intended for current_data_adaptive or current_data_contract_repair evaluation must include an executable proposed_contract.",
            "Patch-like proposed_contracts must be executable against executable_data_catalog when that field is present.",
            "Use only outcomes, predictors, group variables, covariates, and inclusion terms present in executable_data_catalog.",
            "Use validation_evidence_catalog only to describe excluded evaluation evidence. Do not copy holdout cohorts into proposed_contract unless this is a true external-only independent_replication_claim.",
            "True external-only independent_replication_claim contracts may use external cohorts listed in validation_evidence_catalog.external_partitions, but must preserve predictor/group, outcome modality, direction, covariates, gates, and family size.",
            "For stronger_design follow-ups, prefer validation_split=excluded_validation with the parent-compatible proposed_contract; the evaluator will map it to holdout/external evidence.",
            "Set disposition_label to null for generated candidates; do not emit downgrade labels.",
            "If preserving the original predictor/group contrast is impossible with the executable catalog, return candidates=[] instead of substituting a new predictor.",
            "Do not emit placeholder variables such as bench_group, bench_predictor, low_motion_subset, or free-text inclusion descriptions unless they are actual catalog columns.",
            "String values in inclusion filters must be quoted, for example sex == \"F\" rather than sex == female.",
            "Post-hoc follow-up candidates may set validation_split=current_data_adaptive, requires_new_evidence=false, and can_confirm_on_current_data=true.",
            "Same-data adaptive support is labeled exploratory_confirmed by the pipeline, never plain confirmed.",
            "External or holdout evaluation is optional and can upgrade an exploratory_confirmed candidate to holdout_confirmed or external_confirmed.",
            "When validation_evidence_catalog is present, use it only to describe eligible follow-up validation evidence; do not invent holdout results.",
            "Use current_data_contract_repair only for true contract_correction proposals that preserve the original scientific question.",
            "Use only evidence strings supplied in failure_localization.evidence for supported_by_evidence.",
        ],
        "forbidden_actions": [
            "Do not switch to an unrelated outcome modality.",
            "Do not change the predictor or group contrast for exploratory follow-up candidates.",
            "Do not reverse biological direction after seeing results.",
            "Do not weaken gates or lower thresholds.",
            "Do not drop original covariates or required confound covariates.",
            "Do not shrink multiplicity or search-provenance family size.",
            "Do not label same-data adaptive support as final confirmation.",
            "Do not invent cohorts, p-values, effect sizes, or gate results.",
            "Do not invent columns, filters, group labels, or data subsets.",
            "Do not put *_HOLDOUT, *_HOLDOUT_DISC, *_HOLDOUT_REP, *_EXTERNAL_DISC, or *_EXTERNAL_REP cohorts into proposed_contract for ordinary exploratory follow-ups.",
            "Do not use disposition_label unless the output schema explicitly requires a downgrade disposition; for this task use null.",
        ],
        "output_model": "LLMCandidateGenerationResponse",
        "output_schema": LLMCandidateGenerationResponse.model_json_schema(),
    }
    if executable_catalog is not None:
        payload["executable_data_catalog"] = executable_catalog
    if validation_evidence_catalog is not None:
        payload["validation_evidence_catalog"] = validation_evidence_catalog
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
    excluded_validation_available: bool,
    preflight_context: CandidatePreflightContext | None = None,
) -> ProposalValidation:
    """Validate anti-hacking and connection constraints for a candidate."""

    base_payload = _proposal_payload(candidate)
    if candidate.proposal_type != "corrected_contract":
        base_payload["proposed_contract"] = None
    base = validate_new_claim_proposal(original, NewClaimProposal.model_validate(base_payload), localization)
    violations = list(base.violations)
    warnings = list(base.warnings)

    if candidate.round_index > config.max_rounds:
        violations.append("Candidate round_index exceeds configured max_rounds.")
    if candidate.proposal_type not in {"corrected_contract", "exploratory_followup_claim", "independent_replication_claim"}:
        violations.append("Candidate proposals must be executable new claims.")
    if candidate.validation_split == "current_data_adaptive":
        if candidate.proposal_type not in {"exploratory_followup_claim", "independent_replication_claim"}:
            violations.append("Only connected follow-up proposals may use adaptive same-data evaluation.")
        if not candidate.can_confirm_on_current_data:
            violations.append("Adaptive same-data candidates must declare can_confirm_on_current_data=true.")
    if candidate.validation_split == "current_data_contract_repair":
        if candidate.proposal_type != "corrected_contract" or not config.allow_current_data_contract_repairs:
            violations.append("Only allowed contract corrections may use current-data repair evaluation.")
    if candidate.validation_split == "excluded_validation" and not excluded_validation_available:
        violations.append("Candidate requests excluded validation evidence, but none is available.")
    if candidate.proposal_type in {"exploratory_followup_claim", "independent_replication_claim"}:
        if candidate.validation_split == "none":
            violations.append("Follow-up candidates must name future or excluded validation evidence.")
        if candidate.validation_split == "future_required" and not candidate.requires_new_evidence:
            violations.append("Future-only follow-up candidates must require new evidence.")
    if candidate.can_confirm_on_current_data and candidate.proposal_type == "downgraded_claim":
        violations.append("Downgraded claims cannot be confirmable on current data.")
    unsupported_evidence = [item for item in candidate.supported_by_evidence if item not in localization.evidence]
    if unsupported_evidence:
        violations.append("Candidate cites evidence that was not supplied by failure localization.")

    violations.extend(_domain_core_violations(original, candidate))
    violations.extend(_connection_violations(original, candidate))
    if candidate.proposed_contract is not None and candidate.proposal_type != "corrected_contract":
        violations.extend(_followup_contract_connection_violations(original, candidate))
    if candidate.proposed_contract is not None and preflight_context is not None:
        preflight = preflight_context.validate_contract(candidate.proposed_contract)
        violations.extend(preflight.violations)
        warnings.extend(preflight.warnings)

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
    external_evaluator: CandidateEvaluator | None = None,
    excluded_evidence_kind: Literal["holdout", "external"] = "external",
    excluded_validation_available: bool = False,
    preflight_context: CandidatePreflightContext | None = None,
    validation_evidence_catalog: dict[str, Any] | None = None,
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
            validation_evidence_catalog=validation_evidence_catalog,
        )
    else:
        generator = candidate_generator
    effective_excluded_validation_available = excluded_validation_available or external_evaluator is not None
    history: list[CandidateClaimProposal] = []
    duplicate_candidates: list[DuplicateCandidateRecord] = []
    evaluations: list[CandidateEvaluation] = []
    confirmed: list[str] = []
    lineage = {"nodes": [contract.claim_id], "edges": []}
    stopped: StoppedReason = "max_rounds_exhausted"

    for round_index in range(1, cfg.max_rounds + 1):
        candidate_validations: list[tuple[CandidateClaimProposal, ProposalValidation]] = []
        candidates: list[CandidateClaimProposal] = []
        round_duplicates: list[DuplicateCandidateRecord] = []
        for validation_attempt in range(cfg.llm_schema_retries + 1):
            try:
                if hasattr(generator, "candidate_history"):
                    setattr(generator, "candidate_history", list(history))
                candidates = generator(contract, localization, cfg, round_index, contract.claim_id)
            except CandidateGenerationError:
                stopped = "candidate_generation_failed"
                break
            candidates = [_candidate_for_current_data_adaptive(candidate) for candidate in candidates]
            candidates = candidates[: cfg.max_candidates_per_round]
            candidates, round_duplicates = _deduplicate_candidates(candidates, history)
            candidate_validations = [
                (
                    candidate,
                    validate_candidate_claim(
                        contract,
                        candidate,
                        localization,
                        cfg,
                        excluded_validation_available=effective_excluded_validation_available,
                        preflight_context=preflight_context,
                    ),
                )
                for candidate in candidates
            ]
            if not _should_retry_after_validation(candidate_validations, validation_attempt, cfg, generator):
                break
            setattr(generator, "validation_feedback", _validation_retry_feedback(candidate_validations))
        duplicate_candidates.extend(round_duplicates)
        if stopped == "candidate_generation_failed":
            break
        if not candidates:
            stopped = "no_candidates"
            break
        for candidate, validation in candidate_validations:
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
            eligible = _eligible_for_evaluation(candidate, validation, cfg, evaluator, effective_excluded_validation_available)
            evaluation = CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                proposal=candidate,
                validation=validation,
                eligible_for_confirmation=eligible,
                validation_split=candidate.validation_split,
                blocked_reason=None if eligible else _blocked_reason(candidate, validation, evaluator, effective_excluded_validation_available),
            )
            if eligible and evaluator is not None:
                try:
                    _evaluate_candidate(
                        candidate,
                        evaluation,
                        evaluator,
                        external_evaluator,
                        excluded_evidence_kind=excluded_evidence_kind,
                    )
                    if evaluation.confirmed:
                        confirmed.append(candidate.candidate_id)
                except Exception as exc:
                    if candidate.validation_split == "excluded_validation":
                        evaluation.execution_error = _excluded_evidence_error(exc)
                    else:
                        evaluation.execution_error = str(exc)
                    evaluation.evaluated = True
            evaluations.append(evaluation)
            if evaluation.confirmed and cfg.stop_on_first_confirmed:
                stopped = (
                    str(evaluation.final_label)
                    if evaluation.final_label in {"exploratory_confirmed", "holdout_confirmed", "external_confirmed"}
                    else "confirmed"
                )
                return ClaimSearchState(
                    original_claim=contract,
                    failure_localization=localization,
                    lineage_graph=lineage,
                    used_evidence=localization.evidence,
                    candidate_history=history,
                    duplicate_candidates=duplicate_candidates,
                    evaluations=evaluations,
                    confirmed_candidates=confirmed,
                    llm_candidate_prompts=_generator_prompt_records(generator),
                    llm_candidate_responses=_generator_response_records(generator),
                    stopped_reason=stopped,
                )
        if evaluator is None:
            stopped = "no_evaluator"
            break

    return ClaimSearchState(
        original_claim=contract,
        failure_localization=localization,
        lineage_graph=lineage,
        used_evidence=localization.evidence,
        candidate_history=history,
        duplicate_candidates=duplicate_candidates,
        evaluations=evaluations,
        confirmed_candidates=confirmed,
        llm_candidate_prompts=_generator_prompt_records(generator),
        llm_candidate_responses=_generator_response_records(generator),
        stopped_reason=stopped,
    )


def build_claim_search_artifacts(
    contract: ClaimContract,
    verdict: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
    *,
    config: ClaimSearchConfig | None = None,
    llm: LLMClient | None = None,
    candidate_generator: CandidateGenerator | None = None,
    evaluator: CandidateEvaluator | None = None,
    external_evaluator: CandidateEvaluator | None = None,
    excluded_evidence_kind: Literal["holdout", "external"] = "external",
    preflight_context: CandidatePreflightContext | None = None,
    validation_evidence_catalog: dict[str, Any] | None = None,
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
        external_evaluator=external_evaluator,
        excluded_evidence_kind=excluded_evidence_kind,
        excluded_validation_available=external_evaluator is not None,
        preflight_context=preflight_context,
        validation_evidence_catalog=validation_evidence_catalog,
    )
    return {
        "claim_search_config": cfg.model_dump(mode="json"),
        "failure_localization": state.failure_localization.model_dump(mode="json") if state.failure_localization else None,
        "claim_search_trace": state.model_dump(mode="json"),
        "candidate_claims": [item.model_dump(mode="json") for item in state.candidate_history],
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
    duplicates = [duplicate for state in states for duplicate in state.duplicate_candidates]
    valid = [evaluation for evaluation in evaluations if evaluation.validation.ok]
    connected = [evaluation for evaluation in valid if not any("connection" in item.lower() for item in evaluation.validation.violations)]
    gaming = [
        evaluation
        for evaluation in evaluations
        if any(_is_hacking_violation(item) for item in evaluation.validation.violations)
    ]
    no_holdout = [evaluation for evaluation in evaluations if evaluation.blocked_reason == "no_excluded_validation_evidence"]
    any_supported = [
        evaluation
        for evaluation in evaluations
        if not evaluation.execution_error
        and (
            evaluation.final_label in {"exploratory_confirmed", "confirmed", "holdout_confirmed", "external_confirmed"}
            or evaluation.external_confirmed
            or evaluation.holdout_confirmed
        )
    ]
    final_confirmed = [
        evaluation
        for evaluation in evaluations
        if not evaluation.execution_error and evaluation.final_label in {"confirmed", "holdout_confirmed", "external_confirmed"}
    ]
    exploratory_confirmed = [
        evaluation for evaluation in evaluations if not evaluation.execution_error and evaluation.final_label == "exploratory_confirmed"
    ]
    same_data_exploratory = [
        evaluation
        for evaluation in exploratory_confirmed
        if evaluation.validation_split == "current_data_adaptive" or evaluation.same_underlying_data is True
    ]
    external_confirmed = [evaluation for evaluation in evaluations if not evaluation.execution_error and evaluation.external_confirmed]
    holdout_confirmed = [evaluation for evaluation in evaluations if not evaluation.execution_error and evaluation.holdout_confirmed]
    excluded_confirmed = [
        evaluation
        for evaluation in evaluations
        if not evaluation.execution_error
        and (
            evaluation.final_label in {"holdout_confirmed", "external_confirmed"}
            or evaluation.external_confirmed
            or evaluation.holdout_confirmed
        )
    ]
    contract_repair_confirmed = [
        evaluation
        for evaluation in effective_evaluations
        if evaluation.final_label == "confirmed" and evaluation.validation_split == "current_data_contract_repair"
    ]
    execution_errors = [evaluation for evaluation in evaluations if evaluation.execution_error]
    excluded_evidence_errors = [evaluation for evaluation in evaluations if evaluation.excluded_evidence_error]
    execution_error_types = Counter(
        str(evaluation.execution_error).split(":", 1)[0]
        for evaluation in execution_errors
    )
    excluded_evidence_error_types = Counter(
        str(evaluation.excluded_evidence_error).split(":", 1)[0]
        for evaluation in excluded_evidence_errors
    )
    preflight_blocked = [
        evaluation
        for evaluation in evaluations
        if any(str(violation).startswith("Preflight:") for violation in evaluation.validation.violations)
    ]
    preflight_pass_count = len(evaluations) - len(preflight_blocked)
    searches_by_target = Counter(str(state.source_metadata.get("target_family") or "unknown") for state in states)
    searches_by_source_mode = Counter(str(state.source_metadata.get("source_mode") or "unknown") for state in states)
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
    return {
        "n_searches": len(states),
        "candidate_count": len(candidates),
        "duplicate_candidate_count": len(duplicates),
        "valid_connected_candidate_count": len(connected),
        "valid_connected_candidate_rate": len(connected) / len(candidates) if candidates else 0.0,
        "preflight_pass_candidate_count": preflight_pass_count,
        "preflight_pass_candidate_rate": preflight_pass_count / len(candidates) if candidates else 0.0,
        "preflight_block_count": len(preflight_blocked),
        "admissible_evaluation_count": sum(1 for item in evaluations if item.eligible_for_confirmation),
        "exploratory_confirmed_count": len(exploratory_confirmed),
        "same_data_exploratory_confirmed_count": len(same_data_exploratory),
        "confirmed_count": len(final_confirmed),
        "final_confirmed_count": len(final_confirmed),
        "supported_candidate_count": len(any_supported),
        "any_supported_candidate_count": len(any_supported),
        "contract_repair_confirmed_count": len(contract_repair_confirmed),
        "holdout_confirmed_count": len(holdout_confirmed),
        "external_confirmed_count": len(external_confirmed),
        "confirmed_on_external_evidence_count": len(external_confirmed),
        "confirmed_on_excluded_evidence_count": len(excluded_confirmed),
        "false_current_data_confirmation_count": sum(1 for item in evaluations if _false_current_data_confirmation(item)),
        "hacking_block_count": len(gaming),
        "no_holdout_abstention_count": len(no_holdout),
        "execution_error_count": len(execution_errors),
        "execution_error_type_counts": dict(execution_error_types),
        "excluded_evidence_error_count": len(excluded_evidence_errors),
        "excluded_evidence_error_type_counts": dict(excluded_evidence_error_types),
        "raw_final_label_counts": dict(raw_final_label_counts),
        "effective_final_label_counts": dict(effective_final_label_counts),
        "final_label_counts": dict(effective_final_label_counts),
        "stopped_reason_counts": dict(Counter(state.stopped_reason for state in states)),
        "searches_by_target_family": dict(searches_by_target),
        "searches_by_source_mode": dict(searches_by_source_mode),
        "raw_candidate_final_label_counts_by_target_family": raw_final_labels_by_target,
        "effective_candidate_final_label_counts_by_target_family": effective_final_labels_by_target,
        "candidate_final_label_counts_by_target_family": effective_final_labels_by_target,
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
    }
    return CandidateClaimProposal.model_validate(data)


def _candidate_scientific_signature(candidate: CandidateClaimProposal) -> str:
    contract = candidate.proposed_contract.model_dump(mode="json")
    contract.pop("claim_id", None)
    contract.pop("question", None)
    payload = {
        "proposal_type": candidate.proposal_type,
        "transform_type": candidate.transform_type,
        "validation_split": candidate.validation_split,
        "provenance": candidate.provenance,
        "requires_new_evidence": candidate.requires_new_evidence,
        "can_confirm_on_current_data": candidate.can_confirm_on_current_data,
        "contract": contract,
    }
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


def _candidate(
    contract: ClaimContract,
    localization: FailureLocalization,
    parent_claim_id: str,
    round_index: int,
    index: int,
    *,
    proposal_type: str,
    transform_type: str,
    provenance: str,
    validation_split: ValidationSplit,
    question: str,
    rationale: str,
    connection: str,
    proposed_contract: ClaimContract,
    disposition_label: str | None = None,
    requires_new_evidence: bool,
    can_confirm_on_current_data: bool,
    evidence: list[str],
) -> CandidateClaimProposal:
    candidate_id = f"{contract.claim_id}_r{round_index}_c{index + 1}_{transform_type}"
    evidence_policy = CandidateEvidencePolicy(
        provenance=provenance,
        requires_new_evidence=requires_new_evidence,
        can_confirm_on_current_data=can_confirm_on_current_data,
        validation_split=validation_split,
    )
    domain_core = _domain_core_from_contract(contract)
    preservation_check = CandidatePreservationCheck(
        preserves_population=True,
        preserves_cohort_family=True,
        preserves_predictor_or_contrast=True,
        preserves_outcome_modality=True,
        preserves_direction_family=True,
        preserves_scientific_motivation=True,
        changed_fields=_changed_fields_for_transform(transform_type),
        allowed_change_rationale=connection,
    )
    return CandidateClaimProposal.model_validate(
        {
            "candidate_id": candidate_id,
            "parent_claim_id": parent_claim_id,
            "round_index": round_index,
            "transform_type": transform_type,
            "domain_core": domain_core.model_dump(mode="json"),
            "preservation_check": preservation_check.model_dump(mode="json"),
            "evidence_policy": evidence_policy.model_dump(mode="json"),
            "connection_rationale": connection,
            "validation_split": validation_split,
            "source_claim_id": contract.claim_id,
            "proposal_type": proposal_type,
            "rationale": rationale,
            "proposed_question": question,
            "proposed_contract": proposed_contract.model_dump(mode="json"),
            "disposition_label": disposition_label,
            "provenance": provenance,
            "requires_new_evidence": requires_new_evidence,
            "can_confirm_on_current_data": can_confirm_on_current_data,
            "supported_by_evidence": evidence,
        }
    )


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


def _contract_repair_candidate(contract: ClaimContract, localization: FailureLocalization) -> ClaimContract | None:
    if localization.failure_kind == "confound_failure":
        data = contract.model_dump(mode="json")
        if "site" not in data["covariates"] and "site" != data["estimand"]["predictor"]:
            data["covariates"] = [*data["covariates"], "site"]
            data["gates"]["confound"]["require_covariates"] = sorted(
                set(data["gates"]["confound"]["require_covariates"]) | {"site"}
            )
            return ClaimContract.model_validate(data)
    return None


def _executable_proposal_types(proposal_types: list[str]) -> list[str]:
    executable = {"corrected_contract", "exploratory_followup_claim", "independent_replication_claim"}
    return [proposal_type for proposal_type in proposal_types if proposal_type in executable]


def _followup_contract(
    contract: ClaimContract,
    *,
    suffix: str,
    outcome: str | None = None,
    family_size_increment: int = 1,
) -> ClaimContract:
    data = contract.model_dump(mode="json")
    data["claim_id"] = f"{contract.claim_id}_{suffix}"
    data["question"] = f"Adaptive follow-up: {contract.question}"
    if outcome is not None:
        data["estimand"]["outcome"] = outcome
    provenance = dict(data.get("search_provenance") or {})
    provenance["declared"] = True
    provenance["selection"] = "discovery_only"
    provenance["family_size"] = max(int(provenance.get("family_size") or 1), contract.search_provenance.family_size + family_size_increment)
    data["search_provenance"] = provenance
    return ClaimContract.model_validate(data)


def _narrower_outcome_name(contract: ClaimContract) -> str:
    outcome = _outcome_name(contract)
    if outcome.endswith("_mean_abs"):
        return outcome.replace("_mean_abs", "_network_mean_abs")
    if outcome.endswith("_hippocampus"):
        return f"{outcome}_subfield"
    return f"{outcome}_focused"


def _candidate_for_current_data_adaptive(candidate: CandidateClaimProposal) -> CandidateClaimProposal:
    if candidate.validation_split != "future_required":
        return candidate
    if candidate.proposal_type not in {"exploratory_followup_claim", "independent_replication_claim"}:
        return candidate
    evidence_policy = candidate.evidence_policy.model_copy(
        update={
            "validation_split": "current_data_adaptive",
            "requires_new_evidence": False,
            "can_confirm_on_current_data": True,
        }
    )
    return candidate.model_copy(
        update={
            "validation_split": "current_data_adaptive",
            "requires_new_evidence": False,
            "can_confirm_on_current_data": True,
            "evidence_policy": evidence_policy,
        }
    )


def _evaluate_candidate(
    candidate: CandidateClaimProposal,
    evaluation: CandidateEvaluation,
    evaluator: CandidateEvaluator,
    external_evaluator: CandidateEvaluator | None,
    *,
    excluded_evidence_kind: Literal["holdout", "external"] = "external",
) -> None:
    if candidate.validation_split == "excluded_validation":
        if external_evaluator is None:
            evaluation.evaluated = True
            evaluation.final_label = None
            evaluation.blocked_reason = "no_excluded_validation_evidence"
            return
        result = dict(external_evaluator(candidate))
        evaluation.evaluated = True
        raw_label = str(result.get("final_label", result.get("label", "unknown")))
        gate_results = result.get("gate_results")
        evaluation.gate_results = gate_results if isinstance(gate_results, dict) else None
        _apply_evidence_scope(evaluation, evaluation.gate_results)
        actual_evidence_kind = _excluded_evidence_kind_from_gate_results(evaluation.gate_results, excluded_evidence_kind)
        evaluation.excluded_evidence_kind = actual_evidence_kind
        evaluation.excluded_evidence_used = True
        evaluation.final_label = _excluded_final_label(raw_label, actual_evidence_kind)
        if actual_evidence_kind == "holdout":
            evaluation.holdout_label = raw_label
            evaluation.holdout_gate_results = evaluation.gate_results
            evaluation.holdout_confirmed = raw_label == "confirmed"
        else:
            evaluation.external_label = raw_label
            evaluation.external_gate_results = evaluation.gate_results
            evaluation.external_confirmed = raw_label == "confirmed"
            evaluation.external_evidence_used = True
        evaluation.confirmed = evaluation.final_label in {"holdout_confirmed", "external_confirmed"} and not evaluation.execution_error
        return

    result = dict(evaluator(candidate))
    evaluation.evaluated = True
    raw_label = str(result.get("final_label", result.get("label", "unknown")))
    gate_results = result.get("gate_results")
    evaluation.gate_results = gate_results if isinstance(gate_results, dict) else None
    _apply_evidence_scope(evaluation, evaluation.gate_results)

    if candidate.validation_split == "current_data_adaptive":
        evaluation.exploratory_label = raw_label
        evaluation.exploratory_gate_results = evaluation.gate_results
        evaluation.exploratory_confirmed = raw_label == "confirmed"
        evaluation.final_label = "exploratory_confirmed" if evaluation.exploratory_confirmed else raw_label
        if evaluation.exploratory_confirmed and external_evaluator is not None:
            try:
                external_result = dict(external_evaluator(candidate))
                external_label = str(external_result.get("final_label", external_result.get("label", "unknown")))
                external_gate_results = external_result.get("gate_results")
                actual_evidence_kind = _excluded_evidence_kind_from_gate_results(
                    external_gate_results if isinstance(external_gate_results, dict) else None,
                    excluded_evidence_kind,
                )
                evaluation.excluded_evidence_kind = actual_evidence_kind
                evaluation.excluded_evidence_used = True
                evaluation.external_evidence_used = actual_evidence_kind == "external"
                if actual_evidence_kind == "holdout":
                    evaluation.holdout_label = external_label
                    evaluation.holdout_gate_results = external_gate_results if isinstance(external_gate_results, dict) else None
                    evaluation.holdout_confirmed = external_label == "confirmed"
                    if evaluation.holdout_confirmed:
                        evaluation.final_label = "holdout_confirmed"
                else:
                    evaluation.external_label = external_label
                    evaluation.external_gate_results = external_gate_results if isinstance(external_gate_results, dict) else None
                    evaluation.external_confirmed = external_label == "confirmed"
                    if evaluation.external_confirmed:
                        evaluation.final_label = "external_confirmed"
            except Exception as exc:
                evaluation.excluded_evidence_error = _excluded_evidence_error(exc)
        evaluation.confirmed = (
            evaluation.final_label in {"exploratory_confirmed", "holdout_confirmed", "external_confirmed"}
            and not evaluation.execution_error
        )
        return

    evaluation.final_label = raw_label
    if raw_label == "confirmed" and external_evaluator is not None and candidate.validation_split != "excluded_validation":
        try:
            external_result = dict(external_evaluator(candidate))
            external_label = str(external_result.get("final_label", external_result.get("label", "unknown")))
            external_gate_results = external_result.get("gate_results")
            actual_evidence_kind = _excluded_evidence_kind_from_gate_results(
                external_gate_results if isinstance(external_gate_results, dict) else None,
                excluded_evidence_kind,
            )
            evaluation.excluded_evidence_kind = actual_evidence_kind
            evaluation.excluded_evidence_used = True
            evaluation.external_evidence_used = actual_evidence_kind == "external"
            if actual_evidence_kind == "holdout":
                evaluation.holdout_label = external_label
                evaluation.holdout_gate_results = external_gate_results if isinstance(external_gate_results, dict) else None
                evaluation.holdout_confirmed = external_label == "confirmed"
                if evaluation.holdout_confirmed:
                    evaluation.final_label = "holdout_confirmed"
            else:
                evaluation.external_label = external_label
                evaluation.external_gate_results = external_gate_results if isinstance(external_gate_results, dict) else None
                evaluation.external_confirmed = external_label == "confirmed"
                if evaluation.external_confirmed:
                    evaluation.final_label = "external_confirmed"
        except Exception as exc:
            evaluation.excluded_evidence_error = _excluded_evidence_error(exc)
    evaluation.confirmed = evaluation.final_label in {"confirmed", "holdout_confirmed", "external_confirmed"} and not evaluation.execution_error


def _excluded_evidence_error(exc: Exception) -> str:
    return f"excluded_evidence_unavailable_for_candidate: {exc}"


def _excluded_final_label(raw_label: str, excluded_evidence_kind: Literal["holdout", "external"]) -> str:
    if raw_label != "confirmed":
        return raw_label
    return "holdout_confirmed" if excluded_evidence_kind == "holdout" else "external_confirmed"


def _excluded_evidence_kind_from_gate_results(
    gate_results: dict[str, Any] | None,
    fallback: Literal["holdout", "external"],
) -> Literal["holdout", "external"]:
    if isinstance(gate_results, dict):
        scope = gate_results.get("evidence_scope")
        if isinstance(scope, dict) and scope.get("scope") in {"holdout", "external"}:
            return scope["scope"]
    return fallback


def _eligible_for_evaluation(
    candidate: CandidateClaimProposal,
    validation: ProposalValidation,
    config: ClaimSearchConfig,
    evaluator: CandidateEvaluator | None,
    excluded_validation_available: bool,
) -> bool:
    if not validation.ok or evaluator is None:
        return False
    if candidate.validation_split == "current_data_adaptive":
        return candidate.proposal_type in {"exploratory_followup_claim", "independent_replication_claim"}
    if candidate.validation_split == "excluded_validation":
        return excluded_validation_available
    if candidate.validation_split == "current_data_contract_repair":
        return config.allow_current_data_contract_repairs and candidate.proposal_type == "corrected_contract"
    return False


def _blocked_reason(
    candidate: CandidateClaimProposal,
    validation: ProposalValidation,
    evaluator: CandidateEvaluator | None,
    excluded_validation_available: bool,
) -> str | None:
    if not validation.ok:
        return "proposal_validation_failed"
    if evaluator is None and candidate.validation_split in {"current_data_adaptive", "excluded_validation", "current_data_contract_repair"}:
        return "no_evaluator"
    if candidate.validation_split == "future_required":
        return "no_excluded_validation_evidence"
    if candidate.validation_split == "excluded_validation" and not excluded_validation_available:
        return "no_excluded_validation_evidence"
    if candidate.validation_split == "none":
        return "not_confirmatory_candidate"
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
    if any(_only_proposal_shape_violations(validation) for _, validation in candidate_validations):
        return True
    if any(validation.ok for _, validation in candidate_validations):
        return False
    return any(validation.violations for _, validation in candidate_validations)


def _only_proposal_shape_violations(validation: ProposalValidation) -> bool:
    if not validation.violations:
        return False
    snippets = (
        "contract_correction transform must use corrected_contract proposal type",
        "Exploratory transforms must use exploratory_followup_claim proposal type",
        "stronger_design transform must use independent_replication_claim proposal type",
    )
    return all(any(snippet in violation for snippet in snippets) for violation in validation.violations)


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
            "Use only executable_data_catalog fields when present, and do not change predictor, group contrast, direction, gates, or covariates. "
            "If a violation says the discovery or replication cohort changed, keep proposed_contract on the parent cohorts and use validation_split=excluded_validation for holdout/external evaluation. "
            "Use compatible proposal/transform pairs: stronger_design requires independent_replication_claim; narrower_outcome_family, moderator_or_subgroup, and fixed_estimand require exploratory_followup_claim; contract_correction requires corrected_contract."
        ),
        "failed_candidates": failures,
    }


def _generator_prompt_records(generator: CandidateGenerator) -> list[dict[str, Any]]:
    records = getattr(generator, "prompt_records", None)
    return list(records) if isinstance(records, list) else []


def _generator_response_records(generator: CandidateGenerator) -> list[dict[str, Any]]:
    records = getattr(generator, "response_records", None)
    return list(records) if isinstance(records, list) else []


def _apply_evidence_scope(evaluation: CandidateEvaluation, gate_results: dict[str, Any] | None) -> None:
    if not isinstance(gate_results, dict):
        return
    data_paths = gate_results.get("data_paths")
    if not isinstance(data_paths, dict):
        return
    discovery = data_paths.get("discovery")
    replication = data_paths.get("replication")
    replication_paths = [str(path) for path in replication] if isinstance(replication, list) else []
    evaluation.resolved_discovery_path = str(discovery) if discovery is not None else None
    evaluation.resolved_replication_paths = replication_paths
    if evaluation.resolved_discovery_path and replication_paths:
        evaluation.same_underlying_data = any(path == evaluation.resolved_discovery_path for path in replication_paths)


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


def _changed_fields_for_transform(transform_type: str) -> list[str]:
    if transform_type == "narrower_outcome_family":
        return ["outcome_family"]
    if transform_type == "moderator_or_subgroup":
        return ["moderator_or_subgroup"]
    if transform_type == "stronger_design":
        return ["validation_evidence"]
    if transform_type == "fixed_estimand":
        return ["analysis_specification"]
    if transform_type == "contract_correction":
        return ["contract_encoding"]
    return []


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


def _connection_violations(original: ClaimContract, candidate: CandidateClaimProposal) -> list[str]:
    violations: list[str] = []
    if candidate.transform_type == "contract_correction":
        return violations
    if candidate.transform_type == "contract_correction" and candidate.proposal_type != "corrected_contract":
        violations.append("contract_correction transform must use corrected_contract proposal type.")
    if candidate.transform_type in {"narrower_outcome_family", "moderator_or_subgroup", "fixed_estimand"}:
        if candidate.proposal_type != "exploratory_followup_claim":
            violations.append("Exploratory transforms must use exploratory_followup_claim proposal type.")
    if candidate.transform_type == "stronger_design" and candidate.proposal_type != "independent_replication_claim":
        violations.append("stronger_design transform must use independent_replication_claim proposal type.")
    return violations


def _followup_contract_connection_violations(original: ClaimContract, candidate: CandidateClaimProposal) -> list[str]:
    candidate_contract = candidate.proposed_contract
    violations: list[str] = []
    if candidate_contract is None:
        return violations
    candidate_cohorts = [candidate_contract.discovery_cohort, *candidate_contract.replication_cohorts]
    if any(_is_holdout_cohort(cohort) for cohort in candidate_cohorts):
        violations.append(
            "Candidate contract uses holdout partitions as ordinary cohorts; keep parent cohorts and set validation_split=excluded_validation."
        )
    external_contract = any(_is_external_eval_cohort(cohort) for cohort in candidate_cohorts)
    if external_contract and not (
        candidate.proposal_type == "independent_replication_claim"
        and candidate.transform_type == "stronger_design"
        and candidate.validation_split == "excluded_validation"
    ):
        violations.append("Candidate contract uses external evaluation cohorts outside a true external independent-replication proposal.")
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
    if original.discovery_cohort != candidate_contract.discovery_cohort and not external_contract:
        violations.append(
            "Candidate contract changes discovery cohort. Keep proposed_contract on the parent discovery/replication cohorts; holdout evidence is mapped by the evaluator."
        )
    if external_contract and candidate_contract.discovery_cohort == original.discovery_cohort:
        violations.append("External independent-replication contracts must use external discovery evidence when using external cohorts.")
    if not external_contract and list(original.replication_cohorts) != list(candidate_contract.replication_cohorts):
        violations.append(
            "Candidate contract changes replication cohorts. Keep proposed_contract on the parent discovery/replication cohorts; holdout evidence is mapped by the evaluator."
        )
    if candidate_contract.discovery_cohort in set(candidate_contract.replication_cohorts):
        violations.append("Candidate contract uses the discovery cohort as replication evidence.")
    if candidate_contract.gates.multiplicity.alpha > original.gates.multiplicity.alpha:
        violations.append("Candidate contract weakened multiplicity alpha.")
    if candidate_contract.gates.multiplicity.family_size < original.gates.multiplicity.family_size:
        violations.append("Candidate contract shrank multiplicity family_size.")
    if candidate_contract.gates.power.min_power < original.gates.power.min_power:
        violations.append("Candidate contract weakened power threshold.")
    if candidate_contract.gates.multiverse.min_fraction_consistent < original.gates.multiverse.min_fraction_consistent:
        violations.append("Candidate contract weakened multiverse threshold.")
    if candidate_contract.gates.replication.alpha > original.gates.replication.alpha:
        violations.append("Candidate contract weakened replication alpha.")
    if original.gates.replication.require_same_sign and not candidate_contract.gates.replication.require_same_sign:
        violations.append("Candidate contract removes same-sign replication requirement.")
    if original.gates.replication.require_ci_overlap and not candidate_contract.gates.replication.require_ci_overlap:
        violations.append("Candidate contract removes CI-overlap replication requirement.")
    if candidate_contract.gates.replication.pattern_corr_min < original.gates.replication.pattern_corr_min:
        violations.append("Candidate contract weakened pattern-correlation threshold.")
    if candidate_contract.gates.replication.region_replication_frac_min < original.gates.replication.region_replication_frac_min:
        violations.append("Candidate contract weakened region-fraction threshold.")
    if candidate_contract.gates.replication.dice_min < original.gates.replication.dice_min:
        violations.append("Candidate contract weakened Dice threshold.")
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
    if not set(original.covariates).issubset(set(candidate_contract.covariates)):
        violations.append("Candidate contract drops original covariates.")
    return violations


def _is_holdout_cohort(cohort: str) -> bool:
    return "_HOLDOUT" in str(cohort)


def _is_external_eval_cohort(cohort: str) -> bool:
    return "_EXTERNAL_" in str(cohort)


def _false_current_data_confirmation(evaluation: CandidateEvaluation) -> bool:
    if evaluation.final_label not in {"confirmed", "holdout_confirmed", "external_confirmed"}:
        return False
    if evaluation.external_confirmed or evaluation.holdout_confirmed or evaluation.validation_split == "excluded_validation":
        return False
    if (
        evaluation.validation_split == "current_data_contract_repair"
        and evaluation.proposal.proposal_type == "corrected_contract"
        and evaluation.validation.ok
    ):
        return False
    return True


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
