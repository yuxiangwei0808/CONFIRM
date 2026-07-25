"""Self-Refine-style candidate generation for matched feedback controls."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.claim_search import (
    CandidateClaimProposal,
    CandidateGenerationError,
    ClaimSearchConfig,
    LLMCandidateGenerationResponse,
    RoundFailureContext,
    build_candidate_generation_prompt,
    parse_candidate_generation_response,
)
from confirm.contract import ClaimContract
from confirm.llm import (
    LLMClient,
    complete_structured,
    is_non_retryable_provider_error,
)
from confirm.proposals import FailureLocalization

SELF_REFINE_FEEDBACK_SYSTEM_PROMPT = """You are the feedback stage of a Self-Refine baseline for scientific claim generation.
Critique why the supplied claim or candidate set may be scientifically unclear, insufficiently specific, or difficult to execute.
Use only the supplied contracts, immutable constraints, executable data catalog, and the binary fact that the claims were not supported.
Do not infer or invent failed statistical gates, p-values, effect sizes, coefficients, confidence intervals, or validation results.
Do not propose changing the predictor, group contrast, biological direction, required covariates, cohort family, or statistical thresholds.
Return structured JSON only."""

SELF_REFINE_REFINEMENT_SYSTEM_PROMPT = """You are the refinement stage of a Self-Refine baseline for scientific claim generation.
Use the supplied critique to generate connected, executable follow-up claim candidates.
The goal is scientific refinement, not manipulating a contract to obtain a passing result.
Preserve every immutable field and use only the executable source-data catalog.
Do not invent statistical results or validation evidence.
Return structured JSON only using the supplied candidate schema."""


class SelfRefineCandidateCritique(BaseModel):
    """Critique of one failed parent or preceding candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    weaknesses: list[str] = Field(default_factory=list)
    revision_goals: list[str] = Field(default_factory=list)


class SelfRefineFeedback(BaseModel):
    """Structured feedback passed from the critique call to refinement."""

    model_config = ConfigDict(extra="forbid")

    overall_assessment: str
    candidate_feedback: list[SelfRefineCandidateCritique] = Field(
        default_factory=list
    )
    revision_priorities: list[str] = Field(default_factory=list)


class SelfRefineCandidateGenerator:
    """Two-call Self-Refine adapter using the frozen v7 candidate wire schema."""

    def __init__(
        self,
        llm: LLMClient,
        preflight_context: Optional[CandidatePreflightContext] = None,
    ) -> None:
        self.llm = llm
        self.preflight_context = preflight_context
        self.candidate_history: list[CandidateClaimProposal] = []
        self.round_failure_context: Optional[RoundFailureContext] = None
        self.validation_feedback: Optional[dict[str, Any]] = None
        self.validation_retry_index = 0
        self.prompt_records: list[dict[str, Any]] = []
        self.response_records: list[dict[str, Any]] = []
        self._feedback_by_round: dict[int, SelfRefineFeedback] = {}

    def __call__(
        self,
        contract: ClaimContract,
        localization: FailureLocalization,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
    ) -> list[CandidateClaimProposal]:
        feedback = self._feedback_by_round.get(round_index)
        if feedback is None:
            feedback = self._generate_feedback(
                contract,
                config,
                round_index,
                parent_claim_id,
            )
            self._feedback_by_round[round_index] = feedback
        return self._generate_refinement(
            contract,
            localization,
            config,
            round_index,
            parent_claim_id,
            feedback,
        )

    def _safe_failed_candidates(self) -> list[dict[str, Any]]:
        context = self.round_failure_context
        if context is None:
            return []
        history = {
            candidate.candidate_id: candidate for candidate in self.candidate_history
        }
        rows: list[dict[str, Any]] = []
        for failed in context.failed_candidates:
            candidate = history.get(failed.candidate_id)
            rows.append(
                {
                    "candidate_id": failed.candidate_id,
                    "binary_status": "not_source_supported",
                    "question": (
                        candidate.proposed_question if candidate is not None else None
                    ),
                    "contract": failed.effective_contract.model_dump(mode="json"),
                    "rationale": candidate.rationale if candidate is not None else None,
                }
            )
        return rows

    def _feedback_prompt(
        self,
        contract: ClaimContract,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
    ) -> str:
        failed_candidates = self._safe_failed_candidates()
        payload = {
            "task": "critique_claims_for_self_refinement",
            "round_index": round_index,
            "parent_claim_id": parent_claim_id,
            "binary_status": "not_source_supported",
            "original_contract": contract.model_dump(mode="json"),
            "claims_to_critique": (
                failed_candidates
                if failed_candidates
                else [
                    {
                        "candidate_id": parent_claim_id,
                        "binary_status": "not_source_supported",
                        "question": contract.question,
                    }
                ]
            ),
            "immutable_constraints": {
                "predictor": contract.estimand.predictor,
                "group": (
                    contract.estimand.group.model_dump(mode="json")
                    if contract.estimand.group
                    else None
                ),
                "direction": contract.estimand.direction,
                "covariates": list(contract.covariates),
                "required_confound_covariates": list(
                    contract.gates.confound.require_covariates
                ),
                "discovery_cohort": contract.discovery_cohort,
                "replication_cohorts": list(contract.replication_cohorts),
                "gates": contract.gates.model_dump(mode="json"),
                "search_provenance": contract.search_provenance.model_dump(
                    mode="json"
                ),
            },
            "executable_data_catalog": (
                self.preflight_context.prompt_catalog(contract)
                if self.preflight_context is not None
                else None
            ),
            "max_candidates": config.max_candidates_per_round,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def _generate_feedback(
        self,
        contract: ClaimContract,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
    ) -> SelfRefineFeedback:
        prompt = self._feedback_prompt(
            contract,
            config,
            round_index,
            parent_claim_id,
        )
        parsed = self._complete_with_records(
            system=SELF_REFINE_FEEDBACK_SYSTEM_PROMPT,
            prompt=prompt,
            response_model=SelfRefineFeedback,
            retries=config.llm_schema_retries,
            call_type="feedback",
            round_index=round_index,
            parent_claim_id=parent_claim_id,
            candidate_count=0,
        )
        return SelfRefineFeedback.model_validate(parsed)

    def _generate_refinement(
        self,
        contract: ClaimContract,
        localization: FailureLocalization,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
        feedback: SelfRefineFeedback,
    ) -> list[CandidateClaimProposal]:
        generic_config = config.model_copy(update={"feedback_mode": "generic_retry"})
        _, base_user = build_candidate_generation_prompt(
            contract,
            localization,
            generic_config,
            round_index,
            parent_claim_id,
            candidate_history=self.candidate_history,
            executable_catalog=(
                self.preflight_context.prompt_catalog(contract)
                if self.preflight_context is not None
                else None
            ),
            validation_feedback=self.validation_feedback,
            round_failure_context=self.round_failure_context,
        )
        payload = json.loads(base_user)
        payload.pop("output_schema", None)
        payload["candidate_history"] = self._safe_failed_candidates()
        payload["self_refine_feedback"] = feedback.model_dump(mode="json")
        payload["feedback_source"] = (
            "self_critique_using_binary_failure_status_without_gate_localization"
        )
        prompt = json.dumps(payload, indent=2, sort_keys=True)

        parsed_candidates: list[CandidateClaimProposal] = []

        def validate_response(response: LLMCandidateGenerationResponse) -> None:
            nonlocal parsed_candidates
            parsed_candidates = parse_candidate_generation_response(
                response.model_dump_json(),
                contract,
                localization,
                generic_config,
                round_index,
                parent_claim_id,
            )

        self._complete_with_records(
            system=SELF_REFINE_REFINEMENT_SYSTEM_PROMPT,
            prompt=prompt,
            response_model=LLMCandidateGenerationResponse,
            retries=config.llm_schema_retries,
            call_type="refinement",
            round_index=round_index,
            parent_claim_id=parent_claim_id,
            candidate_count=len(parsed_candidates),
            validator=validate_response,
        )
        self.response_records[-1]["candidate_count"] = len(parsed_candidates)
        self.validation_feedback = None
        return parsed_candidates

    def _complete_with_records(
        self,
        *,
        system: str,
        prompt: str,
        response_model: type[BaseModel],
        retries: int,
        call_type: str,
        round_index: int,
        parent_claim_id: str,
        candidate_count: int,
        validator: Any = None,
    ) -> BaseModel:
        active_prompt = prompt
        last_error: Optional[Exception] = None
        for attempt_index in range(retries + 1):
            raw = ""
            error: Optional[Exception] = None
            parsed: Optional[BaseModel] = None
            try:
                raw = complete_structured(
                    self.llm,
                    system,
                    active_prompt,
                    response_model,
                )
                parsed = response_model.model_validate_json(raw)
                if validator is not None:
                    validator(parsed)
            except Exception as exc:
                error = exc
                last_error = exc
            prompt_record = {
                "call_type": call_type,
                "round_index": round_index,
                "parent_claim_id": parent_claim_id,
                "attempt_index": attempt_index,
                "validation_retry_index": int(self.validation_retry_index),
                "model": getattr(self.llm, "model", type(self.llm).__name__),
                "system": system,
                "user": active_prompt,
                "prompt_hash": hashlib.sha256(
                    (system + "\n" + active_prompt).encode("utf-8")
                ).hexdigest(),
            }
            self.prompt_records.append(prompt_record)
            self.response_records.append(
                {
                    "call_type": call_type,
                    "round_index": round_index,
                    "parent_claim_id": parent_claim_id,
                    "attempt_index": attempt_index,
                    "validation_retry_index": int(self.validation_retry_index),
                    "model": prompt_record["model"],
                    "raw_response": raw,
                    "candidate_count": (
                        candidate_count
                        if call_type == "refinement"
                        and error is None
                        else 0
                    ),
                    "parse_error": str(error) if error is not None else None,
                    "schema_valid": error is None,
                    "call_metadata": dict(
                        getattr(self.llm, "last_call_metadata", {}) or {}
                    ),
                }
            )
            if error is None and parsed is not None:
                return parsed
            if error is not None and is_non_retryable_provider_error(error):
                break
            active_prompt = (
                f"{prompt}\n\nPrevious structured-output error: {error}. "
                "Return a corrected response matching the schema exactly."
            )
        raise CandidateGenerationError(
            f"Structured output failed after {retries + 1} attempts: {last_error}"
        )
