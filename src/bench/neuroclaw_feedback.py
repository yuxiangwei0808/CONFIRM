"""Candidate generators that let an existing agent revise its own abstained claims.

Two arms share the same harness, budget, validator, and multiplicity policy, and
differ only in the feedback the agent receives after a claim fails to be
supported:

``NeuroClawSelfCritiqueGenerator``
    The agent critiques its own claim from the binary fact that it was not
    supported, then revises. It never sees which gate failed. This is the
    agent-alone control.

``NeuroClawDiagnosisGenerator``
    The agent revises from CONFIRM's typed gate localization. This is the
    retrofit arm.

Both drive the frozen candidate wire schema, so candidates from either arm face
the identical validator and cumulative multiplicity correction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.claim_evaluation_baselines import NEUROCLAW_PERSONA_ORDER, NEUROCLAW_PERSONAS
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
from confirm.llm import LLMClient, complete_structured, is_non_retryable_provider_error
from confirm.proposals import FailureLocalization

_PANEL = "\n".join(
    f"- {persona.replace('_', ' ')}: {NEUROCLAW_PERSONAS[persona]}"
    for persona in NEUROCLAW_PERSONA_ORDER
)

NEUROCLAW_CRITIQUE_SYSTEM_PROMPT = f"""You are the NeuroClaw statistical-critic panel reviewing one of your own neuroimaging claims that was not supported.
The panel speaks with these perspectives:
{_PANEL}

Critique why the claim may be scientifically unclear, insufficiently specific, or difficult to execute.
Use only the supplied contracts, immutable constraints, executable data catalog, and the binary fact that the claim was not supported.
Do not infer or invent failed statistical gates, p-values, effect sizes, coefficients, confidence intervals, or validation results.
Do not propose changing the predictor, group contrast, biological direction, required covariates, cohort family, or statistical thresholds.
Return structured JSON only."""

NEUROCLAW_REVISION_SYSTEM_PROMPT = f"""You are the NeuroClaw statistical-critic panel revising one of your own neuroimaging claims.
The panel speaks with these perspectives:
{_PANEL}

Generate connected, executable follow-up claim candidates.
The goal is scientific refinement, not manipulating a contract to obtain a passing result.
Preserve every immutable field and use only the executable source-data catalog.
Do not invent statistical results or validation evidence.
Return structured JSON only using the supplied candidate schema."""


class NeuroClawCritique(BaseModel):
    """Panel critique of a claim that was not supported."""

    model_config = ConfigDict(extra="forbid")

    overall_assessment: str
    weaknesses: list[str] = Field(default_factory=list)
    revision_priorities: list[str] = Field(default_factory=list)


class _RecordingGenerator:
    """Shared prompt/response bookkeeping expected by the claim-search harness."""

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

    def _catalog(self, contract: ClaimContract) -> Any:
        if self.preflight_context is None:
            return None
        return self.preflight_context.prompt_catalog(contract)

    def _failed_candidates(self) -> list[dict[str, Any]]:
        context = self.round_failure_context
        if context is None:
            return []
        history = {item.candidate_id: item for item in self.candidate_history}
        rows: list[dict[str, Any]] = []
        for failed in context.failed_candidates:
            candidate = history.get(failed.candidate_id)
            rows.append(
                {
                    "candidate_id": failed.candidate_id,
                    "binary_status": "not_source_supported",
                    "question": candidate.proposed_question if candidate else None,
                    "contract": failed.effective_contract.model_dump(mode="json"),
                    "rationale": candidate.rationale if candidate else None,
                }
            )
        return rows

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
                raw = complete_structured(self.llm, system, active_prompt, response_model)
                parsed = response_model.model_validate_json(raw)
                if validator is not None:
                    validator(parsed)
            except Exception as exc:  # noqa: BLE001 - recorded per attempt
                error = exc
                last_error = exc
            model_name = getattr(self.llm, "model", type(self.llm).__name__)
            self.prompt_records.append(
                {
                    "call_type": call_type,
                    "round_index": round_index,
                    "parent_claim_id": parent_claim_id,
                    "attempt_index": attempt_index,
                    "validation_retry_index": int(self.validation_retry_index),
                    "model": model_name,
                    "system": system,
                    "user": active_prompt,
                    "prompt_hash": hashlib.sha256(
                        (system + "\n" + active_prompt).encode("utf-8")
                    ).hexdigest(),
                }
            )
            self.response_records.append(
                {
                    "call_type": call_type,
                    "round_index": round_index,
                    "parent_claim_id": parent_claim_id,
                    "attempt_index": attempt_index,
                    "validation_retry_index": int(self.validation_retry_index),
                    "model": model_name,
                    "raw_response": raw,
                    "candidate_count": (
                        candidate_count if call_type == "refinement" and error is None else 0
                    ),
                    "parse_error": str(error) if error is not None else None,
                    "schema_valid": error is None,
                    "call_metadata": dict(getattr(self.llm, "last_call_metadata", {}) or {}),
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

    def _refine(
        self,
        contract: ClaimContract,
        localization: FailureLocalization,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
        system: str,
        extra_payload: dict[str, Any],
    ) -> list[CandidateClaimProposal]:
        _, base_user = build_candidate_generation_prompt(
            contract,
            localization,
            config,
            round_index,
            parent_claim_id,
            candidate_history=self.candidate_history,
            executable_catalog=self._catalog(contract),
            validation_feedback=self.validation_feedback,
            round_failure_context=self.round_failure_context,
        )
        payload = json.loads(base_user)
        payload.pop("output_schema", None)
        payload.update(extra_payload)
        prompt = json.dumps(payload, indent=2, sort_keys=True)

        parsed_candidates: list[CandidateClaimProposal] = []

        def validate_response(response: LLMCandidateGenerationResponse) -> None:
            nonlocal parsed_candidates
            parsed_candidates = parse_candidate_generation_response(
                response.model_dump_json(),
                contract,
                localization,
                config,
                round_index,
                parent_claim_id,
            )

        self._complete_with_records(
            system=system,
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


class NeuroClawDiagnosisGenerator(_RecordingGenerator):
    """Agent revises using CONFIRM's typed gate localization (one call per round)."""

    def __call__(
        self,
        contract: ClaimContract,
        localization: FailureLocalization,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
    ) -> list[CandidateClaimProposal]:
        diagnosis_config = config.model_copy(update={"feedback_mode": "structured_diagnosis"})
        return self._refine(
            contract,
            localization,
            diagnosis_config,
            round_index,
            parent_claim_id,
            NEUROCLAW_REVISION_SYSTEM_PROMPT,
            {"feedback_source": "confirm_structured_gate_diagnosis"},
        )


class NeuroClawSelfCritiqueGenerator(_RecordingGenerator):
    """Agent critiques and revises its own claim without gate localization."""

    def __init__(
        self,
        llm: LLMClient,
        preflight_context: Optional[CandidatePreflightContext] = None,
    ) -> None:
        super().__init__(llm, preflight_context)
        self._critique_by_round: dict[int, NeuroClawCritique] = {}

    def __call__(
        self,
        contract: ClaimContract,
        localization: FailureLocalization,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
    ) -> list[CandidateClaimProposal]:
        generic_config = config.model_copy(update={"feedback_mode": "generic_retry"})
        critique = self._critique_by_round.get(round_index)
        if critique is None:
            critique = self._critique(contract, generic_config, round_index, parent_claim_id)
            self._critique_by_round[round_index] = critique
        return self._refine(
            contract,
            localization,
            generic_config,
            round_index,
            parent_claim_id,
            NEUROCLAW_REVISION_SYSTEM_PROMPT,
            {
                "candidate_history": self._failed_candidates(),
                "neuroclaw_panel_critique": critique.model_dump(mode="json"),
                "feedback_source": (
                    "agent_self_critique_using_binary_failure_status_without_gate_localization"
                ),
            },
        )

    def _critique(
        self,
        contract: ClaimContract,
        config: ClaimSearchConfig,
        round_index: int,
        parent_claim_id: str,
    ) -> NeuroClawCritique:
        failed = self._failed_candidates()
        payload = {
            "task": "panel_critique_of_unsupported_claim",
            "round_index": round_index,
            "parent_claim_id": parent_claim_id,
            "binary_status": "not_source_supported",
            "original_contract": contract.model_dump(mode="json"),
            "claims_to_critique": failed
            or [
                {
                    "candidate_id": parent_claim_id,
                    "binary_status": "not_source_supported",
                    "question": contract.question,
                }
            ],
            "immutable_constraints": {
                "predictor": contract.estimand.predictor,
                "group": (
                    contract.estimand.group.model_dump(mode="json")
                    if contract.estimand.group
                    else None
                ),
                "direction": contract.estimand.direction,
                "covariates": list(contract.covariates),
                "required_confound_covariates": list(contract.gates.confound.require_covariates),
                "discovery_cohort": contract.discovery_cohort,
                "replication_cohorts": list(contract.replication_cohorts),
                "gates": contract.gates.model_dump(mode="json"),
                "search_provenance": contract.search_provenance.model_dump(mode="json"),
            },
            "executable_data_catalog": self._catalog(contract),
            "max_candidates": config.max_candidates_per_round,
        }
        parsed = self._complete_with_records(
            system=NEUROCLAW_CRITIQUE_SYSTEM_PROMPT,
            prompt=json.dumps(payload, indent=2, sort_keys=True),
            response_model=NeuroClawCritique,
            retries=config.llm_schema_retries,
            call_type="critique",
            round_index=round_index,
            parent_claim_id=parent_claim_id,
            candidate_count=0,
        )
        return NeuroClawCritique.model_validate(parsed)
