from __future__ import annotations

import json

from confirm.claim_search import (
    ClaimSearchConfig,
    localize_failure,
    run_claim_search,
    validate_candidate_claim,
)
from confirm.contract import ClaimContract
from confirm.self_refine import SelfRefineCandidateGenerator, SelfRefineFeedback


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "parent",
            "question": "Do cases have lower hippocampal volume than controls?",
            "estimand": {
                "type": "group_diff",
                "outcome": "smri_hippocampus",
                "predictor": "dx",
                "group": {"var": "dx", "case": "case", "control": "control"},
                "direction": "negative",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["age", "sex"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
            "search_provenance": {
                "declared": True,
                "family_size": 1,
                "selection": "preregistered",
            },
            "gates": {
                "multiplicity": {
                    "method": "fdr_bh",
                    "alpha": 0.05,
                    "family_size": 1,
                },
                "confound": {
                    "require_covariates": ["age", "sex"],
                    "motion_check": False,
                },
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "combat",
                    "pattern_corr_min": 0.5,
                    "region_replication_frac_min": 0.5,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "fragile"],
        }
    )


def _candidate_response(contract: ClaimContract) -> dict:
    proposed = contract.model_copy(
        update={
            "claim_id": "candidate",
            "question": "Do cases have lower entorhinal volume than controls?",
            "estimand": contract.estimand.model_copy(
                update={"outcome": "smri_entorhinal"}
            ),
        }
    )
    return {
        "candidates": [
            {
                "proposal_type": "exploratory_followup_claim",
                "transform_type": "alternative_same_modality_outcome",
                "domain_core": {
                    "population_or_disease": "case-control",
                    "cohort_family": "DISC and REP",
                    "predictor_or_contrast": "case versus control",
                    "outcome_modality": "smri",
                    "outcome_family": "regional volume",
                    "direction_family": "negative",
                    "scientific_motivation": contract.question,
                },
                "preservation_check": {
                    "preserves_population": True,
                    "preserves_cohort_family": True,
                    "preserves_predictor_or_contrast": True,
                    "preserves_outcome_modality": True,
                    "preserves_direction_family": True,
                    "preserves_scientific_motivation": True,
                    "changed_fields": ["estimand.outcome"],
                    "allowed_change_rationale": "A connected regional biomarker.",
                },
                "proposed_question": proposed.question,
                "proposed_contract": proposed.model_dump(mode="json"),
                "rationale": "Test a connected medial temporal region.",
                "connection_rationale": "Both outcomes are regional sMRI measures.",
                "evidence_policy": {
                    "provenance": "post_hoc_followup",
                    "requires_new_evidence": False,
                    "can_confirm_on_current_data": True,
                    "validation_split": "current_data_adaptive",
                },
                "supported_by_evidence": [],
                "disposition_label": None,
                "responds_to_candidate_ids": [],
            }
        ]
    }


class _StubLLM:
    model = "stub:self-refine"

    def __init__(self, contract: ClaimContract, *, malformed_feedback: bool = False):
        self.contract = contract
        self.malformed_feedback = malformed_feedback
        self.calls: list[tuple[str, str, str]] = []
        self.last_call_metadata = {"usage": {"total_tokens": 10}}

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls.append((response_model.__name__, system, user))
        if response_model is SelfRefineFeedback:
            if self.malformed_feedback:
                self.malformed_feedback = False
                return "{}"
            return json.dumps(
                {
                    "overall_assessment": "The outcome may be too broad.",
                    "candidate_feedback": [
                        {
                            "candidate_id": "parent",
                            "weaknesses": ["The regional hypothesis could be refined."],
                            "revision_goals": ["Use a connected executable region."],
                        }
                    ],
                    "revision_priorities": ["Preserve the contrast and direction."],
                }
            )
        return json.dumps(_candidate_response(self.contract))


def _verdict() -> dict:
    return {
        "label": "fragile",
        "abstained": True,
        "rationale": "not confirmed",
        "gates": {
            "search_provenance": True,
            "confound": True,
            "confound_completeness": True,
            "multiplicity": False,
            "power": True,
            "multiverse": True,
            "replication": True,
        },
    }


def test_self_refine_uses_binary_failure_without_localized_evidence() -> None:
    contract = _contract()
    localization = localize_failure(contract, _verdict(), None)
    llm = _StubLLM(contract)
    generator = SelfRefineCandidateGenerator(llm)
    config = ClaimSearchConfig(
        max_rounds=3,
        max_candidates_per_round=5,
        feedback_mode="generic_retry",
    )
    candidates = generator(contract, localization, config, 1, contract.claim_id)
    assert len(candidates) == 1
    assert [call[0] for call in llm.calls] == [
        "SelfRefineFeedback",
        "LLMCandidateGenerationResponse",
    ]
    feedback_payload = json.loads(llm.calls[0][2])
    serialized = json.dumps(feedback_payload)
    assert "failure_localization" not in serialized
    assert "failed_gates" not in serialized
    assert "gate_results" not in serialized
    assert feedback_payload["binary_status"] == "not_source_supported"
    assert "output_schema" not in feedback_payload
    refinement_payload = json.loads(llm.calls[1][2])
    assert "output_schema" not in refinement_payload
    validation = validate_candidate_claim(
        contract,
        candidates[0],
        localization,
        config,
    )
    assert validation.ok, validation.violations


def test_self_refine_retries_schema_and_caches_round_feedback() -> None:
    contract = _contract()
    localization = localize_failure(contract, _verdict(), None)
    llm = _StubLLM(contract, malformed_feedback=True)
    generator = SelfRefineCandidateGenerator(llm)
    config = ClaimSearchConfig(
        max_rounds=3,
        max_candidates_per_round=5,
        llm_schema_retries=2,
        feedback_mode="generic_retry",
    )
    generator(contract, localization, config, 1, contract.claim_id)
    generator.validation_retry_index = 1
    generator(contract, localization, config, 1, contract.claim_id)
    call_types = [call[0] for call in llm.calls]
    assert call_types.count("SelfRefineFeedback") == 2
    assert call_types.count("LLMCandidateGenerationResponse") == 2
    assert [row["call_type"] for row in generator.response_records].count(
        "feedback"
    ) == 2


def test_self_refine_runs_through_unchanged_search_evaluator() -> None:
    contract = _contract()
    verdict = _verdict()
    localization = localize_failure(contract, verdict, None)
    llm = _StubLLM(contract)
    generator = SelfRefineCandidateGenerator(llm)
    config = ClaimSearchConfig(
        max_rounds=1,
        max_candidates_per_round=5,
        feedback_mode="generic_retry",
    )
    state = run_claim_search(
        contract,
        verdict,
        config=config,
        candidate_generator=generator,
        evaluator=lambda candidate: {
            "final_label": "fragile",
            "gate_results": {
                "contract": candidate.proposed_contract.model_dump(mode="json"),
                "verdict": verdict,
            },
        },
    )
    assert state.generated_candidate_count == 1
    assert state.current_data_evaluated_count == 1
    assert state.internally_supported_candidate_ids == []
    assert [row["call_type"] for row in state.llm_candidate_responses] == [
        "feedback",
        "refinement",
    ]
    assert localization.failed_gates
