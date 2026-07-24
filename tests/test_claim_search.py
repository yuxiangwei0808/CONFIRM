from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.claim_search import (
    CandidateClaimProposal,
    CandidateDomainCore,
    CandidateEvidencePolicy,
    CandidateFailureSummary,
    CandidatePreservationCheck,
    ClaimSearchConfig,
    ClaimSearchState,
    LLMClaimCandidateGenerator,
    LLMCandidateGenerationResponse,
    RoundFailureContext,
    build_claim_search_artifacts,
    build_candidate_generation_prompt,
    localize_failure,
    parse_candidate_generation_response,
    run_claim_search,
    summarize_claim_search,
    validate_candidate_claim,
)
from confirm.contract import ClaimContract


def _contract(**overrides):
    data = {
        "claim_id": "claim",
        "question": "Do Dementia participants differ from CN in smri_hippocampus?",
        "estimand": {
            "type": "group_diff",
            "outcome": "smri_hippocampus",
            "predictor": "dx",
            "group": {"var": "dx", "case": "Dementia", "control": "CN"},
            "direction": "negative",
            "unit": "scalar",
            "region_set": None,
        },
        "covariates": ["age", "sex"],
        "inclusion": None,
        "discovery_cohort": "ADNI",
        "replication_cohorts": ["OASIS3"],
        "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
        "gates": {
            "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
            "confound": {"require_covariates": ["age", "sex"], "motion_check": False},
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
        "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
    }
    _deep_update(data, overrides)
    return ClaimContract.model_validate(data)


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _verdict(label="fragile", failed=None):
    failed = failed or ["multiplicity"]
    gates = {
        "search_provenance": True,
        "confound": True,
        "confound_completeness": True,
        "multiplicity": True,
        "power": True,
        "multiverse": True,
        "replication": True,
    }
    for name in failed:
        gates[name] = False
    return {"label": label, "abstained": label != "confirmed", "rationale": "Failed gates: " + ", ".join(failed), "gates": gates}


def _results(contract):
    return {
        "contract": contract.model_dump(mode="json"),
        "primary": {"p": 0.2, "beta": -0.1, "n": 100},
        "multiverse": {"fraction_consistent": 0.25, "passed": False, "specs": []},
        "power": {"achieved_power": 0.9, "under_powered": False},
        "replication": {"passed": False, "reason": "non_replicated_effect_absent", "cohort_results": []},
    }


def _candidate(contract, **overrides):
    default_contract = contract.model_copy(
        update={
            "claim_id": f"{contract.claim_id}_followup",
            "question": "Does the connected contrast extend to smri_entorhinal?",
            "estimand": contract.estimand.model_copy(update={"outcome": "smri_entorhinal"}),
        }
    )
    domain_core = CandidateDomainCore(
        population_or_disease="Dementia vs CN",
        cohort_family="ADNI;OASIS3",
        predictor_or_contrast="Dementia vs CN",
        outcome_modality="smri",
        outcome_family="smri_hippocampus",
        direction_family="negative",
        scientific_motivation=contract.question,
    )
    preservation_check = CandidatePreservationCheck(
        preserves_population=True,
        preserves_cohort_family=True,
        preserves_predictor_or_contrast=True,
        preserves_outcome_modality=True,
        preserves_direction_family=True,
        preserves_scientific_motivation=True,
        changed_fields=["outcome_family"],
        allowed_change_rationale="Preserves Dementia vs CN and smri modality.",
    )
    evidence_policy = CandidateEvidencePolicy(
        provenance=overrides.get("provenance", "post_hoc_followup"),
        requires_new_evidence=overrides.get("requires_new_evidence", False),
        can_confirm_on_current_data=overrides.get("can_confirm_on_current_data", True),
        validation_split=overrides.get("validation_split", "current_data_adaptive"),
    )
    data = {
        "candidate_id": "candidate",
        "parent_claim_id": contract.claim_id,
        "round_index": 1,
        "transform_type": "narrower_outcome_family",
        "domain_core": domain_core,
        "preservation_check": preservation_check,
        "evidence_policy": evidence_policy,
        "connection_rationale": "Preserves Dementia vs CN and smri modality.",
        "validation_split": "current_data_adaptive",
        "source_claim_id": contract.claim_id,
        "proposal_type": "exploratory_followup_claim",
        "rationale": "Connected follow-up.",
        "proposed_question": "In excluded validation evidence, test a narrower smri Dementia vs CN outcome.",
        "proposed_contract": default_contract,
        "disposition_label": None,
        "provenance": "post_hoc_followup",
        "requires_new_evidence": False,
        "can_confirm_on_current_data": True,
        "supported_by_evidence": [],
    }
    data.update(overrides)
    return CandidateClaimProposal.model_validate(data)


def _stub_candidate_generator(contract, localization, config, round_index, parent_claim_id):
    outcomes = [
        "smri_entorhinal",
        "smri_fusiform",
        "smri_middletemporal",
        "smri_amygdala",
        "smri_thalamus",
    ]
    candidates = []
    for index, outcome in enumerate(outcomes[: config.max_candidates_per_round], start=1):
        proposed_contract = contract.model_copy(
            update={
                "claim_id": f"{contract.claim_id}_r{round_index}_c{index}",
                "question": f"Does the connected contrast extend to {outcome}?",
                "estimand": contract.estimand.model_copy(update={"outcome": outcome}),
            }
        )
        candidates.append(
            _candidate(
                contract,
                candidate_id=f"stub_r{round_index}_c{index}",
                parent_claim_id=parent_claim_id,
                round_index=round_index,
                proposed_question=proposed_contract.question,
                proposed_contract=proposed_contract,
                supported_by_evidence=list(localization.evidence),
                responds_to_candidate_ids=(
                    [f"stub_r{round_index - 1}_c1"] if round_index > 1 else []
                ),
            )
        )
    return candidates


class _FakeCandidateLLM:
    model = "fake-claim-generator"

    def complete(self, system, user):
        payload = json.loads(user)
        evidence = payload["failure_localization"]["evidence"][:1]
        proposed_contract = json.loads(json.dumps(payload["original_contract"]))
        proposed_contract["estimand"]["outcome"] = "smri_entorhinal"
        subgroup_contract = json.loads(json.dumps(payload["original_contract"]))
        subgroup_contract["inclusion"] = 'sex == "F"'
        domain_core = {
            "population_or_disease": "Dementia vs CN",
            "cohort_family": "ADNI;OASIS3",
            "predictor_or_contrast": "Dementia vs CN",
            "outcome_modality": "smri",
            "outcome_family": "smri_hippocampus",
            "direction_family": "negative",
            "scientific_motivation": "Do Dementia participants differ from CN in smri_hippocampus?",
        }
        preservation_check = {
            "preserves_population": True,
            "preserves_cohort_family": True,
            "preserves_predictor_or_contrast": True,
            "preserves_outcome_modality": True,
            "preserves_direction_family": True,
            "preserves_scientific_motivation": True,
            "changed_fields": ["outcome_family"],
            "allowed_change_rationale": "Preserves Dementia vs CN and smri modality.",
        }
        return json.dumps(
            {
                "candidates": [
                    {
                        "proposal_type": "exploratory_followup_claim",
                        "transform_type": "narrower_outcome_family",
                        "domain_core": domain_core,
                        "preservation_check": preservation_check,
                        "proposed_question": "In excluded validation evidence, test a narrower smri Dementia vs CN outcome.",
                        "proposed_contract": proposed_contract,
                        "rationale": "This is a connected same-modality follow-up requiring new evidence.",
                        "connection_rationale": "Preserves Dementia vs CN and smri modality.",
                        "evidence_policy": {
                            "provenance": "post_hoc_followup",
                            "requires_new_evidence": False,
                            "can_confirm_on_current_data": True,
                            "validation_split": "current_data_adaptive",
                        },
                        "supported_by_evidence": evidence,
                        "disposition_label": None,
                    },
                    {
                        "proposal_type": "exploratory_followup_claim",
                        "transform_type": "moderator_or_subgroup",
                        "domain_core": domain_core,
                        "preservation_check": {
                            **preservation_check,
                            "changed_fields": ["inclusion"],
                            "allowed_change_rationale": "Preserves the same claim in a feasible source-data subgroup.",
                        },
                        "proposed_question": "Replicate the original Dementia vs CN smri claim in an independent cohort.",
                        "proposed_contract": subgroup_contract,
                        "rationale": "This is a connected source-data subgroup follow-up.",
                        "connection_rationale": "Preserves the Dementia vs CN contrast, smri modality, and gates.",
                        "evidence_policy": {
                            "provenance": "post_hoc_followup",
                            "requires_new_evidence": False,
                            "can_confirm_on_current_data": True,
                            "validation_split": "current_data_adaptive",
                        },
                        "supported_by_evidence": evidence,
                        "disposition_label": None,
                    },
                ]
            }
        )


class _RetryCandidateLLM:
    model = "retry-claim-generator"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"candidates": [{"proposal_type": "exploratory_followup_claim"}]})
        return _FakeCandidateLLM().complete(system, user)


class _ShapeRetryLLM:
    model = "shape-retry-generator"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        payload = json.loads(_FakeCandidateLLM().complete(system, user))
        if self.calls == 1:
            payload["candidates"][0]["transform_type"] = "stronger_design"
            payload["candidates"][0]["proposal_type"] = "exploratory_followup_claim"
        return json.dumps(payload)


class _PreflightRetryLLM:
    model = "preflight-retry-generator"

    def __init__(self):
        self.calls = 0
        self.saw_validation_feedback = False

    def complete(self, system, user):
        self.calls += 1
        payload = json.loads(user)
        self.saw_validation_feedback = self.saw_validation_feedback or "candidate_validation_retry" in payload
        proposed_contract = payload["original_contract"]
        if self.calls == 1:
            proposed_contract = json.loads(json.dumps(proposed_contract))
            proposed_contract["estimand"]["outcome"] = "smri_missing"
        else:
            proposed_contract = json.loads(json.dumps(proposed_contract))
            proposed_contract["estimand"]["outcome"] = "smri_entorhinal"
        evidence = payload["failure_localization"]["evidence"][:1]
        domain_core = {
            "population_or_disease": "Dementia vs CN",
            "cohort_family": "ADNI;OASIS3",
            "predictor_or_contrast": "Dementia vs CN",
            "outcome_modality": "smri",
            "outcome_family": proposed_contract["estimand"]["outcome"],
            "direction_family": "negative",
            "scientific_motivation": "Do Dementia participants differ from CN in smri_hippocampus?",
        }
        preservation_check = {
            "preserves_population": True,
            "preserves_cohort_family": True,
            "preserves_predictor_or_contrast": True,
            "preserves_outcome_modality": True,
            "preserves_direction_family": True,
            "preserves_scientific_motivation": True,
            "changed_fields": ["outcome_family"],
            "allowed_change_rationale": "Preserves Dementia vs CN and smri modality.",
        }
        return json.dumps(
            {
                "candidates": [
                    {
                        "proposal_type": "exploratory_followup_claim",
                        "transform_type": "narrower_outcome_family",
                        "domain_core": domain_core,
                        "preservation_check": preservation_check,
                        "proposed_question": "Test a connected smri Dementia vs CN follow-up.",
                        "proposed_contract": proposed_contract,
                        "rationale": "This is a connected same-modality follow-up.",
                        "connection_rationale": "Preserves Dementia vs CN and smri modality.",
                        "evidence_policy": {
                            "provenance": "post_hoc_followup",
                            "requires_new_evidence": False,
                            "can_confirm_on_current_data": True,
                            "validation_split": "current_data_adaptive",
                        },
                        "supported_by_evidence": evidence,
                        "disposition_label": None,
                    }
                ]
            }
        )


class _ValidationRetryLLM:
    model = "validation-retry-generator"

    def __init__(self):
        self.calls = 0
        self.saw_validation_feedback = False

    def complete(self, system, user):
        self.calls += 1
        payload = json.loads(user)
        self.saw_validation_feedback = self.saw_validation_feedback or "candidate_validation_retry" in payload
        proposed_contract = json.loads(json.dumps(payload["original_contract"]))
        proposed_contract["estimand"]["outcome"] = "smri_entorhinal"
        if self.calls == 1:
            proposed_contract["estimand"]["group"] = {"var": "dx", "case": "MCI", "control": "CN"}
        evidence = payload["failure_localization"]["evidence"][:1]
        domain_core = {
            "population_or_disease": "Dementia vs CN",
            "cohort_family": "ADNI;OASIS3",
            "predictor_or_contrast": "Dementia vs CN",
            "outcome_modality": "smri",
            "outcome_family": proposed_contract["estimand"]["outcome"],
            "direction_family": "negative",
            "scientific_motivation": "Do Dementia participants differ from CN in smri_hippocampus?",
        }
        preservation_check = {
            "preserves_population": True,
            "preserves_cohort_family": True,
            "preserves_predictor_or_contrast": True,
            "preserves_outcome_modality": True,
            "preserves_direction_family": True,
            "preserves_scientific_motivation": True,
            "changed_fields": ["outcome_family"],
            "allowed_change_rationale": "Preserves Dementia vs CN and smri modality.",
        }
        return json.dumps(
            {
                "candidates": [
                    {
                        "proposal_type": "exploratory_followup_claim",
                        "transform_type": "narrower_outcome_family",
                        "domain_core": domain_core,
                        "preservation_check": preservation_check,
                        "proposed_question": "Test a connected smri Dementia vs CN follow-up.",
                        "proposed_contract": proposed_contract,
                        "rationale": "This is a connected same-modality follow-up.",
                        "connection_rationale": "Preserves Dementia vs CN and smri modality.",
                        "evidence_policy": {
                            "provenance": "post_hoc_followup",
                            "requires_new_evidence": False,
                            "can_confirm_on_current_data": True,
                            "validation_split": "current_data_adaptive",
                        },
                        "supported_by_evidence": evidence,
                        "disposition_label": None,
                    }
                ]
            }
        )


class _PartialValidationLLM:
    model = "partial-validation-generator"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        payload = json.loads(_FakeCandidateLLM().complete(system, user))
        payload["candidates"][1]["proposed_contract"]["estimand"]["group"] = {
            "var": "dx",
            "case": "MCI",
            "control": "CN",
        }
        return json.dumps(payload)


def _preflight_context(tmp_path):
    root = tmp_path / "cohorts"
    root.mkdir()
    base = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(30)],
            "cohort": ["ADNI"] * 30,
            "site": ["site1"] * 30,
            "age": [65 + idx % 5 for idx in range(30)],
            "sex": ["F", "M"] * 15,
            "dx": ["Dementia"] * 15 + ["CN"] * 15,
            "smri_hippocampus": [1.0 + idx for idx in range(30)],
            "smri_entorhinal": [2.0 + idx for idx in range(30)],
        }
    )
    base.to_parquet(root / "ADNI.parquet")
    base.assign(cohort="OASIS3").to_parquet(root / "OASIS3.parquet")
    return CandidatePreflightContext.from_roots([root])


def test_config_enforces_budget_and_stub_generator_respects_candidate_limit():
    with pytest.raises(ValueError):
        ClaimSearchConfig(max_rounds=0)
    with pytest.raises(ValueError):
        ClaimSearchConfig(brainwide_min_features=2)
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    config = ClaimSearchConfig(max_rounds=2, max_candidates_per_round=2)

    candidates = _stub_candidate_generator(contract, loc, config, 1, contract.claim_id)

    assert len(candidates) == 2


def test_v6_search_state_has_no_one_shot_or_excluded_evidence_fields():
    properties = ClaimSearchState.model_json_schema()["properties"]
    assert {
        "internally_supported_candidate_ids",
        "round_failure_contexts",
        "round_summaries",
        "final_search_family_size",
    }.issubset(properties)
    assert {
        "selected_candidate_id",
        "selection_reason",
        "confirmed_candidates",
        "excluded_evidence_query_count",
        "excluded_evidence_status",
    }.isdisjoint(properties)


def test_loop_stops_without_forcing_success_when_no_evaluator_exists():
    contract = _contract()
    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=3, max_candidates_per_round=3),
        candidate_generator=_stub_candidate_generator,
    )

    assert state.internally_supported_candidate_ids == []
    assert state.stopped_reason == "no_evaluator"
    assert state.candidate_history


def test_default_claim_search_requires_llm_or_explicit_generator():
    contract = _contract()
    state = run_claim_search(contract, _verdict(), _results(contract), config=ClaimSearchConfig(max_rounds=1))

    assert state.stopped_reason == "llm_unavailable"
    assert state.candidate_history == []


def test_llm_prompt_and_parser_generate_typed_candidates():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    config = ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2)
    executable_catalog = {"allowed_cohorts": ["ADNI", "OASIS3"]}
    system, user = build_candidate_generation_prompt(
        contract,
        loc,
        config,
        1,
        contract.claim_id,
        executable_catalog=executable_catalog,
    )

    assert "legitimate chance of passing" in system
    assert "do not manipulate the contract" in system
    assert "forbidden_actions" in user
    assert "executable_data_catalog" in user
    assert "output_schema" in user
    assert "immutable_contract_fields" in user
    assert "domain_core" in user
    assert "preservation_check" in user
    assert "evidence_policy" in user
    prompt_payload = json.loads(user)
    assert prompt_payload["max_candidates"] == 2
    assert "external_evidence_sets" not in prompt_payload
    assert "holdout_evaluation_pair" not in prompt_payload
    assert "Generate up to 2 scientifically distinct candidates" in prompt_payload["generation_policy"][0]
    assert "multivariate_pattern" in prompt_payload["allowed_transform_types"]
    assert any("brainwide" in item for item in prompt_payload["generation_policy"])

    generator = LLMClaimCandidateGenerator(_FakeCandidateLLM())
    candidates = generator(contract, loc, config, 1, contract.claim_id)

    assert len(candidates) == 2
    assert candidates[0].candidate_id.startswith("claim_r1_c1_")
    assert generator.prompt_records
    assert generator.response_records[0]["candidate_count"] == 2
    assert candidates[0].domain_core.outcome_modality == "smri"
    assert candidates[0].evidence_policy.validation_split == candidates[0].validation_split


def test_llm_prompt_honors_candidate_limits_above_five():
    contract = _contract()
    localization = localize_failure(contract, _verdict(), _results(contract))
    _, user = build_candidate_generation_prompt(
        contract,
        localization,
        ClaimSearchConfig(max_rounds=1, max_candidates_per_round=10),
        1,
        contract.claim_id,
    )

    payload = json.loads(user)
    assert payload["max_candidates"] == 10
    assert "Generate up to 10 scientifically distinct candidates" in payload["generation_policy"][0]
    assert "Generate 2-5 candidates" not in user


def test_generic_retry_hides_structured_failure_diagnosis_but_keeps_failed_ids():
    contract = _contract()
    localization = localize_failure(contract, _verdict(), _results(contract))
    context = RoundFailureContext(
        round_index=1,
        failed_candidates=[
            CandidateFailureSummary(
                candidate_id="candidate_r1_c1",
                failed_gates=["replication"],
                localized_cause=localization,
                effective_contract=contract,
                execution_status="gate_failed",
            )
        ],
    )
    _, structured_user = build_candidate_generation_prompt(
        contract,
        localization,
        ClaimSearchConfig(feedback_mode="structured_diagnosis"),
        2,
        contract.claim_id,
        round_failure_context=context,
    )
    _, generic_user = build_candidate_generation_prompt(
        contract,
        localization,
        ClaimSearchConfig(feedback_mode="generic_retry"),
        2,
        contract.claim_id,
        round_failure_context=context,
    )

    structured_payload = json.loads(structured_user)
    generic_payload = json.loads(generic_user)
    structured = structured_payload["round_failure_context"]
    generic = generic_payload["round_failure_context"]
    assert structured["failed_candidates"][0]["failed_gates"] == ["replication"]
    assert generic["failed_candidate_ids"] == ["candidate_r1_c1"]
    assert "failed_candidates" not in generic
    assert structured_payload["failure_localization"]["failed_gates"]
    assert generic_payload["failure_localization"]["failed_gates"] == []
    assert generic_payload["failure_localization"]["evidence"] == []


def test_llm_candidate_schema_requires_executable_proposals_but_does_not_enforce_transform_labels():
    schema = LLMCandidateGenerationResponse.model_json_schema()
    candidate_items = schema["properties"]["candidates"]["items"]
    assert candidate_items["$ref"].endswith("/LLMCandidateProposal")
    properties = schema["$defs"]["LLMCandidateProposal"]["properties"]
    assert set(properties["proposal_type"]["enum"]) == {
        "corrected_contract",
        "exploratory_followup_claim",
    }
    assert set(properties["transform_type"]["enum"]) == {
        "narrower_outcome_family",
        "alternative_same_modality_outcome",
        "multivariate_pattern",
        "moderator_or_subgroup",
        "stronger_design",
        "fixed_estimand",
        "contract_correction",
    }
    policy_properties = schema["$defs"]["CandidateEvidencePolicy"]["properties"]
    assert set(policy_properties["validation_split"]["enum"]) == {
        "current_data_adaptive",
        "current_data_contract_repair",
    }


def test_llm_generator_retries_after_schema_failure():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    config = ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2, llm_schema_retries=1)
    llm = _RetryCandidateLLM()
    generator = LLMClaimCandidateGenerator(llm)

    candidates = generator(contract, loc, config, 1, contract.claim_id)

    assert len(candidates) == 2
    assert llm.calls == 2
    assert len(generator.prompt_records) == 2
    assert generator.response_records[0]["parse_error"]
    assert generator.response_records[1]["parse_error"] is None
    assert generator.response_records[0]["retry_kind"] == "none"
    assert generator.response_records[1]["retry_kind"] == "schema"
    assert generator.response_records[1]["is_retry"] is True
    assert "schema_validation_error" in generator.prompt_records[1]["user"]


def test_llm_generator_does_not_retry_a_schema_valid_transform_mismatch():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    config = ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2, llm_schema_retries=1)
    llm = _ShapeRetryLLM()
    generator = LLMClaimCandidateGenerator(llm)

    candidates = generator(contract, loc, config, 1, contract.claim_id)

    assert len(candidates) == 2
    assert llm.calls == 1
    assert generator.response_records[0]["parse_error"] is None


def test_parser_rejects_malformed_llm_output():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))

    with pytest.raises(ValueError):
        parse_candidate_generation_response("{}", contract, loc, ClaimSearchConfig(), 1, contract.claim_id)


def test_parser_accepts_transform_intent_for_later_deterministic_audit():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    raw = json.loads(
        _FakeCandidateLLM().complete(
            "",
            json.dumps({"failure_localization": {"evidence": ["replication failed"]}, "original_contract": contract.model_dump(mode="json")}),
        )
    )
    raw["candidates"][0]["transform_type"] = "stronger_design"
    raw["candidates"][0]["proposal_type"] = "exploratory_followup_claim"

    candidates = parse_candidate_generation_response(
        json.dumps(raw), contract, loc, ClaimSearchConfig(), 1, contract.claim_id
    )

    assert candidates[0].transform_type == "stronger_design"


def test_parser_ignores_round_one_candidate_references():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    raw = json.loads(_FakeCandidateLLM().complete("", json.dumps({
        "failure_localization": {"evidence": ["replication failed"]},
        "original_contract": contract.model_dump(mode="json"),
    })))
    raw["candidates"][0]["responds_to_candidate_ids"] = ["self_reference"]

    candidates = parse_candidate_generation_response(
        json.dumps(raw), contract, loc, ClaimSearchConfig(), 1, contract.claim_id
    )

    assert candidates[0].responds_to_candidate_ids == []


def test_posthoc_candidate_can_use_adaptive_current_data_evaluation():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(
        contract,
        validation_split="current_data_adaptive",
        can_confirm_on_current_data=True,
        requires_new_evidence=False,
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert result.ok


def test_contract_correction_may_use_current_data_when_estimand_preserved():
    contract = _contract()
    loc = localize_failure(contract, _verdict(failed=["confound"]), _results(contract))
    repaired = _contract(covariates=["age", "sex", "site"], gates={"confound": {"require_covariates": ["age", "sex", "site"]}})
    candidate = _candidate(
        contract,
        proposal_type="corrected_contract",
        transform_type="contract_correction",
        provenance="contract_correction",
        validation_split="current_data_contract_repair",
        proposed_contract=repaired,
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
        proposed_question=repaired.question,
        rationale="Add measured site as a confound required by the failed confound audit.",
        connection_rationale="Add measured site while preserving the original estimand and cohorts.",
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert result.ok


def test_added_confound_must_update_covariates_and_required_gate_together():
    contract = _contract()
    loc = localize_failure(contract, _verdict(failed=["confound"]), _results(contract))
    repaired = _contract(covariates=["age", "sex", "site"])
    candidate = _candidate(
        contract,
        proposal_type="corrected_contract",
        transform_type="contract_correction",
        provenance="contract_correction",
        validation_split="current_data_contract_repair",
        proposed_contract=repaired,
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
        rationale="Add measured site as a confound.",
        connection_rationale="Preserve the original estimand and cohorts.",
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert not result.ok
    assert any("both covariates and required" in item for item in result.violations)


def test_candidate_cannot_disable_parent_motion_check():
    contract = _contract(gates={"confound": {"motion_check": True}})
    loc = localize_failure(contract, _verdict(failed=["confound"]), _results(contract))
    repaired = _contract(gates={"confound": {"motion_check": False}})
    candidate = _candidate(
        contract,
        proposed_contract=repaired,
        rationale="Keep the same analysis.",
        connection_rationale="Preserve the original estimand and cohorts.",
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert not result.ok
    assert any("motion confound check" in item for item in result.violations)


def test_noop_contract_correction_is_rejected():
    contract = _contract()
    loc = localize_failure(contract, _verdict(failed=["confound"]), _results(contract))
    candidate = _candidate(
        contract,
        proposal_type="corrected_contract",
        transform_type="contract_correction",
        provenance="contract_correction",
        validation_split="current_data_contract_repair",
        proposed_contract=contract,
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
        proposed_question=contract.question,
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert not result.ok
    assert any("does not change" in item for item in result.violations)


@pytest.mark.parametrize(
    ("parent_outcome", "candidate_outcome"),
    [
        ("smri_hippocampus", "smri_hippocampus_z"),
        ("smri_hippocampus_z", "smri_hippocampus_z_z"),
        ("smri_hippocampus_z_z", "smri_hippocampus"),
    ],
)
def test_z_suffixed_outcome_columns_are_treated_as_distinct_executable_fields(
    parent_outcome,
    candidate_outcome,
):
    contract = _contract(estimand={"outcome": parent_outcome})
    loc = localize_failure(contract, _verdict(), _results(contract))
    revised = _contract(estimand={"outcome": candidate_outcome})
    candidate = _candidate(contract, proposed_contract=revised)

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert result.ok, result.violations


def test_two_outcome_brainwide_pattern_is_retained_but_not_executable():
    contract = _contract(estimand={"outcome": "fc_edge_z", "unit": "scalar"})
    localization = localize_failure(contract, _verdict(), _results(contract))
    revised = _contract(
        estimand={
            "outcome": ["fc_edge_z", "fc_edge"],
            "unit": "brainwide",
            "region_set": "raw_and_standardized_pair",
        }
    )
    candidate = _candidate(contract, proposed_contract=revised)

    result = validate_candidate_claim(
        contract,
        candidate,
        localization,
        ClaimSearchConfig(),
    )

    assert not result.ok
    assert any("at least 3 distinct outcome" in item for item in result.violations)


def test_connected_scalar_to_brainwide_pattern_is_allowed_with_policy_floor_and_feature_burden():
    contract = _contract(
        gates={"replication": {"pattern_corr_min": 0.0}},
    )
    pattern_contract = _contract(
        estimand={
            "outcome": [
                "smri_hippocampus",
                "smri_entorhinal",
                "smri_amygdala",
            ],
            "unit": "brainwide",
            "region_set": "medial_temporal_pattern",
        },
        gates={"replication": {"pattern_corr_min": 0.0}},
    )
    evaluated = []

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [
            _candidate(
                contract,
                transform_type="multivariate_pattern",
                proposed_contract=pattern_contract,
            )
        ]

    def evaluator(candidate):
        evaluated.append(candidate)
        return {"final_label": "fragile", "gate_results": {}}

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1),
        candidate_generator=generator,
        evaluator=evaluator,
    )

    assert len(evaluated) == 1
    effective = evaluated[0]
    assert effective.inferred_transform == "multivariate_pattern"
    assert effective.transform_match is True
    assert effective.proposed_contract.gates.replication.pattern_corr_min == 0.5
    assert effective.policy_adjustments["adaptive_brainwide_pattern_policy"]["effective_pattern_corr_min"] == 0.5
    assert state.evaluations[0].search_hypothesis_count == 4
    assert state.unique_hypotheses_tested_count == 4
    assert state.evaluations[0].effective_family_size == 5
    assert summarize_claim_search([state])["unique_hypotheses_tested_count"] == 4


def test_brainwide_pattern_requires_shared_not_only_per_cohort_features(tmp_path):
    root = tmp_path / "cohorts"
    root.mkdir()
    base = {
        "subject_id": [f"sub-{index}" for index in range(40)],
        "site": ["site1", "site2"] * 20,
        "age": [65 + index % 5 for index in range(40)],
        "sex": ["F", "M"] * 20,
        "dx": ["Dementia", "CN"] * 20,
        "smri_hippocampus": [1.0 + index * 0.01 for index in range(40)],
    }
    pd.DataFrame(
        {
            **base,
            "smri_entorhinal": [2.0 + index * 0.01 for index in range(40)],
            "smri_amygdala": [3.0 + index * 0.01 for index in range(40)],
        }
    ).to_parquet(root / "ADNI.parquet")
    pd.DataFrame(
        {
            **base,
            "smri_precuneus": [4.0 + index * 0.01 for index in range(40)],
            "smri_thalamus": [5.0 + index * 0.01 for index in range(40)],
        }
    ).to_parquet(root / "OASIS3.parquet")
    context = CandidatePreflightContext.from_roots([root])
    contract = _contract()
    pattern_contract = _contract(
        estimand={
            "outcome": "smri_",
            "unit": "brainwide",
            "region_set": "connected_smri_pattern",
        }
    )
    candidate = _candidate(
        contract,
        transform_type="multivariate_pattern",
        proposed_contract=pattern_contract,
    )

    validation = validate_candidate_claim(
        contract,
        candidate,
        localize_failure(contract, _verdict(), _results(contract)),
        ClaimSearchConfig(),
        preflight_context=context,
    )

    assert not validation.ok
    assert any("shared across every source cohort" in item for item in validation.violations)
    assert validation.design_diagnostics["brainwide_pattern"]["shared_outcome_count"] == 1


def test_unrelated_outcome_and_direction_changes_are_rejected():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    unrelated = _contract(estimand={"outcome": "pet_amyloid", "direction": "positive"})
    candidate = _candidate(contract, proposed_contract=unrelated)

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert not result.ok
    assert any("modality" in item or "direction" in item for item in result.violations)


def test_domain_core_mismatch_is_rejected_without_contract():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(
        contract,
        domain_core={
            "population_or_disease": "Dementia vs CN",
            "cohort_family": "ADNI;OASIS3",
            "predictor_or_contrast": "Dementia vs CN",
            "outcome_modality": "pet",
            "outcome_family": "pet_amyloid",
            "direction_family": "negative",
            "scientific_motivation": contract.question,
        },
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert not result.ok
    assert any("domain_core changes outcome modality" in item for item in result.violations)


def test_domain_core_accepts_known_modality_and_direction_aliases():
    contract = _contract(estimand={"outcome": "fc_mean_abs", "direction": "two_sided"})
    loc = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(
        contract,
        proposed_contract=contract.model_copy(
            update={"estimand": contract.estimand.model_copy(update={"outcome": "fc_network_mean_abs"})}
        ),
        domain_core={
            "population_or_disease": "Dementia vs CN",
            "cohort_family": "ADNI;OASIS3",
            "predictor_or_contrast": "Dementia vs CN",
            "outcome_modality": "resting-state fMRI functional connectivity",
            "outcome_family": "fc_mean_abs",
            "direction_family": "two-sided difference",
            "scientific_motivation": contract.question,
        },
        proposed_question="In future evidence, test a connected functional connectivity Dementia vs CN outcome.",
        connection_rationale="Preserves Dementia vs CN and resting-state fMRI functional connectivity.",
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert result.ok


def test_structured_connection_does_not_require_literal_prose_tokens():
    contract = _contract()
    localization = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(
        contract,
        proposed_question="Evaluate the preserved design on future evidence.",
        rationale="The executable contract is unchanged.",
        connection_rationale="All required structured fields are preserved.",
    )

    result = validate_candidate_claim(
        contract,
        candidate,
        localization,
        ClaimSearchConfig(),
    )

    assert result.ok


def test_followup_contract_cannot_weaken_gates():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    weakened = _contract(
        estimand={"outcome": "smri_entorhinal"},
        gates={
            "multiplicity": {"alpha": 0.1, "family_size": 1},
            "power": {"min_power": 0.5},
            "multiverse": {"min_fraction_consistent": 0.1},
        },
    )
    candidate = _candidate(contract, proposed_contract=weakened)

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert not result.ok
    assert any("changes multiplicity alpha" in item for item in result.violations)


def test_narrower_same_modality_contract_is_connected_for_adaptive_source_evaluation():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    narrower = _contract(estimand={"outcome": "smri_entorhinal"})
    candidate = _candidate(contract, proposed_contract=narrower)

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig())

    assert result.ok
    assert not candidate.requires_new_evidence
    assert candidate.validation_split == "current_data_adaptive"


def test_data_preflight_rejects_schema_valid_nonexecutable_contract(tmp_path):
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    invalid = _contract(estimand={"outcome": "smri_missing"})
    candidate = _candidate(contract, proposed_contract=invalid)
    context = _preflight_context(tmp_path)

    result = validate_candidate_claim(
        contract,
        candidate,
        loc,
        ClaimSearchConfig(),
        preflight_context=context,
    )

    assert not result.ok
    assert any("Preflight:" in item and "missing outcome" in item for item in result.violations)


def test_data_preflight_numeric_age_inclusion_accepts_string_age(tmp_path):
    root = tmp_path / "cohorts"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(40)],
            "cohort": ["ADNI"] * 40,
            "site": ["site1"] * 40,
            "age": [str(65 + idx % 5) for idx in range(40)],
            "sex": ["F", "F", "M", "M"] * 10,
            "dx": ["Dementia", "CN"] * 20,
            "smri_hippocampus": [1.0 + idx for idx in range(40)],
        }
    )
    frame.to_parquet(root / "ADNI.parquet")
    frame.assign(cohort="OASIS3").to_parquet(root / "OASIS3.parquet")
    context = CandidatePreflightContext.from_roots([root])
    contract = _contract(inclusion="age <= 67")

    result = context.validate_contract(contract, min_complete_rows=5)

    assert result.ok
    assert not any("inclusion query" in item for item in result.violations)


def test_data_preflight_uses_complete_cases_for_partial_age_and_sex_missingness(tmp_path):
    root = tmp_path / "cohorts"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(40)],
            "cohort": ["ADNI"] * 40,
            "site": ["site1", "site2"] * 20,
            "age": ["unknown", *[str(65 + idx % 5) for idx in range(39)]],
            "sex": ["unknown", *(["F", "F", "M", "M"] * 9), "F", "M", "F"],
            "dx": ["Dementia", "CN"] * 20,
            "smri_hippocampus": [1.0 + idx * 0.1 + (idx % 3) * 0.01 for idx in range(40)],
        }
    )
    frame.to_parquet(root / "ADNI.parquet")
    frame.assign(cohort="OASIS3").to_parquet(root / "OASIS3.parquet")
    context = CandidatePreflightContext.from_roots([root])

    result = context.validate_contract(_contract(), min_complete_rows=20)

    assert result.ok
    assert any("complete-case analysis" in item and "age" in item for item in result.warnings)
    assert any("complete-case analysis" in item and "sex" in item for item in result.warnings)


def test_subgroup_candidates_are_limited_to_parent_feasible_inclusions(tmp_path):
    root = tmp_path / "cohorts"
    root.mkdir()
    n = 80
    frame = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(n)],
            "cohort": ["ADNI"] * n,
            "site": [f"site{idx % 3}" for idx in range(n)],
            "age": [60 + idx % 20 for idx in range(n)],
            "sex": ["F" if idx % 4 < 2 else "M" for idx in range(n)],
            "dx": ["Dementia" if idx % 2 else "CN" for idx in range(n)],
            "smri_hippocampus": [1.0 + idx * 0.01 + (idx % 7) * 0.001 for idx in range(n)],
            "smri_entorhinal": [2.0 + idx * 0.01 + (idx % 5) * 0.001 for idx in range(n)],
        }
    )
    frame.to_parquet(root / "ADNI.parquet")
    frame.assign(cohort="OASIS3").to_parquet(root / "OASIS3.parquet")
    context = CandidatePreflightContext.from_roots([root])
    contract = _contract()
    localization = localize_failure(contract, _verdict(), _results(contract))
    allowed = [
        item
        for item in context.prompt_catalog(contract)["allowed_inclusion_examples"]
        if item is not None
    ]
    assert allowed

    accepted = _candidate(
        contract,
        transform_type="moderator_or_subgroup",
        proposed_contract=contract.model_copy(update={"inclusion": allowed[0]}),
    )
    rejected = _candidate(
        contract,
        transform_type="moderator_or_subgroup",
        proposed_contract=contract.model_copy(update={"inclusion": "age >= 61.234"}),
    )
    compound_contract = contract.model_copy(
        update={
            "estimand": contract.estimand.model_copy(update={"outcome": "smri_entorhinal"}),
            "inclusion": allowed[0],
        }
    )
    compound = _candidate(
        contract,
        transform_type="moderator_or_subgroup",
        proposed_contract=compound_contract,
        rationale=f"Test the connected outcome using the feasible subgroup {allowed[0]}.",
    )

    accepted_result = validate_candidate_claim(
        contract,
        accepted,
        localization,
        ClaimSearchConfig(),
        preflight_context=context,
    )
    rejected_result = validate_candidate_claim(
        contract,
        rejected,
        localization,
        ClaimSearchConfig(),
        preflight_context=context,
    )
    compound_result = validate_candidate_claim(
        contract,
        compound,
        localization,
        ClaimSearchConfig(),
        preflight_context=context,
    )

    assert accepted_result.ok, accepted_result.violations
    assert compound_result.ok, compound_result.violations
    assert any("does not match inferred transform" in item for item in compound_result.warnings)
    assert not rejected_result.ok
    assert any("parent-data-feasible inclusion" in item for item in rejected_result.violations)


def test_candidate_rationale_may_use_group_label_numbers():
    contract = _contract(
        question="Do diagnosis group 2 participants differ from group 0?",
        estimand={
            "outcome": "smri_hippocampus",
            "predictor": "dx",
            "group": {"var": "dx", "case": "2", "control": "0"},
        },
    )
    localization = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(
        contract,
        domain_core=CandidateDomainCore(
            population_or_disease="2 vs 0",
            cohort_family="ADNI;OASIS3",
            predictor_or_contrast="2 vs 0",
            outcome_modality="smri",
            outcome_family="smri_hippocampus",
            direction_family="negative",
            scientific_motivation=contract.question,
        ),
        proposed_contract=_contract(
            question=contract.question,
            estimand={
                "outcome": "smri_entorhinal",
                "predictor": "dx",
                "group": {"var": "dx", "case": "2", "control": "0"},
            },
        ),
        rationale="Keep the diagnosis contrast dx=2 versus dx=0 for a connected structural outcome.",
    )

    result = validate_candidate_claim(
        contract,
        candidate,
        localization,
        ClaimSearchConfig(),
    )

    assert result.ok, result.violations


def test_distinct_z_suffixed_outcomes_are_not_deduplicated_by_name():
    contract = _contract()

    def generator(contract, localization, config, round_index, parent_claim_id):
        first = _candidate(
            contract,
            candidate_id="raw_alias",
            proposed_contract=_contract(estimand={"outcome": "smri_entorhinal"}),
        )
        second = _candidate(
            contract,
            candidate_id="standardized_alias",
            proposed_contract=_contract(estimand={"outcome": "smri_entorhinal_z"}),
        )
        return [first, second]

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2),
        candidate_generator=generator,
        evaluator=lambda candidate: {"final_label": "fragile", "gate_results": {}},
    )

    assert state.unique_candidate_count == 2
    assert state.duplicate_candidates == []


def test_reordered_brainwide_pattern_is_deduplicated_as_same_executable_claim():
    contract = _contract()
    outcomes = ["smri_hippocampus", "smri_entorhinal", "smri_amygdala"]

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [
            _candidate(
                contract,
                candidate_id="pattern_a",
                transform_type="multivariate_pattern",
                proposed_contract=_contract(
                    estimand={
                        "outcome": outcomes,
                        "unit": "brainwide",
                        "region_set": "medial_temporal_pattern",
                    }
                ),
            ),
            _candidate(
                contract,
                candidate_id="pattern_b",
                transform_type="multivariate_pattern",
                proposed_contract=_contract(
                    estimand={
                        "outcome": list(reversed(outcomes)),
                        "unit": "brainwide",
                        "region_set": "medial_temporal_pattern",
                    }
                ),
            ),
        ]

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2),
        candidate_generator=generator,
        evaluator=lambda candidate: {"final_label": "fragile", "gate_results": {}},
    )

    assert state.unique_candidate_count == 1
    assert len(state.duplicate_candidates) == 1
    assert state.duplicate_candidates[0].duplicate_of == "pattern_a"


def test_prompt_catalog_preserves_exact_z_suffixed_outcome_columns(tmp_path):
    root = tmp_path / "cohorts"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(40)],
            "site": ["site1", "site2"] * 20,
            "age": [60 + idx % 10 for idx in range(40)],
            "sex": ["F", "M"] * 20,
            "dx": ["Dementia", "CN"] * 20,
            "fc_edge": [idx * 0.01 for idx in range(40)],
            "fc_edge_z": [idx * 0.02 for idx in range(40)],
            "fc_edge_z_z": [idx * 0.03 for idx in range(40)],
            "fc_other": [idx * 0.04 for idx in range(40)],
            "fc_other_z": [idx * 0.05 for idx in range(40)],
        }
    )
    frame.to_parquet(root / "ADNI.parquet")
    frame.to_parquet(root / "OASIS3.parquet")
    context = CandidatePreflightContext.from_roots([root])
    contract = _contract(estimand={"outcome": "fc_edge_z"})

    catalog = context.prompt_catalog(contract)

    outcomes = catalog["common_outcome_columns_sample"]
    assert "fc_edge_z" in outcomes
    assert "fc_edge" in outcomes
    assert "fc_edge_z_z" in outcomes
    assert "fc_other" in outcomes
    assert "fc_other_z" in outcomes


def test_llm_candidate_generation_retries_after_preflight_failure(tmp_path):
    contract = _contract()
    context = _preflight_context(tmp_path)
    llm = _PreflightRetryLLM()

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1, llm_schema_retries=1),
        llm=llm,
        evaluator=lambda candidate: {"final_label": "confirmed", "gate_results": {"candidate_id": candidate.candidate_id}},
        preflight_context=context,
    )

    assert llm.calls == 2
    assert llm.saw_validation_feedback
    assert state.stopped_reason == "all_candidates_supported"
    assert state.evaluations[0].validation.ok
    assert not any("Preflight:" in item for item in state.evaluations[0].validation.violations)
    assert [item["retry_kind"] for item in state.llm_candidate_responses] == [
        "none",
        "deterministic_validation",
    ]
    assert state.llm_candidate_responses[1]["validation_retry_index"] == 1
    assert state.llm_candidate_responses[1]["is_retry"] is True
    assert len(state.unretained_candidate_attempts) == 1
    assert state.unretained_candidate_attempts[0].validation_retry_index == 0
    assert any(
        "Preflight:" in violation
        for violation in state.unretained_candidate_attempts[0].validation.violations
    )


def test_llm_candidate_generation_retries_after_deterministic_validation_failure():
    contract = _contract()
    llm = _ValidationRetryLLM()

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1, llm_schema_retries=1),
        llm=llm,
        evaluator=lambda candidate: {
            "final_label": "confirmed",
            "gate_results": {
                "candidate_id": candidate.candidate_id,
                "data_paths": {
                    "discovery": "data/ADNI.parquet",
                    "replication": ["data/OASIS3.parquet"],
                },
            },
        },
    )

    assert llm.calls == 2
    assert llm.saw_validation_feedback
    assert state.evaluations[0].validation.ok
    assert state.stopped_reason == "all_candidates_supported"
    assert summarize_claim_search([state])["unretained_generated_candidate_count"] == 1
    assert [item["retry_kind"] for item in state.llm_candidate_responses] == [
        "none",
        "deterministic_validation",
    ]
    assert len(state.unretained_candidate_attempts) == 1
    assert state.unretained_candidate_attempts[0].proposal.proposed_contract.estimand.group.case == "MCI"


def test_partially_valid_llm_response_is_not_retried_or_backfilled():
    contract = _contract()
    llm = _PartialValidationLLM()

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2, llm_schema_retries=2),
        llm=llm,
        evaluator=lambda candidate: {"final_label": "fragile", "gate_results": {}},
    )

    assert llm.calls == 1
    assert state.generated_candidate_count == 2
    assert state.schema_valid_candidate_count == 2
    assert state.valid_candidate_count == 1
    assert state.current_data_evaluated_count == 1


def test_transform_mismatch_is_a_warning_and_does_not_block_execution():
    contract = _contract()

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [_candidate(contract, transform_type="stronger_design")]

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1),
        candidate_generator=generator,
        evaluator=lambda candidate: {"final_label": "confirmed", "gate_results": {}},
    )

    candidate = state.candidate_history[0]
    assert candidate.declared_transform == "stronger_design"
    assert candidate.inferred_transform == "alternative_same_modality_outcome"
    assert candidate.transform_match is False
    assert state.evaluations[0].validation.ok
    assert any("does not match inferred transform" in item for item in state.evaluations[0].validation.warnings)


def test_mixed_pass_fail_round_evaluates_all_candidates_and_continues():
    contract = _contract()
    evaluated_outcomes = []

    def generator(contract, localization, config, round_index, parent_claim_id):
        if round_index == 1:
            specs = [
                ("candidate_r1_c1", "smri_entorhinal", []),
                ("candidate_r1_c2", "smri_amygdala", []),
            ]
        else:
            specs = [("candidate_r2_c1", "smri_temporal", ["candidate_r1_c2"])]
        return [
            _candidate(
                contract,
                candidate_id=candidate_id,
                round_index=round_index,
                proposed_contract=contract.model_copy(
                    update={"estimand": contract.estimand.model_copy(update={"outcome": outcome})}
                ),
                responds_to_candidate_ids=responds_to,
            )
            for candidate_id, outcome, responds_to in specs
        ]

    def evaluator(candidate):
        outcome = candidate.proposed_contract.estimand.outcome
        evaluated_outcomes.append(str(outcome))
        label = "fragile" if outcome == "smri_amygdala" else "confirmed"
        return {"final_label": label, "gate_results": {"verdict": _verdict(label, ["multiplicity"])}}

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=2, max_candidates_per_round=2),
        candidate_generator=generator,
        evaluator=evaluator,
    )

    assert "smri_amygdala" in evaluated_outcomes
    assert any(candidate.round_index == 2 for candidate in state.candidate_history)
    assert state.round_failure_contexts[0].failed_candidates[0].candidate_id == "candidate_r1_c2"
    assert set(state.internally_supported_candidate_ids) == {"candidate_r1_c1", "candidate_r2_c1"}
    assert state.stopped_reason == "all_candidates_supported"


def test_final_realized_family_size_can_retract_an_early_pass():
    contract = _contract()

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [
            _candidate(
                contract,
                candidate_id=f"candidate_r1_c{index}",
                proposed_contract=contract.model_copy(
                    update={"estimand": contract.estimand.model_copy(update={"outcome": outcome})}
                ),
            )
            for index, outcome in enumerate(("smri_entorhinal", "smri_amygdala"), start=1)
        ]

    def evaluator(candidate):
        family_size = candidate.proposed_contract.gates.multiplicity.family_size
        outcome = candidate.proposed_contract.estimand.outcome
        label = "confirmed" if outcome == "smri_entorhinal" and family_size == 2 else "fragile"
        return {"final_label": label, "gate_results": {"verdict": _verdict(label, ["multiplicity"])}}

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2),
        candidate_generator=generator,
        evaluator=evaluator,
    )

    first = state.evaluations[0]
    assert first.provisional_supported is True
    assert first.current_data_supported is False
    assert first.multiplicity_retracted is True
    assert first.final_family_size == 3
    assert state.internally_supported_candidate_ids == []
    summary = summarize_claim_search([state])
    assert summary["provisional_internal_pass_count"] == 1
    assert summary["final_multiplicity_adjusted_internal_pass_count"] == 0
    assert summary["multiplicity_retraction_count"] == 1


def test_later_round_candidate_must_reference_preceding_failed_candidate():
    contract = _contract()
    localization = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(contract, round_index=2, responds_to_candidate_ids=[])

    validation = validate_candidate_claim(
        contract,
        candidate,
        localization,
        ClaimSearchConfig(max_rounds=2),
        preceding_failed_candidate_ids={"candidate_r1_c1"},
    )

    assert not validation.ok
    assert any("must respond" in item for item in validation.violations)


def test_external_routing_proposal_type_is_rejected_by_stage3_schema():
    contract = _contract()
    raw = json.loads(
        _FakeCandidateLLM().complete(
            "",
            json.dumps(
                {
                    "failure_localization": {"evidence": []},
                    "original_contract": contract.model_dump(mode="json"),
                }
            ),
        )
    )
    raw["candidates"][0]["proposal_type"] = "independent_replication_claim"

    with pytest.raises(ValueError):
        LLMCandidateGenerationResponse.model_validate(raw)


def test_external_only_contract_is_blocked_before_source_evaluation():
    contract = _contract()
    external_contract = contract.model_copy(
        update={
            "discovery_cohort": "NACC_EXTERNAL_DISC",
            "replication_cohorts": ["NACC_EXTERNAL_REP"],
        }
    )
    current_contracts = []

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [
            _candidate(
                contract,
                transform_type="stronger_design",
                proposed_contract=external_contract,
                proposed_question="Replicate the parent claim on external evidence.",
            )
        ]

    def current_evaluator(candidate):
        current_contracts.append(candidate.proposed_contract)
        return {"final_label": "confirmed", "gate_results": {}}

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1),
        candidate_generator=generator,
        evaluator=current_evaluator,
    )

    assert current_contracts == []
    assert not state.evaluations[0].validation.ok
    assert any("external evidence" in item for item in state.evaluations[0].validation.violations)


def test_stub_generated_candidate_can_be_exploratory_confirmed_on_same_data():
    contract = _contract()
    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2),
        candidate_generator=_stub_candidate_generator,
        evaluator=lambda candidate: {
            "final_label": "confirmed",
            "gate_results": {
                "candidate_id": candidate.candidate_id,
                "data_paths": {
                    "discovery": "data/ADNI.parquet",
                    "replication": ["data/ADNI.parquet"],
                },
            },
        },
    )

    assert state.internally_supported_candidate_ids
    assert state.evaluations[0].validation_split == "current_data_adaptive"
    assert state.evaluations[0].eligible_for_confirmation
    assert state.evaluations[0].final_label == "exploratory_confirmed"
    assert state.evaluations[0].same_underlying_data is True
    assert state.stopped_reason == "all_candidates_supported"
    summary = summarize_claim_search([state])
    assert summary["same_data_exploratory_confirmed_count"] == 2
    assert summary["confirmed_count"] == 0
    assert summary["final_confirmed_count"] == 0
    assert summary["any_supported_candidate_count"] == 2


def test_duplicate_candidates_are_removed_across_rounds():
    contract = _contract()

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [
            _candidate(
                contract,
                candidate_id=f"candidate_r{round_index}",
                round_index=round_index,
                validation_split="current_data_adaptive",
                requires_new_evidence=False,
                can_confirm_on_current_data=True,
            )
        ]

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=3, max_candidates_per_round=1),
        candidate_generator=generator,
        evaluator=lambda candidate: {"final_label": "fragile", "gate_results": {}},
    )

    assert len(state.candidate_history) == 1
    assert len(state.evaluations) == 1
    assert len(state.duplicate_candidates) == 1
    assert state.duplicate_candidates[0].candidate_id == "candidate_r2"
    assert state.duplicate_candidates[0].duplicate_of == "candidate_r1"
    assert state.stopped_reason == "no_candidates"
    assert len(state.round_summaries) == 2
    assert sum(item.proposals_returned for item in state.round_summaries) == state.generated_candidate_count
    assert state.round_summaries[1].unique_source_tested == 0
    assert summarize_claim_search([state])["duplicate_candidate_count"] == 1


def test_duplicates_from_rejected_validation_attempts_remain_in_ledger():
    contract = _contract()

    class RetryGenerator:
        validation_feedback = None
        validation_retry_index = 0

        def __call__(self, contract, localization, config, round_index, parent_claim_id):
            if round_index == 1:
                return [
                    _candidate(
                        contract,
                        candidate_id="candidate_r1",
                        round_index=1,
                    )
                ]
            if self.validation_retry_index == 0:
                duplicate = _candidate(
                    contract,
                    candidate_id="duplicate_r2",
                    round_index=2,
                    responds_to_candidate_ids=["candidate_r1"],
                )
                invalid_contract = contract.model_copy(
                    update={
                        "claim_id": "invalid_r2_contract",
                        "estimand": contract.estimand.model_copy(
                            update={
                                "outcome": "smri_temporal",
                                "direction": "positive",
                            }
                        ),
                    }
                )
                invalid = _candidate(
                    contract,
                    candidate_id="invalid_r2",
                    round_index=2,
                    responds_to_candidate_ids=["candidate_r1"],
                    proposed_contract=invalid_contract,
                )
                return [duplicate, invalid]
            valid_contract = contract.model_copy(
                update={
                    "claim_id": "valid_r2_contract",
                    "estimand": contract.estimand.model_copy(update={"outcome": "smri_temporal"}),
                }
            )
            return [
                _candidate(
                    contract,
                    candidate_id="valid_r2",
                    round_index=2,
                    responds_to_candidate_ids=["candidate_r1"],
                    proposed_contract=valid_contract,
                )
            ]

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=2, max_candidates_per_round=2, llm_schema_retries=1),
        candidate_generator=RetryGenerator(),
        evaluator=lambda candidate: {"final_label": "fragile", "gate_results": {}},
    )

    assert state.generated_candidate_count == 4
    assert len(state.candidate_history) == 2
    assert len(state.duplicate_candidates) == 1
    assert state.duplicate_candidates[0].candidate_id == "duplicate_r2"
    assert len(state.unretained_candidate_attempts) == 1
    assert state.unretained_candidate_attempts[0].proposal.candidate_id == "invalid_r2"
    assert state.generated_candidate_count == (
        len(state.candidate_history)
        + len(state.duplicate_candidates)
        + len(state.unretained_candidate_attempts)
    )
    summary = summarize_claim_search([state])
    assert summary["duplicate_candidate_count"] == 1
    assert summary["unretained_validation_candidate_count"] == 1


def test_search_api_has_no_excluded_evidence_evaluator():
    contract = _contract()

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1),
        candidate_generator=_stub_candidate_generator,
        evaluator=lambda candidate: {"final_label": "confirmed", "gate_results": {}},
    )

    assert "external_evaluator" not in inspect.signature(run_claim_search).parameters
    assert summarize_claim_search([state])["excluded_evidence_query_count"] == 0


def test_canonical_coverage_requires_successful_current_data_execution():
    contract = _contract()
    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1),
        candidate_generator=_stub_candidate_generator,
        evaluator=lambda candidate: (_ for _ in ()).throw(RuntimeError("stats engine failed")),
    )

    summary = summarize_claim_search([state])
    assert summary["valid_connected_candidate_count"] == 1
    assert summary["valid_connected_executable_candidate_count"] == 0
    assert summary["valid_connected_lineage_count"] == 0
    assert summary["valid_connected_lineage_rate"] == 0.0


def test_excluded_evidence_is_not_queried_without_current_data_support():
    contract = _contract()

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=2, max_candidates_per_round=1),
        candidate_generator=_stub_candidate_generator,
        evaluator=lambda candidate: {"final_label": "fragile", "gate_results": {}},
    )

    assert summarize_claim_search([state])["excluded_evidence_query_count"] == 0
    assert state.stopped_reason in {"no_candidates", "no_supported_candidate"}


def test_effective_family_size_counts_every_unique_candidate_tested():
    contract = _contract(search_provenance={"family_size": 3})
    seen_family_sizes = []
    seen_selections = []

    def generator(contract, localization, config, round_index, parent_claim_id):
        outcomes = [
            f"smri_hippocampus_r{round_index}_c1",
            f"smri_hippocampus_r{round_index}_c2",
        ]
        return [
            _candidate(
                contract,
                candidate_id=f"candidate_r{round_index}_c{index}",
                round_index=round_index,
                proposed_contract=contract.model_copy(
                    update={"estimand": contract.estimand.model_copy(update={"outcome": outcome})}
                ),
                responds_to_candidate_ids=(
                    [f"candidate_r{round_index - 1}_c1"] if round_index > 1 else []
                ),
            )
            for index, outcome in enumerate(outcomes, start=1)
        ]

    def evaluator(candidate):
        seen_family_sizes.append(candidate.proposed_contract.search_provenance.family_size)
        seen_selections.append(candidate.proposed_contract.search_provenance.selection)
        return {"final_label": "fragile", "gate_results": {}}

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=2, max_candidates_per_round=2),
        candidate_generator=generator,
        evaluator=evaluator,
    )

    assert seen_family_sizes == [4, 5, 6, 7]
    assert seen_selections == ["discovery_only"] * 4
    assert [item.effective_family_size for item in state.evaluations] == [4, 5, 6, 7]
    assert state.current_data_evaluated_count == 4
    assert summarize_claim_search([state])["excluded_evidence_query_count"] == 0


def test_hacking_metric_counts_holdout_cohort_misuse():
    contract = _contract()
    invalid_contract = _contract(
        discovery_cohort="ADNI_HOLDOUT_DISC",
        replication_cohorts=["OASIS3_HOLDOUT_REP"],
    )

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [_candidate(contract, proposed_contract=invalid_contract)]

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1),
        candidate_generator=generator,
        evaluator=lambda candidate: {"final_label": "fragile", "gate_results": {}},
    )

    summary = summarize_claim_search([state])
    assert not state.evaluations[0].validation.ok
    assert any("introduces holdout partitions" in item for item in state.evaluations[0].validation.violations)
    assert summary["hacking_block_count"] == 1


def test_claim_search_artifacts_serialize_without_holdout():
    contract = _contract()
    artifacts = build_claim_search_artifacts(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2),
        llm=_FakeCandidateLLM(),
    )

    assert artifacts["claim_search_config"]["max_rounds"] == 1
    assert len(artifacts["candidate_claims"]) == 2
    assert artifacts["failure_localization"]["failure_kind"] == "evidence_failure"
    assert artifacts["llm_candidate_prompts"]
    assert artifacts["llm_candidate_responses"]
    assert artifacts["claim_search_trace"]["stopped_reason"] == "no_evaluator"
