from __future__ import annotations

import json

import pandas as pd
import pytest

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.claim_search import (
    CandidateClaimProposal,
    CandidateDomainCore,
    CandidateEvidencePolicy,
    CandidatePreservationCheck,
    ClaimSearchConfig,
    LLMClaimCandidateGenerator,
    build_claim_search_artifacts,
    build_candidate_generation_prompt,
    generate_connected_candidates,
    localize_failure,
    parse_candidate_generation_response,
    run_claim_search,
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
        requires_new_evidence=overrides.get("requires_new_evidence", True),
        can_confirm_on_current_data=overrides.get("can_confirm_on_current_data", False),
        validation_split=overrides.get("validation_split", "future_required"),
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
        "validation_split": "future_required",
        "source_claim_id": contract.claim_id,
        "proposal_type": "exploratory_followup_claim",
        "rationale": "Connected follow-up.",
        "proposed_question": "In excluded validation evidence, test a narrower smri Dementia vs CN outcome.",
        "proposed_contract": contract,
        "disposition_label": None,
        "provenance": "post_hoc_followup",
        "requires_new_evidence": True,
        "can_confirm_on_current_data": False,
        "supported_by_evidence": [],
    }
    data.update(overrides)
    return CandidateClaimProposal.model_validate(data)


class _FakeCandidateLLM:
    model = "fake-claim-generator"

    def complete(self, system, user):
        payload = json.loads(user)
        evidence = payload["failure_localization"]["evidence"][:1]
        proposed_contract = payload["original_contract"]
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
                            "requires_new_evidence": True,
                            "can_confirm_on_current_data": False,
                            "validation_split": "future_required",
                        },
                        "supported_by_evidence": evidence,
                        "disposition_label": None,
                    },
                    {
                        "proposal_type": "independent_replication_claim",
                        "transform_type": "stronger_design",
                        "domain_core": domain_core,
                        "preservation_check": {
                            **preservation_check,
                            "changed_fields": ["validation_evidence"],
                            "allowed_change_rationale": "Preserves the same claim but moves it to independent evidence.",
                        },
                        "proposed_question": "Replicate the original Dementia vs CN smri claim in an independent cohort.",
                        "proposed_contract": proposed_contract,
                        "rationale": "This asks for independent evidence rather than same-data optimization.",
                        "connection_rationale": "Preserves the Dementia vs CN contrast, smri modality, and gates.",
                        "evidence_policy": {
                            "provenance": "independent_replication",
                            "requires_new_evidence": True,
                            "can_confirm_on_current_data": False,
                            "validation_split": "future_required",
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
                            "requires_new_evidence": True,
                            "can_confirm_on_current_data": False,
                            "validation_split": "future_required",
                        },
                        "supported_by_evidence": evidence,
                        "disposition_label": None,
                    }
                ]
            }
        )


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


def test_config_enforces_budget_and_generator_respects_candidate_limit():
    with pytest.raises(ValueError):
        ClaimSearchConfig(max_rounds=0)
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    config = ClaimSearchConfig(max_rounds=2, max_candidates_per_round=2)

    candidates = generate_connected_candidates(contract, loc, config, 1, contract.claim_id)

    assert len(candidates) == 2


def test_loop_stops_without_forcing_success_when_no_evaluator_exists():
    contract = _contract()
    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=3, max_candidates_per_round=3),
        candidate_generator=generate_connected_candidates,
    )

    assert state.confirmed_candidates == []
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

    assert "goal is not to make the failed claim pass" in system
    assert "forbidden_actions" in user
    assert "executable_data_catalog" in user
    assert "output_schema" in user
    assert "domain_core" in user
    assert "preservation_check" in user
    assert "evidence_policy" in user

    generator = LLMClaimCandidateGenerator(_FakeCandidateLLM())
    candidates = generator(contract, loc, config, 1, contract.claim_id)

    assert len(candidates) == 2
    assert candidates[0].candidate_id.startswith("claim_r1_c1_")
    assert generator.prompt_records
    assert generator.response_records[0]["candidate_count"] == 2
    assert candidates[0].domain_core.outcome_modality == "smri"
    assert candidates[0].evidence_policy.validation_split == candidates[0].validation_split


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
    assert "schema_validation_error" in generator.prompt_records[1]["user"]


def test_parser_rejects_malformed_llm_output():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))

    with pytest.raises(ValueError):
        parse_candidate_generation_response("{}", contract, loc, ClaimSearchConfig(), 1, contract.claim_id)


def test_posthoc_candidate_can_use_adaptive_current_data_evaluation():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(
        contract,
        validation_split="current_data_adaptive",
        can_confirm_on_current_data=True,
        requires_new_evidence=False,
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

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
    )

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

    assert result.ok


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

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

    assert not result.ok
    assert any("does not change" in item for item in result.violations)


def test_unrelated_outcome_and_direction_changes_are_rejected():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    unrelated = _contract(estimand={"outcome": "pet_amyloid", "direction": "positive"})
    candidate = _candidate(contract, proposed_contract=unrelated)

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

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

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

    assert not result.ok
    assert any("domain_core changes outcome modality" in item for item in result.violations)


def test_domain_core_accepts_known_modality_and_direction_aliases():
    contract = _contract(estimand={"outcome": "fc_mean_abs", "direction": "two_sided"})
    loc = localize_failure(contract, _verdict(), _results(contract))
    candidate = _candidate(
        contract,
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

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

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

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

    assert not result.ok
    assert any("weakened" in item for item in result.violations)


def test_narrower_same_modality_contract_is_connected_but_requires_new_evidence():
    contract = _contract()
    loc = localize_failure(contract, _verdict(), _results(contract))
    narrower = _contract(estimand={"outcome": "smri_entorhinal"})
    candidate = _candidate(contract, proposed_contract=narrower)

    result = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)

    assert result.ok
    assert candidate.requires_new_evidence
    assert candidate.validation_split == "future_required"


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
        excluded_validation_available=False,
        preflight_context=context,
    )

    assert not result.ok
    assert any("Preflight:" in item and "missing outcome" in item for item in result.violations)


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
    assert state.stopped_reason == "exploratory_confirmed"
    assert state.evaluations[0].validation.ok
    assert not any("Preflight:" in item for item in state.evaluations[0].validation.violations)


def test_independent_replication_candidate_requires_excluded_validation_evidence():
    contract = _contract()
    loc = localize_failure(contract, _verdict(failed=["replication"]), _results(contract))
    candidate = _candidate(
        contract,
        proposal_type="independent_replication_claim",
        transform_type="stronger_design",
        provenance="independent_replication",
        validation_split="excluded_validation",
        proposed_question="Replicate the Dementia vs CN smri claim in excluded validation evidence.",
    )

    missing = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=False)
    available = validate_candidate_claim(contract, candidate, loc, ClaimSearchConfig(), excluded_validation_available=True)

    assert not missing.ok
    assert any("none is available" in item for item in missing.violations)
    assert available.ok


def test_default_generated_candidate_can_be_exploratory_confirmed_on_same_data():
    contract = _contract()
    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=2),
        candidate_generator=generate_connected_candidates,
        evaluator=lambda candidate: {"final_label": "confirmed", "gate_results": {"candidate_id": candidate.candidate_id}},
    )

    assert state.confirmed_candidates
    assert state.evaluations[0].validation_split == "current_data_adaptive"
    assert state.evaluations[0].eligible_for_confirmation
    assert state.evaluations[0].final_label == "exploratory_confirmed"
    assert state.stopped_reason == "exploratory_confirmed"


def test_controlled_holdout_candidate_can_be_confirmed_by_stub_evaluator():
    contract = _contract()

    def generator(contract, localization, config, round_index, parent_claim_id):
        return [
            _candidate(
                contract,
                validation_split="excluded_validation",
                proposed_question="In excluded validation evidence, test a narrower smri Dementia vs CN outcome.",
            )
        ]

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=2, max_candidates_per_round=2),
        candidate_generator=generator,
        evaluator=lambda candidate: {"final_label": "confirmed", "gate_results": {"candidate_id": candidate.candidate_id}},
        external_evaluator=lambda candidate: {
            "final_label": "confirmed",
            "gate_results": {"external_candidate_id": candidate.candidate_id},
        },
    )

    assert state.confirmed_candidates
    assert state.stopped_reason == "confirmed"
    assert state.evaluations[0].external_confirmed
    assert state.evaluations[0].final_label == "confirmed"


def test_external_evaluator_error_preserves_exploratory_confirmation():
    contract = _contract()

    def external_evaluator(candidate):
        raise RuntimeError("external unavailable")

    state = run_claim_search(
        contract,
        _verdict(),
        _results(contract),
        config=ClaimSearchConfig(max_rounds=1, max_candidates_per_round=1),
        candidate_generator=generate_connected_candidates,
        evaluator=lambda candidate: {"final_label": "confirmed", "gate_results": {"candidate_id": candidate.candidate_id}},
        external_evaluator=external_evaluator,
    )

    assert state.stopped_reason == "exploratory_confirmed"
    assert state.evaluations[0].final_label == "exploratory_confirmed"
    assert state.evaluations[0].exploratory_confirmed
    assert "external_evaluator" in str(state.evaluations[0].execution_error)


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
