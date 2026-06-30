from __future__ import annotations

from confirm.contract import ClaimContract
from confirm.proposals import (
    NewClaimProposal,
    default_new_claim_proposal,
    localization_for_estimand_mismatch,
    localize_failure,
    validate_new_claim_proposal,
)


def _contract(**overrides):
    data = {
        "claim_id": "claim",
        "question": "Question.",
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


def _verdict(label, failed, rationale=None):
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
    return {
        "label": label,
        "abstained": label != "confirmed",
        "rationale": rationale or ("Failed gates: " + ", ".join(failed)),
        "gates": gates,
    }


def _results(contract):
    return {
        "contract": contract.model_dump(mode="json"),
        "primary": {"p": 0.2, "beta": -0.1, "n": 100},
        "multiverse": {"fraction_consistent": 0.25, "passed": False, "specs": []},
        "power": {"achieved_power": 0.3, "under_powered": True, "n_needed_80": 200},
        "replication": {"passed": False, "reason": "non_replicated_effect_absent", "cohort_results": []},
    }


def _proposal(localization, *, proposal_type="downgraded_claim", contract=None, **overrides):
    data = {
        "source_claim_id": localization.claim_id,
        "proposal_type": proposal_type,
        "rationale": "Evidence-grounded proposal.",
        "proposed_question": "Question.",
        "proposed_contract": contract,
        "disposition_label": "fragile" if proposal_type == "downgraded_claim" else None,
        "provenance": "abstention",
        "requires_new_evidence": False,
        "can_confirm_on_current_data": False,
        "supported_by_evidence": localization.evidence,
    }
    data.update(overrides)
    return NewClaimProposal.model_validate(data)


def test_multiplicity_failure_is_evidence_failure_not_current_data_repairable():
    contract = _contract(gates={"multiplicity": {"family_size": 4}})
    loc = localize_failure(contract, _verdict("fragile", ["multiplicity"]), _results(contract))

    assert loc.failure_kind == "evidence_failure"
    assert not loc.current_data_repair_allowed
    assert "corrected_contract" not in loc.allowed_proposal_types


def test_multiverse_failure_allows_fragile_or_future_claim_only():
    contract = _contract()
    loc = localize_failure(contract, _verdict("fragile", ["multiverse"]), _results(contract))
    proposal = default_new_claim_proposal(loc, contract)
    validation = validate_new_claim_proposal(contract, proposal, loc)

    assert proposal.proposal_type == "downgraded_claim"
    assert proposal.disposition_label == "fragile"
    assert validation.ok


def test_replication_and_power_failures_choose_scientific_dispositions():
    contract = _contract()
    rep_loc = localize_failure(contract, _verdict("non_replicated", ["replication"]), _results(contract))
    power_loc = localize_failure(contract, _verdict("under_powered", ["power"]), _results(contract))

    assert rep_loc.failure_kind == "evidence_failure"
    assert default_new_claim_proposal(rep_loc, contract).disposition_label == "non_replicated"
    assert power_loc.failure_kind == "design_limitation"
    assert default_new_claim_proposal(power_loc, contract).disposition_label == "under_powered"


def test_estimand_mismatch_accepts_corrected_contract_rerun():
    original = _contract()
    revised = _contract(estimand={"outcome": "smri_entorhinal"})
    loc = localization_for_estimand_mismatch("claim", mismatches={"outcome": {"expected": "smri_entorhinal"}})
    proposal = _proposal(
        loc,
        proposal_type="corrected_contract",
        contract=revised,
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )

    result = validate_new_claim_proposal(original, proposal, loc)

    assert result.ok
    assert result.checked_contract


def test_confound_missing_allows_covariate_repair_but_structural_confound_does_not():
    original = _contract()
    repaired = _contract(covariates=["age", "sex", "site"], gates={"confound": {"require_covariates": ["age", "sex", "site"]}})
    missing_loc = localize_failure(original, _verdict("fragile", ["confound"]), _results(original))
    missing_proposal = _proposal(
        missing_loc,
        proposal_type="corrected_contract",
        contract=repaired,
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )

    assert validate_new_claim_proposal(original, missing_proposal, missing_loc).ok

    structural_loc = localize_failure(
        original,
        _verdict("fragile", ["confound"], "Failed gates: confound; predictor is nested in a declared confound."),
        _results(original),
    )
    structural_proposal = _proposal(
        structural_loc,
        proposal_type="corrected_contract",
        contract=repaired,
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )
    result = validate_new_claim_proposal(original, structural_proposal, structural_loc)

    assert not result.ok
    assert any("not current-data repairable" in item for item in result.violations)


def test_search_provenance_failure_allows_honest_lineage_correction():
    original = _contract(search_provenance={"declared": True, "family_size": 10, "selection": "discovery_only"})
    repaired = _contract(search_provenance={"declared": True, "family_size": 12, "selection": "discovery_only"})
    loc = localize_failure(original, _verdict("fragile", ["search_provenance"]), _results(original))
    proposal = _proposal(
        loc,
        proposal_type="corrected_contract",
        contract=repaired,
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )

    assert validate_new_claim_proposal(original, proposal, loc).ok

    noop = _proposal(
        loc,
        proposal_type="corrected_contract",
        contract=original,
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )
    result = validate_new_claim_proposal(original, noop, loc)
    assert not result.ok
    assert any("does not change" in item for item in result.violations)

    metadata_only = _proposal(
        loc,
        proposal_type="corrected_contract",
        contract=_contract(
            question="Same executable contract with new wording.",
            search_provenance={"declared": True, "family_size": 10, "selection": "discovery_only"},
        ),
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )
    result = validate_new_claim_proposal(original, metadata_only, loc)
    assert not result.ok
    assert any("executable or governance" in item for item in result.violations)

    relabeled = _proposal(
        loc,
        proposal_type="corrected_contract",
        contract=_contract(search_provenance={"declared": True, "family_size": 10, "selection": "preregistered"}),
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )
    result = validate_new_claim_proposal(original, relabeled, loc)
    assert not result.ok
    assert any("relabeled as preregistered" in item for item in result.violations)


def test_validator_accepts_posthoc_current_data_adaptive_confirmation():
    contract = _contract()
    loc = localize_failure(contract, _verdict("fragile", ["multiplicity"]), _results(contract))
    proposal = _proposal(
        loc,
        proposal_type="exploratory_followup_claim",
        provenance="post_hoc",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )

    result = validate_new_claim_proposal(contract, proposal, loc)

    assert result.ok
    assert result.current_data_confirmability_ok


def test_validator_rejects_gate_weakening_removed_covariates_direction_change_and_same_cohort_replication():
    original = _contract()
    loc = localize_failure(original, _verdict("fragile", ["confound"]), _results(original))
    revised = original.model_dump(mode="json")
    revised["gates"]["multiplicity"]["alpha"] = 0.1
    revised["covariates"] = ["age"]
    revised["gates"]["confound"]["require_covariates"] = ["age"]
    revised["estimand"]["direction"] = "positive"
    revised["replication_cohorts"] = ["ADNI"]
    proposal = _proposal(
        loc,
        proposal_type="corrected_contract",
        contract=ClaimContract.model_validate(revised),
        provenance="contract_correction",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
    )

    result = validate_new_claim_proposal(original, proposal, loc)

    assert not result.ok
    assert any("alpha" in item for item in result.violations)
    assert any("confound covariates" in item for item in result.violations)
    assert any("independent" in item for item in result.violations)


def test_validator_accepts_exploratory_followup_only_with_new_evidence():
    contract = _contract()
    loc = localize_failure(contract, _verdict("fragile", ["multiplicity"]), _results(contract))
    proposal = _proposal(
        loc,
        proposal_type="exploratory_followup_claim",
        provenance="post_hoc",
        requires_new_evidence=True,
        can_confirm_on_current_data=False,
    )

    result = validate_new_claim_proposal(contract, proposal, loc)

    assert result.ok
    assert result.provenance_compliant
