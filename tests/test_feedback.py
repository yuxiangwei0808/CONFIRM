from __future__ import annotations

from confirm.contract import ClaimContract
from confirm.feedback import (
    RevisionResponse,
    feedback_for_estimand_mismatch,
    feedback_from_row,
    feedback_from_verdict,
    validate_revision_response,
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


def _response(contract):
    return RevisionResponse(response_type="revised_contract", revised_contract=contract)


def test_feedback_maps_confound_failure_to_actionable_guidance():
    feedback = feedback_from_row(
        {
            "claim_id": "confounded",
            "final_label": "fragile",
            "rationale": "Failed gates: confound; predictor is nested in a declared confound.",
        }
    )

    assert feedback.primary_failure == "confound"
    assert feedback.repairability == "needs_new_data"
    assert "covariates" in feedback.allowed_contract_changes
    assert any("Do not remove" in item for item in feedback.forbidden_revisions)


def test_feedback_maps_underpowered_nonreplicated_multiverse_and_search():
    cases = [
        (
            {"claim_id": "u", "final_label": "under_powered", "rationale": "Failed gates: power; power=0.2"},
            "power",
        ),
        (
            {
                "claim_id": "r",
                "final_label": "non_replicated",
                "rationale": "Failed gates: replication; replication=non_replicated_effect_absent",
            },
            "replication",
        ),
        (
            {"claim_id": "m", "final_label": "fragile", "rationale": "Failed gates: multiverse"},
            "multiverse",
        ),
        (
            {
                "claim_id": "s",
                "final_label": "fragile",
                "rationale": "Failed gates: multiplicity",
                "search_selection": "discovery_only",
            },
            "search_provenance",
        ),
    ]

    for row, expected in cases:
        feedback = feedback_from_row(row)
        assert feedback.primary_failure == expected
        assert feedback.next_agent_instruction


def test_policy_rejects_gate_weakening_outcome_switch_and_removed_confound():
    original = _contract()
    feedback = feedback_from_row({"claim_id": "x", "final_label": "fragile", "rationale": "Failed gates: confound_incomplete"})

    weakened = original.model_dump(mode="json")
    weakened["gates"]["multiplicity"]["alpha"] = 0.1
    result = validate_revision_response(original, _response(ClaimContract.model_validate(weakened)), feedback)
    assert not result.ok
    assert any("alpha" in item for item in result.violations)

    switched = original.model_dump(mode="json")
    switched["estimand"]["outcome"] = "smri_entorhinal"
    result = validate_revision_response(original, _response(ClaimContract.model_validate(switched)), feedback)
    assert not result.ok
    assert any("Outcome changed" in item for item in result.violations)

    removed = original.model_dump(mode="json")
    removed["covariates"] = ["age"]
    removed["gates"]["confound"]["require_covariates"] = ["age"]
    result = validate_revision_response(original, _response(ClaimContract.model_validate(removed)), feedback)
    assert not result.ok
    assert any("confound covariates" in item for item in result.violations)


def test_policy_accepts_covariate_repair_and_disposition():
    original = _contract()
    feedback = feedback_from_row({"claim_id": "x", "final_label": "fragile", "rationale": "Failed gates: confound_incomplete"})
    repaired = original.model_dump(mode="json")
    repaired["covariates"] = ["age", "sex", "site"]
    repaired["gates"]["confound"]["require_covariates"] = ["age", "sex", "site"]

    result = validate_revision_response(original, _response(ClaimContract.model_validate(repaired)), feedback)
    assert result.ok
    assert result.checked_contract

    disposition = RevisionResponse(
        response_type="claim_disposition",
        disposition_label="fragile",
        rationale="Confounded design cannot support confirmation.",
    )
    result = validate_revision_response(original, disposition, feedback)
    assert result.ok
    assert result.accepted_disposition


def test_policy_rejects_downgrade_only_contract_rewrite():
    original = _contract()
    feedback = feedback_from_row({"claim_id": "x", "final_label": "fragile", "rationale": "Failed gates: multiverse"})

    result = validate_revision_response(original, _response(original), feedback)

    assert not result.ok
    assert any("downgrade-only" in item for item in result.violations)


def test_policy_rejects_needs_new_data_without_new_replication_evidence():
    original = _contract()
    feedback = feedback_from_row({"claim_id": "x", "final_label": "under_powered", "rationale": "Failed gates: power"})

    result = validate_revision_response(original, _response(original), feedback)

    assert not result.ok
    assert any("requires new evidence" in item for item in result.violations)


def test_policy_accepts_allowed_replication_cohort_repair():
    original = _contract()
    feedback = feedback_from_row(
        {"claim_id": "x", "final_label": "non_replicated", "rationale": "Failed gates: replication"}
    )
    repaired = original.model_dump(mode="json")
    repaired["replication_cohorts"] = ["NACC"]

    result = validate_revision_response(original, _response(ClaimContract.model_validate(repaired)), feedback)

    assert result.ok
    assert result.checked_contract


def test_policy_accepts_estimand_mismatch_cohort_and_covariate_correction():
    original = _contract()
    revised = original.model_dump(mode="json")
    revised["estimand"]["type"] = "association"
    revised["estimand"]["predictor"] = "age"
    revised["estimand"]["group"] = None
    revised["discovery_cohort"] = "OASIS3"
    revised["replication_cohorts"] = ["ADNI"]
    revised["covariates"] = ["sex"]
    revised["gates"]["confound"]["require_covariates"] = ["sex"]
    feedback = feedback_for_estimand_mismatch(
        "x",
        mismatches={"predictor": {"expected": "age", "actual": "dx"}, "discovery_cohort": {"expected": "OASIS3", "actual": "ADNI"}},
    )

    result = validate_revision_response(original, _response(ClaimContract.model_validate(revised)), feedback)

    assert result.ok
    assert result.checked_contract


def test_feedback_from_verdict_includes_concrete_gate_evidence():
    contract = _contract(gates={"multiplicity": {"family_size": 4}})
    verdict = {
        "label": "fragile",
        "rationale": "Failed gates: multiplicity, multiverse, replication",
        "gates": {
            "multiplicity": False,
            "multiverse": False,
            "replication": False,
            "multiplicity_effective_family_size": 4,
        },
    }
    results = {
        "contract": contract.model_dump(mode="json"),
        "primary": {"p": 0.5639426762381835, "beta": -0.0038658640042852632, "n": 799},
        "multiverse": {
            "fraction_consistent": 0.0,
            "specs": [
                {"same_sign": True, "significant": False},
                {"same_sign": True, "significant": False},
            ],
        },
        "replication": {
            "passed": False,
            "reason": "non_replicated_effect_absent",
            "cohort_results": [
                {
                    "cohort": "ABIDE1",
                    "reason": "non_replicated_effect_absent:p",
                    "effect": {"beta": -0.00661991517440501, "p": 0.28256708295129807, "n": 799},
                }
            ],
        },
    }

    feedback = feedback_from_verdict("x", verdict, results)

    assert feedback.primary_failure == "multiplicity"
    assert any("primary p=0.5639" in item for item in feedback.evidence)
    assert any("fraction_consistent=0" in item and "consistent_specs=0/2" in item for item in feedback.evidence)
    assert any("p=0.2826" in item for item in feedback.evidence)
    assert any("claim_disposition: fragile" in item for item in feedback.refinement_actions)
