from pathlib import Path

import pytest

from confirm.contract import ClaimContract, load_contract


def test_example_contract_validates():
    contract = load_contract(Path("configs/contracts/example_age_atrophy.yaml"))
    assert contract.claim_id == "age_atrophy_smri"
    assert contract.estimand.outcome == "smri_meanthickness"
    assert contract.gates.replication.harmonize == "combat"


def test_predictor_in_covariates_raises():
    with pytest.raises(Exception, match="predictor 'age'"):
        ClaimContract.model_validate(
            {
                "claim_id": "bad_predictor_covariate",
                "question": "Invalid contract.",
                "estimand": {
                    "type": "association",
                    "outcome": "smri_meanthickness",
                    "predictor": "age",
                    "group": None,
                    "direction": "negative",
                },
                "covariates": ["age", "sex"],
                "inclusion": None,
                "discovery_cohort": "DISC",
                "replication_cohorts": ["REP"],
                "gates": {
                    "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                    "confound": {"require_covariates": ["sex"], "motion_check": False},
                    "power": {"min_power": 0.8, "ref_effect": None},
                    "multiverse": {"min_fraction_consistent": 0.6},
                    "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "none"},
                },
                "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
            }
        )
