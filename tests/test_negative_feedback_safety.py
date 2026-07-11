from __future__ import annotations

import pandas as pd

from bench.injected_nulls import NegativeStressTask
from bench.run_negatives_expansion import materialize_negative_task
from confirm.candidate_preflight import CandidatePreflightContext
from confirm.contract import ClaimContract


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "neg_random_demo_s1",
            "question": "Programmatic known-null stress claim.",
            "estimand": {
                "type": "group_diff",
                "outcome": "fc_demo",
                "predictor": "bench_group",
                "group": {"var": "bench_group", "case": "case", "control": "control"},
                "direction": "two_sided",
                "unit": "scalar",
            },
            "covariates": ["age", "sex", "site"],
            "inclusion": None,
            "discovery_cohort": "DEMO_DISC_s1",
            "replication_cohorts": ["DEMO_REP_s1"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["age", "sex", "site"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {"alpha": 0.05, "require_same_sign": True, "harmonize": "none"},
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def test_materialized_negative_task_is_executable_by_claim_search(tmp_path):
    frame = pd.DataFrame(
        {
            "subject_id": [f"sub-{index}" for index in range(40)],
            "cohort": ["DEMO"] * 40,
            "site": ["site1", "site2"] * 20,
            "age": [20 + index % 10 for index in range(40)],
            "sex": ["F", "M"] * 20,
            "bench_group": ["case", "control"] * 20,
            "fc_demo": [float(index) for index in range(40)],
        }
    )
    task = NegativeStressTask(
        claim_id="neg_random_demo_s1",
        family="random_label",
        label_class="known_null",
        expected_gate="multiplicity_or_replication",
        discovery=frame.copy(),
        replication=frame.copy(),
        contract=_contract(),
        label_row={"discovery_cohort": "DEMO_DISC_s1", "replication_cohort": "DEMO_REP_s1"},
        covariates_min=["age", "sex"],
    )
    data_root = tmp_path / "cohorts"

    materialized = materialize_negative_task(task, data_root)
    context = CandidatePreflightContext.from_roots([data_root])
    preflight = context.validate_contract(materialized.contract)

    assert materialized.contract.discovery_cohort == "neg_random_demo_s1_DISC"
    assert materialized.contract.replication_cohorts == ["neg_random_demo_s1_REP"]
    assert (data_root / "neg_random_demo_s1_DISC.parquet").exists()
    assert (data_root / "neg_random_demo_s1_REP.parquet").exists()
    assert preflight.ok
