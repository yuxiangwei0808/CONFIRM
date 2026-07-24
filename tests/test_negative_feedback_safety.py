from __future__ import annotations

import pandas as pd

from bench.injected_nulls import (
    NegativeCohort,
    NegativeStressTask,
    site_confound_stress_task,
)
from bench.run_iterative_claim_search_replay import _known_negative_or_fragile_source
from bench.run_known_negative_safety import materialize_negative_task
from confirm.analysis import build_analysis_design
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
            "site": ["site1"] * 20 + ["site2"] * 20,
            "age": [20 + index % 7 for index in range(40)],
            "sex": ["F", "F", "M", "M"] * 10,
            "bench_group": ["case", "control", "control", "case"] * 10,
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


def test_site_confound_has_both_groups_per_site_and_disjoint_holdouts(tmp_path):
    n_rows = 240
    frame = pd.DataFrame(
        {
            "subject_id": [f"sub-{index}" for index in range(n_rows)],
            "cohort": ["DEMO"] * n_rows,
            "site": [f"original-{index % 3}" for index in range(n_rows)],
            "age": [20 + index % 30 for index in range(n_rows)],
            "sex": ["F", "M", "F", "M"] * (n_rows // 4),
            "fc_demo": [float(index % 17) for index in range(n_rows)],
        }
    )
    cohort = NegativeCohort(
        name="DEMO",
        path=tmp_path / "demo.parquet",
        frame=frame,
        features=["fc_demo"],
        modality="fMRI-FC",
    )

    task = site_confound_stress_task(cohort, seed=3)
    partitions = [
        task.discovery,
        task.replication,
        task.holdout_discovery,
        task.holdout_replication,
    ]
    subject_sets = [set(partition["subject_id"].astype(str)) for partition in partitions]
    assert all(not left & right for index, left in enumerate(subject_sets) for right in subject_sets[index + 1 :])
    for partition in partitions:
        counts = pd.crosstab(partition["site"], partition["bench_group"])
        assert set(counts.columns) == {"case", "control"}
        assert (counts > 0).all().all()
    assert task.metadata["direct_group_effect"] == 0.0
    assert build_analysis_design(task.discovery, task.contract).diagnostics["full_rank"] is True

    materialized = materialize_negative_task(task, tmp_path / "materialized" / "cohorts")
    assert not materialized.holdout_discovery.empty
    assert (tmp_path / "materialized" / "cohorts" / f"{task.claim_id}_HOLDOUT_DISC.parquet").exists()
    assert (tmp_path / "materialized" / "cohorts" / f"{task.claim_id}_HOLDOUT_REP.parquet").exists()
    assert len(materialized.metadata["_evidence_partition_records"]) == 4


def test_all_150_synthetic_source_rows_are_recognized_as_known_negative():
    families = ["random_label", "site_confound", "p_fishing", "underpowered", "cross_nonreplication"]
    rows = [
        {
            "claim_id": f"neg_{families[index % len(families)]}_{index}",
            "source_scoring_label": "known_null" if index % 5 else "fragile",
        }
        for index in range(150)
    ]

    assert sum(_known_negative_or_fragile_source(row) for row in rows) == 150
