import pytest

from bench.metrics import exact_binomial_ci, summarize_rows


def test_exact_binomial_ci_zero_successes_has_nonzero_upper_bound():
    lo, hi = exact_binomial_ci(0, 7)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 7))


def test_fcr_denominator_includes_known_null_and_fragile():
    rows = [
        {"claim_id": "ad_smri_adni_oasis3", "scoring_label": "known_positive", "scoring_bucket": "positive", "+replication": True},
        {"claim_id": "injected_null_random_hcp", "scoring_label": "known_null", "scoring_bucket": "negative", "+replication": False},
        {"claim_id": "adhd_region_adhd200_abcd", "scoring_label": "fragile", "scoring_bucket": "negative", "+replication": True},
        {
            "claim_id": "cognition_fc_ukb_abcd",
            "scoring_label": "underpowered_small_positive",
            "scoring_bucket": "small_positive",
            "+replication": False,
        },
    ]
    summary = summarize_rows(rows, ["+replication"])["summary"]["+replication"]
    assert summary["known_positive_recall_count"] == 1
    assert summary["known_positive_recall_denominator"] == 1
    assert summary["FCR_count"] == 1
    assert summary["FCR_denominator"] == 2
    assert summary["small_positive_recovery_denominator"] == 1
    assert summary["small_positive_denominator"] == 1
    assert summary["coverage_count"] == 2


def test_gate_ladder_reports_full_and_main_subsets():
    rows = [
        {
            "claim_id": "ad_smri_adni_oasis3",
            "scoring_label": "known_positive",
            "scoring_bucket": "positive",
            "label_authority": "main",
            "+replication": True,
        },
        {
            "claim_id": "nacc_age_smri_split",
            "scoring_label": "known_positive",
            "scoring_bucket": "positive",
            "label_authority": "supplementary",
            "+replication": True,
        },
        {
            "claim_id": "injected_null_random_hcp",
            "scoring_label": "known_null",
            "scoring_bucket": "negative",
            "label_authority": "main",
            "+replication": False,
        },
        {
            "claim_id": "asd_fc_abide2_internal_split",
            "scoring_label": "fragile",
            "scoring_bucket": "negative",
            "label_authority": "supplementary",
            "+replication": True,
        },
    ]
    metrics = summarize_rows(rows, ["+replication"])
    full = metrics["summary_full"]["+replication"]
    main = metrics["summary_main"]["+replication"]

    assert metrics["label_authority_counts"] == {"main": 2, "supplementary": 2}
    assert full["TPR_count"] == 2
    assert full["TPR_denominator"] == 2
    assert full["FCR_count"] == 1
    assert full["FCR_denominator"] == 2
    assert main["TPR_count"] == 1
    assert main["TPR_denominator"] == 1
    assert main["FCR_count"] == 0
    assert main["FCR_denominator"] == 1
    assert "TPR_ci95" in main
    assert "FCR_ci95_exact" in main


def test_generated_label_metadata_registers_claim_for_metrics():
    label_row = {
        "claim_id": "generated_known_null_for_metrics",
        "phenotype": "synthetic generated null",
        "modality": "fMRI-FC",
        "cohorts": "SYN;SYN",
        "discovery_cohort": "SYN",
        "replication_cohort": "SYN",
        "label_class": "known_null",
        "label_basis": "synthetic_stress",
        "adjudication_status": "preregistered",
        "expected_direction": "two_sided",
        "expected_effect_scale": "none",
        "mde_assumption": "synthetic",
        "cohort_role": "generated test",
        "forbidden_evidence": "none",
        "confound_set": "age;sex",
        "site_scanner_handling": "synthetic",
        "decision_target": "FCR",
        "construct_validity_notes": "test row",
        "label_confidence": "high",
        "source_citation": "synthetic_stress",
    }
    metrics = summarize_rows(
        [
            {
                "claim_id": "generated_known_null_for_metrics",
                "scoring_label": "known_null",
                "scoring_bucket": "negative",
                "label_metadata": label_row,
                "+replication": False,
            }
        ],
        ["+replication"],
    )
    assert metrics["summary_main"]["+replication"]["FCR_denominator"] == 1
