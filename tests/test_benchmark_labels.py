import csv
from pathlib import Path

import pytest
import yaml

from bench.labels import label_authority, load_claim_label_table, load_label_table, scoring_bucket, scoring_label_for_claim, validate_label_coverage


def test_fmri_cognition_claims_are_small_positive_not_fcr():
    claim = {"claim_id": "cognition_fc_ukb_abcd", "expected_label": "fragile_or_small_effect", "claim_type": "brain_behavior"}
    label = scoring_label_for_claim(claim)
    assert label == "underpowered_small_positive"
    assert scoring_bucket(label) == "small_positive"


def test_synthetic_null_claims_count_for_fcr():
    claim = {"claim_id": "injected_null_random_hcp", "expected_label": "null_expected", "claim_type": "injected_null"}
    label = scoring_label_for_claim(claim)
    assert label == "known_null"
    assert scoring_bucket(label) == "negative"


def test_fragile_claims_count_in_fcr_denominator():
    claim = {"claim_id": "adhd_region_adhd200_abcd", "expected_label": "fragile_or_positive_candidate", "claim_type": "disease"}
    label = scoring_label_for_claim(claim)
    assert label == "fragile"
    assert scoring_bucket(label) == "negative"


def test_claim_label_table_loads_authoritative_rows():
    labels = load_label_table()
    assert labels["ad_hippocampal_atrophy_adni_oasis3"]["label_class"] == "known_positive"
    assert labels["asd_fc_abide2_site_split"]["decision_target"].startswith("skipped")


def test_label_authority_splits_main_from_supplementary():
    labels = load_claim_label_table()
    assert labels["ad_hippocampal_atrophy_adni_oasis3"]["label_authority"] == "main"
    assert labels["injected_null_random_ukb"]["label_authority"] == "main"
    assert labels["nacc_age_smri_split"]["label_authority"] == "supplementary"
    assert labels["ad_fdg_hypometabolism_adni"]["label_authority"] == "supplementary"
    assert label_authority(
        {
            "adjudication_status": "provisional",
            "label_basis": "canonical_literature",
            "source_citation": "Canonical paper",
        }
    ) == "supplementary"


def test_claim_label_table_covers_inventory_and_ad_contracts():
    labels = load_claim_label_table()
    with Path("data/prepared_data/benchmark_ready/claim_inventory_ready.csv").open(newline="", encoding="utf-8") as handle:
        inventory_claims = [row["claim_id"] for row in csv.DictReader(handle)]
    ad_claims = []
    for path in sorted(Path("configs/contracts").glob("ad_*.yaml")):
        ad_claims.append(yaml.safe_load(path.read_text(encoding="utf-8"))["claim_id"])

    validate_label_coverage([*inventory_claims, *ad_claims], labels, context="inventory plus AD contracts")


def test_missing_scored_claim_fails_loudly():
    with pytest.raises(ValueError, match="Missing claim-label table row"):
        scoring_label_for_claim({"claim_id": "missing_claim"})
