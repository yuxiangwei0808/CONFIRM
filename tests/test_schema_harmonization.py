from __future__ import annotations

import pandas as pd
import pytest

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.contract import ClaimContract
from confirm.derived_columns import cohort_base
from confirm.schema import idp_columns, validate_canonical


def _nacc_style_frame(cohort: str, n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"{cohort}-{idx}" for idx in range(n)],
            "cohort": ["NACC"] * n,
            "site": [f"site-{idx % 4}" for idx in range(n)],
            "age": [70 + idx % 8 for idx in range(n)],
            "sex": ["F", "M"] * (n // 2),
            "dx": ["AD"] * (n // 2) + ["CN"] * (n // 2),
            "smri_icv": [1400.0 + idx for idx in range(n)],
            "smri_midtemporal": [20.0 - 0.1 * idx for idx in range(n)],
            "smri_lateralventricle": [30.0 + 0.1 * idx for idx in range(n)],
            "smri_hippocampus": [8.0 - 0.05 * idx for idx in range(n)],
        }
    )


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "nacc_alias_contract",
            "question": "Is middle-temporal volume lower in AD than CN?",
            "estimand": {
                "type": "group_diff",
                "outcome": "smri_midtemp",
                "predictor": "confirm_dx",
                "group": {"var": "confirm_dx", "case": "case", "control": "control"},
                "direction": "negative",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["age", "sex", "eTIV"],
            "inclusion": None,
            "discovery_cohort": "NACC_EXTERNAL_DISC",
            "replication_cohorts": ["NACC_EXTERNAL_REP"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {
                    "require_covariates": ["age", "sex", "eTIV"],
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
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def test_validate_canonical_exposes_structural_aliases_without_counting_alias_idps_twice():
    canonical = validate_canonical(_nacc_style_frame("NACC"))

    assert canonical["eTIV"].equals(canonical["smri_icv"])
    assert canonical["smri_midtemp"].equals(canonical["smri_midtemporal"])
    assert canonical["smri_ventricles"].equals(canonical["smri_lateralventricle"])
    assert "smri_midtemp" in idp_columns(canonical.columns)
    assert "smri_midtemporal" not in idp_columns(canonical.columns)
    assert "smri_lateralventricle" not in idp_columns(canonical.columns)
    assert "smri_icv" not in idp_columns(canonical.columns)


def test_analysis_mode_drops_invalid_demographics_while_default_validation_remains_strict():
    frame = _nacc_style_frame("NACC")
    frame["age"] = frame["age"].astype(object)
    frame.loc[0, "age"] = "unknown"
    frame.loc[1, "sex"] = "unknown"

    with pytest.raises(ValueError):
        validate_canonical(frame)

    canonical = validate_canonical(frame, drop_invalid_demographics=True)

    assert len(canonical) == len(frame) - 2
    assert canonical["age"].notna().all()
    assert canonical["sex"].notna().all()


def test_candidate_preflight_resolves_canonical_smri_names_from_external_parquet(tmp_path):
    _nacc_style_frame("disc").to_parquet(tmp_path / "NACC_EXTERNAL_DISC.parquet", index=False)
    _nacc_style_frame("rep").to_parquet(tmp_path / "NACC_EXTERNAL_REP.parquet", index=False)
    context = CandidatePreflightContext.from_roots([tmp_path])

    result = context.validate_contract(_contract())
    catalog = context.prompt_catalog(_contract())

    assert result.ok, result.violations
    assert result.resolved_outcome_columns["NACC_EXTERNAL_DISC"] == ["smri_midtemp"]
    assert "smri_midtemp" in catalog["common_outcome_columns_sample"]
    assert "smri_midtemporal" not in catalog["common_outcome_columns_sample"]
    assert "eTIV" in catalog["resolved_parent_cohorts"]["NACC_EXTERNAL_DISC"]["columns"]


def test_virtual_diagnosis_mapping_ignores_modality_and_evidence_suffixes():
    assert cohort_base("PK_MPRC_fMRI_EXTERNAL_DISC") == "PK_MPRC"
    assert cohort_base("AIBL_sMRI_EXTERNAL_REP") == "AIBL"
