from __future__ import annotations

import json

import pandas as pd

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.contract import ClaimContract
from confirm.excluded_evidence import mapped_contract_for_evidence
from confirm.evidence_partitions import (
    EvidencePartitionManifest,
    EvidencePartitionRecord,
    ExternalEvidenceSetRecord,
    _filter_valid_evaluation_rows,
    build_evidence_partitions,
    canonical_base_cohort,
    infer_target_family,
    validate_manifest_no_overlap,
)


def _frame(cohort: str, n: int = 600) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"{cohort}_{idx}" for idx in range(n)],
            "cohort": [cohort] * n,
            "site": [f"site{idx % 6}" for idx in range(n)],
            "age": [65 + idx % 8 for idx in range(n)],
            "sex": ["F", "M"] * (n // 2),
            "dx": ["Dementia"] * (n // 2) + ["CN"] * (n // 2),
            "smri_hippocampus": [float(idx) for idx in range(n)],
        }
    )


def _contract(**overrides) -> ClaimContract:
    data = {
        "claim_id": "ad_claim",
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
    data.update(overrides)
    return ClaimContract.model_validate(data)


def _write_config(tmp_path) -> tuple[object, object]:
    source = tmp_path / "source"
    source.mkdir()
    for cohort in ("ADNI", "OASIS3", "NACC"):
        _frame(cohort).to_parquet(source / f"{cohort}.parquet")
    config = {
        "seed": 7,
        "output_root": str(tmp_path / "parts"),
        "default_split": {"discovery": 0.6, "replication": 0.2, "holdout": 0.2},
        "min_rows": {"default_partition_rows": 5, "continuous_rows": 5},
        "datasets": [
            {
                "dataset": "ADNI",
                "source": str(source / "ADNI.parquet"),
                "target_families": ["ad_aging"],
                "split_method": "site_or_stratified",
            },
            {
                "dataset": "OASIS3",
                "source": str(source / "OASIS3.parquet"),
                "target_families": ["ad_aging"],
                "split_method": "site_or_stratified",
            },
            {
                "dataset": "NACC",
                "source": str(source / "NACC.parquet"),
                "target_families": ["ad_aging"],
                "external_eval": True,
                "split_method": "site_or_stratified",
            },
        ],
    }
    path = tmp_path / "config.yml"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, tmp_path / "parts"


def test_evidence_partition_builder_writes_non_overlapping_manifest_and_reserves_nacc(tmp_path):
    config_path, out_root = _write_config(tmp_path)

    manifest = build_evidence_partitions(config_path, out_root)

    assert not validate_manifest_no_overlap(manifest)
    assert {"ADNI_DISC", "ADNI_REP", "ADNI_HOLDOUT", "NACC_EXTERNAL_DISC", "NACC_EXTERNAL_REP"}.issubset(
        manifest.partition_ids()
    )
    assert "NACC" in manifest.external_only_bases()
    assert manifest.has_excluded_evidence_for_contract(_contract())


def test_evaluation_partition_filter_drops_invalid_age_or_sex_rows():
    frame = pd.DataFrame(
        {
            "subject_id": ["ok", "bad_age", "bad_sex"],
            "cohort": ["C"] * 3,
            "site": ["s"] * 3,
            "age": ["60", "unknown", "61"],
            "sex": ["Female", "M", "unknown"],
            "smri_hippocampus": [1.0, 2.0, 3.0],
        }
    )

    filtered = _filter_valid_evaluation_rows(frame)

    assert filtered["subject_id"].tolist() == ["ok"]
    assert filtered["age"].tolist() == [60.0]
    assert filtered["sex"].tolist() == ["F"]


def test_holdout_partition_resolves_to_actual_split_not_base_alias(tmp_path):
    config_path, out_root = _write_config(tmp_path)
    build_evidence_partitions(config_path, out_root)
    context = CandidatePreflightContext.from_roots([out_root / "cohorts"])

    holdout = context.resolve("ADNI_HOLDOUT")
    base = context.resolve("ADNI")

    assert holdout is not None
    assert holdout.path.endswith("ADNI_HOLDOUT.parquet")
    assert base is None


def test_target_family_and_base_cohort_helpers_for_ad_claim():
    contract = _contract(discovery_cohort="ADNI_DISC", replication_cohorts=["OASIS3_REP"])

    assert canonical_base_cohort(contract.discovery_cohort) == "ADNI"
    assert infer_target_family(contract) == "ad_aging"


def test_manifest_maps_contract_to_distinct_holdout_paths_outcome_blind(tmp_path):
    config_path, out_root = _write_config(tmp_path)
    manifest = build_evidence_partitions(config_path, out_root)
    mapped, discovery, replications, evidence_set_id = mapped_contract_for_evidence(
        _contract(), manifest, "holdout"
    )

    assert mapped.discovery_cohort == "ADNI_HOLDOUT_DISC"
    assert mapped.replication_cohorts == ["OASIS3_HOLDOUT_REP"]
    assert discovery.path.endswith("ADNI_HOLDOUT_DISC.parquet")
    assert replications[0].path.endswith("OASIS3_HOLDOUT_REP.parquet")
    assert evidence_set_id is None


def test_same_base_contract_uses_distinct_holdout_evaluation_pair(tmp_path):
    config_path, out_root = _write_config(tmp_path)
    manifest = build_evidence_partitions(config_path, out_root)
    contract = _contract(discovery_cohort="ADNI_DISC", replication_cohorts=["ADNI_REP"])

    pair = manifest.holdout_evaluation_pair_for_contract(contract)

    assert pair is not None
    discovery, replications = pair
    assert discovery.partition_id == "ADNI_HOLDOUT_DISC"
    assert replications[0].partition_id == "ADNI_HOLDOUT_REP"
    assert discovery.path != replications[0].path
    assert manifest.has_excluded_evidence_for_contract(contract)


def test_holdout_mapping_deduplicates_repeated_replication_bases(tmp_path):
    config_path, out_root = _write_config(tmp_path)
    manifest = build_evidence_partitions(config_path, out_root)
    contract = _contract(
        discovery_cohort="ADNI_DISC",
        replication_cohorts=["OASIS3_DISC", "ADNI_REP", "OASIS3_REP"],
    )

    pair = manifest.holdout_evaluation_pair_for_contract(contract)

    assert pair is not None
    discovery, replications = pair
    assert discovery.partition_id == "ADNI_HOLDOUT_DISC"
    assert [record.partition_id for record in replications] == [
        "OASIS3_HOLDOUT_REP",
        "ADNI_HOLDOUT_REP",
    ]
    assert len({discovery.path, *[record.path for record in replications]}) == 3


def test_contract_that_already_used_holdout_cannot_reuse_internal_holdout(tmp_path):
    config_path, out_root = _write_config(tmp_path)
    manifest = build_evidence_partitions(config_path, out_root)
    contract = _contract(
        discovery_cohort="ADNI_HOLDOUT",
        replication_cohorts=["ADNI_REP"],
    )

    assert manifest.holdout_evaluation_pair_for_contract(contract) is None


def _external_record(
    partition_id: str,
    base_dataset: str,
    evaluation_role: str,
    *,
    group_levels: dict[str, list[str]] | None = None,
    columns: list[str] | None = None,
) -> EvidencePartitionRecord:
    return EvidencePartitionRecord(
        partition_id=partition_id,
        base_dataset=base_dataset,
        target_family="ad_aging",
        role="external_eval",
        evaluation_role=evaluation_role,
        path=f"/{partition_id}.parquet",
        source_path=f"/{base_dataset}.parquet",
        split_method="test",
        seed=7,
        n_rows=100,
        site_count=2,
        exclusion_role="excluded_evaluation",
        subject_id_sha256=partition_id,
        source_row_count=200,
        columns=columns
        or [
            "subject_id",
            "cohort",
            "site",
            "age",
            "sex",
            "dx",
            "smri_hippocampus",
        ],
        modality="sMRI",
        feature_families=["regional_volume"],
        units={"smri_*": "mm3"},
        group_levels=group_levels or {"dx": ["CN", "Dementia"]},
    )


def _evidence_set(
    evidence_set_id: str,
    base_dataset: str,
    *,
    priority: int,
    role: str,
) -> ExternalEvidenceSetRecord:
    return ExternalEvidenceSetRecord(
        evidence_set_id=evidence_set_id,
        target_family="ad_aging",
        modality="sMRI",
        feature_family="regional_volume",
        discovery_partition_id=f"{base_dataset}_EXTERNAL_DISC",
        replication_partition_ids=[f"{base_dataset}_EXTERNAL_REP"],
        supported_predictors=["dx"],
        supported_group_vars=["dx"],
        priority=priority,
        confirmation_role=role,
        units={"smri_*": "mm3"},
    )


def test_contract_compatible_external_selection_prefers_primary_priority_not_first_pair():
    records = []
    for base in ("NACC", "BLSA", "AIBL"):
        records.extend(
            [
                _external_record(f"{base}_EXTERNAL_DISC", base, "discovery"),
                _external_record(f"{base}_EXTERNAL_REP", base, "replication"),
            ]
        )
    manifest = EvidencePartitionManifest(
        seed=7,
        records=records,
        external_evidence_sets=[
            _evidence_set("nacc_primary", "NACC", priority=20, role="primary"),
            _evidence_set("blsa_primary", "BLSA", priority=5, role="primary"),
            _evidence_set("aibl_secondary", "AIBL", priority=1, role="secondary"),
        ],
    )

    selected = manifest.external_pair_for_contract(_contract())

    assert selected is not None
    assert selected[2].evidence_set_id == "blsa_primary"
    assert [item.evidence_set_id for item in manifest.external_sets_for_contract(_contract())] == [
        "blsa_primary",
        "nacc_primary",
        "aibl_secondary",
    ]


def test_secondary_external_set_cannot_control_confirmation_but_remains_addressable_for_robustness():
    records = [
        _external_record("AIBL_EXTERNAL_DISC", "AIBL", "discovery"),
        _external_record("AIBL_EXTERNAL_REP", "AIBL", "replication"),
    ]
    secondary = _evidence_set("aibl_secondary", "AIBL", priority=1, role="secondary")
    manifest = EvidencePartitionManifest(seed=7, records=records, external_evidence_sets=[secondary])

    assert manifest.primary_external_set_for_contract(_contract()) is None
    assert manifest.external_pair_for_contract(_contract()) is None
    robustness = manifest.external_pair_for_contract(_contract(), evidence_set_id="aibl_secondary")
    assert robustness is not None
    assert robustness[2].confirmation_role == "secondary"


def test_external_selection_rejects_incompatible_group_levels_or_missing_outcome():
    records = [
        _external_record(
            "NACC_EXTERNAL_DISC",
            "NACC",
            "discovery",
            group_levels={"dx": ["AD", "CN"]},
        ),
        _external_record(
            "NACC_EXTERNAL_REP",
            "NACC",
            "replication",
            group_levels={"dx": ["AD", "CN"]},
        ),
    ]
    manifest = EvidencePartitionManifest(
        seed=7,
        records=records,
        external_evidence_sets=[_evidence_set("nacc_primary", "NACC", priority=1, role="primary")],
    )
    assert manifest.external_pair_for_contract(_contract()) is None

    for record in records:
        record.group_levels = {"dx": ["CN", "Dementia"]}
        record.columns = [column for column in record.columns if column != "smri_hippocampus"]
    assert manifest.external_pair_for_contract(_contract()) is None


def test_cnp_external_compatibility_derives_confirm_dx_from_raw_manifest_levels():
    records = [
        _external_record(
            "ds000030_EXTERNAL_DISC",
            "ds000030",
            "discovery",
            group_levels={"confirm_dx": ["control"], "dx": ["SCHZ", "CONTROL", "BIPOLAR", "ADHD"]},
        ),
        _external_record(
            "ds000030_EXTERNAL_REP",
            "ds000030",
            "replication",
            group_levels={"confirm_dx": ["control"], "dx": ["SCHZ", "CONTROL", "BIPOLAR", "ADHD"]},
        ),
    ]
    records = [record.model_copy(update={"target_family": "psychosis"}) for record in records]
    evidence_set = ExternalEvidenceSetRecord(
        evidence_set_id="psychosis_cnp_smri",
        target_family="psychosis",
        modality="sMRI",
        feature_family="regional_volume",
        discovery_partition_id="ds000030_EXTERNAL_DISC",
        replication_partition_ids=["ds000030_EXTERNAL_REP"],
        supported_predictors=["confirm_dx"],
        supported_group_vars=["confirm_dx"],
        priority=1,
    )
    contract = _contract(
        claim_id="psychosis_claim",
        question="Do psychosis cases differ from controls?",
        estimand={
            "type": "group_diff",
            "outcome": "smri_hippocampus",
            "predictor": "confirm_dx",
            "group": {"var": "confirm_dx", "case": "case", "control": "control"},
            "direction": "negative",
            "unit": "scalar",
            "region_set": None,
        },
        discovery_cohort="COBRE_DISC",
        replication_cohorts=["FBIRN_REP"],
    )
    manifest = EvidencePartitionManifest(seed=7, records=records, external_evidence_sets=[evidence_set])

    compatible = manifest.external_pair_for_contract(contract)

    assert compatible is not None
    assert compatible[2].evidence_set_id == "psychosis_cnp_smri"
