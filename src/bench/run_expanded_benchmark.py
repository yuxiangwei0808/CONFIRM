"""Run Phase-C expanded benchmark claims without mutating prepared data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bench.labels import claim_label_for_claim, label_provenance, scoring_bucket, scoring_label_for_claim
from bench.metrics import summarize_rows
from bench.run_benchmark_ready import (
    RUNGS,
    AnalysisSpec,
    PreparedLayer,
    _any_region_significant,
    _best_effect,
    _complete_required_rows,
    _confound_violation,
    _contract_with_covariates,
    _json_safe,
    _label_counts,
    _ready_fmri_claims,
    _shared_features,
    evaluate_claim,
)
from confirm.analysis import directionally_consistent, fit_effect
from confirm.brainwide import run_brainwide
from confirm.contract import (
    ClaimContract,
    ConfoundGate,
    Estimand,
    Gates,
    GroupSpec,
    MultiplicityGate,
    MultiverseGate,
    PowerGate,
    ReplicationGate,
)
from confirm.ingest.oasis3 import DEFAULT_DATA_DIR as DEFAULT_OASIS3_DATA_DIR
from confirm.ingest.oasis3 import Oasis3Adapter
from confirm.multiverse import run_brainwide_multiverse, run_multiverse
from confirm.power import power_check
from confirm.replication import replicate, replicate_brainwide
from confirm.schema import normalize_sex
from confirm.verdict import decide, decide_brainwide

SEED = 20260617


def _fmri_claim(
    claim_id: str,
    modality: str,
    claim_type: str,
    discovery: str,
    replication: str,
    predictor: str,
    outcome_family: str,
    expected: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "modality": modality,
        "claim_type": claim_type,
        "discovery_cohort": discovery,
        "replication_cohort": replication,
        "predictor_or_group": predictor,
        "outcome_family": outcome_family,
        "expected_label": expected,
        "prepared_status": "expanded_phase_c",
        "notes": notes,
        "discovery_ready": True,
        "replication_ready": True,
        "feature_ready": True,
        "shared_feature_count": None,
        "benchmark_ready": True,
    }


EXTRA_FMRI_CLAIMS = [
    _fmri_claim("sz_fc_within_cobre_fbirn", "fMRI-FC", "disease", "COBRE", "FBIRN", "diagnosis", "fc_self_descriptors", "known_positive", "Round 4 recovered NeuroMark-160 summary within-network SZ hypoconnectivity."),
    _fmri_claim("sz_fc_mean_abs_cobre_fbirn", "fMRI-FC", "disease", "COBRE", "FBIRN", "diagnosis", "fc_self_descriptors", "known_positive", "Predeclared NeuroMark-160 mean absolute FC SZ hypoconnectivity descriptor."),
    _fmri_claim("asd_fc_abide1_site_split", "fMRI-FC", "disease", "ABIDE1", "ABIDE1", "diagnosis", "fc_self_descriptors", "fragile", "Round 4 recovered ABIDE1 ASD/HC real-site split."),
    _fmri_claim("asd_fc_mean_abs_abide1_site_split", "fMRI-FC", "disease", "ABIDE1", "ABIDE1", "diagnosis", "fc_self_descriptors", "fragile", "ABIDE1 ASD/HC real-site split on a predeclared mean absolute FC descriptor."),
    _fmri_claim("injected_null_random_ukb", "fMRI-FC", "injected_null", "UKB", "UKB", "random label", "fc_self_descriptors", "null_expected", "Expanded random-label null in UKB split-half."),
    _fmri_claim("injected_null_random_abcd", "fMRI-FC", "injected_null", "ABCD", "ABCD", "random label", "fc_self_descriptors", "null_expected", "Expanded random-label null in ABCD split-half."),
    _fmri_claim("injected_null_random_abide2", "fMRI-FC", "injected_null", "ABIDE2", "ABIDE2", "random label", "fc_self_descriptors", "null_expected", "Expanded random-label null in ABIDE2 split-half."),
    _fmri_claim("injected_null_random_adhd200", "fMRI-FC", "injected_null", "ADHD200", "ADHD200", "random label", "fc_self_descriptors", "null_expected", "Expanded random-label null in ADHD200 split-half."),
    _fmri_claim("injected_null_random_hcpaging", "fMRI-FC", "injected_null", "HCP_Aging", "HCP_Aging", "random label", "fc_self_descriptors", "null_expected", "Expanded random-label null in HCP-Aging split-half."),
    _fmri_claim("injected_null_random_cobre", "fMRI-FC", "injected_null", "COBRE", "COBRE", "random label", "fc_self_descriptors", "null_expected", "Expanded random-label null in COBRE split-half."),
    _fmri_claim("injected_null_random_fbirn", "fMRI-FC", "injected_null", "FBIRN", "FBIRN", "random label", "fc_self_descriptors", "null_expected", "Expanded random-label null in FBIRN split-half."),
    _fmri_claim("injected_null_site_hcp", "fMRI-FC", "injected_null", "HCP", "HCP", "feature-derived synthetic label", "fc_self_descriptors", "null_expected", "Expanded injected-confound null in HCP split-half."),
    _fmri_claim("injected_null_site_hcpaging", "fMRI-FC", "injected_null", "HCP_Aging", "HCP_Aging", "feature-derived synthetic label", "fc_self_descriptors", "null_expected", "Expanded injected-confound null in HCP-Aging split-half."),
    _fmri_claim("injected_null_site_cobre", "fMRI-FC", "injected_null", "COBRE", "COBRE", "feature-derived synthetic label", "fc_self_descriptors", "null_expected", "Expanded injected-confound null in COBRE split-half."),
    _fmri_claim("injected_null_site_gsp_smri", "sMRI", "injected_null", "GSP", "GSP", "feature-derived synthetic label", "smri_descriptors", "null_expected", "Expanded injected-confound null in GSP structural split-half."),
    _fmri_claim("injected_null_fishing_ukb", "fMRI-FC", "injected_null", "UKB", "UKB", "selected random label", "fc_self_descriptors", "null_expected", "Expanded discovery-fishing null in UKB split-half."),
    _fmri_claim("injected_null_fishing_abcd", "fMRI-FC", "injected_null", "ABCD", "ABCD", "selected random label", "fc_self_descriptors", "null_expected", "Expanded discovery-fishing null in ABCD split-half."),
    _fmri_claim("injected_null_fishing_hcp", "fMRI-FC", "injected_null", "HCP", "HCP", "selected random label", "fc_self_descriptors", "null_expected", "Expanded discovery-fishing null in HCP split-half."),
    _fmri_claim("adhd_dyno_adhd200_abcd", "fMRI-dynamics", "disease", "ADHD200", "ABCD", "ADHD/had_adhd", "ica_dyno_descriptors", "fragile_or_positive_candidate", "Expanded ADHD dynamics fragile cross-cohort claim."),
    _fmri_claim("asd_fc_abide2_internal_split", "fMRI-FC", "disease", "ABIDE2", "ABIDE2", "diagnosis", "fc_self_descriptors", "fragile_or_positive_candidate", "ABIDE2 internal split because original site labels are not in the prepared table."),
    _fmri_claim("adhd_fc_adhd200_internal_split", "fMRI-FC", "disease", "ADHD200", "ADHD200", "diagnosis", "fc_self_descriptors", "fragile_or_positive_candidate", "ADHD200 internal split because original site labels are not in the prepared table."),
]

SKIPPED_REGISTERED_CLAIMS = [
    {"claim_id": "brain_aging_atrophy_ukb_hcpaging", "reason": "registered UKB/HCP_Aging prepared cohort tables contain fMRI descriptors only; no local sMRI/eTIV columns"},
    {"claim_id": "sex_smri_ukb_hcp", "reason": "registered UKB/HCP prepared cohort tables contain fMRI descriptors only; no local eTIV-adjusted sMRI volume or thickness columns"},
    {"claim_id": "sex_smri_abcd_hcp", "reason": "registered ABCD/HCP prepared cohort tables do not provide matching local eTIV-adjusted sMRI volume or thickness columns"},
    {"claim_id": "asd_fc_abide2_site_split", "reason": "prepared ABIDE2 site column is constant; original site split cannot be recovered without fabricating labels"},
    {"claim_id": "adhd_fc_adhd200_site_split", "reason": "prepared ADHD200 site column is constant; original site split cannot be recovered without fabricating labels"},
]


@dataclass(frozen=True)
class SmriTask:
    claim_id: str
    modality: str
    claim_type: str
    notes: str
    discovery: pd.DataFrame
    replication: pd.DataFrame
    outcomes: list[str]
    unit: str
    direction: str
    covariates_full: list[str]
    covariates_min: list[str]
    replication_case_definition: str
    predictor_name: str = "dx"
    group_spec: GroupSpec | None = None
    estimand_type: str = "group_diff"
    outcome_family: str = "ADNI/OASIS3 sMRI"
    source_kind: str = "smri_disease"

    @property
    def predictor(self) -> str:
        return self.predictor_name

    @property
    def group(self) -> GroupSpec | None:
        if self.estimand_type != "group_diff":
            return None
        if self.group_spec is not None:
            return self.group_spec
        return GroupSpec(var="dx", case="Dementia", control="CN")


def _oasis3_case_definition(dementia_cdr_min: float | None) -> str:
    if dementia_cdr_min is None or dementia_cdr_min <= 0:
        return "OASIS3 Dementia: CDR>0"
    return f"OASIS3 Dementia: CDR>={dementia_cdr_min:g}; 0<CDR<{dementia_cdr_min:g} excluded"


def _load_smri(
    root: Path,
    *,
    oasis3_dementia_cdr_min: float | None = None,
    oasis3_data_dir: str | Path = DEFAULT_OASIS3_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    adni = pd.read_parquet(root / "ADNI.parquet").copy()
    if oasis3_dementia_cdr_min is None:
        oasis = pd.read_parquet(root / "OASIS3.parquet").copy()
    else:
        oasis = Oasis3Adapter(oasis3_data_dir, dementia_cdr_min=oasis3_dementia_cdr_min).to_canonical()
    for df in [adni, oasis]:
        df["subject_id"] = df["subject_id"].astype(str)
        df["cohort"] = df["cohort"].astype(str)
        df["site"] = df["site"].astype(str)
        df["sex"] = df["sex"].astype("string")
        df["dx"] = df["dx"].astype("string")
    return adni, oasis, _oasis3_case_definition(oasis3_dementia_cdr_min)


def _smri_tasks(
    root: Path,
    *,
    cluster_root: Path,
    oasis3_dementia_cdr_min: float | None = None,
    oasis3_data_dir: str | Path = DEFAULT_OASIS3_DATA_DIR,
) -> list[SmriTask]:
    adni, oasis, case_definition = _load_smri(
        root,
        oasis3_dementia_cdr_min=oasis3_dementia_cdr_min,
        oasis3_data_dir=oasis3_data_dir,
    )
    negative_regions = ["smri_hippocampus", "smri_entorhinal", "smri_fusiform", "smri_midtemp", "smri_wholebrain"]
    shared = [col for col in negative_regions if col in adni.columns and col in oasis.columns]
    tasks = [
        SmriTask(
            claim_id="ad_smri_adni_oasis3",
            modality="sMRI",
            claim_type="known_positive",
            notes="Registered ADNI/OASIS3 scalar hippocampal AD positive control.",
            discovery=adni,
            replication=oasis,
            outcomes=["smri_hippocampus"],
            unit="scalar",
            direction="negative",
            covariates_full=["age", "sex", "eTIV"],
            covariates_min=["age", "sex"],
            replication_case_definition=case_definition,
        ),
        SmriTask(
            claim_id="ad_entorhinal_atrophy_adni_oasis3",
            modality="sMRI",
            claim_type="known_positive",
            notes="AD entorhinal atrophy scalar positive control.",
            discovery=adni,
            replication=oasis,
            outcomes=["smri_entorhinal"],
            unit="scalar",
            direction="negative",
            covariates_full=["age", "sex", "eTIV"],
            covariates_min=["age", "sex"],
            replication_case_definition=case_definition,
        ),
        SmriTask(
            claim_id="ad_midtemp_atrophy_adni_oasis3",
            modality="sMRI",
            claim_type="known_positive",
            notes="AD middle temporal atrophy scalar positive control.",
            discovery=adni,
            replication=oasis,
            outcomes=["smri_midtemp"],
            unit="scalar",
            direction="negative",
            covariates_full=["age", "sex", "eTIV"],
            covariates_min=["age", "sex"],
            replication_case_definition=case_definition,
        ),
        SmriTask(
            claim_id="ad_wholebrain_atrophy_adni_oasis3",
            modality="sMRI",
            claim_type="known_positive",
            notes="AD whole-brain atrophy scalar positive control.",
            discovery=adni,
            replication=oasis,
            outcomes=["smri_wholebrain"],
            unit="scalar",
            direction="negative",
            covariates_full=["age", "sex", "eTIV"],
            covariates_min=["age", "sex"],
            replication_case_definition=case_definition,
        ),
        SmriTask(
            claim_id="ad_brainwide_smri_adni_oasis3",
            modality="sMRI",
            claim_type="known_positive",
            notes="Registered ADNI/OASIS3 brainwide AD atrophy positive control.",
            discovery=adni,
            replication=oasis,
            outcomes=shared,
            unit="brainwide",
            direction="negative",
            covariates_full=["age", "sex", "eTIV"],
            covariates_min=["age", "sex"],
            replication_case_definition=case_definition,
        ),
        SmriTask(
            claim_id="ad_hippocampal_atrophy_adni_oasis3",
            modality="sMRI",
            claim_type="known_positive",
            notes="AD hippocampal atrophy scalar positive control.",
            discovery=adni,
            replication=oasis,
            outcomes=["smri_hippocampus"],
            unit="scalar",
            direction="negative",
            covariates_full=["age", "sex", "eTIV"],
            covariates_min=["age", "sex"],
            replication_case_definition=case_definition,
        ),
        SmriTask(
            claim_id="brainwide_adni_oasis_ad_signature",
            modality="sMRI",
            claim_type="known_positive",
            notes="AD regional atrophy signature positive control.",
            discovery=adni,
            replication=oasis,
            outcomes=shared,
            unit="brainwide",
            direction="negative",
            covariates_full=["age", "sex", "eTIV"],
            covariates_min=["age", "sex"],
            replication_case_definition=case_definition,
        ),
    ]
    adni_cn = adni[adni["dx"].astype(str).eq("CN")].copy()
    oasis_cn = oasis[oasis["dx"].astype(str).eq("CN")].copy()
    adni_cn["cohort"] = "ADNI_CN"
    oasis_cn["cohort"] = "OASIS3_CN"
    tasks.append(
        SmriTask(
            claim_id="brain_aging_hippocampus_adni_oasis3_cn",
            modality="sMRI",
            claim_type="known_positive",
            notes="CN-only hippocampal atrophy with age positive control.",
            discovery=adni_cn,
            replication=oasis_cn,
            outcomes=["smri_hippocampus"],
            unit="scalar",
            direction="negative",
            covariates_full=["sex", "eTIV"],
            covariates_min=["sex", "eTIV"],
            replication_case_definition="OASIS3 CN only; Dementia excluded",
            predictor_name="age",
            estimand_type="association",
            outcome_family="ADNI/OASIS3 CN sMRI",
        )
    )
    tasks.extend(_sex_smri_tasks(root, cluster_root))
    return tasks


def _standardize_subject_frame(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = df.copy()
    if "session" not in out.columns:
        out["session"] = "ses-01"
    if "site" not in out.columns:
        out["site"] = cohort
    out["subject_id"] = out["subject_id"].astype(str)
    out["session"] = out["session"].fillna("ses-01").astype(str)
    out["cohort"] = cohort
    out["site"] = out["site"].fillna(cohort).astype(str)
    out["sex"] = normalize_sex(out["sex"])
    if "dx" in out.columns:
        out["dx"] = out["dx"].astype("string")
    for col in ["age", "eTIV", "smri_hippocampus", "smri_hippocampus_total"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sex_smri_tasks(root: Path, cluster_root: Path) -> list[SmriTask]:
    gsp_path = cluster_root / "GSP.parquet"
    adni_path = root / "ADNI.parquet"
    oasis_path = root / "OASIS3.parquet"
    if not gsp_path.exists() or not adni_path.exists() or not oasis_path.exists():
        return []

    gsp = _standardize_subject_frame(pd.read_parquet(gsp_path), "GSP")
    if "smri_hippocampus_total" in gsp.columns:
        gsp["smri_hippocampus"] = gsp["smri_hippocampus_total"]
    adni = _standardize_subject_frame(pd.read_parquet(adni_path), "ADNI_CN")
    adni = adni[adni["dx"].astype(str).eq("CN")].copy()
    oasis = _standardize_subject_frame(pd.read_parquet(oasis_path), "OASIS3_CN")
    oasis = oasis[oasis["dx"].astype(str).eq("CN")].copy()
    return [
        SmriTask(
            claim_id="sex_smri_hippocampus_gsp_adni_cn",
            modality="sMRI",
            claim_type="candidate_unknown",
            notes="Round 4 candidate sex contrast on shared eTIV-adjusted hippocampal volume.",
            discovery=gsp,
            replication=adni,
            outcomes=["smri_hippocampus"],
            unit="scalar",
            direction="negative",
            covariates_full=["age", "eTIV"],
            covariates_min=["age", "eTIV"],
            replication_case_definition="ADNI CN only; MCI and Dementia excluded",
            predictor_name="sex",
            group_spec=GroupSpec(var="sex", case="F", control="M"),
            outcome_family="GSP/ADNI CN sMRI",
            source_kind="smri_sex",
        ),
        SmriTask(
            claim_id="sex_smri_hippocampus_gsp_oasis3_cn",
            modality="sMRI",
            claim_type="candidate_unknown",
            notes="Round 5 candidate sex contrast on shared eTIV-adjusted hippocampal volume.",
            discovery=gsp,
            replication=oasis,
            outcomes=["smri_hippocampus"],
            unit="scalar",
            direction="negative",
            covariates_full=["age", "eTIV"],
            covariates_min=["age", "eTIV"],
            replication_case_definition="OASIS3 CN only; Dementia excluded",
            predictor_name="sex",
            group_spec=GroupSpec(var="sex", case="F", control="M"),
            outcome_family="GSP/OASIS3 CN sMRI",
            source_kind="smri_sex",
        ),
    ]


def _build_smri_contract(task: SmriTask, harmonize: str) -> ClaimContract:
    outcome: str | list[str] = task.outcomes[0] if task.unit == "scalar" else task.outcomes
    estimand = Estimand(
        type=task.estimand_type,  # type: ignore[arg-type]
        outcome=outcome,
        predictor=task.predictor,
        group=task.group,
        direction=task.direction,  # type: ignore[arg-type]
        unit=task.unit,  # type: ignore[arg-type]
        region_set="ad_signature" if task.unit == "brainwide" else None,
    )
    gates = Gates(
        multiplicity=MultiplicityGate(method="fdr_bh", alpha=0.05, family_size=max(1, len(task.outcomes))),
        confound=ConfoundGate(require_covariates=task.covariates_full, motion_check=False),
        power=PowerGate(min_power=0.8, ref_effect=None),
        multiverse=MultiverseGate(min_fraction_consistent=0.6),
        replication=ReplicationGate(
            alpha=0.05,
            require_same_sign=True,
            require_ci_overlap=False,
            harmonize=harmonize,  # type: ignore[arg-type]
            pattern_corr_min=0.5,
            region_replication_frac_min=0.5,
            dice_min=0.0,
        ),
    )
    return ClaimContract(
        claim_id=task.claim_id,
        question=task.notes,
        estimand=estimand,
        covariates=task.covariates_full,
        inclusion=None,
        discovery_cohort=str(task.discovery["cohort"].iloc[0]),
        replication_cohorts=[str(task.replication["cohort"].iloc[0])],
        gates=gates,
        reporting_language_allowed=["confirmed", "non_replicated", "under_powered", "fragile"],
    )


def _label_fields(claim_id: str) -> tuple[str, str, dict[str, str]]:
    label_row = claim_label_for_claim({"claim_id": claim_id}) or {}
    scoring_label = scoring_label_for_claim({"claim_id": claim_id})
    return scoring_label, scoring_bucket(scoring_label), label_row


def _scalar_significant(effect: Any, contract: ClaimContract) -> bool:
    return bool(effect.p <= contract.gates.multiplicity.alpha and directionally_consistent(effect.beta, contract))


def _base_smri_result(task: SmriTask, contract: ClaimContract, disc: pd.DataFrame, rep: pd.DataFrame) -> dict[str, Any]:
    scoring_label, bucket, label_row = _label_fields(task.claim_id)
    return {
        "claim_id": task.claim_id,
        "modality": task.modality,
        "claim_type": task.claim_type,
        "ground_truth": scoring_label,
        "scoring_label": scoring_label,
        "scoring_bucket": bucket,
        "label_provenance": label_provenance(scoring_label),
        "label_basis": label_row.get("label_basis"),
        "adjudication_status": label_row.get("adjudication_status"),
        "label_authority": label_row.get("label_authority", "supplementary"),
        "label_confidence": label_row.get("label_confidence"),
        "source_citation": label_row.get("source_citation"),
        "label_metadata": label_row,
        "expected_label": scoring_label,
        "outcome_family": task.outcome_family,
        "source_kind": task.source_kind,
        "discovery_cohort": str(disc["cohort"].iloc[0]),
        "replication_cohort": str(rep["cohort"].iloc[0]),
        "n_discovery": int(len(disc)),
        "n_replication": int(len(rep)),
        "label_counts_discovery": _label_counts(disc, AnalysisSpec("group_diff", task.predictor, task.group, task.direction, task.covariates_full, task.covariates_min)),
        "label_counts_replication": _label_counts(rep, AnalysisSpec("group_diff", task.predictor, task.group, task.direction, task.covariates_full, task.covariates_min)),
        "replication_case_definition": task.replication_case_definition,
        "n_features": int(len(task.outcomes)),
        "covariates_full": task.covariates_full,
        "covariates_min": task.covariates_min,
        "contract": contract.model_dump(mode="json"),
    }


def evaluate_smri_task(task: SmriTask, harmonize: str) -> dict[str, Any]:
    if not task.outcomes:
        raise ValueError("No shared sMRI outcomes")
    spec = AnalysisSpec(task.estimand_type, task.predictor, task.group, task.direction, task.covariates_full, task.covariates_min)
    disc = _complete_required_rows(task.discovery, task.outcomes, spec)
    rep = _complete_required_rows(task.replication, task.outcomes, spec)
    if len(disc) < 20 or len(rep) < 20:
        raise ValueError(f"Too few complete rows after filtering: discovery={len(disc)}, replication={len(rep)}")
    contract = _build_smri_contract(task, harmonize)
    min_contract = _contract_with_covariates(contract, task.covariates_min)
    base = _base_smri_result(task, contract, disc, rep)
    confound_valid = not _confound_violation(disc, spec)

    if task.unit == "scalar":
        primary = fit_effect(disc, contract, covariates=contract.covariates, model="ols")
        min_primary = fit_effect(disc, min_contract, covariates=min_contract.covariates, model="ols")
        power = power_check(primary, contract)
        multiverse = run_multiverse(disc, contract)
        replication = replicate(primary, disc, [rep], contract)
        verdict = decide(primary, multiverse, power, replication, contract)
        exec_only = _scalar_significant(min_primary, min_contract)
        confound = _scalar_significant(primary, contract) and confound_valid
        best_region = task.outcomes[0]
        primary_sig_regions = int(_scalar_significant(primary, contract))
        primary_region_table = None
    else:
        min_regions = run_brainwide(disc, min_contract)
        full_regions = run_brainwide(disc, contract)
        power = power_check(_best_effect(full_regions), contract)
        multiverse = run_brainwide_multiverse(disc, full_regions, contract, min_covariates=task.covariates_min)
        replication = replicate_brainwide(full_regions, disc, [rep], contract)
        verdict = decide_brainwide(full_regions, multiverse, power, replication, contract)
        exec_only = _any_region_significant(min_regions)
        confound = _any_region_significant(full_regions) and confound_valid
        primary = _best_effect(full_regions)
        best_region = next(region.region for region in sorted(full_regions.regions, key=lambda item: item.effect.p))
        primary_sig_regions = int(sum(region.significant for region in full_regions.regions))
        primary_region_table = full_regions.to_dict()

    power_pass = confound and not power.under_powered
    multiverse_pass = power_pass and multiverse.passed
    replication_pass = multiverse_pass and replication.passed
    final_label = verdict.label
    rationale = verdict.rationale
    if not confound_valid:
        final_label = "fragile"
        rationale = "Failed gates: confound; predictor is nested in a declared confound."

    base.update(
        {
            "best_region": best_region,
            "best_beta": float(primary.beta),
            "best_p": float(primary.p),
            "best_standardized_effect": float(primary.standardized_effect),
            "primary_effect": primary.to_dict(),
            "primary_region_table": primary_region_table,
            "primary_sig_regions": primary_sig_regions,
            "confound_valid": bool(confound_valid),
            "multiverse_fraction_consistent": float(multiverse.fraction_consistent),
            "multiverse_specs": [item.to_dict() for item in multiverse.specs],
            "exec_only": bool(exec_only),
            "+confound": bool(confound),
            "+power": bool(power_pass),
            "+multiverse": bool(multiverse_pass),
            "+replication": bool(replication_pass),
            "final_label": final_label,
            "confirmation_subtype": verdict.confirmation_subtype,
            "heterogeneity_i2": verdict.heterogeneity_i2,
            "abstained": bool(final_label != "confirmed"),
            "rationale": rationale,
            "power": power.to_dict(),
            "replication": replication.to_dict(),
        }
    )
    return base


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lockfile(args: argparse.Namespace, claims: list[dict[str, Any]], skipped: list[dict[str, str]]) -> dict[str, Any]:
    source_files = [
        Path(args.data_root) / "claim_inventory_ready.csv",
        Path(args.data_root) / "feature_dictionary.csv",
        Path(args.smri_root) / "ADNI.parquet",
        Path(args.smri_root) / "OASIS3.parquet",
        Path(args.cluster_root) / "ABIDE1.parquet",
        Path(args.cluster_root) / "COBRE.parquet",
        Path(args.cluster_root) / "FBIRN.parquet",
        Path(args.cluster_root) / "GSP.parquet",
        Path("data/labels/claim_label_table.csv"),
    ]
    code_files = [
        Path("src/bench/labels.py"),
        Path("src/bench/metrics.py"),
        Path("src/bench/run_benchmark_ready.py"),
        Path("src/bench/run_expanded_benchmark.py"),
        Path("src/bench/injected_nulls.py"),
        Path("src/confirm/analysis.py"),
        Path("src/confirm/contract.py"),
        Path("src/confirm/ingest/oasis3.py"),
        Path("src/confirm/verdict.py"),
        Path("src/confirm/replication.py"),
        Path("src/confirm/results.py"),
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "harmonize": args.harmonize,
        "oasis3_dementia_cdr_min": args.oasis3_dementia_cdr_min,
        "cluster_root": str(args.cluster_root),
        "feature_limit": args.feature_limit,
        "source_files": [{"path": str(path), "sha256": _file_sha256(path)} for path in source_files if path.exists()],
        "code_files": [{"path": str(path), "sha256": _file_sha256(path)} for path in code_files if path.exists()],
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "scoring_label": claim["scoring_label"],
                "scoring_bucket": claim["scoring_bucket"],
                "label_authority": claim.get("label_authority"),
                "discovery_cohort": claim["discovery_cohort"],
                "replication_cohort": claim["replication_cohort"],
                "replication_case_definition": claim.get("replication_case_definition"),
                "n_features": claim["n_features"],
                "reported_feature_count": claim.get("reported_feature_count"),
                "search_provenance": claim.get("search_provenance"),
                "covariates_full": claim["covariates_full"],
                "covariates_min": claim["covariates_min"],
            }
            for claim in claims
        ],
        "skipped": skipped,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"expanded_benchmark_results_{timestamp}.json"
    csv_path = out_dir / f"expanded_benchmark_claims_{timestamp}.csv"
    audit_path = out_dir / f"expanded_benchmark_audit_{timestamp}.csv"
    risk_path = out_dir / f"expanded_benchmark_risk_coverage_{timestamp}.csv"
    lock_path = out_dir / f"expanded_benchmark_lockfile_{timestamp}.json"
    latest_json = out_dir / "expanded_benchmark_results.json"
    latest_csv = out_dir / "expanded_benchmark_claims.csv"
    latest_audit = out_dir / "expanded_benchmark_audit.csv"
    latest_risk = out_dir / "expanded_benchmark_risk_coverage.csv"
    latest_lock = out_dir / "expanded_benchmark_lockfile.json"

    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    pd.DataFrame(payload["claims"]).to_csv(csv_path, index=False)
    audit_rows = []
    for claim in payload["claims"]:
        rep = claim.get("replication", {})
        heterogeneity = rep.get("heterogeneity") if isinstance(rep, dict) else None
        audit_rows.append(
            {
                "claim_id": claim["claim_id"],
                "modality": claim["modality"],
                "scoring_label": claim["scoring_label"],
                "scoring_bucket": claim["scoring_bucket"],
                "label_basis": claim.get("label_basis"),
                "adjudication_status": claim.get("adjudication_status"),
                "label_authority": claim.get("label_authority"),
                "final_label": claim["final_label"],
                "confirmation_subtype": claim.get("confirmation_subtype"),
                "confirmation_i2": claim.get("heterogeneity_i2"),
                "exec_only": claim["exec_only"],
                "+confound": claim["+confound"],
                "+power": claim["+power"],
                "+multiverse": claim["+multiverse"],
                "+replication": claim["+replication"],
                "n_discovery": claim["n_discovery"],
                "n_replication": claim["n_replication"],
                "replication_case_definition": claim.get("replication_case_definition"),
                "n_features": claim["n_features"],
                "reported_feature_count": claim.get("reported_feature_count"),
                "search_family_size": (claim.get("search_provenance") or {}).get("family_size"),
                "search_selection": (claim.get("search_provenance") or {}).get("selection"),
                "best_region": claim["best_region"],
                "best_beta": claim["best_beta"],
                "best_p": claim["best_p"],
                "best_standardized_effect": claim["best_standardized_effect"],
                "multiverse_fraction_consistent": claim["multiverse_fraction_consistent"],
                "replication_passed": rep.get("passed") if isinstance(rep, dict) else None,
                "replication_reason": rep.get("reason") if isinstance(rep, dict) else None,
                "heterogeneity_i2": heterogeneity.get("i2") if isinstance(heterogeneity, dict) else None,
                "replication_confirmation_subtype": rep.get("confirmation_subtype") if isinstance(rep, dict) else None,
                "replicated_but_heterogeneous": rep.get("replicated_but_heterogeneous") if isinstance(rep, dict) else None,
                "leakage_audit": json.dumps(_json_safe(claim.get("leakage_audit")), sort_keys=True),
                "rationale": claim["rationale"],
            }
        )
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    pd.DataFrame(payload.get("risk_coverage", [])).to_csv(risk_path, index=False)
    lock_path.write_text(json.dumps(_json_safe(payload["lockfile"]), indent=2), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(csv_path, latest_csv)
    shutil.copyfile(audit_path, latest_audit)
    shutil.copyfile(risk_path, latest_risk)
    shutil.copyfile(lock_path, latest_lock)
    return json_path, csv_path, latest_json


def _wanted(claim_id: str, claim_ids: list[str] | None) -> bool:
    return not claim_ids or claim_id in set(claim_ids)


def run(args: argparse.Namespace) -> dict[str, Any]:
    layer = PreparedLayer.load(args.data_root)
    base_claims = _ready_fmri_claims(layer.claim_inventory)
    extra_claims = pd.DataFrame(EXTRA_FMRI_CLAIMS)
    claims = pd.concat([base_claims, extra_claims], ignore_index=True, sort=False)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = [
        item for item in SKIPPED_REGISTERED_CLAIMS if _wanted(item["claim_id"], args.claim_id)
    ]

    for _, row in claims.iterrows():
        claim_id = str(row["claim_id"])
        if not _wanted(claim_id, args.claim_id):
            continue
        if args.limit and len(rows) >= args.limit:
            break
        try:
            if not _shared_features(layer, row, args.feature_limit):
                skipped.append({"claim_id": claim_id, "reason": "no_shared_features"})
                print(f"[skip] {claim_id}: no shared features")
                continue
            result = evaluate_claim(
                layer,
                row,
                feature_limit=args.feature_limit,
                seed=args.seed,
                harmonize=args.harmonize,
            )
            rows.append(result)
            print(f"[ok] {result['claim_id']} final={result['final_label']} features={result['n_features']}")
        except Exception as exc:
            errors.append({"claim_id": claim_id, "error": str(exc)})
            print(f"[error] {claim_id}: {exc}")

    for task in _smri_tasks(
        Path(args.smri_root),
        cluster_root=Path(args.cluster_root),
        oasis3_dementia_cdr_min=args.oasis3_dementia_cdr_min,
        oasis3_data_dir=args.oasis3_data_dir,
    ):
        if not _wanted(task.claim_id, args.claim_id):
            continue
        if args.limit and len(rows) >= args.limit:
            break
        try:
            result = evaluate_smri_task(task, args.harmonize)
            rows.append(result)
            print(f"[ok] {result['claim_id']} final={result['final_label']} features={result['n_features']}")
        except Exception as exc:
            errors.append({"claim_id": task.claim_id, "error": str(exc)})
            print(f"[error] {task.claim_id}: {exc}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": {
            "data_root": str(args.data_root),
            "smri_root": str(args.smri_root),
            "feature_limit": args.feature_limit,
            "limit": args.limit,
            "claim_id": args.claim_id,
            "seed": args.seed,
            "harmonize": args.harmonize,
            "oasis3_dementia_cdr_min": args.oasis3_dementia_cdr_min,
            "oasis3_data_dir": str(args.oasis3_data_dir),
            "cluster_root": str(args.cluster_root),
        },
        **summarize_rows(rows, RUNGS),
        "claims": rows,
        "errors": errors,
        "skipped": skipped,
    }
    payload["lockfile"] = _lockfile(args, rows, skipped)
    json_path, csv_path, latest_json = write_outputs(Path(args.out_dir), payload)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"latest {latest_json}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/prepared_data/benchmark_ready")
    parser.add_argument("--smri-root", default="data/prepared_data/smri_disease")
    parser.add_argument("--out-dir", default="review-stage/expanded-core-combat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--claim-id", action="append", default=None)
    parser.add_argument("--feature-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--harmonize", choices=["none", "combat"], default="combat")
    parser.add_argument("--oasis3-dementia-cdr-min", type=float, default=None)
    parser.add_argument("--oasis3-data-dir", default=str(DEFAULT_OASIS3_DATA_DIR))
    parser.add_argument("--cluster-root", default="data/prepared_data/cluster_recovered")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
