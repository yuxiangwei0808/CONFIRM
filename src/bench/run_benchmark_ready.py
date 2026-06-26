"""Run CONFIRM gate-ladder experiments on the benchmark-ready fMRI layer.

This runner consumes:

  data/prepared_data/benchmark_ready/

It intentionally does not mutate the prepared data. For each ready fMRI claim it
builds a CONFIRM brain-wide contract, runs a compact gate ladder, and writes
parseable JSON/CSV results for the review loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bench.injected_nulls import leakage_audit
from bench.labels import LABEL_PROVENANCE, claim_label_for_claim, label_provenance, scoring_bucket, scoring_label_for_claim
from bench.metrics import summarize_rows
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
from confirm.multiverse import run_brainwide_multiverse, run_multiverse
from confirm.power import power_check
from confirm.replication import replicate, replicate_brainwide
from confirm.results import EffectResult, RegionTable
from confirm.schema import normalize_sex
from confirm.verdict import decide, decide_brainwide

SEED = 20260615
RUNGS = ["exec_only", "+confound", "+power", "+multiverse", "+replication"]

FAMILY_TO_SOURCE_KIND = {
    "fc_self_descriptors": "fc_self_descriptors",
    "region_self_descriptors": "region_self_descriptors",
    "ica_dyno_descriptors": "ica_dyno_descriptors",
    "smri_descriptors": "smri_descriptors",
}

@dataclass(frozen=True)
class PreparedLayer:
    """Lightweight accessor for the prepared benchmark data layer."""

    root: Path
    cohorts_dir: Path
    recovered_dir: Path
    feature_dictionary: pd.DataFrame
    claim_inventory: pd.DataFrame

    @classmethod
    def load(cls, root: str | Path) -> "PreparedLayer":
        path = Path(root)
        feature_dictionary = pd.read_csv(path / "feature_dictionary.csv")
        claim_inventory = pd.read_csv(path / "claim_inventory_ready.csv")
        return cls(path, path / "cohorts", path.parent / "cluster_recovered", feature_dictionary, claim_inventory)

    def load_cohort(self, cohort: str) -> pd.DataFrame:
        path = self.cohorts_dir / f"{cohort}.parquet"
        if not path.exists():
            recovered_path = self.recovered_dir / f"{cohort}.parquet"
            if not recovered_path.exists():
                raise FileNotFoundError(f"Prepared cohort not found: {path} or {recovered_path}")
            path = recovered_path
        return _normalize_prepared_frame(pd.read_parquet(path))

    def feature_columns(self, cohort: str, source_kind: str) -> list[str]:
        rows = self.feature_dictionary[
            (self.feature_dictionary["cohort"] == cohort)
            & (self.feature_dictionary["source_kind"] == source_kind)
            & (self.feature_dictionary["is_feature"].astype(bool))
        ]
        columns = [str(col) for col in rows["column"].tolist()]
        if columns:
            return columns
        frame = self.load_cohort(cohort)
        if source_kind == "fc_self_descriptors":
            return sorted(col for col in frame.columns if col.startswith("fc_"))
        if source_kind == "smri_descriptors":
            return sorted(col for col in frame.columns if col.startswith("smri_"))
        return []


@dataclass(frozen=True)
class AnalysisSpec:
    """Claim-specific statistical setup after dataset harmonization."""

    estimand_type: str
    predictor: str
    group: GroupSpec | None
    direction: str
    covariates_full: list[str]
    covariates_min: list[str]
    unit: str = "brainwide"
    outcomes: list[str] | None = None
    searched_feature_count: int | None = None
    search_provenance: dict[str, Any] | None = None
    leakage_audit: dict[str, Any] | None = None


def _normalize_prepared_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "session" not in out.columns:
        out["session"] = "ses-01"
    if "site" not in out.columns:
        out["site"] = out.get("cohort", "unknown")
    out["subject_id"] = out["subject_id"].astype(str)
    out["session"] = out["session"].fillna("ses-01").astype(str)
    out["cohort"] = out["cohort"].astype(str)
    out["site"] = out["site"].fillna(out["cohort"]).astype(str)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["sex"] = normalize_sex(out["sex"])
    if "dx" in out.columns:
        out["dx"] = out["dx"].astype("string")

    feature_cols = [col for col in out.columns if col.startswith(("fc_", "raw_fmri_", "smri_"))]
    numeric_cols = feature_cols + [col for col in out.columns if col.startswith("phen_")]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col.startswith("phen_"):
            out.loc[out[col].isin([999, 999.0, -999, -999.0]), col] = np.nan
    return out


def _truth_label(row: pd.Series) -> str:
    return scoring_label_for_claim(row)


def _ready_fmri_claims(claims: pd.DataFrame) -> pd.DataFrame:
    ready = claims["benchmark_ready"].astype(bool)
    fmri = claims["modality"].astype(str).str.startswith("fMRI")
    family = claims["outcome_family"].isin(FAMILY_TO_SOURCE_KIND)
    return claims[ready & fmri & family].copy()


def _split_same_cohort(df: pd.DataFrame, seed: int, discovery: str, replication: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if "subject_id" in df.columns:
        subjects = df["subject_id"].astype(str).drop_duplicates().to_numpy()
        rng.shuffle(subjects)
        half = len(subjects) // 2
        disc_subjects = set(subjects[:half])
        disc = df[df["subject_id"].astype(str).isin(disc_subjects)].copy()
        rep = df[~df["subject_id"].astype(str).isin(disc_subjects)].copy()
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        half = len(idx) // 2
        disc = df.iloc[idx[:half]].copy()
        rep = df.iloc[idx[half:]].copy()
    disc["cohort"] = f"{discovery}_DISC"
    rep["cohort"] = f"{replication}_REP"
    return disc, rep


def _split_site_cohort(df: pd.DataFrame, seed: int, discovery: str, replication: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = sorted(str(site) for site in df["site"].dropna().astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(sites)
    site_counts = df.assign(__site=df["site"].astype(str)).groupby("__site").size().to_dict()
    left: list[str] = []
    right: list[str] = []
    left_n = 0
    right_n = 0
    for site in sites:
        if left_n <= right_n:
            left.append(site)
            left_n += int(site_counts[site])
        else:
            right.append(site)
            right_n += int(site_counts[site])
    if not left or not right:
        raise ValueError("Site split requires at least two non-empty sites")
    disc = df[df["site"].astype(str).isin(left)].copy()
    rep = df[df["site"].astype(str).isin(right)].copy()
    disc["cohort"] = f"{discovery}_DISC_SITES"
    rep["cohort"] = f"{replication}_REP_SITES"
    return disc, rep


def _load_claim_frames(layer: PreparedLayer, row: pd.Series, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = str(row["discovery_cohort"])
    replication = str(row["replication_cohort"])
    disc = layer.load_cohort(discovery)
    if str(row["claim_id"]) in {"asd_fc_abide1_site_split", "asd_fc_mean_abs_abide1_site_split"}:
        return _split_site_cohort(disc, seed, discovery, replication)
    if discovery == replication:
        return _split_same_cohort(disc, seed, discovery, replication)
    return disc, layer.load_cohort(replication)


def _shared_features(layer: PreparedLayer, row: pd.Series, feature_limit: int | None) -> list[str]:
    discovery = str(row["discovery_cohort"])
    replication = str(row["replication_cohort"])
    source_kind = FAMILY_TO_SOURCE_KIND[str(row["outcome_family"])]
    disc_cols = set(layer.feature_columns(discovery, source_kind))
    rep_cols = set(layer.feature_columns(replication, source_kind))
    shared = sorted(disc_cols & rep_cols)
    if feature_limit and feature_limit > 0:
        return shared[:feature_limit]
    return shared


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _copy_predictor(disc: pd.DataFrame, rep: pd.DataFrame, candidates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    disc_col = _first_existing(disc, candidates)
    rep_col = _first_existing(rep, candidates)
    if not disc_col or not rep_col:
        raise ValueError(f"No compatible predictor found among {candidates}")
    disc = disc.copy()
    rep = rep.copy()
    disc["bench_predictor"] = pd.to_numeric(disc[disc_col], errors="coerce")
    rep["bench_predictor"] = pd.to_numeric(rep[rep_col], errors="coerce")
    return disc, rep, "bench_predictor"


def _label_from_dx(values: pd.Series, case_values: set[str], control_values: set[str]) -> pd.Series:
    text = values.astype("string")
    out = pd.Series(pd.NA, index=values.index, dtype="string")
    out[text.isin(case_values)] = "case"
    out[text.isin(control_values)] = "control"
    return out


def _disease_labels(row: pd.Series, disc: pd.DataFrame, rep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    claim_id = str(row["claim_id"])
    disc = disc.copy()
    rep = rep.copy()
    if claim_id.startswith("ad_"):
        case = {"2", "2.0", "Dementia", "AD", "AD dementia"}
        control = {"0", "0.0", "CN", "Control"}
        disc["bench_group"] = _label_from_dx(disc["dx"], case, control)
        rep["bench_group"] = _label_from_dx(rep["dx"], case, control)
    elif claim_id.startswith("asd_") or claim_id.startswith("underpowered_asd"):
        if "phen_had_asd" in disc.columns:
            disc["bench_group"] = np.where(disc["phen_had_asd"] == 1, "case", np.where(disc["phen_had_asd"] == 0, "control", pd.NA))
        else:
            disc["bench_group"] = _label_from_dx(disc["dx"], {"1", "1.0", "ASD"}, {"2", "2.0", "0", "0.0", "HC", "Control", "TC"})
        if "phen_had_asd" in rep.columns:
            rep["bench_group"] = np.where(rep["phen_had_asd"] == 1, "case", np.where(rep["phen_had_asd"] == 0, "control", pd.NA))
        else:
            rep["bench_group"] = _label_from_dx(rep["dx"], {"1", "1.0", "ASD"}, {"2", "2.0", "0", "0.0", "HC", "Control", "TC"})
    elif claim_id.startswith("adhd_") or claim_id.startswith("underpowered_adhd"):
        if "phen_had_adhd" in disc.columns:
            disc["bench_group"] = np.where(disc["phen_had_adhd"] == 1, "case", np.where(disc["phen_had_adhd"] == 0, "control", pd.NA))
        else:
            disc["bench_group"] = _label_from_dx(disc["dx"], {"1", "1.0", "2", "2.0", "3", "3.0", "ADHD"}, {"0", "0.0", "Control"})
        if "phen_had_adhd" in rep.columns:
            rep["bench_group"] = np.where(rep["phen_had_adhd"] == 1, "case", np.where(rep["phen_had_adhd"] == 0, "control", pd.NA))
        else:
            rep["bench_group"] = _label_from_dx(rep["dx"], {"1", "1.0", "2", "2.0", "3", "3.0", "ADHD"}, {"0", "0.0", "Control"})
    elif claim_id.startswith("sz_"):
        disc["bench_group"] = _label_from_dx(disc["dx"], {"SZ", "Schizophrenia", "case", "1", "1.0"}, {"HC", "Control", "control", "0", "0.0"})
        rep["bench_group"] = _label_from_dx(rep["dx"], {"SZ", "Schizophrenia", "case", "1", "1.0"}, {"HC", "Control", "control", "0", "0.0"})
    else:
        raise ValueError(f"No disease-label rule for {claim_id}")
    disc["bench_group"] = pd.Series(disc["bench_group"], index=disc.index, dtype="string")
    rep["bench_group"] = pd.Series(rep["bench_group"], index=rep.index, dtype="string")
    return disc, rep


def _assign_random_group(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    values = np.random.default_rng(seed).choice(["case", "control"], size=len(out))
    out["bench_group"] = values
    return out


def _assign_feature_leakage_group(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out[feature], errors="coerce")
    median = x.median()
    out["synthetic_site"] = np.where(x >= median, "site_high", "site_low")
    out["bench_group"] = np.where(x >= median, "case", "control")
    out.loc[x.isna(), ["synthetic_site", "bench_group"]] = pd.NA
    return out


def _selection_contract(row: pd.Series, outcome: str, spec: AnalysisSpec, discovery_name: str, replication_name: str) -> ClaimContract:
    estimand = Estimand(
        type=spec.estimand_type,
        outcome=outcome,
        predictor=spec.predictor,
        group=spec.group,
        direction=spec.direction,  # type: ignore[arg-type]
        unit="scalar",
    )
    gates = Gates(
        multiplicity=MultiplicityGate(method="fdr_bh", alpha=0.05, family_size=1),
        confound=ConfoundGate(require_covariates=spec.covariates_full, motion_check=False),
        power=PowerGate(min_power=0.8, ref_effect=_predeclared_ref_effect(row)),
        multiverse=MultiverseGate(min_fraction_consistent=0.6),
        replication=ReplicationGate(alpha=0.05, require_same_sign=True, require_ci_overlap=False, harmonize="none"),
    )
    return ClaimContract(
        claim_id=str(row["claim_id"]),
        question=str(row.get("notes", row["claim_id"])),
        estimand=estimand,
        covariates=spec.covariates_full,
        inclusion=None,
        discovery_cohort=discovery_name,
        replication_cohorts=[replication_name],
        gates=gates,
        reporting_language_allowed=["confirmed", "non_replicated", "under_powered", "fragile"],
    )


def _select_fishing_outcome(row: pd.Series, disc: pd.DataFrame, rep: pd.DataFrame, outcomes: list[str], spec: AnalysisSpec) -> tuple[str, list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    discovery_name = str(disc["cohort"].iloc[0]) if "cohort" in disc.columns and len(disc) else str(row["discovery_cohort"])
    replication_name = str(rep["cohort"].iloc[0]) if "cohort" in rep.columns and len(rep) else str(row["replication_cohort"])
    for outcome in outcomes:
        try:
            contract = _selection_contract(row, outcome, spec, discovery_name, replication_name)
            effect = fit_effect(disc, contract, covariates=spec.covariates_full, model="ols")
            scored.append({"outcome": outcome, "p": float(effect.p), "beta": float(effect.beta), "n": int(effect.n)})
        except Exception as exc:
            scored.append({"outcome": outcome, "p": float("inf"), "beta": float("nan"), "n": 0, "error": str(exc)})
    fitted = [item for item in scored if np.isfinite(float(item["p"]))]
    if not fitted:
        raise ValueError("Fishing null could not fit any searched feature")
    best = min(fitted, key=lambda item: float(item["p"]))
    return str(best["outcome"]), scored


def _configure_injected_null(
    row: pd.Series,
    disc: pd.DataFrame,
    rep: pd.DataFrame,
    outcomes: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, AnalysisSpec]:
    claim_id = str(row["claim_id"])
    if "random" in claim_id:
        disc = _assign_random_group(disc, seed)
        rep = _assign_random_group(rep, seed + 1)
        extra = []
    elif "fishing" in claim_id:
        if not outcomes:
            raise ValueError("Fishing null requires at least one outcome feature")
        disc = _assign_random_group(disc, seed)
        rep = _assign_random_group(rep, seed + 1)
        covars_full = _available_covariates(disc, rep, ["age", "sex", "site"])
        covars_min = _available_covariates(disc, rep, ["age", "sex"])
        search_spec = AnalysisSpec(
            estimand_type="group_diff",
            predictor="bench_group",
            group=GroupSpec(var="bench_group", case="case", control="control"),
            direction="two_sided",
            covariates_full=covars_full,
            covariates_min=covars_min,
            unit="scalar",
        )
        selected, search_scores = _select_fishing_outcome(row, disc, rep, outcomes, search_spec)
        audit = leakage_audit(disc, rep, group_col="bench_group", covariates=covars_full)
        audit["selected_outcome"] = selected
        audit["searched_feature_count"] = int(len(outcomes))
        audit["top_discovery_features"] = sorted(
            search_scores,
            key=lambda item: float(item["p"]) if np.isfinite(float(item["p"])) else float("inf"),
        )[:10]
        return disc, rep, AnalysisSpec(
            estimand_type="group_diff",
            predictor="bench_group",
            group=GroupSpec(var="bench_group", case="case", control="control"),
            direction="two_sided",
            covariates_full=covars_full,
            covariates_min=covars_min,
            unit="scalar",
            outcomes=[selected],
            searched_feature_count=len(outcomes),
            search_provenance={"declared": True, "family_size": len(outcomes), "selection": "discovery_only"},
            leakage_audit=audit,
        )
    else:
        if not outcomes:
            raise ValueError("Injected null requires at least one outcome feature")
        disc = _assign_feature_leakage_group(disc, outcomes[0])
        if "site" in claim_id:
            rep = _assign_feature_leakage_group(rep, outcomes[0])
            extra = ["synthetic_site"]
        else:
            rep = _assign_random_group(rep, seed + 2)
            extra = []
    covars_full = _available_covariates(disc, rep, ["age", "sex", "site", *extra])
    covars_min = _available_covariates(disc, rep, ["age", "sex"])
    return disc, rep, AnalysisSpec(
        estimand_type="group_diff",
        predictor="bench_group",
        group=GroupSpec(var="bench_group", case="case", control="control"),
        direction="two_sided",
        covariates_full=covars_full,
        covariates_min=covars_min,
    )


def _available_covariates(disc: pd.DataFrame, rep: pd.DataFrame, desired: list[str]) -> list[str]:
    covars: list[str] = []
    for col in desired:
        if col in disc.columns and col in rep.columns and (disc[col].notna().any() or rep[col].notna().any()):
            covars.append(col)
    return list(dict.fromkeys(covars))


def _configure_claim(
    row: pd.Series,
    disc: pd.DataFrame,
    rep: pd.DataFrame,
    outcomes: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, AnalysisSpec]:
    claim_type = str(row["claim_type"])
    predictor_or_group = str(row["predictor_or_group"]).lower()
    if claim_type == "injected_null":
        return _configure_injected_null(row, disc, rep, outcomes, seed)
    if predictor_or_group == "age":
        covars_full = _available_covariates(disc, rep, ["sex", "site"])
        covars_min = _available_covariates(disc, rep, ["sex"])
        return disc, rep, AnalysisSpec("association", "age", None, "two_sided", covars_full, covars_min)
    if predictor_or_group == "sex":
        covars_full = _available_covariates(disc, rep, ["age", "site"])
        covars_min = _available_covariates(disc, rep, ["age"])
        return disc, rep, AnalysisSpec(
            "group_diff",
            "sex",
            GroupSpec(var="sex", case="F", control="M"),
            "two_sided",
            covars_full,
            covars_min,
        )
    if "fluid" in predictor_or_group or "cognition" in str(row["claim_id"]):
        disc, rep, predictor = _copy_predictor(
            disc,
            rep,
            ["phen_fluid_intelligence", "phen_fluid_composite", "phen_fluid_composite_z"],
        )
        covars_full = _available_covariates(disc, rep, ["age", "sex", "site"])
        covars_min = _available_covariates(disc, rep, ["age", "sex"])
        return disc, rep, AnalysisSpec("association", predictor, None, "two_sided", covars_full, covars_min)
    if claim_type in {"disease", "underpowered_or_fragile"}:
        disc, rep = _disease_labels(row, disc, rep)
        if str(row["claim_id"]) in {"sz_fc_within_cobre_fbirn", "sz_fc_mean_abs_cobre_fbirn"}:
            outcome = "fc_mean_abs" if str(row["claim_id"]) == "sz_fc_mean_abs_cobre_fbirn" else "fc_within_network"
            covars = _available_covariates(disc, rep, ["age", "sex"])
            return disc, rep, AnalysisSpec(
                "group_diff",
                "bench_group",
                GroupSpec(var="bench_group", case="case", control="control"),
                "negative",
                covars,
                covars,
                unit="scalar",
                outcomes=[outcome],
                searched_feature_count=1,
                search_provenance={"declared": True, "family_size": 1, "selection": "preregistered"},
            )
        if str(row["claim_id"]) == "asd_fc_mean_abs_abide1_site_split":
            covars = _available_covariates(disc, rep, ["age", "sex"])
            return disc, rep, AnalysisSpec(
                "group_diff",
                "bench_group",
                GroupSpec(var="bench_group", case="case", control="control"),
                "two_sided",
                covars,
                covars,
                unit="scalar",
                outcomes=["fc_mean_abs"],
                searched_feature_count=1,
                search_provenance={"declared": True, "family_size": 1, "selection": "preregistered"},
            )
        if str(row["claim_id"]) == "asd_fc_abide1_site_split":
            covars = _available_covariates(disc, rep, ["age", "sex"])
            return disc, rep, AnalysisSpec(
                "group_diff",
                "bench_group",
                GroupSpec(var="bench_group", case="case", control="control"),
                "two_sided",
                covars,
                covars,
            )
        if claim_type == "underpowered_or_fragile":
            disc = _balanced_subsample(disc, "bench_group", max_per_group=24, seed=seed)
            rep = _balanced_subsample(rep, "bench_group", max_per_group=24, seed=seed + 1)
        covars_full = _available_covariates(disc, rep, ["age", "sex", "site"])
        covars_min = _available_covariates(disc, rep, ["age", "sex"])
        return disc, rep, AnalysisSpec(
            "group_diff",
            "bench_group",
            GroupSpec(var="bench_group", case="case", control="control"),
            "two_sided",
            covars_full,
            covars_min,
        )
    raise ValueError(f"Unsupported claim setup: {row['claim_id']}")


def _balanced_subsample(df: pd.DataFrame, group_col: str, max_per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    rng = np.random.default_rng(seed)
    for _, group_df in df.dropna(subset=[group_col]).groupby(group_col):
        idx = np.arange(len(group_df))
        rng.shuffle(idx)
        parts.append(group_df.iloc[idx[:max_per_group]])
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True, sort=False)


def _predeclared_ref_effect(row: pd.Series) -> float | None:
    claim_type = str(row["claim_type"])
    claim_id = str(row["claim_id"])
    if claim_type == "underpowered_or_fragile":
        return 0.2
    if "cognition" in claim_id:
        return 0.03
    return None


def _build_contract(
    row: pd.Series,
    outcomes: list[str],
    spec: AnalysisSpec,
    discovery_name: str,
    replication_name: str,
    harmonize: str,
) -> ClaimContract:
    unit = spec.unit
    outcome: str | list[str] = outcomes[0] if unit == "scalar" else outcomes
    estimand = Estimand(
        type=spec.estimand_type,
        outcome=outcome,
        predictor=spec.predictor,
        group=spec.group,
        direction=spec.direction,  # type: ignore[arg-type]
        unit=unit,  # type: ignore[arg-type]
        region_set=None if unit == "scalar" else str(row["outcome_family"]),
    )
    gates = Gates(
        multiplicity=MultiplicityGate(method="fdr_bh", alpha=0.05, family_size=max(1, len(outcomes))),
        confound=ConfoundGate(require_covariates=spec.covariates_full, motion_check=False),
        power=PowerGate(min_power=0.8, ref_effect=_predeclared_ref_effect(row)),
        multiverse=MultiverseGate(min_fraction_consistent=0.6),
        replication=ReplicationGate(
            alpha=0.05,
            require_same_sign=True,
            require_ci_overlap=False,
            harmonize=harmonize,  # type: ignore[arg-type]
            pattern_corr_min=0.25,
            region_replication_frac_min=0.05,
            dice_min=0.0,
        ),
    )
    return ClaimContract(
        claim_id=str(row["claim_id"]),
        question=str(row.get("notes", row["claim_id"])),
        estimand=estimand,
        covariates=spec.covariates_full,
        inclusion=None,
        discovery_cohort=discovery_name,
        replication_cohorts=[replication_name],
        search_provenance=spec.search_provenance
        or {"declared": True, "family_size": 1, "selection": "preregistered"},
        gates=gates,
        reporting_language_allowed=["confirmed", "non_replicated", "under_powered", "fragile"],
    )


def _contract_with_covariates(contract: ClaimContract, covariates: list[str]) -> ClaimContract:
    gates = contract.gates.model_copy(update={"confound": ConfoundGate(require_covariates=covariates)})
    return contract.model_copy(update={"covariates": covariates, "gates": gates})


def _best_effect(regions: RegionTable) -> EffectResult:
    ordered = sorted(regions.regions, key=lambda region: (not region.significant, region.effect.p))
    return ordered[0].effect


def _any_region_significant(regions: RegionTable) -> bool:
    return any(region.significant for region in regions.regions)


def _scalar_significant(effect: EffectResult, contract: ClaimContract) -> bool:
    return bool(effect.p <= contract.gates.multiplicity.alpha and directionally_consistent(effect.beta, contract))


def _group_nested_in_confound(df: pd.DataFrame, group_col: str, confound_col: str) -> bool:
    data = df[[group_col, confound_col]].dropna().astype(str)
    if data.empty:
        return False
    if data[group_col].nunique(dropna=True) < 2 or data[confound_col].nunique(dropna=True) < 2:
        return False
    per_confound = data.groupby(confound_col)[group_col].nunique(dropna=True)
    per_group = data.groupby(group_col)[confound_col].nunique(dropna=True)
    return bool((per_confound <= 1).all() or (per_group <= 1).all())


def _confound_violation(df: pd.DataFrame, spec: AnalysisSpec) -> bool:
    if spec.group is None:
        return False
    group_col = spec.group.var
    if group_col not in df.columns:
        return False
    for confound_col in spec.covariates_full:
        if confound_col == group_col or confound_col not in df.columns:
            continue
        if _group_nested_in_confound(df, group_col, confound_col):
            return True
    return False


def _complete_required_rows(df: pd.DataFrame, outcomes: list[str], spec: AnalysisSpec) -> pd.DataFrame:
    needed = ["subject_id", "cohort", "site", "age", "sex", spec.predictor, *spec.covariates_full]
    if spec.group is not None:
        needed.append(spec.group.var)
    base = [col for col in dict.fromkeys(needed) if col in df.columns]
    out = df.dropna(subset=base).copy()
    if outcomes:
        out = out[out[outcomes].notna().any(axis=1)].copy()
    return out


def _label_counts(df: pd.DataFrame, spec: AnalysisSpec) -> dict[str, int]:
    if spec.group is None or spec.group.var not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[spec.group.var].value_counts(dropna=False).items()}


def evaluate_claim(
    layer: PreparedLayer,
    row: pd.Series,
    *,
    feature_limit: int | None,
    seed: int,
    harmonize: str,
) -> dict[str, Any]:
    outcomes = _shared_features(layer, row, feature_limit)
    if not outcomes:
        raise ValueError(f"No shared features for {row['claim_id']}")
    disc, rep = _load_claim_frames(layer, row, seed)
    disc, rep, spec = _configure_claim(row, disc, rep, outcomes, seed)
    reported_outcomes = spec.outcomes or outcomes
    disc = _complete_required_rows(disc, reported_outcomes, spec)
    rep = _complete_required_rows(rep, reported_outcomes, spec)
    if len(disc) < 20 or len(rep) < 20:
        raise ValueError(f"Too few complete rows after filtering: discovery={len(disc)}, replication={len(rep)}")

    contract = _build_contract(
        row,
        reported_outcomes,
        spec,
        discovery_name=str(disc["cohort"].iloc[0]),
        replication_name=str(rep["cohort"].iloc[0]),
        harmonize=harmonize,
    )
    min_contract = _contract_with_covariates(contract, spec.covariates_min)

    confound_valid = not _confound_violation(disc, spec)
    if spec.unit == "scalar":
        primary = fit_effect(disc, contract, covariates=contract.covariates, model="ols")
        min_primary = fit_effect(disc, min_contract, covariates=min_contract.covariates, model="ols")
        power = power_check(primary, contract)
        multiverse = run_multiverse(disc, contract)
        replication = replicate(primary, disc, [rep], contract)
        verdict = decide(primary, multiverse, power, replication, contract)
        exec_only = _scalar_significant(min_primary, min_contract)
        best = primary
        best_region = reported_outcomes[0]
        primary_sig_regions = int(bool(verdict.gates.get("multiplicity", False)))
        primary_region_table = None
    else:
        min_regions = run_brainwide(disc, min_contract)
        full_regions = run_brainwide(disc, contract)
        power = power_check(_best_effect(full_regions), contract)
        multiverse = run_brainwide_multiverse(disc, full_regions, contract, min_covariates=spec.covariates_min)
        replication = replicate_brainwide(full_regions, disc, [rep], contract)
        verdict = decide_brainwide(full_regions, multiverse, power, replication, contract)
        exec_only = _any_region_significant(min_regions)
        best = _best_effect(full_regions)
        best_region = next(region.region for region in sorted(full_regions.regions, key=lambda item: item.effect.p))
        primary_sig_regions = int(sum(region.significant for region in full_regions.regions))
        primary_region_table = full_regions.to_dict()

    admissible_primary = bool(verdict.gates.get("search_provenance", True)) and bool(verdict.gates.get("multiplicity", False))
    confound = admissible_primary and confound_valid
    power_pass = confound and not power.under_powered
    multiverse_pass = power_pass and multiverse.passed
    replication_pass = multiverse_pass and replication.passed
    final_label = verdict.label
    rationale = verdict.rationale
    if not confound_valid:
        final_label = "fragile"
        rationale = "Failed gates: confound; predictor is nested in a declared confound."
    scoring_label = _truth_label(row)
    bucket = scoring_bucket(scoring_label)
    label_row = claim_label_for_claim(row) or {}
    searched_feature_count = spec.searched_feature_count or len(outcomes)

    return {
        "claim_id": str(row["claim_id"]),
        "modality": str(row["modality"]),
        "claim_type": str(row["claim_type"]),
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
        "expected_label": str(row["expected_label"]),
        "outcome_family": str(row["outcome_family"]),
        "source_kind": FAMILY_TO_SOURCE_KIND[str(row["outcome_family"])],
        "discovery_cohort": str(disc["cohort"].iloc[0]),
        "replication_cohort": str(rep["cohort"].iloc[0]),
        "n_discovery": int(len(disc)),
        "n_replication": int(len(rep)),
        "label_counts_discovery": _label_counts(disc, spec),
        "label_counts_replication": _label_counts(rep, spec),
        "n_features": int(searched_feature_count),
        "reported_features": list(reported_outcomes),
        "reported_feature_count": int(len(reported_outcomes)),
        "covariates_full": spec.covariates_full,
        "covariates_min": spec.covariates_min,
        "best_region": best_region,
        "best_beta": float(best.beta),
        "best_p": float(best.p),
        "best_standardized_effect": float(best.standardized_effect),
        "primary_effect": best.to_dict(),
        "primary_region_table": primary_region_table,
        "primary_sig_regions": primary_sig_regions,
        "search_provenance": contract.search_provenance.model_dump(mode="json"),
        "leakage_audit": spec.leakage_audit,
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
        "contract": contract.model_dump(mode="json"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_rows(rows, RUNGS)


def _json_safe(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_json_safe(value) for value in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lockfile(layer: PreparedLayer, claims: list[dict[str, Any]], skipped: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    source_files = [
        layer.root / "claim_inventory_ready.csv",
        layer.root / "feature_dictionary.csv",
        layer.root / "cohort_manifest.csv",
        Path("data/labels/claim_label_table.csv"),
    ]
    code_files = [
        Path("src/bench/labels.py"),
        Path("src/bench/metrics.py"),
        Path("src/bench/run_benchmark_ready.py"),
        Path("src/confirm/brainwide.py"),
        Path("src/confirm/multiverse.py"),
        Path("src/confirm/replication.py"),
        Path("src/confirm/power.py"),
        Path("src/confirm/verdict.py"),
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "harmonize": args.harmonize,
        "feature_limit": args.feature_limit,
        "label_provenance": LABEL_PROVENANCE,
        "source_files": [{"path": str(path), "sha256": _file_sha256(path)} for path in source_files if path.exists()],
        "code_files": [{"path": str(path), "sha256": _file_sha256(path)} for path in code_files if path.exists()],
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "ground_truth": claim["ground_truth"],
                "scoring_label": claim["scoring_label"],
                "scoring_bucket": claim["scoring_bucket"],
                "label_authority": claim.get("label_authority"),
                "expected_label": claim["expected_label"],
                "label_provenance": claim["label_provenance"],
                "discovery_cohort": claim["discovery_cohort"],
                "replication_cohort": claim["replication_cohort"],
                "outcome_family": claim["outcome_family"],
                "source_kind": claim["source_kind"],
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
    json_path = out_dir / f"benchmark_ready_results_{timestamp}.json"
    csv_path = out_dir / f"benchmark_ready_claims_{timestamp}.csv"
    audit_path = out_dir / f"benchmark_ready_audit_{timestamp}.csv"
    risk_path = out_dir / f"benchmark_ready_risk_coverage_{timestamp}.csv"
    lock_path = out_dir / f"benchmark_ready_lockfile_{timestamp}.json"
    latest_json = out_dir / "benchmark_ready_results.json"
    latest_csv = out_dir / "benchmark_ready_claims.csv"
    latest_audit = out_dir / "benchmark_ready_audit.csv"
    latest_risk = out_dir / "benchmark_ready_risk_coverage.csv"
    latest_lock = out_dir / "benchmark_ready_lockfile.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    pd.DataFrame(payload["claims"]).to_csv(csv_path, index=False)
    audit_rows = []
    for claim in payload["claims"]:
        rep = claim.get("replication", {})
        audit_rows.append(
            {
                "claim_id": claim["claim_id"],
                "ground_truth": claim["ground_truth"],
                "scoring_label": claim["scoring_label"],
                "scoring_bucket": claim["scoring_bucket"],
                "label_basis": claim.get("label_basis"),
                "adjudication_status": claim.get("adjudication_status"),
                "label_authority": claim.get("label_authority"),
                "label_provenance": claim["label_provenance"],
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
                "n_features": claim["n_features"],
                "reported_feature_count": claim.get("reported_feature_count"),
                "search_family_size": (claim.get("search_provenance") or {}).get("family_size"),
                "search_selection": (claim.get("search_provenance") or {}).get("selection"),
                "best_region": claim["best_region"],
                "best_beta": claim["best_beta"],
                "best_p": claim["best_p"],
                "best_standardized_effect": claim["best_standardized_effect"],
                "achieved_power": claim["power"].get("achieved_power"),
                "ref_effect": claim["power"].get("ref_effect"),
                "under_powered": claim["power"].get("under_powered"),
                "multiverse_fraction_consistent": claim["multiverse_fraction_consistent"],
                "replication_passed": rep.get("passed"),
                "replication_reason": rep.get("reason"),
                "replication_pattern_corr": rep.get("pattern_corr"),
                "replication_dice": rep.get("dice"),
                "replication_region_fraction": rep.get("region_replication_fraction"),
                "leakage_audit": json.dumps(_json_safe(claim.get("leakage_audit")), sort_keys=True),
                "rationale": claim["rationale"],
            }
        )
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    pd.DataFrame(payload.get("risk_coverage", [])).to_csv(risk_path, index=False)
    Path(lock_path).write_text(json.dumps(_json_safe(payload["lockfile"]), indent=2), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(csv_path, latest_csv)
    shutil.copyfile(audit_path, latest_audit)
    shutil.copyfile(risk_path, latest_risk)
    shutil.copyfile(lock_path, latest_lock)
    return json_path, csv_path, latest_json


def run(args: argparse.Namespace) -> dict[str, Any]:
    layer = PreparedLayer.load(args.data_root)
    claims = _ready_fmri_claims(layer.claim_inventory)
    if args.claim_id:
        claims = claims[claims["claim_id"].isin(args.claim_id)]
    if args.limit:
        claims = claims.head(args.limit)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for _, row in claims.iterrows():
        try:
            if not _shared_features(layer, row, args.feature_limit):
                skipped.append({"claim_id": str(row["claim_id"]), "reason": "no_shared_features"})
                print(f"[skip] {row['claim_id']}: no shared features")
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
            error = {"claim_id": str(row["claim_id"]), "error": str(exc)}
            errors.append(error)
            print(f"[error] {error['claim_id']}: {error['error']}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": {
            "data_root": str(args.data_root),
            "feature_limit": args.feature_limit,
            "limit": args.limit,
            "claim_id": args.claim_id,
            "seed": args.seed,
            "harmonize": args.harmonize,
        },
        **summarize(rows),
        "claims": rows,
        "errors": errors,
        "skipped": skipped,
    }
    payload["lockfile"] = _lockfile(layer, rows, skipped, args)
    json_path, csv_path, latest_json = write_outputs(Path(args.out_dir), payload)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"latest {latest_json}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/prepared_data/benchmark_ready")
    parser.add_argument("--out-dir", default="review-stage")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N ready fMRI claims.")
    parser.add_argument("--claim-id", action="append", default=None, help="Run a specific claim ID; may repeat.")
    parser.add_argument("--feature-limit", type=int, default=None, help="Limit shared feature count per claim for sanity runs.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--harmonize", choices=["none", "combat"], default="none")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
