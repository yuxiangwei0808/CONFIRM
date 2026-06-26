"""Deterministic injected-null task builders."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from confirm.analysis import fit_effect
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
    SearchProvenance,
)
from confirm.schema import idp_columns, normalize_sex

SEED = 20260615


def _first_idp(df: pd.DataFrame) -> str:
    cols = idp_columns(df.columns)
    if not cols:
        raise ValueError("Injected-null builders require at least one IDP column")
    return cols[0]


def _base_contract(
    df: pd.DataFrame,
    claim_id: str,
    outcome: str,
    search_provenance: dict[str, Any] | None = None,
) -> ClaimContract:
    cohort = str(df["cohort"].iloc[0]) if "cohort" in df.columns and len(df) else "SYNTH"
    data = {
        "claim_id": claim_id,
        "question": "Injected-null benchmark task.",
        "estimand": {
            "type": "group_diff",
            "outcome": outcome,
            "predictor": "arm_code",
            "group": {"var": "arm_code", "case": "A", "control": "B"},
            "direction": "two_sided",
        },
        "covariates": [col for col in ["age", "sex", "eTIV", "site"] if col in df.columns],
        "inclusion": None,
        "discovery_cohort": cohort,
        "replication_cohorts": [cohort],
        "search_provenance": search_provenance
        or {"declared": True, "family_size": 1, "selection": "preregistered"},
        "gates": {
            "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
            "confound": {"require_covariates": [col for col in ["sex", "site"] if col in df.columns], "motion_check": True},
            "power": {"min_power": 0.8, "ref_effect": None},
            "multiverse": {"min_fraction_consistent": 0.6},
            "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "combat"},
        },
        "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
    }
    return ClaimContract.model_validate(data)


def _available_covariates(df: pd.DataFrame) -> list[str]:
    return [col for col in ["age", "sex", "eTIV", "site"] if col in df.columns]


def _assign_random_arm(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["arm_code"] = np.random.default_rng(seed).choice(["A", "B"], size=len(out))
    return out


def _cramers_v(x: pd.Series, y: pd.Series) -> float | None:
    data = pd.DataFrame({"x": x, "y": y}).dropna().astype(str)
    if data.empty or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return None
    from scipy.stats import chi2_contingency

    table = pd.crosstab(data["x"], data["y"])
    chi2 = float(chi2_contingency(table, correction=False)[0])
    n = float(table.to_numpy().sum())
    denom = n * max(min(table.shape) - 1, 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else None


def _max_group_delta_by_level(df: pd.DataFrame, group_col: str, level_col: str) -> float | None:
    data = df[[group_col, level_col]].dropna().astype(str)
    if data.empty or data[group_col].nunique() != 2 or data[level_col].nunique() < 2:
        return None
    groups = sorted(data[group_col].unique())
    target = groups[0]
    overall = float((data[group_col] == target).mean())
    by_level = data.groupby(level_col)[group_col].apply(lambda values: float((values == target).mean()))
    return float((by_level - overall).abs().max())


def _covariate_predictability(df: pd.DataFrame, group_col: str, covariates: list[str]) -> dict[str, Any]:
    data = df[[group_col, *covariates]].dropna().copy()
    if data.empty or data[group_col].nunique() != 2:
        return {
            "n": int(len(data)),
            "covariates": covariates,
            "baseline_accuracy": None,
            "cv_accuracy": None,
            "note": "requires two non-missing groups",
        }

    y = data[group_col].astype(str)
    baseline = float(y.value_counts(normalize=True).max())
    x = pd.get_dummies(data[covariates], drop_first=True, dtype=float)
    if x.empty:
        return {
            "n": int(len(data)),
            "covariates": covariates,
            "baseline_accuracy": baseline,
            "cv_accuracy": baseline,
            "note": "no usable covariate columns",
        }

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    min_class = int(y.value_counts().min())
    if min_class < 2:
        return {
            "n": int(len(data)),
            "covariates": covariates,
            "baseline_accuracy": baseline,
            "cv_accuracy": None,
            "note": "too few rows in one group for cross-validation",
        }
    n_splits = min(5, min_class)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear"))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accuracy = float(cross_val_score(model, x, y, cv=cv, scoring="accuracy").mean())
    return {
        "n": int(len(data)),
        "covariates": covariates,
        "baseline_accuracy": baseline,
        "cv_accuracy": accuracy,
        "cv_splits": int(n_splits),
    }


def leakage_audit(
    discovery: pd.DataFrame,
    replication: pd.DataFrame,
    *,
    group_col: str = "arm_code",
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Audit synthetic-null labels for covariate, subject, cohort, and site leakage."""

    covars = list(covariates) if covariates is not None else [
        col for col in ["age", "sex", "site", "cohort"] if col in discovery.columns or col in replication.columns
    ]
    disc_covars = [col for col in covars if col in discovery.columns]
    rep_covars = [col for col in covars if col in replication.columns]
    disc_subjects = set(discovery["subject_id"].astype(str)) if "subject_id" in discovery.columns else set()
    rep_subjects = set(replication["subject_id"].astype(str)) if "subject_id" in replication.columns else set()
    overlap = disc_subjects & rep_subjects

    def structure(df: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for col in ["site", "cohort"]:
            if group_col in df.columns and col in df.columns:
                out[col] = {
                    "cramers_v": _cramers_v(df[group_col], df[col]),
                    "max_group_rate_delta": _max_group_delta_by_level(df, group_col, col),
                    "n_levels": int(df[col].dropna().astype(str).nunique()),
                }
        return out

    return {
        "group_col": group_col,
        "covariate_predictability_discovery": _covariate_predictability(discovery, group_col, disc_covars),
        "covariate_predictability_replication": _covariate_predictability(replication, group_col, rep_covars),
        "subject_overlap": {
            "n_overlap": int(len(overlap)),
            "fraction_discovery": float(len(overlap) / len(disc_subjects)) if disc_subjects else None,
            "fraction_replication": float(len(overlap) / len(rep_subjects)) if rep_subjects else None,
        },
        "arm_code_tracks_structure_discovery": structure(discovery),
        "arm_code_tracks_structure_replication": structure(replication),
    }


def fishing_label_null(
    discovery: pd.DataFrame,
    replication: pd.DataFrame,
    seed: int = SEED,
    feature_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, ClaimContract, str, dict[str, Any]]:
    """Select the best null feature in discovery only and declare its searched family."""

    features = sorted(set(idp_columns(discovery.columns)) & set(idp_columns(replication.columns)))
    if feature_limit and feature_limit > 0:
        features = features[:feature_limit]
    if not features:
        raise ValueError("Fishing null requires shared IDP features")

    disc = _assign_random_arm(discovery, seed)
    rep = _assign_random_arm(replication, seed + 1)
    covariates = _available_covariates(disc)
    search_provenance = {"declared": True, "family_size": len(features), "selection": "discovery_only"}
    scored: list[tuple[float, str]] = []
    for feature in features:
        try:
            contract = _base_contract(disc, "injected_fishing_null", feature, search_provenance)
            contract = contract.model_copy(update={"covariates": covariates})
            effect = fit_effect(disc, contract, covariates=covariates)
            scored.append((float(effect.p), feature))
        except Exception:
            continue
    if not scored:
        raise ValueError("No fishing-null feature could be fitted")
    _, outcome = min(scored, key=lambda item: item[0])
    contract = _base_contract(disc, "injected_fishing_null", outcome, search_provenance)
    contract = contract.model_copy(update={"covariates": covariates})
    return disc, rep, contract, "non_replicated", leakage_audit(disc, rep, covariates=covariates)


def inject_site_confound(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, ClaimContract, str]:
    """Create a fake group assignment correlated with site."""

    rng = np.random.default_rng(seed)
    out = df.copy()
    outcome = _first_idp(out)
    sites = sorted(out["site"].astype(str).unique()) if "site" in out.columns else ["unknown"]
    favored = set(sites[: max(1, len(sites) // 2)])
    probs = out["site"].astype(str).map(lambda site: 0.85 if site in favored else 0.15) if "site" in out.columns else 0.5
    out["arm_code"] = np.where(rng.random(len(out)) < np.asarray(probs, dtype=float), "A", "B")
    contract = _base_contract(out, "injected_site_confound", outcome)
    return out, contract, "non_replicated"


def inject_motion_leakage(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, ClaimContract, str]:
    """Create a masked motion-related nuisance variable and confounded group."""

    rng = np.random.default_rng(seed)
    out = df.copy()
    outcome = _first_idp(out)
    out["beh_quality_index"] = rng.normal(0, 1, len(out))
    out["arm_code"] = np.where(out["beh_quality_index"] > np.median(out["beh_quality_index"]), "A", "B")
    out[outcome] = pd.to_numeric(out[outcome], errors="coerce") + 0.2 * out["beh_quality_index"]
    contract = _base_contract(out, "injected_motion_nuisance", outcome)
    return out, contract, "fragile"


def inject_label_leakage(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, ClaimContract, str]:
    """Create a non-biological group label from an analysis variable."""

    rng = np.random.default_rng(seed)
    out = df.copy()
    outcome = _first_idp(out)
    jitter = rng.normal(0, 1e-6, len(out))
    out["arm_code"] = np.where(pd.to_numeric(out[outcome], errors="coerce") + jitter > out[outcome].median(), "A", "B")
    contract = _base_contract(out, "injected_derived_label", outcome)
    return out, contract, "fragile"


@dataclass(frozen=True)
class NegativeCohort:
    """Local cohort table available for synthetic negative stress tasks."""

    name: str
    path: Path
    frame: pd.DataFrame
    features: list[str]
    modality: str


@dataclass(frozen=True)
class NegativeStressTask:
    """Executable synthetic negative claim plus its generated label-table row."""

    claim_id: str
    family: str
    label_class: str
    expected_gate: str
    discovery: pd.DataFrame
    replication: pd.DataFrame
    contract: ClaimContract
    label_row: dict[str, str]
    covariates_min: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


LOCAL_NEGATIVE_COHORT_FILES: tuple[tuple[str, str], ...] = (
    ("UKB", "data/prepared_data/benchmark_ready/cohorts/UKB.parquet"),
    ("ABCD", "data/prepared_data/benchmark_ready/cohorts/ABCD.parquet"),
    ("HCP", "data/prepared_data/benchmark_ready/cohorts/HCP.parquet"),
    ("HCP_Aging", "data/prepared_data/benchmark_ready/cohorts/HCP_Aging.parquet"),
    ("ABIDE2", "data/prepared_data/benchmark_ready/cohorts/ABIDE2.parquet"),
    ("ADHD200", "data/prepared_data/benchmark_ready/cohorts/ADHD200.parquet"),
    ("ADNI_fMRI", "data/prepared_data/benchmark_ready/cohorts/ADNI_fMRI.parquet"),
    ("OASIS3_fMRI", "data/prepared_data/benchmark_ready/cohorts/OASIS3_fMRI.parquet"),
    ("COBRE", "data/prepared_data/cluster_recovered/COBRE.parquet"),
    ("FBIRN", "data/prepared_data/cluster_recovered/FBIRN.parquet"),
    ("ABIDE1", "data/prepared_data/cluster_recovered/ABIDE1.parquet"),
    ("GSP", "data/prepared_data/cluster_recovered/GSP.parquet"),
    ("ADNI", "data/prepared_data/smri_disease/ADNI.parquet"),
    ("OASIS3", "data/prepared_data/smri_disease/OASIS3.parquet"),
)

DEFAULT_NEGATIVE_SEEDS = (1, 2, 3)


def load_local_negative_cohorts(root: str | Path = ".") -> list[NegativeCohort]:
    """Load all local cohorts requested for the negatives expansion."""

    base = Path(root)
    cohorts: list[NegativeCohort] = []
    for name, rel_path in LOCAL_NEGATIVE_COHORT_FILES:
        path = base / rel_path
        if not path.exists():
            continue
        frame = _normalize_negative_frame(pd.read_parquet(path), name)
        features = sorted(idp_columns(frame.columns))
        if not features:
            continue
        modality = "sMRI" if all(feature.startswith("smri_") for feature in features) else "fMRI-FC"
        cohorts.append(NegativeCohort(name=name, path=path, frame=frame, features=features, modality=modality))
    return cohorts


def generate_negative_stress_tasks(
    root: str | Path = ".",
    *,
    seeds: tuple[int, ...] = DEFAULT_NEGATIVE_SEEDS,
    fishing_feature_limit: int = 24,
    underpowered_cohort_limit: int = 7,
    cross_pairs: tuple[tuple[str, str], ...] = (("UKB", "HCP_Aging"), ("ABIDE2", "ABCD"), ("ADHD200", "ABCD")),
) -> tuple[list[NegativeStressTask], list[dict[str, str]]]:
    """Generate the local known-null/fragile negative benchmark expansion."""

    cohorts = load_local_negative_cohorts(root)
    by_name = {cohort.name: cohort for cohort in cohorts}
    tasks: list[NegativeStressTask] = []
    skipped: list[dict[str, str]] = []

    def add(build_name: str, fn: Any) -> None:
        try:
            tasks.append(fn())
        except Exception as exc:
            skipped.append({"builder": build_name, "reason": str(exc)})

    for cohort in cohorts:
        for seed in seeds:
            add(f"random_label:{cohort.name}:s{seed}", lambda cohort=cohort, seed=seed: random_label_stress_task(cohort, seed))
            add(f"site_confound:{cohort.name}:s{seed}", lambda cohort=cohort, seed=seed: site_confound_stress_task(cohort, seed))
            add(
                f"p_fishing:{cohort.name}:s{seed}",
                lambda cohort=cohort, seed=seed: p_fishing_stress_task(cohort, seed, feature_limit=fishing_feature_limit),
            )

    eligible_underpowered = [cohort for cohort in cohorts if _has_two_sex_groups(cohort.frame)]
    for cohort in eligible_underpowered[:underpowered_cohort_limit]:
        for seed in seeds:
            add(f"underpowered:{cohort.name}:s{seed}", lambda cohort=cohort, seed=seed: underpowered_stress_task(cohort, seed))

    for i, (discovery, replication) in enumerate(cross_pairs, start=1):
        if discovery not in by_name or replication not in by_name:
            skipped.append({"builder": f"cross_cohort_nonreplication:{discovery}:{replication}", "reason": "cohort_not_loaded"})
            continue
        seed = seeds[(i - 1) % len(seeds)]
        add(
            f"cross_cohort_nonreplication:{discovery}:{replication}:s{seed}",
            lambda discovery=discovery, replication=replication, seed=seed: cross_cohort_nonreplication_task(
                by_name[discovery],
                by_name[replication],
                seed,
                feature_limit=fishing_feature_limit,
            ),
        )
    return tasks, skipped


def random_label_stress_task(cohort: NegativeCohort, seed: int) -> NegativeStressTask:
    disc, rep = _split_negative_frame(cohort.frame, seed, cohort.name, cohort.name)
    disc = _assign_bench_group_random(disc, seed)
    rep = _assign_bench_group_random(rep, seed + 10_000)
    outcome = _seeded_feature(cohort.features, seed)
    covariates = _negative_covariates(disc, rep, include_site=True)
    contract = _negative_scalar_contract(
        claim_id=f"neg_random_label_{_slug(cohort.name)}_s{seed}",
        outcome=outcome,
        covariates=covariates,
        discovery=str(disc["cohort"].iloc[0]),
        replication=str(rep["cohort"].iloc[0]),
        search_family_size=1,
    )
    return _negative_task(
        family="random_label",
        label_class="known_null",
        expected_gate="multiplicity_or_replication",
        cohort=cohort,
        discovery=disc,
        replication=rep,
        contract=contract,
        covariates_min=[cov for cov in covariates if cov in {"age", "sex", "eTIV"}],
        seed=seed,
    )


def site_confound_stress_task(cohort: NegativeCohort, seed: int) -> NegativeStressTask:
    disc, rep = _split_negative_frame(cohort.frame, seed, cohort.name, cohort.name)
    outcome = _seeded_feature(cohort.features, seed + 17)
    disc = _assign_synthetic_site_confound(disc, outcome)
    rep = _assign_synthetic_site_confound(rep, outcome)
    covariates = _negative_covariates(disc, rep, include_site=False)
    contract = _negative_scalar_contract(
        claim_id=f"neg_site_confound_{_slug(cohort.name)}_s{seed}",
        outcome=outcome,
        covariates=covariates,
        discovery=str(disc["cohort"].iloc[0]),
        replication=str(rep["cohort"].iloc[0]),
        search_family_size=1,
    )
    return _negative_task(
        family="site_confound",
        label_class="known_null",
        expected_gate="confound_completeness",
        cohort=cohort,
        discovery=disc,
        replication=rep,
        contract=contract,
        covariates_min=[cov for cov in covariates if cov in {"age", "sex", "eTIV"}],
        seed=seed,
        metadata={"site_source": "synthetic_median_split"},
    )


def p_fishing_stress_task(cohort: NegativeCohort, seed: int, *, feature_limit: int) -> NegativeStressTask:
    disc, rep = _split_negative_frame(cohort.frame, seed, cohort.name, cohort.name)
    disc = _assign_bench_group_random(disc, seed + 20_000)
    rep = _assign_bench_group_random(rep, seed + 30_000)
    features = cohort.features[: max(1, min(feature_limit, len(cohort.features)))]
    covariates = _negative_covariates(disc, rep, include_site=True)
    selected, search_scores = _select_best_negative_feature(
        claim_id=f"neg_p_fishing_{_slug(cohort.name)}_s{seed}",
        disc=disc,
        features=features,
        covariates=covariates,
    )
    contract = _negative_scalar_contract(
        claim_id=f"neg_p_fishing_{_slug(cohort.name)}_s{seed}",
        outcome=selected,
        covariates=covariates,
        discovery=str(disc["cohort"].iloc[0]),
        replication=str(rep["cohort"].iloc[0]),
        search_family_size=len(features),
        search_selection="discovery_only",
    )
    return _negative_task(
        family="p_fishing",
        label_class="known_null",
        expected_gate="search_provenance_multiplicity",
        cohort=cohort,
        discovery=disc,
        replication=rep,
        contract=contract,
        covariates_min=[cov for cov in covariates if cov in {"age", "sex", "eTIV"}],
        seed=seed,
        metadata={"searched_feature_count": len(features), "selected_outcome": selected, "top_discovery_features": search_scores[:10]},
    )


def underpowered_stress_task(cohort: NegativeCohort, seed: int) -> NegativeStressTask:
    disc, rep = _split_negative_frame(cohort.frame, seed, cohort.name, cohort.name)
    rng = np.random.default_rng(seed + 40_000)
    n_per_group = int(rng.integers(15, 26))
    disc = _balanced_subsample(disc, "sex", max_per_group=n_per_group, seed=seed + 41_000)
    rep = _balanced_subsample(rep, "sex", max_per_group=n_per_group, seed=seed + 42_000)
    outcome = _seeded_feature(cohort.features, seed + 31)
    covariates = _negative_covariates(disc, rep, include_site=True, exclude=("sex",))
    contract = _negative_scalar_contract(
        claim_id=f"neg_underpowered_{_slug(cohort.name)}_s{seed}",
        outcome=outcome,
        covariates=covariates,
        discovery=str(disc["cohort"].iloc[0]),
        replication=str(rep["cohort"].iloc[0]),
        group_var="sex",
        case="F",
        control="M",
        search_family_size=1,
    )
    return _negative_task(
        family="underpowered",
        label_class="fragile",
        expected_gate="power",
        cohort=cohort,
        discovery=disc,
        replication=rep,
        contract=contract,
        covariates_min=[cov for cov in covariates if cov in {"age", "eTIV"}],
        seed=seed,
        metadata={"n_per_group": n_per_group},
    )


def cross_cohort_nonreplication_task(
    discovery: NegativeCohort,
    replication: NegativeCohort,
    seed: int,
    *,
    feature_limit: int,
) -> NegativeStressTask:
    shared = sorted(set(discovery.features) & set(replication.features))
    if not shared:
        raise ValueError("cross-cohort nonreplication requires shared features")
    features = shared[: max(1, min(feature_limit, len(shared)))]
    disc = _assign_bench_group_random(discovery.frame.copy(), seed + 50_000)
    rep = _assign_bench_group_random(replication.frame.copy(), seed + 60_000)
    covariates = _negative_covariates(disc, rep, include_site=True)
    claim_id = f"neg_cross_nonrep_{_slug(discovery.name)}_{_slug(replication.name)}_s{seed}"
    selected, search_scores = _select_best_negative_feature(
        claim_id=claim_id,
        disc=disc,
        features=features,
        covariates=covariates,
    )
    contract = _negative_scalar_contract(
        claim_id=claim_id,
        outcome=selected,
        covariates=covariates,
        discovery=discovery.name,
        replication=replication.name,
        search_family_size=len(features),
        search_selection="discovery_only",
    )
    label_row = _negative_label_row(
        claim_id=claim_id,
        family="cross_cohort_nonreplication",
        label_class="known_null",
        modality=discovery.modality,
        discovery=discovery.name,
        replication=replication.name,
        covariates=covariates,
    )
    return NegativeStressTask(
        claim_id=claim_id,
        family="cross_cohort_nonreplication",
        label_class="known_null",
        expected_gate="replication",
        discovery=disc,
        replication=rep,
        contract=contract,
        label_row=label_row,
        covariates_min=[cov for cov in covariates if cov in {"age", "sex", "eTIV"}],
        metadata={"searched_feature_count": len(features), "selected_outcome": selected, "top_discovery_features": search_scores[:10]},
    )


def _negative_task(
    *,
    family: str,
    label_class: str,
    expected_gate: str,
    cohort: NegativeCohort,
    discovery: pd.DataFrame,
    replication: pd.DataFrame,
    contract: ClaimContract,
    covariates_min: list[str],
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> NegativeStressTask:
    label_row = _negative_label_row(
        claim_id=contract.claim_id,
        family=family,
        label_class=label_class,
        modality=cohort.modality,
        discovery=cohort.name,
        replication=cohort.name,
        covariates=contract.covariates,
    )
    task_metadata = {"seed": seed, "cohort_path": str(cohort.path), **(metadata or {})}
    return NegativeStressTask(
        claim_id=contract.claim_id,
        family=family,
        label_class=label_class,
        expected_gate=expected_gate,
        discovery=discovery,
        replication=replication,
        contract=contract,
        label_row=label_row,
        covariates_min=covariates_min,
        metadata=task_metadata,
    )


def _normalize_negative_frame(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = df.copy()
    if "subject_id" not in out.columns:
        out["subject_id"] = [f"{cohort}_{i}" for i in range(len(out))]
    if "session" not in out.columns:
        out["session"] = "ses-01"
    if "cohort" not in out.columns:
        out["cohort"] = cohort
    if "site" not in out.columns:
        out["site"] = cohort
    if "dx" not in out.columns:
        out["dx"] = pd.NA
    out["subject_id"] = out["subject_id"].astype(str)
    out["session"] = out["session"].fillna("ses-01").astype(str)
    out["cohort"] = cohort
    out["site"] = out["site"].fillna(cohort).astype(str)
    out["sex"] = normalize_sex(out["sex"]) if "sex" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
    out["age"] = pd.to_numeric(out["age"], errors="coerce") if "age" in out.columns else np.nan
    if "eTIV" in out.columns:
        out["eTIV"] = pd.to_numeric(out["eTIV"], errors="coerce")
    for feature in idp_columns(out.columns):
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    return out


def _split_negative_frame(df: pd.DataFrame, seed: int, discovery: str, replication: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 70_000)
    subjects = df["subject_id"].astype(str).drop_duplicates().to_numpy()
    rng.shuffle(subjects)
    half = len(subjects) // 2
    if half < 10:
        raise ValueError("cohort too small for split-half negative stress task")
    left = set(subjects[:half])
    disc = df[df["subject_id"].astype(str).isin(left)].copy()
    rep = df[~df["subject_id"].astype(str).isin(left)].copy()
    disc["cohort"] = f"{discovery}_DISC_s{seed}"
    rep["cohort"] = f"{replication}_REP_s{seed}"
    return disc, rep


def _assign_bench_group_random(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["bench_group"] = np.random.default_rng(seed).choice(["case", "control"], size=len(out))
    return out


def _assign_synthetic_site_confound(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out[outcome], errors="coerce")
    median = float(x.median())
    out["original_site"] = out["site"].astype(str)
    out["site"] = np.where(x >= median, "synthetic_site_high", "synthetic_site_low")
    out["bench_group"] = np.where(x >= median, "case", "control")
    out.loc[x.isna(), ["site", "bench_group"]] = pd.NA
    return out


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


def _negative_covariates(
    disc: pd.DataFrame,
    rep: pd.DataFrame,
    *,
    include_site: bool,
    exclude: tuple[str, ...] = (),
) -> list[str]:
    desired = ["age", "sex", "eTIV"]
    if include_site:
        desired.append("site")
    blocked = set(exclude)
    covariates: list[str] = []
    for col in desired:
        if col in blocked:
            continue
        if col in disc.columns and col in rep.columns and (disc[col].notna().any() or rep[col].notna().any()):
            covariates.append(col)
    return covariates


def _negative_scalar_contract(
    *,
    claim_id: str,
    outcome: str,
    covariates: list[str],
    discovery: str,
    replication: str,
    group_var: str = "bench_group",
    case: str = "case",
    control: str = "control",
    search_family_size: int = 1,
    search_selection: str = "preregistered",
) -> ClaimContract:
    return ClaimContract(
        claim_id=claim_id,
        question="Programmatic synthetic known-null/fragile negative stress claim.",
        estimand=Estimand(
            type="group_diff",
            outcome=outcome,
            predictor=group_var,
            group=GroupSpec(var=group_var, case=case, control=control),
            direction="two_sided",
            unit="scalar",
        ),
        covariates=covariates,
        inclusion=None,
        discovery_cohort=discovery,
        replication_cohorts=[replication],
        search_provenance=SearchProvenance(
            declared=True,
            family_size=max(1, int(search_family_size)),
            selection=search_selection,  # type: ignore[arg-type]
        ),
        gates=Gates(
            multiplicity=MultiplicityGate(method="fdr_bh", alpha=0.05, family_size=1),
            confound=ConfoundGate(require_covariates=covariates, motion_check=False),
            power=PowerGate(min_power=0.8, ref_effect=None),
            multiverse=MultiverseGate(min_fraction_consistent=0.6),
            replication=ReplicationGate(alpha=0.05, require_same_sign=True, require_ci_overlap=False, harmonize="none"),
        ),
        reporting_language_allowed=["confirmed", "non_replicated", "under_powered", "fragile"],
    )


def _select_best_negative_feature(
    *,
    claim_id: str,
    disc: pd.DataFrame,
    features: list[str],
    covariates: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    for feature in features:
        try:
            contract = _negative_scalar_contract(
                claim_id=claim_id,
                outcome=feature,
                covariates=covariates,
                discovery=str(disc["cohort"].iloc[0]),
                replication=str(disc["cohort"].iloc[0]),
            )
            effect = fit_effect(disc, contract, covariates=covariates, model="ols")
            scored.append({"outcome": feature, "p": float(effect.p), "beta": float(effect.beta), "n": int(effect.n)})
        except Exception as exc:
            scored.append({"outcome": feature, "p": float("inf"), "beta": float("nan"), "n": 0, "error": str(exc)})
    fitted = [item for item in scored if np.isfinite(float(item["p"]))]
    if not fitted:
        raise ValueError("no searched features could be fitted")
    ordered = sorted(fitted, key=lambda item: float(item["p"]))
    return str(ordered[0]["outcome"]), ordered


def _negative_label_row(
    *,
    claim_id: str,
    family: str,
    label_class: str,
    modality: str,
    discovery: str,
    replication: str,
    covariates: list[str],
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "phenotype": f"synthetic {family} negative stress",
        "modality": modality,
        "cohorts": f"{discovery};{replication}",
        "discovery_cohort": discovery,
        "replication_cohort": replication,
        "label_class": label_class,
        "label_basis": "synthetic_stress",
        "adjudication_status": "preregistered",
        "expected_direction": "two_sided",
        "expected_effect_scale": "no admissible biological confirmation expected",
        "mde_assumption": "programmatic negative stress benchmark",
        "cohort_role": "generated local negative benchmark",
        "forbidden_evidence": "no outcome-informed relabeling after execution",
        "confound_set": ";".join(covariates) if covariates else "none",
        "site_scanner_handling": "local synthetic contract; no source data mutation",
        "decision_target": "confirmation counts as FCR",
        "construct_validity_notes": f"{family} generated by src/bench/injected_nulls.py",
        "label_confidence": "high: synthetic construction",
        "source_citation": "synthetic_stress",
    }


def _seeded_feature(features: list[str], seed: int) -> str:
    if not features:
        raise ValueError("negative stress task requires at least one feature")
    return features[seed % len(features)]


def _has_two_sex_groups(df: pd.DataFrame, min_per_group: int = 40) -> bool:
    if "sex" not in df.columns:
        return False
    counts = df["sex"].value_counts(dropna=True)
    return bool({"F", "M"}.issubset(set(counts.index.astype(str))) and counts.min() >= min_per_group)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def random_label_null(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, ClaimContract, str]:
    """Assign a random null label independent of the outcome."""

    rng = np.random.default_rng(seed)
    out = deepcopy(df)
    outcome = _first_idp(out)
    out["arm_code"] = rng.choice(["A", "B"], size=len(out))
    contract = _base_contract(out, "injected_random_label", outcome)
    return out, contract, "non_replicated"
