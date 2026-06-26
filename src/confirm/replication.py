"""Cross-cohort replication gate."""

from __future__ import annotations

import logging
import math

import pandas as pd

from confirm.analysis import ci_overlap, directionally_consistent, fit_effect
from confirm.brainwide import run_brainwide
from confirm.contract import ClaimContract
from confirm.harmonize import combat_harmonize
from confirm.results import (
    BrainwideReplicationResult,
    CohortEffectSummary,
    CohortBrainwideReplicationResult,
    CohortReplicationResult,
    EffectResult,
    HeterogeneityResult,
    RegionTable,
    ReplicationResult,
    scalar_confirmation_subtype,
)
from confirm.schema import idp_columns

LOGGER = logging.getLogger(__name__)


def _feature_coverage(disc_df: pd.DataFrame, rep_df: pd.DataFrame) -> float:
    disc_idps = set(idp_columns(disc_df.columns))
    rep_idps = set(idp_columns(rep_df.columns))
    if not disc_idps:
        return 0.0
    return len(disc_idps & rep_idps) / len(disc_idps)


def _region_feature_coverage(disc_regions: list[str], rep_df: pd.DataFrame) -> float:
    if not disc_regions:
        return 0.0
    return len(set(disc_regions) & set(rep_df.columns)) / len(set(disc_regions))


def _brainwide_contract(contract: ClaimContract, regions: list[str]) -> ClaimContract:
    estimand = contract.estimand.model_copy(update={"outcome": regions, "unit": "brainwide"})
    multiplicity = contract.gates.multiplicity.model_copy(update={"family_size": len(regions)})
    gates = contract.gates.model_copy(update={"multiplicity": multiplicity})
    return contract.model_copy(update={"estimand": estimand, "gates": gates})


def _effect_map(table: RegionTable) -> dict[str, float]:
    return {region.region: float(region.effect.standardized_effect) for region in table.regions}


def _sig_set(table: RegionTable) -> set[str]:
    return {region.region for region in table.regions if region.significant}


def _cohort_name(df: pd.DataFrame, fallback: str) -> str:
    return str(df["cohort"].iloc[0]) if "cohort" in df.columns and len(df) else fallback


def _standardized_se(effect: EffectResult) -> float:
    if (
        math.isfinite(effect.beta)
        and math.isfinite(effect.se)
        and math.isfinite(effect.standardized_effect)
        and effect.beta != 0
    ):
        value = abs(effect.se * effect.standardized_effect / effect.beta)
        if value > 0 and math.isfinite(value):
            return float(value)
    return float(1.0 / math.sqrt(max(effect.n, 1)))


def _effect_summary(cohort: str, effect: EffectResult) -> CohortEffectSummary:
    standardized = float(effect.standardized_effect)
    sign = "positive" if standardized > 0 else "negative" if standardized < 0 else "zero"
    return CohortEffectSummary(
        cohort=cohort,
        standardized_effect=standardized,
        se_standardized_effect=_standardized_se(effect),
        sign=sign,
        p=float(effect.p),
        n=int(effect.n),
    )


def _heterogeneity_audit(
    discovery_effect: EffectResult,
    disc_df: pd.DataFrame,
    cohort_results: list[CohortReplicationResult],
    *,
    high_i2_threshold: float = 75.0,
) -> HeterogeneityResult | None:
    summaries = [_effect_summary(_cohort_name(disc_df, "discovery"), discovery_effect)]
    for result in cohort_results:
        if result.effect is not None:
            summaries.append(_effect_summary(result.cohort, result.effect))
    if len(summaries) < 2:
        return None

    effects = [summary.standardized_effect for summary in summaries]
    variances = [max(summary.se_standardized_effect**2, 1e-12) for summary in summaries]
    weights = [1.0 / variance for variance in variances]
    fixed_mean = sum(w * y for w, y in zip(weights, effects)) / sum(weights)
    q = sum(w * (y - fixed_mean) ** 2 for w, y in zip(weights, effects))
    df = len(effects) - 1
    c = sum(weights) - (sum(w * w for w in weights) / sum(weights))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    random_weights = [1.0 / (variance + tau2) for variance in variances]
    random_effect = sum(w * y for w, y in zip(random_weights, effects)) / sum(random_weights)
    random_se = math.sqrt(1.0 / sum(random_weights))
    i2 = max(0.0, ((q - df) / q) * 100.0) if q > 0 else 0.0
    return HeterogeneityResult(
        cohort_effects=summaries,
        random_effect=float(random_effect),
        random_effect_se=float(random_se),
        ci_low=float(random_effect - 1.96 * random_se),
        ci_high=float(random_effect + 1.96 * random_se),
        q=float(q),
        tau2=float(tau2),
        i2=float(i2),
        high_i2=bool(i2 >= high_i2_threshold),
    )


def _pattern_corr(discovery: RegionTable, replication: RegionTable) -> float:
    disc = _effect_map(discovery)
    rep = _effect_map(replication)
    shared = sorted(set(disc) & set(rep))
    if len(shared) < 2:
        return float("nan")
    x = pd.Series([disc[region] for region in shared], dtype=float)
    y = pd.Series([rep[region] for region in shared], dtype=float)
    finite = x.notna() & y.notna()
    if finite.sum() < 2 or x[finite].std(ddof=0) == 0 or y[finite].std(ddof=0) == 0:
        return float("nan")
    return float(x[finite].corr(y[finite], method="pearson"))


def _dice(discovery: RegionTable, replication: RegionTable) -> float:
    a = _sig_set(discovery)
    b = _sig_set(replication)
    if not a and not b:
        return 1.0
    denom = len(a) + len(b)
    return float(2 * len(a & b) / denom) if denom else 0.0


def _region_replication_fraction(discovery: RegionTable, replication: RegionTable, contract: ClaimContract) -> float:
    disc_sig = [region for region in discovery.regions if region.significant]
    if not disc_sig:
        return 0.0
    rep_by_region = {region.region: region for region in replication.regions}
    replicated = 0
    for disc_region in disc_sig:
        rep_region = rep_by_region.get(disc_region.region)
        if rep_region is None:
            continue
        same_sign = directionally_consistent(rep_region.effect.beta, contract)
        if contract.gates.replication.require_same_sign:
            same_sign = same_sign and (rep_region.effect.beta * disc_region.effect.beta > 0)
        if same_sign and rep_region.effect.p < contract.gates.replication.alpha:
            replicated += 1
    return float(replicated / len(disc_sig))


def replicate(
    discovery_effect: EffectResult,
    disc_df: pd.DataFrame,
    rep_dfs: list[pd.DataFrame],
    contract: ClaimContract,
) -> ReplicationResult:
    """Re-fit the same contract model in each replication cohort."""

    outcome = contract.estimand.outcome
    if not rep_dfs:
        return ReplicationResult(False, "non_replicated_cohort_mismatch:no_replication_data", [], harmonized=False)

    harmonized = False
    all_frames: list[pd.DataFrame] = []
    disc_copy = disc_df.copy()
    disc_copy["__confirm_frame_id__"] = "discovery"
    all_frames.append(disc_copy)
    for i, rep_df in enumerate(rep_dfs):
        rep_copy = rep_df.copy()
        rep_copy["__confirm_frame_id__"] = f"rep_{i}"
        all_frames.append(rep_copy)

    if contract.gates.replication.harmonize == "combat":
        shared_idps = set(idp_columns(disc_df.columns))
        for rep_df in rep_dfs:
            shared_idps &= set(idp_columns(rep_df.columns))
        if outcome not in shared_idps:
            results = [
                CohortReplicationResult(
                    cohort=str(rep_df["cohort"].iloc[0]) if "cohort" in rep_df.columns and len(rep_df) else f"rep_{i}",
                    passed=False,
                    reason="non_replicated_cohort_mismatch:outcome_missing_for_harmonization",
                    effect=None,
                    feature_coverage=_feature_coverage(disc_df, rep_df),
                )
                for i, rep_df in enumerate(rep_dfs)
            ]
            return ReplicationResult(False, "non_replicated_cohort_mismatch", results, harmonized=False)
        combined = pd.concat(all_frames, ignore_index=True, sort=False)
        preserve = list(dict.fromkeys([contract.estimand.predictor, "age", "sex", *contract.covariates]))
        combined = combat_harmonize(combined, sorted(shared_idps), batch_col="site", covars=preserve)
        harmonized = True
        rep_frames = [combined[combined["__confirm_frame_id__"] == f"rep_{i}"].drop(columns=["__confirm_frame_id__"]) for i in range(len(rep_dfs))]
    else:
        rep_frames = rep_dfs

    cohort_results: list[CohortReplicationResult] = []
    for i, rep_df in enumerate(rep_frames):
        cohort = _cohort_name(rep_df, f"rep_{i}")
        coverage = _feature_coverage(disc_df, rep_df)
        if outcome not in rep_df.columns or coverage < 0.5:
            cohort_results.append(
                CohortReplicationResult(
                    cohort=cohort,
                    passed=False,
                    reason="non_replicated_cohort_mismatch",
                    effect=None,
                    feature_coverage=float(coverage),
                )
            )
            continue
        try:
            effect = fit_effect(rep_df, contract, covariates=contract.covariates, model="ols")
            same_sign = directionally_consistent(effect.beta, contract)
            if contract.gates.replication.require_same_sign:
                same_sign = same_sign and (effect.beta * discovery_effect.beta > 0)
            significant = effect.p < contract.gates.replication.alpha
            overlaps = ci_overlap(discovery_effect, effect) if contract.gates.replication.require_ci_overlap else True
            passed = bool(same_sign and significant and overlaps)
            failed = []
            if not same_sign:
                failed.append("sign")
            if not significant:
                failed.append("p")
            if not overlaps:
                failed.append("ci")
            reason = "passed" if passed else "non_replicated_effect_absent:" + ",".join(failed)
            cohort_results.append(CohortReplicationResult(cohort, passed, reason, effect, float(coverage)))
        except Exception as exc:
            LOGGER.warning("Replication fit failed for %s: %s", cohort, exc)
            cohort_results.append(
                CohortReplicationResult(cohort, False, f"non_replicated_cohort_mismatch:{exc}", None, float(coverage))
            )

    passed = bool(cohort_results) and all(result.passed for result in cohort_results)
    if passed:
        reason = "passed"
    elif any("cohort_mismatch" in result.reason for result in cohort_results):
        reason = "non_replicated_cohort_mismatch"
    else:
        reason = "non_replicated_effect_absent"
    heterogeneity = _heterogeneity_audit(discovery_effect, disc_df, cohort_results)
    subtype = scalar_confirmation_subtype(heterogeneity) if passed else None
    return ReplicationResult(
        passed,
        reason,
        cohort_results,
        harmonized=harmonized,
        heterogeneity=heterogeneity,
        replicated_but_heterogeneous=bool(passed and heterogeneity and heterogeneity.high_i2),
        confirmation_subtype=subtype,
        confirmation_i2=float(heterogeneity.i2) if passed and heterogeneity else None,
    )


def replicate_brainwide(
    disc_regions: RegionTable,
    disc_df: pd.DataFrame,
    rep_dfs: list[pd.DataFrame],
    contract: ClaimContract,
) -> BrainwideReplicationResult:
    """Replicate a brain-wide effect map in independent cohorts."""

    discovery_regions = [region.region for region in disc_regions.regions]
    if not rep_dfs:
        return BrainwideReplicationResult(
            False,
            "non_replicated_cohort_mismatch:no_replication_data",
            [],
            harmonized=False,
            pattern_corr=float("nan"),
            dice=0.0,
            region_replication_fraction=0.0,
        )

    all_results: list[CohortBrainwideReplicationResult] = []
    harmonized = False

    for i, rep_df in enumerate(rep_dfs):
        cohort = str(rep_df["cohort"].iloc[0]) if "cohort" in rep_df.columns and len(rep_df) else f"rep_{i}"
        shared_regions = sorted(set(discovery_regions) & set(idp_columns(rep_df.columns)))
        coverage = _region_feature_coverage(discovery_regions, rep_df)
        if len(shared_regions) < 2 or coverage < 0.5:
            all_results.append(
                CohortBrainwideReplicationResult(
                    cohort=cohort,
                    passed=False,
                    reason="non_replicated_cohort_mismatch",
                    pattern_corr=float("nan"),
                    dice=0.0,
                    region_replication_fraction=0.0,
                    region_table=None,
                    feature_coverage=float(coverage),
                )
            )
            continue

        disc_work = disc_df.copy()
        rep_work = rep_df.copy()
        disc_for_metrics = disc_regions
        if contract.gates.replication.harmonize == "combat":
            disc_work["__confirm_frame_id__"] = "discovery"
            rep_work["__confirm_frame_id__"] = "replication"
            combined = pd.concat([disc_work, rep_work], ignore_index=True, sort=False)
            preserve = list(dict.fromkeys([contract.estimand.predictor, "age", "sex", *contract.covariates]))
            try:
                combined = combat_harmonize(combined, shared_regions, batch_col="site", covars=preserve)
                harmonized = True
                disc_work = combined[combined["__confirm_frame_id__"] == "discovery"].drop(columns=["__confirm_frame_id__"])
                rep_work = combined[combined["__confirm_frame_id__"] == "replication"].drop(columns=["__confirm_frame_id__"])
                disc_for_metrics = run_brainwide(disc_work, _brainwide_contract(contract, shared_regions))
            except Exception as exc:
                LOGGER.warning("Brain-wide ComBat failed for %s: %s", cohort, exc)
                disc_work = disc_work.drop(columns=["__confirm_frame_id__"], errors="ignore")
                rep_work = rep_work.drop(columns=["__confirm_frame_id__"], errors="ignore")

        try:
            rep_table = run_brainwide(rep_work, _brainwide_contract(contract, shared_regions))
            corr = _pattern_corr(disc_for_metrics, rep_table)
            dice = _dice(disc_for_metrics, rep_table)
            frac = _region_replication_fraction(disc_for_metrics, rep_table, contract)
            gate = contract.gates.replication
            passed = bool(corr >= gate.pattern_corr_min and frac >= gate.region_replication_frac_min and dice >= gate.dice_min)
            failed = []
            if not corr >= gate.pattern_corr_min:
                failed.append("pattern_corr")
            if not frac >= gate.region_replication_frac_min:
                failed.append("region_replication_fraction")
            if not dice >= gate.dice_min:
                failed.append("dice")
            reason = "passed" if passed else "non_replicated_effect_absent:" + ",".join(failed)
            all_results.append(
                CohortBrainwideReplicationResult(
                    cohort=cohort,
                    passed=passed,
                    reason=reason,
                    pattern_corr=float(corr),
                    dice=float(dice),
                    region_replication_fraction=float(frac),
                    region_table=rep_table,
                    feature_coverage=float(coverage),
                )
            )
        except Exception as exc:
            LOGGER.warning("Brain-wide replication fit failed for %s: %s", cohort, exc)
            all_results.append(
                CohortBrainwideReplicationResult(
                    cohort=cohort,
                    passed=False,
                    reason=f"non_replicated_cohort_mismatch:{exc}",
                    pattern_corr=float("nan"),
                    dice=0.0,
                    region_replication_fraction=0.0,
                    region_table=None,
                    feature_coverage=float(coverage),
                )
            )

    passed = bool(all_results) and all(result.passed for result in all_results)
    if passed:
        reason = "passed"
    elif any("cohort_mismatch" in result.reason for result in all_results):
        reason = "non_replicated_cohort_mismatch"
    else:
        reason = "non_replicated_effect_absent"

    valid = [result for result in all_results if result.region_table is not None]
    if valid:
        pattern_corr = float(pd.Series([result.pattern_corr for result in valid], dtype=float).mean())
        dice = float(pd.Series([result.dice for result in valid], dtype=float).mean())
        frac = float(pd.Series([result.region_replication_fraction for result in valid], dtype=float).mean())
    else:
        pattern_corr = float("nan")
        dice = 0.0
        frac = 0.0
    return BrainwideReplicationResult(
        passed=passed,
        reason=reason,
        cohort_results=all_results,
        harmonized=harmonized,
        pattern_corr=pattern_corr,
        dice=dice,
        region_replication_fraction=frac,
    )
