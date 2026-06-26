"""Primary statistical analyses for CONFIRM contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

from confirm.contract import ClaimContract
from confirm.results import EffectResult

ModelKind = Literal["ols", "robust"]
STRUCTURAL_CONFOUND_COLUMNS = ("site", "scanner", "field_strength", "fs_version")


def direction_sign(contract: ClaimContract) -> int:
    """Return expected effect sign; ``0`` means two-sided."""

    if contract.estimand.direction == "negative":
        return -1
    if contract.estimand.direction == "positive":
        return 1
    return 0


def directionally_consistent(beta: float, contract: ClaimContract) -> bool:
    """Check whether an estimate matches the declared direction."""

    sign = direction_sign(contract)
    if sign == 0:
        return True
    return math.copysign(1.0, beta) == float(sign) if beta != 0 else False


def effective_multiplicity_family_size(contract: ClaimContract, observed_family_size: int = 1) -> int:
    """Return the maximum declared, observed, and search-lineage family size."""

    return max(
        1,
        int(observed_family_size),
        int(contract.gates.multiplicity.family_size),
        int(contract.search_provenance.family_size),
    )


def multiplicity_threshold(contract: ClaimContract, observed_family_size: int = 1, alpha: float | None = None) -> float:
    """Conservative per-hypothesis threshold for a possibly selected scalar claim."""

    gate_alpha = contract.gates.multiplicity.alpha if alpha is None else float(alpha)
    return gate_alpha / effective_multiplicity_family_size(contract, observed_family_size)


def covariates_for_fitting(contract: ClaimContract, covariates: Iterable[str]) -> list[str]:
    """Return covariates after removing columns that define the estimand predictor."""

    estimand_terms = {contract.estimand.predictor}
    if contract.estimand.group is not None:
        estimand_terms.add(contract.estimand.group.var)
    return [cov for cov in dict.fromkeys(covariates) if cov not in estimand_terms]


def fdr_bh_q_values(p_values: Iterable[float], family_size: int | None = None) -> list[float]:
    """Benjamini-Hochberg q-values with optional unobserved family members."""

    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q.tolist()
    idx = np.where(finite)[0]
    p_finite = p[idx]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    m = float(max(len(ranked), int(family_size or len(ranked)), 1))
    adjusted = ranked * m / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[idx[order]] = adjusted
    return q.tolist()


def ci_overlap(a: EffectResult, b: EffectResult) -> bool:
    """Return true when two confidence intervals overlap."""

    return max(a.ci_low, b.ci_low) <= min(a.ci_high, b.ci_high)


def _numeric_if_complete(series: pd.Series) -> pd.Series | None:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype(float) if numeric.notna().all() else None


def _association_p_value(work: pd.DataFrame, predictor_col: str, confound_col: str, predictor_is_group: bool) -> tuple[str, float] | None:
    from scipy import stats
    from scipy.stats import chi2_contingency

    predictor = work[predictor_col]
    confound = work[confound_col]
    predictor_numeric = None if predictor_is_group else _numeric_if_complete(predictor)
    confound_numeric = _numeric_if_complete(confound)

    if predictor_is_group or predictor_numeric is None:
        table = pd.crosstab(predictor.astype("string"), confound.astype("string"))
        if table.shape[0] < 2 or table.shape[1] < 2:
            return None
        return "chi_square", float(chi2_contingency(table, correction=False).pvalue)

    if confound_numeric is None:
        groups = [
            predictor_numeric.loc[confound.astype("string") == level].dropna()
            for level in sorted(confound.astype("string").dropna().unique())
        ]
        groups = [group for group in groups if len(group) >= 2]
        if len(groups) < 2:
            return None
        return "anova", float(stats.f_oneway(*groups).pvalue)

    if len(work) < 3 or predictor_numeric.nunique(dropna=True) < 2 or confound_numeric.nunique(dropna=True) < 2:
        return None
    return "pearson", float(stats.pearsonr(predictor_numeric, confound_numeric).pvalue)


def audit_confound_completeness(
    discovery_df: pd.DataFrame,
    contract: ClaimContract,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Audit omitted structural confounds for association with the estimand predictor/group."""

    data = _apply_inclusion(discovery_df, contract)
    predictor_col = contract.estimand.predictor
    predictor_is_group = False
    if contract.estimand.type == "group_diff":
        group = contract.estimand.group
        if group is None:
            raise ValueError("group_diff estimand requires group spec")
        predictor_col = group.var
        if predictor_col not in data.columns:
            raise ValueError(f"Group variable {predictor_col!r} not found")
        data = data[data[predictor_col].isin([group.case, group.control])].copy()
        predictor_is_group = True
    elif predictor_col not in data.columns:
        raise ValueError(f"Predictor {predictor_col!r} not found")

    declared_covariates = set(contract.covariates)
    details: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for confound_col in STRUCTURAL_CONFOUND_COLUMNS:
        if confound_col in declared_covariates or confound_col not in data.columns:
            continue
        if confound_col == predictor_col:
            values = data[predictor_col].dropna()
            if values.nunique(dropna=True) < 2:
                continue
            item = {
                "confound": confound_col,
                "test": "identity",
                "p": 0.0,
                "n": int(len(values)),
                "levels": int(values.nunique(dropna=True)),
                "associated": True,
            }
            details.append(item)
            failures.append(item)
            continue
        work = data[[predictor_col, confound_col]].dropna().copy()
        if work.empty or work[predictor_col].nunique(dropna=True) < 2 or work[confound_col].nunique(dropna=True) < 2:
            continue
        tested = _association_p_value(work, predictor_col, confound_col, predictor_is_group)
        if tested is None:
            continue
        test_name, p_value = tested
        associated = bool(math.isfinite(p_value) and p_value < alpha)
        item = {
            "confound": confound_col,
            "test": test_name,
            "p": p_value,
            "n": int(len(work)),
            "levels": int(work[confound_col].nunique(dropna=True)),
            "associated": associated,
        }
        details.append(item)
        if associated:
            failures.append(item)

    return {
        "passed": not failures,
        "reason": "passed" if not failures else "confound_incomplete",
        "alpha": float(alpha),
        "predictor": predictor_col,
        "tested_confound_count": len(details),
        "failures": failures,
        "details": details,
    }


def _apply_inclusion(df: pd.DataFrame, contract: ClaimContract) -> pd.DataFrame:
    if not contract.inclusion:
        return df.copy()
    try:
        return df.query(contract.inclusion, engine="python").copy()
    except Exception as exc:
        raise ValueError(f"Invalid inclusion query {contract.inclusion!r}: {exc}") from exc


def _analysis_frame(df: pd.DataFrame, contract: ClaimContract, covariates: list[str]) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    data = _apply_inclusion(df, contract)
    outcome = contract.estimand.outcome
    if not isinstance(outcome, str):
        raise ValueError("fit_effect requires a scalar outcome; use run_brainwide for region profiles")
    if outcome not in data.columns:
        raise ValueError(f"Outcome {outcome!r} not found")

    predictor_name = "__confirm_predictor__"
    if contract.estimand.type == "group_diff":
        group = contract.estimand.group
        if group is None:
            raise ValueError("group_diff estimand requires group spec")
        if group.var not in data.columns:
            raise ValueError(f"Group variable {group.var!r} not found")
        data = data[data[group.var].isin([group.case, group.control])].copy()
        data[predictor_name] = (data[group.var] == group.case).astype(float)
        skip_covars = {outcome, group.var, predictor_name}
    else:
        predictor = contract.estimand.predictor
        if predictor not in data.columns:
            raise ValueError(f"Predictor {predictor!r} not found")
        data[predictor_name] = pd.to_numeric(data[predictor], errors="coerce")
        skip_covars = {outcome, predictor, predictor_name}

    needed = [outcome, predictor_name] + [cov for cov in covariates if cov not in skip_covars]
    missing = [col for col in needed if col not in data.columns]
    if missing:
        raise ValueError(f"Analysis columns not found: {missing}")
    data = data[needed].dropna().copy()
    if len(data) < max(8, len(needed) + 3):
        raise ValueError(f"Too few complete rows for analysis: n={len(data)}")

    y = pd.to_numeric(data[outcome], errors="coerce")
    x_parts = [pd.Series(pd.to_numeric(data[predictor_name], errors="coerce"), name=predictor_name)]
    for cov in covariates:
        if cov in skip_covars:
            continue
        series = data[cov]
        if pd.api.types.is_numeric_dtype(series):
            x_parts.append(pd.Series(pd.to_numeric(series, errors="coerce"), name=cov))
        else:
            dummies = pd.get_dummies(series.astype("string"), prefix=cov, drop_first=True, dtype=float)
            if not dummies.empty:
                x_parts.append(dummies)
    x = pd.concat(x_parts, axis=1)
    complete = pd.concat([y.rename(outcome), x], axis=1).dropna()
    y = complete[outcome].astype(float)
    x = complete.drop(columns=[outcome]).astype(float)
    x = x.loc[:, x.nunique(dropna=False) > 1]
    if predictor_name not in x.columns:
        raise ValueError("Predictor has no variation after filtering")
    return y, x, data.loc[complete.index]


def _standardized_effect(beta: float, se: float, dof: float, y: pd.Series, predictor: pd.Series, contract: ClaimContract) -> float:
    if contract.estimand.type == "group_diff":
        groups = predictor.astype(int)
        case = y[groups == 1]
        control = y[groups == 0]
        pooled = math.sqrt(((len(case) - 1) * case.var(ddof=1) + (len(control) - 1) * control.var(ddof=1)) / max(len(case) + len(control) - 2, 1))
        return float((case.mean() - control.mean()) / pooled) if pooled > 0 else float("nan")
    if se <= 0 or dof <= 0:
        return float("nan")
    t_value = beta / se
    return float(math.copysign(math.sqrt((t_value * t_value) / (t_value * t_value + dof)), beta))


def _diagnostics(y: pd.Series, x: pd.DataFrame, fitted: object) -> dict[str, object]:
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.stats.diagnostic import het_breuschpagan

    resid = pd.Series(getattr(fitted, "resid", np.array([])), index=y.index).astype(float)
    out: dict[str, object] = {
        "residual_mean": float(resid.mean()),
        "residual_sd": float(resid.std(ddof=1)),
    }
    try:
        if 3 <= len(resid) <= 5000:
            out["normality_p"] = float(stats.shapiro(resid).pvalue)
        elif len(resid) > 5000:
            out["normality_p"] = float(stats.normaltest(resid).pvalue)
    except Exception as exc:
        out["normality_error"] = str(exc)
    try:
        exog = sm.add_constant(x, has_constant="add")
        out["homoscedasticity_p"] = float(het_breuschpagan(resid, exog)[1])
    except Exception as exc:
        out["homoscedasticity_error"] = str(exc)
    try:
        influence = fitted.get_influence()
        cooks = pd.Series(influence.cooks_distance[0], index=y.index).sort_values(ascending=False).head(5)
        out["cooks_distance_top"] = [{"row": str(idx), "value": float(value)} for idx, value in cooks.items()]
    except Exception as exc:
        out["cooks_distance_error"] = str(exc)
    return out


def fit_effect(df: pd.DataFrame, contract: ClaimContract, covariates: list[str] | None = None, model: ModelKind = "ols") -> EffectResult:
    """Fit the contract estimand with the provided covariate set."""

    import statsmodels.api as sm

    covars = covariates_for_fitting(contract, contract.covariates if covariates is None else covariates)
    y, x, _ = _analysis_frame(df, contract, covars)
    exog = sm.add_constant(x, has_constant="add")
    predictor_col = "__confirm_predictor__"
    if model == "robust":
        fitted = sm.RLM(y, exog, M=sm.robust.norms.HuberT()).fit()
        dof = float(len(y) - exog.shape[1])
    else:
        fitted = sm.OLS(y, exog).fit()
        dof = float(fitted.df_resid)

    params = pd.Series(fitted.params, index=exog.columns)
    bse = pd.Series(fitted.bse, index=exog.columns)
    pvalues = pd.Series(fitted.pvalues, index=exog.columns)
    ci_all = fitted.conf_int(alpha=0.05)
    if not isinstance(ci_all, pd.DataFrame):
        ci_all = pd.DataFrame(ci_all, index=exog.columns)
    beta = float(params[predictor_col])
    se = float(bse[predictor_col])
    p = float(pvalues[predictor_col])
    ci = ci_all.loc[predictor_col]
    standardized = _standardized_effect(beta, se, dof, y, x[predictor_col], contract)
    diagnostics = _diagnostics(y, x, fitted) if model == "ols" else {"model": "robust_hubert"}
    return EffectResult(
        beta=beta,
        se=se,
        ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]),
        p=p,
        n=int(len(y)),
        dof=dof,
        standardized_effect=standardized,
        diagnostics=diagnostics,
    )


def run_primary(df: pd.DataFrame, contract: ClaimContract) -> EffectResult:
    """Run the primary OLS analysis declared by a claim contract."""

    return fit_effect(df, contract, covariates=contract.covariates, model="ols")
