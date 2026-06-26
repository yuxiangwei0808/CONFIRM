"""Bounded multiverse stability analysis."""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd

from confirm.analysis import directionally_consistent, fit_effect, multiplicity_threshold
from confirm.brainwide import run_brainwide
from confirm.contract import ClaimContract
from confirm.results import EffectResult, MultiverseResult, MultiverseSpecResult, RegionTable

LOGGER = logging.getLogger(__name__)


DEFAULT_FORKS: dict[str, list[Any]] = {
    "etiv": [True, False],
    "site": [True, False],
    "outliers": ["none", "drop_z3", "winsorize"],
    "model": ["ols", "robust"],
}


def _fork_grid(forks: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    grid = DEFAULT_FORKS if forks is None else forks
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def _apply_outlier_handling(df: pd.DataFrame, contract: ClaimContract, method: str) -> pd.DataFrame:
    if method == "none":
        return df.copy()
    cols = [contract.estimand.outcome]
    if contract.estimand.type == "association":
        cols.append(contract.estimand.predictor)
    data = df.copy()
    numeric = data[cols].apply(pd.to_numeric, errors="coerce")
    if method == "drop_z3":
        z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
        mask = (z.abs() <= 3).all(axis=1) | z.isna().all(axis=1)
        return data.loc[mask].copy()
    if method == "winsorize":
        # NOTE: Use 1st/99th percentile winsorization as the bounded default.
        for col in cols:
            lo, hi = numeric[col].quantile([0.01, 0.99])
            data[col] = numeric[col].clip(lo, hi)
        return data
    raise ValueError(f"Unknown outlier fork: {method}")


def _covariates_for_spec(contract: ClaimContract, spec: dict[str, Any]) -> list[str]:
    covars = list(contract.covariates)
    if not spec.get("etiv", True):
        covars = [cov for cov in covars if cov != "eTIV"]
    if not spec.get("site", True):
        covars = [cov for cov in covars if cov != "site"]
    return covars


def run_multiverse(df: pd.DataFrame, contract: ClaimContract, forks: dict[str, list[Any]] | None = None) -> MultiverseResult:
    """Run the declared bounded fork grid and summarize stability."""

    spec_results: list[MultiverseSpecResult] = []
    for i, spec in enumerate(_fork_grid(forks)):
        spec_id = ",".join(f"{key}={value}" for key, value in spec.items())
        try:
            fork_df = _apply_outlier_handling(df, contract, str(spec.get("outliers", "none")))
            effect = fit_effect(
                fork_df,
                contract,
                covariates=_covariates_for_spec(contract, spec),
                model="robust" if spec.get("model") == "robust" else "ols",
            )
            same_sign = directionally_consistent(effect.beta, contract)
            significant = effect.p < multiplicity_threshold(contract)
            spec_results.append(
                MultiverseSpecResult(
                    spec_id=spec_id or f"spec_{i}",
                    same_sign=same_sign,
                    significant=significant,
                    beta=effect.beta,
                    p=effect.p,
                    n=effect.n,
                )
            )
        except Exception as exc:
            LOGGER.warning("Multiverse fork failed (%s): %s", spec_id, exc)
            spec_results.append(
                MultiverseSpecResult(
                    spec_id=spec_id or f"spec_{i}",
                    same_sign=False,
                    significant=False,
                    beta=float("nan"),
                    p=float("nan"),
                    n=0,
                    status="error",
                    error=str(exc),
                )
            )
    ok = [spec for spec in spec_results if spec.status == "ok"]
    consistent = [spec for spec in ok if spec.same_sign and spec.significant]
    fraction = len(consistent) / len(ok) if ok else 0.0
    passed = fraction >= contract.gates.multiverse.min_fraction_consistent
    return MultiverseResult(fraction_consistent=float(fraction), passed=passed, specs=spec_results)


def _best_region_effect(regions: RegionTable) -> EffectResult:
    ordered = sorted(regions.regions, key=lambda region: (not region.significant, region.effect.p))
    return ordered[0].effect


def _any_region_significant(regions: RegionTable) -> bool:
    return any(region.significant for region in regions.regions)


def _apply_brainwide_outlier_handling(df: pd.DataFrame, contract: ClaimContract, method: str) -> pd.DataFrame:
    if method == "none":
        return df.copy()
    outcomes = contract.estimand.outcome if isinstance(contract.estimand.outcome, list) else [contract.estimand.outcome]
    cols = [col for col in outcomes if col in df.columns]
    if contract.estimand.type == "association" and contract.estimand.predictor in df.columns:
        cols.append(contract.estimand.predictor)
    data = df.copy()
    numeric = data[cols].apply(pd.to_numeric, errors="coerce")
    if method == "drop_z3":
        z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
        mask = (z.abs() <= 3).all(axis=1) | z.isna().all(axis=1)
        return data.loc[mask].copy()
    if method == "winsorize":
        for col in cols:
            lo, hi = numeric[col].quantile([0.01, 0.99])
            data[col] = numeric[col].clip(lo, hi)
        return data
    raise ValueError(f"Unknown brain-wide outlier fork: {method}")


def _brainwide_contract_with_covariates(contract: ClaimContract, covariates: list[str]) -> ClaimContract:
    return contract.model_copy(update={"covariates": covariates})


def run_brainwide_multiverse(
    df: pd.DataFrame,
    primary_regions: RegionTable,
    contract: ClaimContract,
    min_covariates: list[str] | None = None,
    outlier_options: list[str] | None = None,
) -> MultiverseResult:
    """Run a bounded brain-wide fork grid around the primary regional result.

    A fork is consistent when at least half of the primary significant regions
    remain significant with the same beta sign under the fork. If the primary
    analysis has no significant regions, the multiverse is considered
    inconsistent rather than passing vacuously.
    """

    primary = {region.region: region for region in primary_regions.regions}
    primary_sig = {name: region for name, region in primary.items() if region.significant}
    min_covars = list(min_covariates) if min_covariates is not None else [
        cov for cov in contract.covariates if cov not in {"site", "eTIV"}
    ]
    covariate_options = [
        ("full", list(contract.covariates)),
        ("min", min_covars),
    ]
    outliers = outlier_options or ["none", "winsorize", "drop_z3"]
    spec_results: list[MultiverseSpecResult] = []

    for cov_label, covariates in covariate_options:
        for outlier in outliers:
            spec_id = f"covariates={cov_label},outliers={outlier}"
            try:
                spec_contract = _brainwide_contract_with_covariates(contract, covariates)
                spec_df = _apply_brainwide_outlier_handling(df, spec_contract, outlier)
                table = run_brainwide(spec_df, spec_contract)
                table_by_region = {region.region: region for region in table.regions}
                replicated = 0
                for region_name, primary_region in primary_sig.items():
                    spec_region = table_by_region.get(region_name)
                    if spec_region and spec_region.significant and spec_region.effect.beta * primary_region.effect.beta > 0:
                        replicated += 1
                frac = replicated / len(primary_sig) if primary_sig else 0.0
                consistent = frac >= 0.5
                effect = _best_region_effect(table)
                spec_results.append(
                    MultiverseSpecResult(
                        spec_id=spec_id,
                        same_sign=consistent,
                        significant=_any_region_significant(table),
                        beta=effect.beta,
                        p=effect.p,
                        n=effect.n,
                    )
                )
            except Exception as exc:
                LOGGER.warning("Brain-wide multiverse fork failed (%s): %s", spec_id, exc)
                spec_results.append(
                    MultiverseSpecResult(
                        spec_id=spec_id,
                        same_sign=False,
                        significant=False,
                        beta=float("nan"),
                        p=float("nan"),
                        n=0,
                        status="error",
                        error=str(exc),
                    )
                )

    ok = [spec for spec in spec_results if spec.status == "ok"]
    consistent = [spec for spec in ok if spec.same_sign and spec.significant]
    fraction = len(consistent) / len(ok) if ok else 0.0
    return MultiverseResult(
        fraction_consistent=float(fraction),
        passed=fraction >= contract.gates.multiverse.min_fraction_consistent,
        specs=spec_results,
    )
