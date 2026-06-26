"""Brain-wide regional analysis engine."""

from __future__ import annotations

import fnmatch
import logging

import numpy as np
import pandas as pd

from confirm.analysis import directionally_consistent, effective_multiplicity_family_size, fdr_bh_q_values, fit_effect
from confirm.contract import ClaimContract
from confirm.results import RegionEffectResult, RegionTable
from confirm.schema import idp_columns

LOGGER = logging.getLogger(__name__)


def select_region_columns(df: pd.DataFrame, contract: ClaimContract) -> list[str]:
    """Select regional IDP columns declared by a brain-wide contract."""

    candidates = idp_columns(df.columns)
    outcome = contract.estimand.outcome
    if isinstance(outcome, list):
        selected = [col for col in outcome if col in df.columns]
    elif outcome in {"*", "brainwide", "all"}:
        selected = candidates
    elif any(ch in outcome for ch in "*?[]"):
        selected = [col for col in candidates if fnmatch.fnmatch(col, outcome)]
    elif outcome in df.columns and contract.estimand.unit == "scalar":
        selected = [outcome]
    elif outcome in df.columns:
        selected = [outcome]
    else:
        prefix = outcome if outcome.endswith("_") else f"{outcome}_"
        selected = [col for col in candidates if col.startswith(prefix)]

    region_set = contract.estimand.region_set
    if region_set and region_set not in {"all", "shared", "shared_ad_signature", "ad_signature", "dk", "aparc"}:
        if any(col.startswith(region_set) for col in selected):
            selected = [col for col in selected if col.startswith(region_set)]

    selected = sorted(dict.fromkeys(selected))
    if not selected:
        raise ValueError(f"No brain-wide region columns matched outcome={outcome!r} region_set={region_set!r}")
    return selected


def _contract_for_region(contract: ClaimContract, region: str) -> ClaimContract:
    estimand = contract.estimand.model_copy(update={"outcome": region, "unit": "scalar"})
    return contract.model_copy(update={"estimand": estimand})


def run_brainwide(df: pd.DataFrame, contract: ClaimContract) -> RegionTable:
    """Fit the contract estimand per regional IDP and FDR-correct the family."""

    regions = select_region_columns(df, contract)
    effects = []
    fitted_regions = []
    for region in regions:
        try:
            effect = fit_effect(df, _contract_for_region(contract, region), covariates=contract.covariates, model="ols")
            effects.append(effect)
            fitted_regions.append(region)
        except Exception as exc:
            LOGGER.warning("Brain-wide region fit failed for %s: %s", region, exc)
    if not effects:
        raise ValueError("No brain-wide regions could be fitted")

    family_size = effective_multiplicity_family_size(contract, observed_family_size=len(effects))
    q_values = fdr_bh_q_values((effect.p for effect in effects), family_size=family_size)
    alpha = contract.gates.multiplicity.alpha
    region_results = [
        RegionEffectResult(
            region=region,
            effect=effect,
            q=float(q),
            significant=bool(np.isfinite(q) and q <= alpha and directionally_consistent(effect.beta, contract)),
        )
        for region, effect, q in zip(fitted_regions, effects, q_values)
    ]
    return RegionTable(
        regions=region_results,
        alpha=alpha,
        method=contract.gates.multiplicity.method,
        region_set=contract.estimand.region_set,
    )
