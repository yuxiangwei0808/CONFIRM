"""Site/scanner harmonization helpers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _fallback_site_residualization(df: pd.DataFrame, idp_cols: list[str], batch_col: str, covars: list[str]) -> pd.DataFrame:
    """Remove additive site effects while preserving biological covariates."""

    out = df.copy()
    covar_cols = [col for col in covars if col in out.columns and col not in idp_cols and col != batch_col]
    covar_design_parts: list[pd.DataFrame] = []
    for col in covar_cols:
        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            covar_design_parts.append(pd.DataFrame({col: pd.to_numeric(series, errors="coerce")}, index=out.index))
        else:
            covar_design_parts.append(pd.get_dummies(series.astype("string"), prefix=col, drop_first=True, dtype=float))
    site_design = pd.get_dummies(out[batch_col].astype("string"), prefix=batch_col, drop_first=True, dtype=float)
    if covar_design_parts:
        covar_design = pd.concat(covar_design_parts, axis=1)
    else:
        covar_design = pd.DataFrame(index=out.index)
    design = pd.concat([pd.Series(1.0, index=out.index, name="intercept"), covar_design, site_design], axis=1).astype(float)
    site_cols = list(site_design.columns)
    for col in idp_cols:
        y = pd.to_numeric(out[col], errors="coerce")
        complete = pd.concat([y.rename(col), design], axis=1).dropna()
        if len(complete) <= design.shape[1] + 1 or not site_cols:
            continue
        x = complete.drop(columns=[col]).to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, complete[col].to_numpy(dtype=float), rcond=None)
        params = pd.Series(beta, index=complete.drop(columns=[col]).columns)
        site_effect = complete[site_cols].dot(params[site_cols])
        out.loc[complete.index, col] = complete[col] - site_effect
    return out


def combat_harmonize(
    df: pd.DataFrame,
    idp_cols: list[str],
    batch_col: str = "site",
    covars: list[str] | None = None,
    min_batch_size: int = 5,
) -> pd.DataFrame:
    """Harmonize IDP columns across batches while preserving covariates.

    ``neuroHarmonize`` is attempted first. If it is unavailable or rejects the
    local data shape, CONFIRM falls back to deterministic additive site-effect
    removal and logs the reason.
    """

    covars = list(covars or [])
    if batch_col not in df.columns:
        raise ValueError(f"Batch column {batch_col!r} not found")
    missing = [col for col in idp_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot harmonize missing IDP columns: {missing}")
    out = df.copy()
    if out[batch_col].nunique(dropna=False) < 2:
        # Single batch (e.g. within-site split-half replication): nothing to harmonize.
        # ComBat would divide by a zero between-site variance and return NaNs.
        LOGGER.info("Single batch in %r; skipping harmonization.", batch_col)
        return out
    counts = out[batch_col].value_counts(dropna=False)
    if (counts < min_batch_size).any():
        small = counts[counts < min_batch_size].index.astype(str).tolist()
        LOGGER.warning("Skipping ComBat for small batches %s; using fallback residualization", small)
        return _fallback_site_residualization(out, idp_cols, batch_col, covars)

    try:
        from neuroHarmonize import harmonizationLearn

        covar_frame = pd.DataFrame({"SITE": out[batch_col].astype(str)}, index=out.index)
        for covar in covars:
            if covar in out.columns and covar not in idp_cols and covar != batch_col:
                series = out[covar]
                if pd.api.types.is_numeric_dtype(series):
                    covar_frame[covar] = pd.to_numeric(series, errors="coerce")
                else:
                    covar_frame[covar] = pd.Categorical(series.astype("string")).codes
        data = out[idp_cols].apply(pd.to_numeric, errors="coerce")
        complete = pd.concat([covar_frame, data], axis=1).dropna()
        if len(complete) < len(out):
            LOGGER.warning("ComBat complete-case harmonization dropped %d rows", len(out) - len(complete))
        model, harmonized = harmonizationLearn(data.loc[complete.index].to_numpy(dtype=float), covar_frame.loc[complete.index])
        del model
        out.loc[complete.index, idp_cols] = harmonized
        return out
    except Exception as exc:
        LOGGER.warning("neuroHarmonize failed (%s); using deterministic fallback", exc)
        return _fallback_site_residualization(out, idp_cols, batch_col, covars)

