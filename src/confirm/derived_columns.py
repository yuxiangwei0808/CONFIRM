"""Runtime-derived columns used by CONFIRM without rewriting parquet files."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

CONFIRM_DX = "confirm_dx"
_VIRTUAL_COLUMN_ALIASES = {
    "NACC": {
        "smri_lateralventricle": "smri_ventricles",
        "smri_midtemporal": "smri_midtemp",
    },
}


def cohort_base(cohort: str) -> str:
    """Return the base dataset name for a split or role-specific cohort."""

    base = str(cohort)
    suffixes = (
        "_EXTERNAL_DISC",
        "_EXTERNAL_REP",
        "_HOLDOUT_DISC",
        "_HOLDOUT_REP",
        "_DISC_SITES",
        "_REP_SITES",
        "_HOLDOUT",
        "_DISC",
        "_REP",
        "_CN",
    )
    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for split in ("_DISC_s", "_REP_s", "_HOLDOUT_s"):
        if split in base:
            base = base.split(split, 1)[0]
            break
    for modality_suffix in ("_fMRI", "_sMRI"):
        if base.lower().endswith(modality_suffix.lower()):
            return base[: -len(modality_suffix)]
    return base


def _norm_label(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    numeric = re.fullmatch(r"[-+]?\d+(?:\.0+)?", lowered)
    if numeric:
        return str(int(float(lowered)))
    return lowered


def confirm_dx_mapping(cohort: str) -> dict[str, str]:
    """Return normalized dx-label mapping to case/control for a known cohort."""

    base = cohort_base(cohort).upper()
    if base == "ABCD":
        return {"1": "case", "0": "control"}
    if base in {"ADHD200", "ADHD_SUIJING", "PKU_ADHD"}:
        return {
            "1": "case", "2": "case", "3": "case", "0": "control",
            "adhd": "case", "hc": "control", "control": "control",
        }
    if base == "ABIDE1":
        return {"asd": "case", "autism": "case", "hc": "control", "control": "control"}
    if base == "ABIDE2":
        return {"1": "case", "2": "control", "asd": "case", "hc": "control"}
    if base in {"ADNI", "OASIS3", "OASIS1", "NACC", "AIBL", "BLSA"}:
        return {"dementia": "case", "ad": "case", "cn": "control", "control": "control"}
    if base in {
        "COBRE", "FBIRN", "BSNIP", "BSNIP2", "CHINESESZ", "JH", "OLIN_SZ",
        "CNP", "DS000030", "LA5C", "PK_MPRC", "SHILE_NANJING",
    }:
        return {
            "sz": "case",
            "schz": "case",
            "schizophrenia": "case",
            "psychosis": "case",
            "hc": "control",
            "control": "control",
        }
    return {}


def confirm_dx_levels(cohort: str, dx_levels: Iterable[object]) -> list[str]:
    """Return mapped case/control levels present in a cohort's dx levels."""

    mapping = confirm_dx_mapping(cohort)
    if not mapping:
        return []
    values = {mapping[_norm_label(level)] for level in dx_levels if _norm_label(level) in mapping}
    return [level for level in ("case", "control") if level in values]


def has_confirm_dx(cohort: str, columns: Iterable[str], dx_levels: Iterable[object] | None = None) -> bool:
    """Return true when confirm_dx can be derived for this cohort."""

    column_set = set(columns)
    if CONFIRM_DX in column_set:
        return True
    if "dx" not in column_set or not confirm_dx_mapping(cohort):
        return False
    if dx_levels is None:
        return True
    return set(confirm_dx_levels(cohort, dx_levels)) == {"case", "control"}


def columns_with_virtuals(cohort: str, columns: Iterable[str], dx_levels: Iterable[object] | None = None) -> list[str]:
    """Append virtual columns that are available for the cohort."""

    out = list(columns)
    if CONFIRM_DX not in out and has_confirm_dx(cohort, out, dx_levels):
        out.append(CONFIRM_DX)
    aliases = _VIRTUAL_COLUMN_ALIASES.get(cohort_base(cohort).upper(), {})
    for alias, source in aliases.items():
        if alias not in out and source in out:
            out.append(alias)
    return out


def add_virtual_columns(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Add runtime-derived columns used by contracts."""

    out = df.copy()
    if CONFIRM_DX not in out.columns and "dx" in out.columns:
        mapping = confirm_dx_mapping(cohort)
        if mapping:
            mapped = out["dx"].map(lambda value: mapping.get(_norm_label(value), pd.NA))
            out[CONFIRM_DX] = mapped.astype("string")
    aliases = _VIRTUAL_COLUMN_ALIASES.get(cohort_base(cohort).upper(), {})
    for alias, source in aliases.items():
        if alias not in out.columns and source in out.columns:
            out[alias] = out[source]
    return out
