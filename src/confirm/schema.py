"""Canonical table schema and validation utilities."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

CANONICAL_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "session",
    "cohort",
    "site",
    "field_strength",
    "fs_version",
    "age",
    "sex",
    "dx",
    "eTIV",
)

REQUIRED_COLUMNS: tuple[str, ...] = ("subject_id", "cohort", "site", "age", "sex")
OPTIONAL_COLUMNS: tuple[str, ...] = ("session", "field_strength", "fs_version", "dx", "eTIV")
IDP_PREFIXES: tuple[str, ...] = ("smri_", "pet_", "fc_")
RAW_IDP_PREFIXES: tuple[str, ...] = ("raw_",)
BEHAVIORAL_PREFIXES: tuple[str, ...] = ("beh_",)

CANONICAL_IDPS: dict[str, str] = {
    "smri_meanthickness": "Global mean cortical thickness.",
    "smri_gm_total": "Total gray-matter proxy from VBM maps.",
    "smri_hippocampus_lh": "Left hippocampal volume.",
    "smri_hippocampus_rh": "Right hippocampal volume.",
    "smri_hippocampus_total": "Bilateral hippocampal volume.",
    "pet_fdg_suvr": "FDG PET SUVR.",
    "fc_mean_abs": "Mean absolute functional connectivity.",
    "fc_mean_positive": "Mean positive functional connectivity.",
    "fc_within_network": "Mean within-network functional connectivity.",
    "fc_between_network": "Mean between-network functional connectivity.",
}


class SubjectRow(BaseModel):
    """Loose pydantic row model for documentation and optional external validation."""

    model_config = ConfigDict(extra="allow")

    subject_id: str
    session: str = "ses-01"
    cohort: str
    site: str
    field_strength: Optional[float] = None
    fs_version: Optional[str] = None
    age: float
    sex: str
    dx: Optional[str] = None
    eTIV: Optional[float] = None


def normalize_sex(series: pd.Series) -> pd.Series:
    """Normalize common sex encodings into ``M``/``F`` with missing values preserved."""

    def one(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip().upper()
        if text in {"M", "MALE", "MAN", "1", "TRUE"}:
            return "M"
        if text in {"F", "FEMALE", "WOMAN", "2", "0", "FALSE"}:
            return "F"
        return pd.NA

    return series.map(one).astype("string")


def idp_columns(columns: Iterable[str]) -> list[str]:
    """Return canonical imaging-derived phenotype columns."""

    return [col for col in columns if col.startswith(IDP_PREFIXES + RAW_IDP_PREFIXES)]


def behavioral_columns(columns: Iterable[str]) -> list[str]:
    """Return canonical behavioral columns."""

    return [col for col in columns if col.startswith(BEHAVIORAL_PREFIXES)]


def validate_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a canonical subject-level table.

    The function returns a defensive copy with canonical defaults and numeric
    coercions applied. It raises ``ValueError`` for missing required columns,
    invalid sex encodings, or absence of imaging-derived phenotypes.
    """

    out = df.copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}")

    if "session" not in out.columns:
        out["session"] = "ses-01"
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["subject_id"] = out["subject_id"].astype(str)
    out["session"] = out["session"].fillna("ses-01").astype(str)
    out["cohort"] = out["cohort"].astype(str)
    out["site"] = out["site"].fillna("unknown").astype(str)
    out["sex"] = normalize_sex(out["sex"])

    invalid_sex = out["sex"].isna()
    if invalid_sex.any():
        raise ValueError(f"Invalid sex encoding in {int(invalid_sex.sum())} rows")

    for col in ("age", "field_strength", "eTIV"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["age"].isna().any():
        raise ValueError("Canonical age contains missing or non-numeric values")

    idps = idp_columns(out.columns)
    if not idps:
        raise ValueError("Canonical table must contain at least one smri_*, pet_*, fc_*, or raw_* IDP column")
    for col in idps + behavioral_columns(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    ordered = [col for col in CANONICAL_COLUMNS if col in out.columns]
    extras = [col for col in out.columns if col not in ordered]
    return out[ordered + extras]
