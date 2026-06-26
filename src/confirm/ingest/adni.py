"""ADNI ADNIMERGE adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from confirm.ingest.base import CohortAdapter
from confirm.schema import normalize_sex, validate_canonical

ADNIMERGE_COLUMNS: tuple[str, ...] = (
    "PTID",
    "VISCODE",
    "SITE",
    "AGE",
    "PTGENDER",
    "DX",
    "ICV",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
    "Ventricles",
    "FDG",
    "AV45",
    "MMSE",
    "CDRSB",
    "ADAS13",
    "MOCA",
    "PTEDUCAT",
    "APOE4",
)

IDP_MAP: dict[str, str] = {
    "smri_hippocampus": "Hippocampus",
    "smri_wholebrain": "WholeBrain",
    "smri_entorhinal": "Entorhinal",
    "smri_fusiform": "Fusiform",
    "smri_midtemp": "MidTemp",
    "smri_ventricles": "Ventricles",
    "pet_fdg_suvr": "FDG",
    "pet_av45": "AV45",
}

BEHAVIORAL_MAP: dict[str, str] = {
    "beh_mmse": "MMSE",
    "beh_cdrsb": "CDRSB",
    "beh_adas13": "ADAS13",
    "beh_moca": "MOCA",
    "beh_educ": "PTEDUCAT",
    "beh_apoe4": "APOE4",
}


class AdniAdapter(CohortAdapter):
    """Normalize ADNI ADNIMERGE tabular derivatives into canonical rows."""

    def __init__(self, adnimerge: str | Path | pd.DataFrame = "data/raw/ADNIMERGE.xlsx", *, all_visits: bool = False):
        raw_paths = {} if isinstance(adnimerge, pd.DataFrame) else {"adnimerge": adnimerge}
        super().__init__("ADNI", raw_paths=raw_paths)
        self.adnimerge = adnimerge
        self.all_visits = all_visits

    def _load_raw(self) -> pd.DataFrame:
        if isinstance(self.adnimerge, pd.DataFrame):
            return self.adnimerge.copy()
        return pd.read_excel(self.adnimerge, usecols=lambda column: column in ADNIMERGE_COLUMNS)

    @staticmethod
    def _source(raw: pd.DataFrame, column: str) -> pd.Series:
        if column in raw.columns:
            return raw[column]
        return pd.Series(pd.NA, index=raw.index)

    def to_canonical(self) -> pd.DataFrame:
        raw = self._load_raw()
        if not self.all_visits:
            raw = raw[self._source(raw, "VISCODE").astype("string") == "bl"].copy()

        out = pd.DataFrame(index=raw.index)
        out["subject_id"] = self._source(raw, "PTID").astype("string")
        out["session"] = self._source(raw, "VISCODE").astype("string")
        out["cohort"] = "ADNI"
        out["site"] = self._source(raw, "SITE").astype("string")
        out["field_strength"] = pd.NA
        out["fs_version"] = pd.NA
        out["age"] = pd.to_numeric(self._source(raw, "AGE"), errors="coerce")
        out["sex"] = normalize_sex(self._source(raw, "PTGENDER"))
        out["dx"] = self._source(raw, "DX").astype("string")
        out["eTIV"] = self._source(raw, "ICV")

        for canonical, source in IDP_MAP.items():
            out[canonical] = self._source(raw, source)
        for canonical, source in BEHAVIORAL_MAP.items():
            out[canonical] = self._source(raw, source)

        out = out[out["age"].notna() & out["sex"].notna()].reset_index(drop=True)
        return validate_canonical(out)
