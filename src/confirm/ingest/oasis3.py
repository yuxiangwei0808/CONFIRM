"""OASIS-3 FreeSurfer adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from confirm.ingest.base import CohortAdapter
from confirm.schema import normalize_sex, validate_canonical

DEFAULT_DATA_DIR = Path("data/raw/oasis3_extracted/data/qneuromark/Data/OASIS/OASIS3/Data_info")

FS_REL = Path("OASIS3_data_files/scans/FS-Freesurfer_output/resources/csv/files/OASIS3_Freesurfer_output.csv")
DEMO_REL = Path("OASIS3_data_files/scans/demo-demographics/resources/csv/files/OASIS3_demographics.csv")
CDR_REL = Path(
    "OASIS3_data_files/scans/"
    "UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/resources/csv/files/OASIS3_UDSb4_cdr.csv"
)


def _session_day(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.extract(r"_d(\d+)", expand=False), errors="coerce")


def _sum_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.Series(pd.NA, index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def _cdr_to_dx(value: object, dementia_cdr_min: float = 0.0) -> object:
    cdr = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(cdr):
        return pd.NA
    if cdr == 0:
        return "CN"
    if dementia_cdr_min <= 0:
        return "Dementia"
    return "Dementia" if cdr >= dementia_cdr_min else pd.NA


class Oasis3Adapter(CohortAdapter):
    """Normalize local OASIS-3 FreeSurfer and CDR CSV derivatives."""

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR, *, dementia_cdr_min: float = 0.0):
        super().__init__("OASIS3", raw_paths={"data_dir": data_dir})
        self.data_dir = Path(data_dir)
        self.dementia_cdr_min = float(dementia_cdr_min)

    def _read_csv(self, rel_path: Path) -> pd.DataFrame:
        path = self.data_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Required OASIS3 CSV not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _first_freesurfer_visit(fs: pd.DataFrame) -> pd.DataFrame:
        qc = fs["FS QC Status"].astype("string").str.lower()
        fs = fs[qc.str.startswith("passed", na=False)].copy()
        fs["mr_day"] = _session_day(fs["MR_session"])
        return fs.sort_values(["Subject", "mr_day", "MR_session"]).drop_duplicates("Subject", keep="first")

    @staticmethod
    def _nearest_cdr(fs: pd.DataFrame, cdr: pd.DataFrame) -> pd.DataFrame:
        cdr = cdr.copy()
        cdr["visit_day"] = pd.to_numeric(cdr["days_to_visit"], errors="coerce")
        cdr_groups = {subject: frame for subject, frame in cdr.groupby("OASISID", sort=False)}
        matched: list[pd.Series] = []
        for _, row in fs.iterrows():
            candidates = cdr_groups.get(row["Subject"])
            if candidates is None or candidates.empty:
                matched.append(pd.Series(dtype=object))
                continue
            day = row["mr_day"]
            if pd.isna(day):
                selected = candidates.sort_values("visit_day").iloc[0]
            else:
                selected = candidates.loc[(candidates["visit_day"] - day).abs().idxmin()]
            matched.append(selected)
        return pd.DataFrame(matched, index=fs.index)

    def to_canonical(self) -> pd.DataFrame:
        fs = self._first_freesurfer_visit(self._read_csv(FS_REL))
        demo = self._read_csv(DEMO_REL)
        cdr = self._nearest_cdr(fs, self._read_csv(CDR_REL))

        merged = fs.merge(demo, left_on="Subject", right_on="OASISID", how="left", suffixes=("", "_demo"))
        cdr = cdr.add_prefix("cdr_")
        merged = pd.concat([merged.reset_index(drop=True), cdr.reset_index(drop=True)], axis=1)

        out = pd.DataFrame(index=merged.index)
        out["subject_id"] = merged["Subject"].astype("string")
        out["session"] = merged["MR_session"].astype("string")
        out["cohort"] = "OASIS3"
        out["site"] = "OASIS3"
        out["field_strength"] = pd.NA
        out["fs_version"] = merged["version"].astype("string")
        out["age"] = pd.to_numeric(merged["cdr_age at visit"], errors="coerce")
        out["sex"] = normalize_sex(merged["GENDER"])
        out["dx"] = merged["cdr_CDRTOT"].map(lambda value: _cdr_to_dx(value, self.dementia_cdr_min))
        out["eTIV"] = merged["IntraCranialVol"]

        out["smri_hippocampus"] = merged["TOTAL_HIPPOCAMPUS_VOLUME"]
        out["smri_wholebrain"] = merged["SupraTentorialVol"]
        out["smri_entorhinal"] = _sum_columns(merged, ["lh_entorhinal_volume", "rh_entorhinal_volume"])
        out["smri_fusiform"] = _sum_columns(merged, ["lh_fusiform_volume", "rh_fusiform_volume"])
        out["smri_midtemp"] = _sum_columns(merged, ["lh_middletemporal_volume", "rh_middletemporal_volume"])
        out["smri_ventricles"] = _sum_columns(
            merged,
            [
                "3rd-Ventricle_volume",
                "4th-Ventricle_volume",
                "5th-Ventricle_volume",
                "Left-Inf-Lat-Vent_volume",
                "Left-Lateral-Ventricle_volume",
                "Right-Inf-Lat-Vent_volume",
                "Right-Lateral-Ventricle_volume",
            ],
        )
        if "cdr_MMSE" in merged.columns:
            out["beh_mmse"] = merged["cdr_MMSE"]
        if "cdr_CDRTOT" in merged.columns:
            out["beh_cdr_global"] = merged["cdr_CDRTOT"]
        if "cdr_CDRSUM" in merged.columns:
            out["beh_cdrsb"] = merged["cdr_CDRSUM"]

        out = out[out["age"].notna() & out["sex"].notna()].reset_index(drop=True)
        return validate_canonical(out)
