"""Prepare the NACC cohort as an UNSEEN external benchmark cohort for CONFIRM.

NACC was never used in CONFIRM development. This script joins the provider
FreeSurfer-style ROI volume table (investigator_mri) to the UDS clinical
diagnosis (investigator_ftldlbd, which carries NACCUDSD/NACCALZD/CDRGLOB),
matching each MRI session to its nearest clinical visit, and writes a single
CONFIRM-ready table.

Output: data/prepared_data/external/NACC.parquet with columns
  subject_id, cohort, site, age, sex, dx, smri_icv, smri_* (ROIs)
where dx in {AD, MCI, CN, other}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MRI_PARQUET = "data/prepared_data/misc_tables/data_qneuromark_Data_NACC_data_investigator_mri_nacc65.parquet"
DX_CSV = "data/external_benchmark/nacc_dx.csv"
OUT = "data/prepared_data/external/NACC.parquet"

# NACC ROI column -> standardized smri_ name. L/R columns are summed (bilateral).
# AD-affected (expected atrophy / positives) and AD-spared primary cortices (nulls).
ROI_MAP = {
    "smri_hippocampus": ["HIPPOVOL"],            # AD positive (gold, d~-1.5)
    "smri_entorhinal": ["LENT", "RENT"],          # AD positive (d~-1.5)
    "smri_parahippocampal": ["LPARHIP", "RPARHIP"],  # AD positive
    "smri_midtemporal": ["LMIDTEMP", "RMIDTEMP"],    # AD positive
    "smri_inferiortemporal": ["LINFTEMP", "RINFTEMP"],  # AD positive
    "smri_fusiform": ["LFUS", "RFUS"],            # AD positive
    "smri_lateralventricle": ["LATVENT"],         # AD positive (enlargement)
    "smri_wholebrain": ["NACCBRNV"],              # AD positive (global atrophy)
    # AD-spared primary sensorimotor / visual cortices -> known-null / negative controls
    "smri_pericalcarine": ["LPERCAL", "RPERCAL"],
    "smri_precentral": ["LPRECEN", "RPRECEN"],
    "smri_postcentral": ["LPOSCEN", "RPOSCEN"],
    "smri_paracentral": ["LPARCEN", "RPARCEN"],
    "smri_cuneus": ["LCUN", "RCUN"],
    "smri_cerebellum": ["CERETISS"],            # AD-spared (negative control)
}

DX_COLS = ["NACCID", "NACCVNUM", "VISITYR", "VISITMO", "VISITDAY",
           "NACCUDSD", "NACCALZD", "CDRGLOB", "SEX", "BIRTHYR", "EDUC"]


def _to_date(y, m, d):
    return pd.to_datetime(dict(year=pd.to_numeric(y, errors="coerce"),
                               month=pd.to_numeric(m, errors="coerce").clip(1, 12),
                               day=pd.to_numeric(d, errors="coerce").clip(1, 28)),
                          errors="coerce")


def main() -> None:
    mri = pd.read_parquet(MRI_PARQUET)
    # one MRI per subject: earliest session
    mri["mri_date"] = _to_date(mri["MRIYR"], mri["MRIMO"], mri["MRIDY"])
    mri = mri.sort_values("mri_date").drop_duplicates("NACCID", keep="first").copy()

    dx = pd.read_csv(DX_CSV, usecols=lambda c: c in DX_COLS, low_memory=False)
    dx["visit_date"] = _to_date(dx["VISITYR"], dx["VISITMO"], dx["VISITDAY"])

    # match each MRI to nearest clinical visit (same subject)
    m = mri[["NACCID", "mri_date"]].merge(dx, on="NACCID", how="left")
    m["gap"] = (m["mri_date"] - m["visit_date"]).abs().dt.days
    m = m.sort_values("gap").drop_duplicates("NACCID", keep="first")
    m = m[m["gap"] <= 365]  # require a clinical visit within 1 year of MRI

    out = mri.merge(m[["NACCID", "NACCUDSD", "NACCALZD", "CDRGLOB", "SEX", "BIRTHYR", "EDUC", "gap"]],
                    on="NACCID", how="inner")

    # labels
    udsd = pd.to_numeric(out["NACCUDSD"], errors="coerce")
    alzd = pd.to_numeric(out["NACCALZD"], errors="coerce")
    dx_label = pd.Series("other", index=out.index, dtype="object")
    dx_label[(udsd == 1)] = "CN"
    dx_label[(udsd == 3)] = "MCI"
    dx_label[(udsd == 4) & (alzd == 1)] = "AD"
    out["dx"] = dx_label

    # covariates
    out["age"] = pd.to_numeric(out["MRIYR"], errors="coerce") - pd.to_numeric(out["BIRTHYR"], errors="coerce")
    out["sex"] = pd.to_numeric(out["SEX"], errors="coerce").map({1: "M", 2: "F"})
    out["site"] = out["NACCADC"].astype("string")
    out["subject_id"] = out["NACCID"].astype(str)
    out["cohort"] = "NACC"
    def _clean(s):
        # NACC codes missing as repeating 8s/9s at the column's scale
        # (e.g. 88.8888 for cm^3 ROIs, 888.8888 ventricles, 8888.888/9999.999 ICV).
        s = pd.to_numeric(s, errors="coerce")
        bad = s >= 3000  # cm^3-scale sentinels: 8888.888 / 9999.999 / 10000
        for sentinel in (8.8888, 88.8888, 888.8888, 9.9999, 99.9999, 999.9999, 100.0, 1000.0):
            bad = bad | np.isclose(s, sentinel, rtol=1e-4, atol=1e-4)
        return s.where(~bad & (s > 0))

    out["smri_icv"] = _clean(out["NACCICV"])

    # ROIs: mask NACC missing codes per source column BEFORE bilateral sum,
    # so a subject needs both hemispheres valid (NaN propagates otherwise).
    roi_cols = []
    for name, srcs in ROI_MAP.items():
        if all(s in out.columns for s in srcs):
            vals = None
            for s in srcs:
                cleaned = _clean(out[s])
                vals = cleaned if vals is None else (vals + cleaned)
            out[name] = vals
            roi_cols.append(name)

    keep = ["subject_id", "cohort", "site", "age", "sex", "dx", "smri_icv", *roi_cols]
    final = out[keep].copy()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT, index=False)

    print(f"wrote {OUT}  shape={final.shape}")
    print("dx counts:\n", final["dx"].value_counts(dropna=False))
    print("n sites (centers):", final["site"].nunique())
    print("ROI columns:", roi_cols)
    print("\nmean ROI by dx (AD vs CN sanity — AD should be LOWER on temporal/hippo):")
    for col in ["smri_hippocampus", "smri_entorhinal", "smri_lateralventricle", "smri_pericalcarine"]:
        if col in final.columns:
            g = final[final.dx.isin(["AD", "CN"])].groupby("dx")[col].mean()
            print(f"  {col}: CN={g.get('CN', float('nan')):.0f}  AD={g.get('AD', float('nan')):.0f}")
    print("\nage by dx:\n", final.groupby("dx")["age"].describe()[["mean", "min", "max"]])


if __name__ == "__main__":
    main()
