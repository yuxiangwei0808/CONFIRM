"""TRUE cross-cohort must-win: AD-vs-CN atrophy discovered in ADNI, replicated in OASIS-3
(different cohorts, scanners, FreeSurfer versions). Shared AD-signature regions; ComBat harmonized.

OASIS-3 join: baseline FreeSurfer scan per subject + nearest CDR assessment (by days) -> dx
(CDR==0 -> CN, CDR>=1 -> Dementia). Run: python -m pilots.run_adni_oasis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from confirm.schema import normalize_sex
from pilots.run_adni import load_adni
from pilots.run_open_pilots import build_contract, run_all

OA = "data/raw/oasis3_extracted"

# shared regions present in BOTH cohorts: (canonical, OASIS3 source cols to sum, direction)
SHARED = [
    ("smri_hippocampus", ["TOTAL_HIPPOCAMPUS_VOLUME"], "negative"),
    ("smri_entorhinal", ["lh_entorhinal_volume", "rh_entorhinal_volume"], "negative"),
    ("smri_midtemp", ["lh_middletemporal_volume", "rh_middletemporal_volume"], "negative"),
    ("smri_fusiform", ["lh_fusiform_volume", "rh_fusiform_volume"], "negative"),
    ("smri_ventricles", ["Left-Lateral-Ventricle_volume", "Right-Lateral-Ventricle_volume"], "positive"),
]


def _find(name):
    return next(Path(OA).rglob(name))


def load_oasis3() -> pd.DataFrame:
    fs = pd.read_csv(_find("OASIS3_Freesurfer_output.csv"))
    demo = pd.read_csv(_find("OASIS3_demographics.csv"))
    cdr = pd.read_csv(_find("OASIS3_UDSb4_cdr.csv"))
    fs["days"] = fs["MR_session"].str.extract(r"_d(\d+)").astype(float)
    base = fs.sort_values("days").groupby("Subject", as_index=False).first()  # baseline scan/subject

    cdr = cdr.dropna(subset=["days_to_visit", "CDRTOT"])
    def nearest_cdr(row):
        sub = cdr[cdr["OASISID"] == row["Subject"]]
        if sub.empty:
            return np.nan
        return sub.loc[(sub["days_to_visit"] - row["days"]).abs().idxmin(), "CDRTOT"]
    base["CDRTOT"] = base.apply(nearest_cdr, axis=1)

    demo_map = demo.set_index("OASISID")
    out = pd.DataFrame()
    out["subject_id"] = base["Subject"].astype(str)
    out["session"] = "bl"
    out["cohort"] = "OASIS3"
    out["site"] = "OASIS3"
    age_entry = base["Subject"].map(demo_map["AgeatEntry"])
    out["age"] = pd.to_numeric(age_entry, errors="coerce") + base["days"] / 365.25
    out["sex"] = normalize_sex(base["Subject"].map(demo_map["GENDER"]))
    cdrt = pd.to_numeric(base["CDRTOT"], errors="coerce")
    out["dx"] = np.where(cdrt == 0, "CN", np.where(cdrt >= 1, "Dementia", "MCI"))
    out["eTIV"] = pd.to_numeric(base["IntraCranialVol"], errors="coerce")
    for canon, srcs, _ in SHARED:
        out[canon] = base[srcs].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return out


def main(argv=None) -> int:
    adni = load_adni().copy()
    adni["cohort"] = "ADNI"  # single discovery cohort
    oa = load_oasis3()
    print(f"ADNI disc: n={len(adni)} dx={adni['dx'].value_counts(dropna=False).to_dict()}")
    print(f"OASIS-3 rep: n={len(oa)} dx={oa['dx'].value_counts(dropna=False).to_dict()}")

    print("\n==== TRUE cross-cohort: AD vs CN, discovery ADNI -> replicate OASIS-3 (ComBat harmonized) ====")
    print(f"{'region':<18}{'verdict':<16}{'ADNI beta':>12}{'ADNI p':>11}{'repl?':>8}  reason")
    for canon, _, direction in SHARED:
        c = build_contract(
            f"xcohort_{canon}", kind="group_diff", outcome=canon, predictor="dx",
            direction=direction, covars=["age", "sex", "eTIV"], disc="ADNI", rep="OASIS3",
            group={"var": "dx", "case": "Dementia", "control": "CN"}, require=["age", "sex", "eTIV"],
        )
        try:
            v, pr = run_all(adni, oa, c)
            print(f"{canon:<18}{v.label:<16}{pr.beta:>12.1f}{pr.p:>11.1e}{str(v.gates.get('replication')):>8}  {v.rationale[:70]}")
        except Exception as exc:
            print(f"{canon:<18}{'ERROR':<16}  {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
