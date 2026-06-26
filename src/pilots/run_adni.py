"""Real ADNI run: AD (Dementia) vs CN atrophy across AD-signature regions, with an
ADNI1 -> ADNI-later phase split as the replication cohort (different scanner generations).
A brain-wide-ish preview using the scalar B0 engine in a loop over regions.

Run: python -m pilots.run_adni
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from confirm.schema import normalize_sex
from pilots.run_open_pilots import build_contract, run_all

ADNIMERGE = Path("data/raw/ADNIMERGE.xlsx")

# (canonical IDP name, ADNIMERGE column, expected AD direction, covariates)
REGIONS = [
    ("smri_hippocampus", "Hippocampus", "negative", ["age", "sex", "eTIV"]),
    ("smri_entorhinal", "Entorhinal", "negative", ["age", "sex", "eTIV"]),
    ("smri_midtemp", "MidTemp", "negative", ["age", "sex", "eTIV"]),
    ("smri_fusiform", "Fusiform", "negative", ["age", "sex", "eTIV"]),
    ("smri_wholebrain", "WholeBrain", "negative", ["age", "sex", "eTIV"]),
    ("smri_ventricles", "Ventricles", "positive", ["age", "sex", "eTIV"]),
    ("pet_fdg", "FDG", "negative", ["age", "sex"]),
]


def load_adni() -> pd.DataFrame:
    df = pd.read_excel(ADNIMERGE)
    df = df[df["VISCODE"] == "bl"].copy()  # baseline visit -> one row per subject
    out = pd.DataFrame()
    out["subject_id"] = df["RID"].astype(str)
    out["session"] = "bl"
    # discovery vs replication = ADNI1 vs later phases (different scanner generations)
    out["cohort"] = np.where(df["ORIGPROT"].astype(str) == "ADNI1", "ADNI1", "ADNI_later")
    out["site"] = df["PTID"].astype(str).str.split("_").str[0].fillna("unknown")  # site code for ComBat
    out["field_strength"] = df["FLDSTRENG"]
    out["age"] = pd.to_numeric(df["AGE"], errors="coerce")
    out["sex"] = normalize_sex(df["PTGENDER"])
    out["dx"] = df["DX"]  # CN / MCI / Dementia
    out["eTIV"] = pd.to_numeric(df["ICV"], errors="coerce")
    for canon, col, _, _ in REGIONS:
        out[canon] = pd.to_numeric(df[col], errors="coerce")
    return out


def main(argv=None) -> int:
    adni = load_adni()
    print(f"ADNI baseline: n={len(adni)} | cohorts={adni['cohort'].value_counts().to_dict()} "
          f"| dx={adni['dx'].value_counts(dropna=False).to_dict()}")
    disc = adni[adni["cohort"] == "ADNI1"].copy()
    rep = adni[adni["cohort"] == "ADNI_later"].copy()

    print("\n==== ADNI: AD (Dementia) vs CN atrophy — discovery ADNI1, replicate ADNI-later ====")
    print(f"{'region':<18}{'verdict':<16}{'disc beta':>12}{'disc p':>12}  replication")
    confirmed = []
    for canon, col, direction, covars in REGIONS:
        c = build_contract(
            f"adni_AD_{canon}", kind="group_diff", outcome=canon, predictor="dx",
            direction=direction, covars=covars, disc="ADNI1", rep="ADNI_later",
            group={"var": "dx", "case": "Dementia", "control": "CN"}, require=covars,
        )
        try:
            v, pr = run_all(disc, rep, c)
            rep_reason = v.gates.get("replication")
            print(f"{canon:<18}{v.label:<16}{pr.beta:>12.1f}{pr.p:>12.1e}  repl_pass={rep_reason}")
            if v.label == "confirmed":
                confirmed.append(canon)
        except Exception as exc:
            print(f"{canon:<18}{'ERROR':<16}  {exc}")

    print(f"\nCONFIRMED (replicated AD effect) regions: {confirmed}")
    print("Expected: medial-temporal atrophy (hippocampus/entorhinal/midtemp/fusiform) + FDG should confirm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
