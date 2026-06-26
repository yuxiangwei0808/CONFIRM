"""Prepare ds000030 (UCLA CNP / LA5c) as an UNSEEN external benchmark cohort.

Downloads per-subject FreeSurfer aseg.stats from the LEGACY OpenNeuro mirror
(s3://openneuro/ds000030/ds000030_R1.0.5/.../derivatives/freesurfer/), parses
ENIGMA-relevant subcortical volumes + eTIV, joins participants.tsv diagnosis,
and writes a CONFIRM-ready table.

Output: data/prepared_data/external/ds000030.parquet with columns
  subject_id, cohort, site, age, sex, dx, smri_icv, smri_<region>
where dx in {CONTROL, SCHZ, BIPOLAR, ADHD}.
"""
from __future__ import annotations

import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PARTICIPANTS = "data/raw_external/ds000030/participants.tsv"
CACHE = Path("data/raw_external/ds000030/fs")
OUT = "data/prepared_data/external/ds000030.parquet"
BASE = ("https://s3.amazonaws.com/openneuro/ds000030/ds000030_R1.0.5/"
        "uncompressed/derivatives/freesurfer/{sub}/stats/aseg.stats")

# ENIGMA subcortical structures (FreeSurfer aseg StructName -> bilateral sum)
ROI_MAP = {
    "smri_hippocampus": ["Left-Hippocampus", "Right-Hippocampus"],
    "smri_amygdala": ["Left-Amygdala", "Right-Amygdala"],
    "smri_thalamus": ["Left-Thalamus-Proper", "Right-Thalamus-Proper", "Left-Thalamus", "Right-Thalamus"],
    "smri_accumbens": ["Left-Accumbens-area", "Right-Accumbens-area"],
    "smri_pallidum": ["Left-Pallidum", "Right-Pallidum"],
    "smri_caudate": ["Left-Caudate", "Right-Caudate"],
    "smri_putamen": ["Left-Putamen", "Right-Putamen"],
    "smri_lateralventricle": ["Left-Lateral-Ventricle", "Right-Lateral-Ventricle"],
}


def fetch(sub: str) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sub}_aseg.stats"
    if f.exists() and f.stat().st_size > 0:
        return f.read_text(errors="ignore")
    try:
        with urllib.request.urlopen(BASE.format(sub=sub), timeout=30) as r:
            txt = r.read().decode(errors="ignore")
        f.write_text(txt)
        return txt
    except Exception:
        return None


def parse_aseg(text: str) -> tuple[dict[str, float], float | None]:
    vols: dict[str, float] = {}
    etiv = None
    for line in text.splitlines():
        if line.startswith("# Measure EstimatedTotalIntraCranialVol"):
            try:
                etiv = float(line.split(",")[-2].strip())
            except Exception:
                pass
        elif line and not line.startswith("#"):
            p = line.split()
            if len(p) >= 5:
                try:
                    vols[p[4]] = float(p[3])
                except ValueError:
                    continue
    return vols, etiv


def main() -> None:
    part = pd.read_csv(PARTICIPANTS, sep="\t")
    subs = part["participant_id"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=8) as ex:
        texts = dict(zip(subs, ex.map(fetch, subs)))

    rows = []
    for _, r in part.iterrows():
        sub = str(r["participant_id"])
        txt = texts.get(sub)
        if not txt:
            continue
        vols, etiv = parse_aseg(txt)
        if etiv is None or not vols:
            continue
        row = {
            "subject_id": sub, "cohort": "CNP",
            "site": str(r.get("ScannerSerialNumber", "UCLA")),
            "age": pd.to_numeric(r.get("age"), errors="coerce"),
            "sex": str(r.get("gender", "")).strip().upper()[:1],
            "dx": str(r.get("diagnosis", "")).strip(),
            "smri_icv": etiv,
        }
        for name, srcs in ROI_MAP.items():
            present = [s for s in srcs if s in vols]  # only the 2 (L+R) that exist for this FS version
            row[name] = sum(vols[s] for s in present) if present else None
        rows.append(row)

    df = pd.DataFrame(rows)
    roi_cols = [c for c in ROI_MAP if c in df.columns]
    keep = ["subject_id", "cohort", "site", "age", "sex", "dx", "smri_icv", *roi_cols]
    df = df[keep].copy()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"wrote {OUT}  shape={df.shape}")
    print("dx counts:\n", df["dx"].value_counts(dropna=False).to_dict())
    print("sites:", df["site"].nunique(), "| sex:", df["sex"].value_counts(dropna=False).to_dict())
    print("ROI cols:", roi_cols)
    print("\nsanity (SCHZ should be LOWER hippo, HIGHER ventricle vs CONTROL):")
    for col in ["smri_hippocampus", "smri_lateralventricle", "smri_amygdala"]:
        if col in df.columns:
            g = df[df.dx.isin(["SCHZ", "CONTROL"])].groupby("dx")[col].mean()
            print(f"  {col}: CONTROL={g.get('CONTROL', float('nan')):.0f}  SCHZ={g.get('SCHZ', float('nan')):.0f}")
    miss = df[roi_cols].isna().mean().round(3).to_dict()
    print("ROI missingness:", miss)


if __name__ == "__main__":
    main()
