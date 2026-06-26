#!/usr/bin/env python
"""Remote extraction (run on arcdev): COBRE + FBIRN schizophrenia canonical tables.

Reads READ-ONLY from /data/qneuromark/Data/{COBRE,FBIRN}. Writes only to the
caller-provided scratch output dir. Produces canonical parquet rows:
    subject_id, cohort, site, age, sex, dx (SZ/HC) + 4 fc_ descriptors.

FNC: per-subject RC_ROI.npy (T x 160) -> 160x160 Pearson -> Fisher-z off-diagonal.
Descriptors: fc_mean_abs, fc_mean_positive, fc_within_network, fc_between_network.

Network partition: no native 160-component NeuroMark label file exists on the
cluster (templates present are 53 / 100 / 30 / 175). Per instruction, we apply a
FIXED documented partition IDENTICALLY to both cohorts: the canonical NeuroMark-1.0
7-domain block structure (SC,AU,SM,VI,CC,DM,CB with 53-ICN sizes 5,2,9,9,17,7,4)
scaled proportionally to 160 contiguous components. Within/between are defined on
this fixed labeling. Internal consistency (identical partition for COBRE & FBIRN)
is what the cross-cohort replication relies on.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

N_COMP = 160

# Canonical NeuroMark-1.0 domain sizes (sum = 53), in template order.
_NM1_DOMAINS = [("SC", 5), ("AU", 2), ("SM", 9), ("VI", 9), ("CC", 17), ("DM", 7), ("CB", 4)]


def build_partition(n_comp: int = N_COMP) -> np.ndarray:
    """Deterministic network label (0..6) per component, scaled NM1 -> n_comp.

    Largest-remainder allocation so block sizes sum exactly to n_comp; contiguous
    blocks in the fixed NM1 domain order. Identical for every subject and cohort.
    """
    total53 = sum(s for _, s in _NM1_DOMAINS)
    raw = [n_comp * s / total53 for _, s in _NM1_DOMAINS]
    floor = [int(np.floor(x)) for x in raw]
    rem = n_comp - sum(floor)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floor[i], reverse=True)
    for i in range(rem):
        floor[order[i]] += 1
    labels = np.concatenate([np.full(sz, di, dtype=int) for di, sz in enumerate(floor)])
    assert labels.shape[0] == n_comp, (labels.shape, n_comp)
    return labels


PARTITION = build_partition(N_COMP)
_IU = np.triu_indices(N_COMP, k=1)
_SAME_NET = PARTITION[_IU[0]] == PARTITION[_IU[1]]  # boolean mask over off-diagonal edges


def fc_descriptors(roi_tc: np.ndarray) -> dict[str, float]:
    """RC_ROI (T x 160) -> 4 fc_ scalar descriptors on Fisher-z off-diagonal FNC."""
    if roi_tc.ndim != 2 or roi_tc.shape[1] != N_COMP:
        raise ValueError(f"expected (T,{N_COMP}), got {roi_tc.shape}")
    corr = np.corrcoef(roi_tc.T)  # 160 x 160
    r = corr[_IU]
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    pos = z[z > 0]
    within = z[_SAME_NET]
    between = z[~_SAME_NET]
    return {
        "fc_mean_abs": float(np.mean(np.abs(z))),
        "fc_mean_positive": float(np.mean(pos)) if pos.size else float("nan"),
        "fc_within_network": float(np.mean(within)) if within.size else float("nan"),
        "fc_between_network": float(np.mean(between)) if between.size else float("nan"),
    }


def feature_subjects(fdir: str) -> dict[str, str]:
    """Map subject-dir name -> RC_ROI.npy path, only for dirs that have the file."""
    out = {}
    for d in sorted(os.listdir(fdir)):
        p = os.path.join(fdir, d, "RC_ROI.npy")
        if os.path.isfile(p):
            out[d] = p
    return out


def extract_cobre() -> pd.DataFrame:
    fdir = "/data/qneuromark/Data/COBRE/ZN_Neuromark/ZN_Prep_fMRI/"
    feats = feature_subjects(fdir)
    pheno = pd.read_csv(
        "/data/qneuromark/Data/COBRE/Data_info/pheno_comb_cobre_all.csv", dtype=str
    )
    # JOIN KEY: URSI == feature dir name (full coverage of RC_ROI subjects).
    dx_map = {"Schizophrenia": "SZ", "Control": "HC"}  # excludes Schizoaffective + Bipolar
    pheno = pheno.set_index("URSI")
    rows = []
    excluded = {"Schizoaffective": 0, "Bipolar": 0}
    no_pheno = 0
    for sid, path in feats.items():
        if sid not in pheno.index:
            no_pheno += 1
            continue
        di=pheno.loc[sid, "Diagnosis"]
        if di not in dx_map:
            if di in excluded:
                excluded[di] += 1
            continue
        roi = np.load(path)
        rec = {
            "subject_id": sid,
            "cohort": "COBRE",
            "site": "COBRE",
            "age": pheno.loc[sid, "Age"],
            "sex": pheno.loc[sid, "Sex"],
            "dx": dx_map[di],
        }
        rec.update(fc_descriptors(roi))
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(
        f"[COBRE] RC_ROI subjects={len(feats)} | matched_pheno={len(feats)-no_pheno} "
        f"| excluded Schizoaffective={excluded['Schizoaffective']} Bipolar={excluded['Bipolar']} "
        f"| no_pheno={no_pheno} | final rows={len(df)}",
        flush=True,
    )
    return df


def extract_fbirn() -> pd.DataFrame:
    fdir = "/data/qneuromark/Data/FBIRN/ZN_Neuromark/ZN_Prep_fMRI/"
    feats = feature_subjects(fdir)
    pheno = pd.read_csv(
        "/data/qneuromark/Data/FBIRN/Data_info/fBIRN_CMINDS_4rsfMRI2_G.csv", dtype=str
    )
    # JOIN KEY: SubjectID == feature dir name.
    pheno = pheno.set_index("SubjectID")
    rows = []
    no_pheno = 0
    for sid, path in feats.items():
        if sid not in pheno.index:
            no_pheno += 1
            continue
        roi = np.load(path)
        rec = {
            "subject_id": sid,
            "cohort": "FBIRN",
            "site": "FBIRN",  # no per-subject site column in this file
            "age": pheno.loc[sid, "nDEMOG_CUR_AGE"],
            "sex": pheno.loc[sid, "sDEMOG_GENDER"],
            "dx": pheno.loc[sid, "sDEMOG_DIAGNOSIS"],  # already SZ / HC
        }
        rec.update(fc_descriptors(roi))
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(
        f"[FBIRN] RC_ROI subjects={len(feats)} | matched_pheno={len(feats)-no_pheno} "
        f"| no_pheno(dropped)={no_pheno} | final rows={len(df)}",
        flush=True,
    )
    return df


def extract_abide1() -> pd.DataFrame:
    fdir = "/data/qneuromark/Data/Autism/ABIDE1/ZN_Neuromark/ZN_Prep_fMRI/"
    feats = feature_subjects(fdir)
    pheno = pd.read_csv(
        "/data/qneuromark/Data/Autism/ABIDE1/Data_info/Phenotypic_V1_0b_preprocessed1.csv",
        dtype=str,
    )
    # JOIN KEY: FILE_ID == feature dir name (e.g. Pitt_0050003). Drop no_filename.
    pheno = pheno[pheno["FILE_ID"].astype(str) != "no_filename"].set_index("FILE_ID")
    dx_map = {"1": "ASD", "2": "HC"}  # ABIDE: 1=autism, 2=control
    rows = []
    no_pheno = 0
    for sid, path in feats.items():
        if sid not in pheno.index:
            no_pheno += 1
            continue
        roi = np.load(path)
        rec = {
            "subject_id": sid,
            "cohort": "ABIDE1",
            "site": pheno.loc[sid, "SITE_ID"],  # real site
            "age": pheno.loc[sid, "AGE_AT_SCAN"],
            "sex": pheno.loc[sid, "SEX"],  # 1/2 -> normalize_sex handles
            "dx": dx_map.get(pheno.loc[sid, "DX_GROUP"], pd.NA),
        }
        rec.update(fc_descriptors(roi))
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(
        f"[ABIDE1] RC_ROI subjects={len(feats)} | matched_pheno={len(feats)-no_pheno} "
        f"| no_pheno(dropped)={no_pheno} | final rows={len(df)}",
        flush=True,
    )
    return df


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/sz_scratch")
    os.makedirs(outdir, exist_ok=True)
    # Document the partition block sizes for the report.
    sizes = np.bincount(PARTITION, minlength=7)
    names = [n for n, _ in _NM1_DOMAINS]
    print("[partition] domain block sizes (n=160):", dict(zip(names, sizes.tolist())), flush=True)

    cobre = extract_cobre()
    fbirn = extract_fbirn()
    cobre.to_parquet(os.path.join(outdir, "COBRE.parquet"), index=False)
    fbirn.to_parquet(os.path.join(outdir, "FBIRN.parquet"), index=False)

    for name, df in [("COBRE", cobre), ("FBIRN", fbirn)]:
        print(f"\n=== {name} summary ===", flush=True)
        print("dx counts:\n", df["dx"].value_counts(dropna=False), flush=True)
        print("sex counts (raw):\n", df["sex"].value_counts(dropna=False), flush=True)
        print("site counts:\n", df["site"].value_counts(dropna=False), flush=True)
        fc_cols = [c for c in df.columns if c.startswith("fc_")]
        print("fc means:\n", df[fc_cols].astype(float).mean(), flush=True)
        print("age range:", df["age"].astype(float).min(), "-", df["age"].astype(float).max(), flush=True)
    print("\nWROTE:", outdir, flush=True)


if __name__ == "__main__":
    main()
