#!/usr/bin/env python
"""Remote extraction (run on arcdev): psychosis cohorts -> canonical SZ/HC parquet.

Cohorts: JH, BSNIP (allPR), BSNIP2. READ-ONLY on /data; writes only to scratch outdir.

Method IDENTICAL to COBRE/FBIRN (scripts/_remote_extract_sz.py) for comparability:
  per subject, ICA component timecourses (compSet.tc) -> N x N Pearson static FNC
  -> Fisher-z off-diagonal -> 4 scalar descriptors
  (fc_mean_abs, fc_mean_positive, fc_within_network, fc_between_network).

Component count: these NeuroMark cohorts use N=100 (NOT 160). Per instruction we apply
the SAME 7-domain NeuroMark-1.0 block partition (SC,AU,SM,VI,CC,DM,CB; 53-ICN sizes
5,2,9,9,17,7,4) scaled by largest-remainder to N=100, applied IDENTICALLY across all
three cohorts. *** FLAG: N=100 differs from the local COBRE/FBIRN N=160; within/between
descriptors are therefore only comparable AMONG the 100-comp cohorts, and the partition
is the analogous-scaled one, not the native one. ***

Join key: GIFT subject order. <COHORT>_ica_br<k>.mat (k=1..numOfSub, contiguous) maps
to the k-th entry of files.name in <COHORT>Subject.mat. Subject ID parsed from that path.

Output canonical columns: subject_id, cohort, site, age, sex, dx (SZ/HC) + 4 fc_.
"""
from __future__ import annotations
import os, sys, re, glob
import numpy as np
import pandas as pd

# ---- partition (scaled NeuroMark-1.0 7-domain), parameterized by N --------------
_NM1_DOMAINS = [("SC", 5), ("AU", 2), ("SM", 9), ("VI", 9), ("CC", 17), ("DM", 7), ("CB", 4)]


def build_partition(n_comp: int) -> np.ndarray:
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


def fc_descriptors(tc: np.ndarray, n_comp: int, iu, same_net) -> dict:
    """tc normalized to (T, N). Returns 4 fc_ scalars on Fisher-z off-diagonal FNC."""
    if tc.ndim != 2 or tc.shape[1] != n_comp:
        raise ValueError(f"expected (T,{n_comp}), got {tc.shape}")
    corr = np.corrcoef(tc.T)
    r = corr[iu]
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    pos = z[z > 0]
    within = z[same_net]
    between = z[~same_net]
    return {
        "fc_mean_abs": float(np.mean(np.abs(z))),
        "fc_mean_positive": float(np.mean(pos)) if pos.size else float("nan"),
        "fc_within_network": float(np.mean(within)) if within.size else float("nan"),
        "fc_between_network": float(np.mean(between)) if between.size else float("nan"),
    }


# ---- robust compSet.tc loader (old scipy + v7.3 HDF5) ---------------------------
def load_tc(path: str, n_comp: int) -> np.ndarray:
    """Return component timecourses normalized to (T, n_comp)."""
    import scipy.io as sio
    try:
        m = sio.loadmat(path)
        tc = m["compSet"]["tc"][0, 0]
        tc = np.asarray(tc, dtype=float)
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as h:
            tc = np.asarray(h["compSet"]["tc"][()], dtype=float)
    # normalize to (T, n_comp)
    if tc.shape[1] != n_comp and tc.shape[0] == n_comp:
        tc = tc.T
    return tc


# ---- subject-order path lists (the join) ----------------------------------------
def _h5_paths(subj_mat: str) -> list[str]:
    import h5py
    out = []
    with h5py.File(subj_mat, "r") as h:
        namerefs = np.array(h["files"]["name"]).ravel()
        for r in namerefs:
            a = np.array(h[r])
            # MATLAB char matrix is column-major; first run's path = F-order, up to first ','
            s = "".join(chr(int(c)) for c in a.ravel(order="F") if 0 < int(c) < 0x110000)
            out.append(s.split(",")[0].strip())
    return out


def _scipy_paths(subj_mat: str) -> list[str]:
    import scipy.io as sio
    m = sio.loadmat(subj_mat, squeeze_me=True, struct_as_record=False)
    files = np.atleast_1d(m["files"])
    out = []
    for f in files.ravel():
        nm = np.atleast_1d(getattr(f, "name")).ravel()
        out.append(str(nm[0]).split(",")[0].strip())
    return out


def subject_order(subj_mat: str) -> list[str]:
    import scipy.io as sio
    try:
        sio.loadmat(subj_mat)  # raises NotImplementedError if v7.3
        return _scipy_paths(subj_mat)
    except NotImplementedError:
        return _h5_paths(subj_mat)


# ---- per-cohort config -----------------------------------------------------------
def _ica_file(icadir: str, prefix: str, k: int) -> str:
    return os.path.join(icadir, f"{prefix}_ica_br{k}.mat")


def extract_generic(cohort, icadir, prefix, subj_mat, id_from_path, pheno_lookup,
                    n_comp=100):
    """Generic driver. id_from_path(path)->subject_id; pheno_lookup(sid)->dict|None
    with keys site,age,sex,dx (dx already 'SZ'/'HC' or None to drop)."""
    iu = np.triu_indices(n_comp, k=1)
    part = build_partition(n_comp)
    same = part[iu[0]] == part[iu[1]]

    paths = subject_order(subj_mat)
    n_sub = len(paths)
    rows, n_no_pheno, n_excluded, n_no_file = [], 0, 0, 0
    dx_excl_detail = {}
    for k in range(1, n_sub + 1):
        f = _ica_file(icadir, prefix, k)
        sid = id_from_path(paths[k - 1])
        ph = pheno_lookup(sid)
        if ph is None:
            n_no_pheno += 1
            continue
        if ph.get("dx") not in ("SZ", "HC"):
            n_excluded += 1
            raw = ph.get("dx_raw", ph.get("dx"))
            dx_excl_detail[raw] = dx_excl_detail.get(raw, 0) + 1
            continue
        if not os.path.isfile(f):
            n_no_file += 1
            continue
        tc = load_tc(f, n_comp)
        rec = {"subject_id": sid, "cohort": cohort, "site": ph["site"],
               "age": ph["age"], "sex": ph["sex"], "dx": ph["dx"]}
        rec.update(fc_descriptors(tc, n_comp, iu, same))
        rows.append(rec)
    df = pd.DataFrame(rows)
    # Drop rows missing required age/sex (cannot fabricate; would fail validate_canonical).
    n_pre = len(df)
    if n_pre:
        bad_age = df["age"].isna() | df["age"].astype(str).str.strip().str.lower().isin(["nan", "none", ""])
        bad_sex = df["sex"].isna() | df["sex"].astype(str).str.strip().str.lower().isin(["nan", "none", ""])
        df = df[~(bad_age | bad_sex)].reset_index(drop=True)
    n_dropped = n_pre - len(df)
    print(f"[{cohort}] numOfSub(order)={n_sub} | no_pheno={n_no_pheno} | "
          f"excluded(non-SZ/HC)={n_excluded} {dx_excl_detail} | no_ica_file={n_no_file} | "
          f"dropped_missing_age_or_sex={n_dropped} | final rows={len(df)}", flush=True)
    return df


# ---------- JH ----------
def jh():
    pheno = pd.read_csv("/data/qneuromark/Data/SZ_JH/Data_info/demo.csv", dtype=str)
    # JOIN KEY: scan_id == path subject folder (e.g. FEP1001_1_140116)
    dx_map = {"SZ": "SZ", "HC": "HC"}  # 'HC' is value for controls? verify: dx col has SZ + clinical
    # demo.csv: group Control/Patient; dx has SZ/BPADI/SA/.../ and HC for controls? -> controls dx blank
    # We saw dx counts include 'HC'? No: dx had SZ,BPADI,SA,MDD,Others,SF,Substance_induced,NOS + HC=94.
    pheno = pheno.set_index("scan_id")

    def id_from_path(p):
        # .../Raw_Data/FEP1001_1_140116/ses_01/func/SmNSprest.nii.nii
        m = re.search(r"/Raw_Data/([^/]+)/", p)
        return m.group(1) if m else os.path.basename(os.path.dirname(p))

    def lookup(sid):
        if sid not in pheno.index:
            return None
        row = pheno.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        dxraw = row["dx"]
        dx = "SZ" if dxraw == "SZ" else ("HC" if dxraw == "HC" else None)
        return {"site": "JH", "age": row["age"], "sex": row["gender"],
                "dx": dx, "dx_raw": dxraw}

    return extract_generic("JH", "/data/qneuromark/Results/ICA/JH", "JH",
                           "/data/qneuromark/Results/ICA/JH/JHSubject.mat",
                           id_from_path, lookup, n_comp=100)


# ---------- BSNIP2 ----------
def bsnip2():
    x = pd.read_excel(
        "/data/collaboration/bsnip2/Data_Info/bsnip2_ad_preliminary_20201221.xlsx",
        sheet_name="Sheet1", dtype=str)
    # JOIN KEY: subject_id (e.g. 22059) == path folder id.
    x = x.drop_duplicates(subset="subject_id").set_index("subject_id")
    # group: NC, SZ, SAD, BP, BPnon, OTH. Clean binary: SZ -> SZ, NC -> HC.
    dx_map = {"SZ": "SZ", "NC": "HC"}  # excludes SAD (schizoaffective), BP/BPnon (bipolar), OTH

    def id_from_path(p):
        # /data/collaboration/bsnip2/Data_BIDS/Raw_Data/Boston/22059/ses_01/func1/SmNSprest.nii
        m = re.search(r"/Raw_Data/[^/]+/([^/]+)/ses", p)
        return m.group(1) if m else None

    def lookup(sid):
        if sid is None or sid not in x.index:
            return None
        row = x.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        graw = row["group"]
        return {"site": row["site"], "age": row["age"], "sex": row["sex"],
                "dx": dx_map.get(graw), "dx_raw": graw}

    return extract_generic("BSNIP2", "/data/qneuromark/Results/ICA/BSNIP2", "BSNIP2",
                           "/data/qneuromark/Results/ICA/BSNIP2/BSNIP2Subject.mat",
                           id_from_path, lookup, n_comp=100)


# ---------- BSNIP (allPR, NeuroMark1) ----------
def bsnip():
    """Demographics/dx from BSNIP_Clinic.xlsx (subject_information mat was on a defunct mount).

    JOIN KEY: Scan_ID (9-char, e.g. S0015SRH1) == last 9 chars of the ICA path folder
    (Baltimore_S0015SRH1). dx from DXGROUP_2 (SADBP*/SADDEP* collapsed to SADP/SADR there).
    Clean binary: SZP -> SZ ; NC -> HC. EXCLUDE SZR (unaffected relatives), BPP/BPR, SADP/SADR.
    """
    cl = pd.read_excel(
        "/data/qneuromark/Results/Data_info/BSNIP/BSNIP_Clinic.xlsx", sheet_name=0, dtype=str)
    cl["Scan_ID"] = cl["Scan_ID"].astype(str).str.strip()
    cl = cl.dropna(subset=["Scan_ID"]).drop_duplicates(subset="Scan_ID").set_index("Scan_ID")
    dx_map = {"SZP": "SZ", "NC": "HC"}

    # NOTE: BSNIP_Clinic.xlsx Scan_IDs only cover the Hartford site (clinic site code 'GP')
    # in the ICA-path ID space; the other 5 sites' crosswalk lived in a now-defunct mount.
    # The join therefore yields a verified Hartford-only BSNIP cohort.
    def id_from_path(p):
        folder = p.split("/ZN_Prep_fMRI/")[-1].split("/")[0]  # e.g. Hartford_S0015SRH1
        return folder.split("_", 1)[1][-9:] if "_" in folder else folder[-9:]

    def lookup(sid):
        if sid not in cl.index:
            return None
        row = cl.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        graw = row["DXGROUP_2"]
        return {"site": "Hartford", "age": row["Age_cal"], "sex": row["sex"],
                "dx": dx_map.get(graw), "dx_raw": graw}

    return extract_generic("BSNIP", "/data/qneuromark/Results/ICA/BSNIP/allPR", "BSNIP_allPR",
                           "/data/qneuromark/Results/ICA/BSNIP/allPR/BSNIP_allPRSubject.mat",
                           id_from_path, lookup, n_comp=100)


def _summary(df, name, outdir):
    df.to_parquet(os.path.join(outdir, f"{name}.parquet"), index=False)
    print(f"\n=== {name} summary ===")
    print("dx:", df["dx"].value_counts(dropna=False).to_dict())
    print("sex(raw):", df["sex"].value_counts(dropna=False).to_dict())
    print("site:", df["site"].value_counts(dropna=False).to_dict())
    a = pd.to_numeric(df["age"], errors="coerce")
    print("age range:", a.min(), "-", a.max(), "| n null age:", int(a.isna().sum()))
    print("fc means:\n", df[[c for c in df.columns if c.startswith("fc_")]].astype(float).mean())
    print("WROTE", os.path.join(outdir, f"{name}.parquet"))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/sz_scratch")
    which = sys.argv[2] if len(sys.argv) > 2 else "jh"
    bsnip1_info = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(outdir, exist_ok=True)
    part = build_partition(100)
    print("[partition n=100] sizes:",
          dict(zip([n for n, _ in _NM1_DOMAINS], np.bincount(part, minlength=7).tolist())), flush=True)
    if which == "jh":
        _summary(jh(), "JH", outdir)
    elif which == "bsnip2":
        _summary(bsnip2(), "BSNIP2", outdir)
    elif which == "bsnip":
        _summary(bsnip(), "BSNIP", outdir)


if __name__ == "__main__":
    main()
