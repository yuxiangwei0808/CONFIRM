#!/usr/bin/env python
"""Remote extraction (run on arcdev): EDGE-LEVEL FNC for the 100-component SZ cohorts.

Cohorts: ChineseSZ, BSNIP2, BSNIP-1 (the "PC" subset), JH. READ-ONLY on /data; writes
only to the caller-provided scratch outdir.

WHY this script exists: the prior extraction kept only 4 summary FC scalars (individually
null for SZ here, since SZ dysconnectivity is a DISTRIBUTED edge pattern). It also mixed
GIFT's precomputed mean-CENTERED FNC (ChineseSZ/BSNIP-1) with raw Pearson (JH/BSNIP2).
This script fixes BOTH: it recomputes the FULL static FNC IDENTICALLY for all four cohorts
directly from the 100 component TIMECOURSES (GIFT compSet.tc), never from any precomputed
fnc_corrs_all.

FNC METHOD (identical for all 4): per subject, compSet.tc -> normalize to (T,100)
-> 100x100 Pearson -> Fisher-z (arctanh) the off-diagonal. Store the full upper triangle
as fc_edge_0001 .. fc_edge_4950 (100*99/2 = 4950), PLUS 4 summary descriptors
(fc_mean_abs, fc_mean_positive, fc_within_network, fc_between_network) computed on the same
Fisher-z edges, using the scaled 7-domain NeuroMark-1.0 partition
(SC=9, AU=4, SM=17, VI=17, CC=32, DM=13, CB=8; sums to 100).

SUBJECT SET + LABELS: reuse exactly the subject sets and clean SZ-vs-HC labels the prior
agents established, by joining on subject_id against the recovered canonical parquet for
each cohort (passed in as --labels). dx/age/sex/site are carried over verbatim from that
parquet; only the FC columns are recomputed. This guarantees identical N and identical
clean definitions (high-risk / relatives / bipolar / schizoaffective / other already
excluded upstream) while replacing the feature block.

The join from the GIFT subject order (ica_br<k> <-> k-th files.name path) to subject_id is
parsed from the path per cohort (see SUBJECT_ID_PARSERS).

Output canonical columns: subject_id, cohort, site, age, sex, dx + 4950 fc_edge_* + 4 fc_.
"""
from __future__ import annotations
import os, sys, re
import numpy as np
import pandas as pd

N_COMP = 100

# ---- scaled NeuroMark-1.0 7-domain partition (identical to prior scalar extraction) ----
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


PARTITION = build_partition(N_COMP)
_IU = np.triu_indices(N_COMP, k=1)            # 4950 off-diagonal upper-triangle indices
_SAME_NET = PARTITION[_IU[0]] == PARTITION[_IU[1]]
N_EDGES = _IU[0].size                          # 4950
EDGE_COLS = [f"fc_edge_{i + 1:04d}" for i in range(N_EDGES)]


def fnc_edges_and_descriptors(tc: np.ndarray):
    """tc -> (edge_vector[4950], descriptor_dict). Fisher-z off-diagonal static FNC.

    tc is normalized to (T, N_COMP). Pearson 100x100 -> arctanh off-diagonal.
    """
    if tc.ndim != 2:
        raise ValueError(f"tc not 2D: {tc.shape}")
    if tc.shape[1] != N_COMP and tc.shape[0] == N_COMP:
        tc = tc.T
    if tc.shape[1] != N_COMP:
        raise ValueError(f"expected (T,{N_COMP}), got {tc.shape}")
    corr = np.corrcoef(tc.T)                   # 100 x 100
    r = corr[_IU]
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)).astype(np.float32)
    pos = z[z > 0]
    within = z[_SAME_NET]
    between = z[~_SAME_NET]
    desc = {
        "fc_mean_abs": float(np.mean(np.abs(z))),
        "fc_mean_positive": float(np.mean(pos)) if pos.size else float("nan"),
        "fc_within_network": float(np.mean(within)) if within.size else float("nan"),
        "fc_between_network": float(np.mean(between)) if between.size else float("nan"),
    }
    return z, desc


# ---- robust compSet.tc loader (old scipy + v7.3 HDF5) ----------------------------
def load_tc(path: str) -> np.ndarray:
    import scipy.io as sio
    try:
        m = sio.loadmat(path)
        tc = np.asarray(m["compSet"]["tc"][0, 0], dtype=float)
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as h:
            tc = np.asarray(h["compSet"]["tc"][()], dtype=float)
    return tc


# ---- subject-order decode (GIFT *Subject.mat files.name -> path list) ------------
def _h5_char(h, obj) -> str:
    a = np.array(obj).ravel(order="F")
    return "".join(chr(int(c)) for c in a if 0 < int(c) < 0x110000).split(",")[0].strip()


def subject_order(subj_mat: str) -> list[str]:
    import scipy.io as sio
    try:
        m = sio.loadmat(subj_mat, squeeze_me=True, struct_as_record=False)
        files = np.atleast_1d(m["files"])
        out = []
        for f in files.ravel():
            nm = np.atleast_1d(getattr(f, "name")).ravel()
            out.append(str(nm[0]).split(",")[0].strip())
        return out
    except NotImplementedError:
        import h5py
        out = []
        with h5py.File(subj_mat, "r") as h:
            namerefs = np.array(h["files"]["name"]).ravel()
            for r in namerefs:
                out.append(_h5_char(h, h[r]))
        return out


# ---- per-cohort subject_id parsers (path -> subject_id matching recovered parquet) ----
def _id_chinese(p: str):
    # .../RawData_nii/<SITE>/<ID>/fMRI/SmNSprest.nii   ID e.g. NC_04-0001 / SZ_10_0101 / HR_04-0001
    m = re.search(r"/RawData_nii/[^/]+/([^/]+)/fMRI/", p)
    return m.group(1) if m else None


def _id_bsnip2(p: str):
    # .../Raw_Data/<Site>/<ID>/ses_01/func1/...   ID e.g. 22059
    m = re.search(r"/Raw_Data/[^/]+/([^/]+)/ses", p)
    return m.group(1) if m else None


def _id_bsnip_pc(p: str):
    # .../ZN_Prep_fMRI/<Site>_[<extra>_]<ID>/SM.nii   ID = trailing SRC code, e.g.
    #   Baltimore_S0015SRH1            -> S0015SRH1
    #   Hartford_1720_TR_S0084UJI3     -> S0084UJI3   (extra "1720_TR_" segment present)
    folder = p.split("/ZN_Prep_fMRI/")[-1].split("/")[0]
    m = re.search(r"(S\d{4}[A-Za-z]{3}\d)$", folder)
    if m:
        return m.group(1)
    return folder.split("_")[-1]  # fallback: last underscore-delimited token


def _id_jh(p: str):
    # .../Raw_Data/<ID>/ses_01/func/...   ID e.g. FEP1001_1_140116
    m = re.search(r"/Raw_Data/([^/]+)/ses", p)
    return m.group(1) if m else None


COHORTS = {
    "ChineseSZ": dict(icadir="/data/qneuromark/Results/ICA/ChineseSZ", prefix="CSZ",
                      subj_mat="/data/qneuromark/Results/ICA/ChineseSZ/CSZSubject.mat",
                      id_fn=_id_chinese),
    "BSNIP2": dict(icadir="/data/qneuromark/Results/ICA/BSNIP2", prefix="BSNIP2",
                   subj_mat="/data/qneuromark/Results/ICA/BSNIP2/BSNIP2Subject.mat",
                   id_fn=_id_bsnip2),
    "Olin_SZ": dict(icadir="/data/qneuromark/Results/ICA/BSNIP/PC", prefix="BSNIP",
                    subj_mat="/data/qneuromark/Results/ICA/BSNIP/PC/BSNIPSubject.mat",
                    id_fn=_id_bsnip_pc),
    "JH": dict(icadir="/data/qneuromark/Results/ICA/JH", prefix="JH",
               subj_mat="/data/qneuromark/Results/ICA/JH/JHSubject.mat",
               id_fn=_id_jh),
}


def extract(cohort: str, labels_parquet: str) -> pd.DataFrame:
    cfg = COHORTS[cohort]
    icadir, prefix, subj_mat, id_fn = cfg["icadir"], cfg["prefix"], cfg["subj_mat"], cfg["id_fn"]

    lab = pd.read_parquet(labels_parquet)
    lab["subject_id"] = lab["subject_id"].astype(str)
    n_lab_rows = len(lab)
    # Some recovered parquets contain exact-duplicate rows for a subject that appeared
    # twice in the GIFT subject order (identical dx/age/sex/site). Dedupe to one row per
    # subject: a unique-subject FNC table is correct (the same person's FNC must not be
    # counted twice). Flagged below as label_dup_rows.
    lab = lab.drop_duplicates(subset="subject_id", keep="first")
    n_lab_dup = n_lab_rows - len(lab)
    # Authoritative clean labels: subject_id -> dx/age/sex/site (carried over verbatim).
    lab_map = lab.set_index("subject_id")[["dx", "age", "sex", "site"]].to_dict("index")
    target_ids = set(lab_map)

    paths = subject_order(subj_mat)
    n_sub = len(paths)

    rows = []
    edge_rows = []
    seen = set()
    n_no_id = n_not_target = n_no_file = n_dup = 0
    for k in range(1, n_sub + 1):
        sid = id_fn(paths[k - 1])
        if sid is None:
            n_no_id += 1
            continue
        if sid not in target_ids:
            n_not_target += 1
            continue
        if sid in seen:
            n_dup += 1
            continue
        f = os.path.join(icadir, f"{prefix}_ica_br{k}.mat")
        if not os.path.isfile(f):
            n_no_file += 1
            continue
        tc = load_tc(f)
        z, desc = fnc_edges_and_descriptors(tc)
        seen.add(sid)
        meta = lab_map[sid]
        rec = {"subject_id": sid, "cohort": cohort, "site": meta["site"],
               "age": meta["age"], "sex": meta["sex"], "dx": meta["dx"]}
        rec.update(desc)
        rows.append(rec)
        edge_rows.append(z)

    meta_df = pd.DataFrame(rows)
    edge_df = pd.DataFrame(np.asarray(edge_rows, dtype=np.float32), columns=EDGE_COLS)
    df = pd.concat([meta_df.reset_index(drop=True), edge_df], axis=1)

    n_missing = len(target_ids - seen)
    print(f"[{cohort}] subj_order={n_sub} | label_rows={n_lab_rows} label_dup_rows={n_lab_dup} | "
          f"labels_target(unique)={len(target_ids)} | matched={len(df)} | "
          f"not_in_target={n_not_target} | no_id={n_no_id} | no_ica_file={n_no_file} | "
          f"dup_id_skipped={n_dup} | target_not_found_on_disk={n_missing}", flush=True)
    if n_missing:
        miss = sorted(target_ids - seen)
        print(f"[{cohort}] WARNING missing {n_missing} target ids, e.g.: {miss[:10]}", flush=True)
    return df


def main():
    outdir = sys.argv[1]
    cohort = sys.argv[2]
    labels_parquet = sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    sizes = np.bincount(PARTITION, minlength=7).tolist()
    print("[partition n=100] sizes:",
          dict(zip([n for n, _ in _NM1_DOMAINS], sizes)), "| n_edges:", N_EDGES, flush=True)

    df = extract(cohort, labels_parquet)
    out = os.path.join(outdir, f"{cohort}.parquet")
    df.to_parquet(out, index=False)

    # ---- report ----
    print(f"\n=== {cohort} summary ===", flush=True)
    print("rows:", len(df), "| cols:", df.shape[1],
          "| n edge cols:", len([c for c in df.columns if c.startswith('fc_edge_')]), flush=True)
    print("dx:", df["dx"].value_counts(dropna=False).to_dict(), flush=True)
    print("sex(raw):", df["sex"].value_counts(dropna=False).to_dict(), flush=True)
    print("site:", df["site"].value_counts(dropna=False).to_dict(), flush=True)
    a = pd.to_numeric(df["age"], errors="coerce")
    print("age range:", float(a.min()), "-", float(a.max()), "| n null age:", int(a.isna().sum()), flush=True)
    w = df["fc_within_network"].astype(float).mean()
    b = df["fc_between_network"].astype(float).mean()
    print(f"SANITY within={w:.4f} between={b:.4f} within>between={w > b}", flush=True)
    print("fc summary means:\n",
          df[["fc_mean_abs", "fc_mean_positive", "fc_within_network", "fc_between_network"]].astype(float).mean(),
          flush=True)
    # edge stats sanity
    em = edge_block_stats(df)
    print(f"edge block: mean={em[0]:.4f} std={em[1]:.4f} n_nan_cols={em[2]} n_const_cols={em[3]}", flush=True)
    print("WROTE", out, flush=True)


def edge_block_stats(df):
    ecols = [c for c in df.columns if c.startswith("fc_edge_")]
    E = df[ecols].to_numpy(dtype=float)
    nan_cols = int(np.isnan(E).all(axis=0).sum())
    const_cols = int((np.nanstd(E, axis=0) == 0).sum())
    return float(np.nanmean(E)), float(np.nanstd(E)), nan_cols, const_cols


if __name__ == "__main__":
    main()
