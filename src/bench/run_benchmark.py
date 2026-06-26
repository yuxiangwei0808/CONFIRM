"""NeuroDecide-Bench-lite + gate-ablation ladder — the comparative must-win.

A frozen benchmark of claims with ground-truth labels (positive / null / fragile) is run
through a cumulative GATE LADDER:
  exec_only -> +confound -> +power -> +multiverse -> +replication (= full CONFIRM)
For each rung we compute:
  TPR = confirmed known-positives / known-positives        (want HIGH, preserved)
  FCR = confirmed null+fragile claims / null+fragile        (want LOW, driven to ~0)
Shows WHERE false confirmations disappear -> isolates each gate's contribution.

Baselines are ablations of CONFIRM (gates removed): `exec_only` == an execution-valid runner
that reports any significant discovery-cohort effect as "confirmed" (no validity gates).
Real competitor systems (NeuroClaw/NIAgent) are not available as executable baselines.

Run: python -m bench.run_benchmark
Outputs: review-stage/benchmark_results.json, review-stage/benchmark_fcr_tpr.png,
         review-stage/benchmark_heatmap.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from confirm.analysis import directionally_consistent, fit_effect, run_primary
from confirm.multiverse import run_multiverse
from confirm.power import power_check
from confirm.replication import replicate
from pilots.run_adni import load_adni
from pilots.run_adni_oasis import load_oasis3
from pilots.run_open_pilots import build_contract

RUNGS = ["exec_only", "+confound", "+power", "+multiverse", "+replication"]
OUT = Path("review-stage")
SEED = 20260615


def _split(df, a, b, seed):
    idx = np.arange(len(df)); np.random.default_rng(seed).shuffle(idx)
    h = len(idx) // 2
    A = df.iloc[idx[:h]].copy(); A["cohort"] = a
    B = df.iloc[idx[h:]].copy(); B["cohort"] = b
    return A, B


def evaluate(disc, rep, *, kind, outcome, predictor, direction, group, covars_full, covars_min):
    full = build_contract("c", kind=kind, outcome=outcome, predictor=predictor, direction=direction,
                          covars=covars_full, disc=disc["cohort"].iloc[0], rep=rep["cohort"].iloc[0],
                          group=group, require=covars_full)
    eff_full = run_primary(disc, full)
    eff_min = fit_effect(disc, full, covariates=covars_min)
    mv = run_multiverse(disc, full)
    pw = power_check(eff_full, full)
    rp = replicate(eff_full, disc, [rep], full)

    def sig(e):
        return bool(e.p < 0.05 and directionally_consistent(e.beta, full))

    r = {}
    r["exec_only"] = sig(eff_min)
    r["+confound"] = sig(eff_full)
    r["+power"] = r["+confound"] and (not pw.under_powered)
    r["+multiverse"] = r["+power"] and mv.passed
    r["+replication"] = r["+multiverse"] and rp.passed
    return r


def build_claims():
    adni = load_adni(); adni["cohort"] = "ADNI"
    oasis3 = load_oasis3()
    oasis1 = pd.read_parquet("data/canonical/OASIS1.parquet")
    abide = pd.read_parquet("data/canonical/ABIDE.parquet")
    adni1 = load_adni(); a1 = adni1[adni1["cohort"] == "ADNI1"].copy(); a2 = adni1[adni1["cohort"] == "ADNI_later"].copy()
    claims = []

    # ---- POSITIVES (should remain confirmed) ----
    ad_regions = [("smri_hippocampus", "negative"), ("smri_entorhinal", "negative"),
                  ("smri_midtemp", "negative"), ("smri_fusiform", "negative"), ("smri_ventricles", "positive")]
    for reg, d in ad_regions:
        claims.append((f"pos_AD_{reg}", "positive", dict(
            disc=adni, rep=oasis3, kind="group_diff", outcome=reg, predictor="dx", direction=d,
            group={"var": "dx", "case": "Dementia", "control": "CN"},
            covars_full=["age", "sex", "eTIV"], covars_min=["age", "sex"])))
    oa_a, oa_b = _split(oasis1.dropna(subset=["age", "smri_nwbv", "sex"]), "OA_A", "OA_B", 1)
    claims.append(("pos_aging_nwbv", "positive", dict(
        disc=oa_a, rep=oa_b, kind="association", outcome="smri_nwbv", predictor="age", direction="negative",
        group=None, covars_full=["sex", "eTIV"], covars_min=["sex"])))
    claims.append(("pos_ADNI_age_hippo", "positive", dict(
        disc=a1, rep=a2, kind="association", outcome="smri_hippocampus", predictor="age", direction="negative",
        group=None, covars_full=["sex", "eTIV"], covars_min=["sex"])))

    # ---- NULLS (should NOT confirm) ----
    cn1 = oasis1[oasis1["dx"] == "CN"].dropna(subset=["smri_nwbv", "sex", "eTIV", "age"]).copy()
    cn1["arm"] = np.random.default_rng(0).choice(["A", "B"], len(cn1))
    na, nb = _split(cn1, "N1A", "N1B", 2)
    claims.append(("null_random_oasis1", "null", dict(
        disc=na, rep=nb, kind="group_diff", outcome="smri_nwbv", predictor="arm", direction="two_sided",
        group={"var": "arm", "case": "A", "control": "B"}, covars_full=["sex", "eTIV"], covars_min=["sex"])))
    cnA = adni[adni["dx"] == "CN"].dropna(subset=["smri_hippocampus", "sex", "eTIV", "age"]).copy()
    cnA["arm"] = np.random.default_rng(1).choice(["A", "B"], len(cnA))
    na2, nb2 = _split(cnA, "N2A", "N2B", 3)
    claims.append(("null_random_adni", "null", dict(
        disc=na2, rep=nb2, kind="group_diff", outcome="smri_hippocampus", predictor="arm", direction="two_sided",
        group={"var": "arm", "case": "A", "control": "B"}, covars_full=["age", "sex", "eTIV"], covars_min=["age", "sex"])))
    # site-confounded null: group assigned by site-mean hippocampus (apparent effect driven by site)
    cf = adni[adni["dx"] == "CN"].dropna(subset=["smri_hippocampus", "sex", "eTIV", "age", "site"]).copy()
    site_mean = cf.groupby("site")["smri_hippocampus"].transform("mean")
    cf["arm"] = np.where(site_mean >= site_mean.median(), "A", "B")  # confounded with site
    ca, cb = _split(cf, "CFA", "CFB", 4)
    claims.append(("null_site_confounded", "null", dict(
        disc=ca, rep=cb, kind="group_diff", outcome="smri_hippocampus", predictor="arm", direction="two_sided",
        group={"var": "arm", "case": "A", "control": "B"},
        covars_full=["age", "sex", "eTIV", "site"], covars_min=["age", "sex"])))  # +confound adds site -> collinear -> gone

    # ---- FRAGILE (should NOT confirm) ----
    sites = abide["site"].value_counts().index.tolist()
    pairs = [(sites[0], sites[1], "fc_mean_abs"), (sites[0], sites[1], "fc_mean_positive"), (sites[0], sites[2], "fc_mean_abs")]
    for i, (s1, s2, out) in enumerate(pairs):
        d1 = abide[abide["site"] == s1].copy(); d1["cohort"] = f"AB_{s1}"
        d2 = abide[abide["site"] == s2].copy(); d2["cohort"] = f"AB_{s2}"
        claims.append((f"frag_ASD_{out}_{s1}_{s2}", "fragile", dict(
            disc=d1, rep=d2, kind="group_diff", outcome=out, predictor="dx", direction="two_sided",
            group={"var": "dx", "case": "ASD", "control": "TC"}, covars_full=["age", "sex"], covars_min=["age", "sex"])))
    # ---- SELECTION / WINNER'S-CURSE NULLS (discovery-significant but non-replicable) ----
    # Search random labelings for one significant in the discovery half; an independent half
    # will NOT replicate it. This is the scenario the replication gate is meant to catch.
    def fishing(src, outcome, covars, label, seeds=150, seed0=100):
        d = src.dropna(subset=[outcome, *[c for c in covars if c in src.columns]]).copy()
        if len(d) < 40:
            return None
        base = build_contract("f", kind="group_diff", outcome=outcome, predictor="arm", direction="two_sided",
                              covars=covars, disc=label + "A", rep=label + "B",
                              group={"var": "arm", "case": "A", "control": "B"}, require=covars)
        for s in range(seed0, seed0 + seeds):
            d["arm"] = np.random.default_rng(s).choice(["A", "B"], len(d))
            A, B = _split(d, label + "A", label + "B", s)
            try:
                if fit_effect(A, base, covariates=covars).p < 0.05:
                    return A, B
            except Exception:
                continue
        return None

    for label, src, outcome, covars in [
        ("null_fish_adni", adni[adni["dx"] == "CN"], "smri_hippocampus", ["age", "sex", "eTIV"]),
        ("null_fish_oasis1", oasis1[oasis1["dx"] == "CN"], "smri_nwbv", ["age", "sex", "eTIV"]),
        ("null_fish_abide", abide, "fc_mean_abs", ["age", "sex"]),
    ]:
        fb = fishing(src, outcome, covars, label)
        if fb is not None:
            A, B = fb
            claims.append((label, "null", dict(
                disc=A, rep=B, kind="group_diff", outcome=outcome, predictor="arm", direction="two_sided",
                group={"var": "arm", "case": "A", "control": "B"}, covars_full=covars, covars_min=covars)))
    return claims


def main(argv=None) -> int:
    OUT.mkdir(exist_ok=True)
    claims = build_claims()
    rows = []
    for cid, gt, kw in claims:
        try:
            r = evaluate(kw.pop("disc"), kw.pop("rep"), **kw)
        except Exception as exc:
            r = {k: False for k in RUNGS}; r["error"] = str(exc)[:80]
        rows.append({"claim": cid, "ground_truth": gt, **r})
    df = pd.DataFrame(rows)

    pos = df[df["ground_truth"] == "positive"]
    neg = df[df["ground_truth"].isin(["null", "fragile"])]
    summary = {}
    for rung in RUNGS:
        tpr = float(pos[rung].mean()) if len(pos) else float("nan")
        fcr = float(neg[rung].mean()) if len(neg) else float("nan")
        summary[rung] = {"TPR": round(tpr, 3), "FCR": round(fcr, 3)}

    print(f"\n==== NeuroDecide-Bench-lite: {len(claims)} claims "
          f"({len(pos)} positive, {len(neg)} null/fragile) ====")
    print(f"{'rung':<16}{'TPR (keep high)':>18}{'FCR (drive to 0)':>20}")
    for rung in RUNGS:
        print(f"{rung:<16}{summary[rung]['TPR']:>18}{summary[rung]['FCR']:>20}")
    print("\nper-claim confirmed by rung:")
    print(df[["claim", "ground_truth", *RUNGS]].to_string(index=False))

    (OUT / "benchmark_results.json").write_text(json.dumps(
        {"summary": summary, "claims": rows, "n_positive": int(len(pos)), "n_neg": int(len(neg))}, indent=2))

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = range(len(RUNGS))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(x, [summary[r]["TPR"] for r in RUNGS], "o-", label="TPR (known-positive recall)", color="#2a7")
        ax.plot(x, [summary[r]["FCR"] for r in RUNGS], "s--", label="FCR (false-confirmed rate)", color="#d33")
        ax.set_xticks(list(x)); ax.set_xticklabels(RUNGS, rotation=20, ha="right")
        ax.set_ylim(-0.05, 1.05); ax.set_ylabel("rate"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("CONFIRM gate ladder: false confirmations fall, positives preserved")
        fig.tight_layout(); fig.savefig(OUT / "benchmark_fcr_tpr.png", dpi=130); plt.close(fig)

        order = df.sort_values("ground_truth").reset_index(drop=True)
        mat = order[RUNGS].astype(int).to_numpy()
        fig, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.5))
        ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(RUNGS))); ax.set_xticklabels(RUNGS, rotation=20, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{c}  [{g}]" for c, g in zip(order["claim"], order["ground_truth"])], fontsize=7)
        ax.set_title("Confirmed (green) vs not (red) per claim × gate")
        fig.tight_layout(); fig.savefig(OUT / "benchmark_heatmap.png", dpi=130); plt.close(fig)
        print(f"\nwrote {OUT}/benchmark_fcr_tpr.png, {OUT}/benchmark_heatmap.png")
    except Exception as exc:
        print(f"[figure skipped: {exc}]")

    print(f"\nHEADLINE: FCR {summary['exec_only']['FCR']} (exec-only) -> {summary['+replication']['FCR']} (full CONFIRM); "
          f"TPR preserved {summary['exec_only']['TPR']} -> {summary['+replication']['TPR']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
