"""Generalized external preregistered benchmark: CONFIRM (frozen gates) on any
UNSEEN disease cohort with a CONFIRM-ready parquet + an ENIGMA-anchored claims CSV.

Generalizes run_nacc_external.py to arbitrary cohorts. The cohort parquet must
have columns: subject_id, site, dx, age, sex, smri_icv, smri_<region>...
The claims CSV must have: claim_id, label_class, outcome, case, control, expected_sign.

Same frozen gate ladder, disjoint-center discovery/replication, blinded, run once.
Random-label negative controls are drawn within the --control-dx group.

Example:
  PYTHONPATH=src python -m bench.run_external_generic \
    --cohort data/prepared_data/external/ds000030.parquet \
    --claims data/external_benchmark/ds000030_claims.csv \
    --control-dx CONTROL --cohort-name CNP --out-dir review-stage/external-cnp
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from confirm.analysis import directionally_consistent, fit_effect
from confirm.contract import (
    ClaimContract, ConfoundGate, Estimand, Gates, GroupSpec,
    MultiplicityGate, MultiverseGate, PowerGate, ReplicationGate,
)
from confirm.multiverse import run_multiverse
from confirm.power import power_check
from confirm.replication import replicate
from confirm.schema import normalize_sex
from confirm.verdict import decide

SEED = 20260619
MIN_PER_GROUP = 10
RANDOM_NULL_SEEDS = [101, 202]


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return (lo, hi)


def center_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = sorted(df["site"].dropna().astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(sites)
    counts = df.groupby(df["site"].astype(str)).size().to_dict()
    left, right, ln, rn = [], [], 0, 0
    for s in sites:
        if ln <= rn:
            left.append(s); ln += int(counts[s])
        else:
            right.append(s); rn += int(counts[s])
    if not right:  # single-site cohort: random subject-level split as a fallback
        idx = np.arange(len(df)); np.random.default_rng(seed).shuffle(idx)
        half = len(df) // 2
        disc = df.iloc[idx[:half]].copy(); rep = df.iloc[idx[half:]].copy()
    else:
        disc = df[df["site"].astype(str).isin(left)].copy()
        rep = df[df["site"].astype(str).isin(right)].copy()
    disc["cohort"], rep["cohort"] = "DISC", "REP"
    return disc, rep


def build_contract(claim_id, outcome, group_var, case, control, direction, covariates) -> ClaimContract:
    est = Estimand(type="group_diff", outcome=outcome, predictor=group_var,
                   group=GroupSpec(var=group_var, case=case, control=control),
                   direction=direction, unit="scalar")
    gates = Gates(
        multiplicity=MultiplicityGate(method="fdr_bh", alpha=0.05, family_size=1),
        confound=ConfoundGate(require_covariates=covariates, motion_check=False),
        power=PowerGate(min_power=0.8, ref_effect=None),
        multiverse=MultiverseGate(min_fraction_consistent=0.6),
        replication=ReplicationGate(alpha=0.05, require_same_sign=True, require_ci_overlap=False, harmonize="combat"),
    )
    return ClaimContract(
        claim_id=claim_id, question=claim_id, estimand=est, covariates=covariates, inclusion=None,
        discovery_cohort="DISC", replication_cohorts=["REP"],
        search_provenance={"declared": True, "family_size": 1, "selection": "preregistered"},
        gates=gates, reporting_language_allowed=["confirmed", "non_replicated", "under_powered", "fragile"],
    )


def evaluate(disc_all, rep_all, *, claim_id, label_class, outcome, group_var, case, control, direction, covariates) -> dict[str, Any]:
    def prep(df):
        d = df[df[group_var].isin([case, control])].copy()
        d = d.dropna(subset=[outcome, *covariates, group_var])
        d["sex"] = normalize_sex(d["sex"])
        return d
    disc, rep = prep(disc_all), prep(rep_all)
    n = lambda d, v: int((d[group_var] == v).sum())
    if min(n(disc, case), n(disc, control), n(rep, case), n(rep, control)) < MIN_PER_GROUP:
        return {"claim_id": claim_id, "label_class": label_class, "outcome": outcome, "final_label": "skipped_n"}
    contract = build_contract(claim_id, outcome, group_var, case, control, direction, covariates)
    eff = fit_effect(disc, contract, covariates=covariates, model="ols")
    pw = power_check(eff, contract)
    mv = run_multiverse(disc, contract)
    rp = replicate(eff, disc, [rep], contract)
    verdict = decide(eff, mv, pw, rp, contract)
    return {
        "claim_id": claim_id, "label_class": label_class, "outcome": outcome,
        "contrast": f"{case} vs {control}", "expected_sign": direction,
        "n_disc_case": n(disc, case), "n_disc_control": n(disc, control),
        "n_rep_case": n(rep, case), "n_rep_control": n(rep, control),
        "beta": float(eff.beta), "p": float(eff.p), "std_effect": float(eff.standardized_effect),
        "achieved_power": pw.to_dict().get("achieved_power"), "under_powered": bool(pw.under_powered),
        "multiverse_consistent": float(mv.fraction_consistent),
        "replication_passed": bool(rp.passed), "replication_reason": rp.reason,
        "final_label": verdict.label, "confirmation_subtype": verdict.confirmation_subtype,
        "baseline_significant": bool(eff.p <= 0.05 and directionally_consistent(eff.beta, contract)),
        "rationale": verdict.rationale,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    covariates = [c.strip() for c in args.covariates.split(",")]
    df = pd.read_parquet(args.cohort)
    df["site"] = df["site"].astype(str)
    claims = pd.read_csv(args.claims)
    regions = [c for c in df.columns if c.startswith("smri_") and c != "smri_icv"]
    disc_all, rep_all = center_split(df, SEED)
    rows: list[dict[str, Any]] = []

    for _, c in claims.iterrows():
        rows.append(evaluate(disc_all, rep_all, claim_id=str(c["claim_id"]), label_class=str(c["label_class"]),
                             outcome=str(c["outcome"]), group_var="dx", case=str(c["case"]),
                             control=str(c["control"]), direction=str(c["expected_sign"]), covariates=covariates))

    ctrl_disc = disc_all[disc_all["dx"] == args.control_dx]
    ctrl_rep = rep_all[rep_all["dx"] == args.control_dx]
    for seed in RANDOM_NULL_SEEDS:
        d_disc = ctrl_disc.assign(rand_group=np.random.default_rng(seed).choice(["case", "control"], size=len(ctrl_disc)))
        d_rep = ctrl_rep.assign(rand_group=np.random.default_rng(seed + 1).choice(["case", "control"], size=len(ctrl_rep)))
        for region in regions:
            rows.append(evaluate(d_disc, d_rep, claim_id=f"random_{region.replace('smri_','')}_s{seed}",
                                 label_class="random_null", outcome=region, group_var="rand_group",
                                 case="case", control="control", direction="two_sided", covariates=covariates))

    res = pd.DataFrame(rows)
    scored = res[~res["final_label"].isin(["skipped_n", "error"])]

    def rate(sub):
        k, nn = int((sub["final_label"] == "confirmed").sum()), len(sub)
        return {"k": k, "n": nn, "rate": (k / nn) if nn else None, "ci95": clopper_pearson(k, nn)}

    pos, rnull = scored[scored.label_class == "known_positive"], scored[scored.label_class == "random_null"]
    base_k = int(rnull["baseline_significant"].sum()) if len(rnull) else 0
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "cohort": f"{args.cohort_name} (unseen)",
        "n_rows": len(res), "n_scored": len(scored), "n_skipped": int((res.final_label == "skipped_n").sum()),
        "discovery_centers": sorted(disc_all["site"].unique().tolist()),
        "replication_centers": sorted(rep_all["site"].unique().tolist()),
        "CONFIRM_external": {"TPR_known_positive": rate(pos), "FCR_random_null": rate(rnull)},
        "baseline_significance_only": {"FCR_random_null": {"k": base_k, "n": len(rnull),
            "rate": (base_k / len(rnull)) if len(rnull) else None, "ci95": clopper_pearson(base_k, len(rnull))}},
        "prereg_bar": {"FCR_upper95_below_0.10": clopper_pearson(rate(rnull)["k"], rate(rnull)["n"])[1] < 0.10,
                       "TPR_above_0.70": (rate(pos)["rate"] or 0) > 0.70},
        "lockfile": {"seed": SEED, "covariates": covariates, "control_dx": args.control_dx,
            "claims_sha256": hashlib.sha256(Path(args.claims).read_bytes()).hexdigest(),
            "cohort_sha256": hashlib.sha256(Path(args.cohort).read_bytes()).hexdigest()},
        "claims": rows,
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.cohort_name}_external_results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    res.to_csv(out / f"{args.cohort_name}_external_audit.csv", index=False)
    print(json.dumps({k: summary[k] for k in ["CONFIRM_external", "baseline_significance_only", "prereg_bar", "n_scored", "n_skipped"]}, indent=2, default=str))
    print(f"\nwrote {out}/{args.cohort_name}_external_results.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--control-dx", required=True, help="dx value for the healthy/control group (random-null arm)")
    ap.add_argument("--cohort-name", default="EXT")
    ap.add_argument("--out-dir", default="review-stage/external-generic")
    ap.add_argument("--covariates", default="age,sex,smri_icv")
    run(ap.parse_args())
