"""External preregistered benchmark: CONFIRM (frozen gates) on UNSEEN NACC.

NACC was never used in CONFIRM development. Three claim groups, all scored
through the SAME frozen gate ladder, blinded, run once:

  * known_positive   - literature/ENIGMA AD & MCI atrophy effects (TPR).
  * random_null      - within-CN random-label negative controls on real NACC
                       regions: genuinely null by construction (FCR).
  * ad_spared_control- "AD-spared" primary cortices; reported separately for
                       SPECIFICITY (they carry small real global-atrophy effects,
                       so they are NOT clean nulls and are excluded from FCR).

Discovery and replication are DISJOINT sets of NACC centers (independent-site
replication). Gates/thresholds are frozen below. Report honestly.
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
    ClaimContract,
    ConfoundGate,
    Estimand,
    Gates,
    GroupSpec,
    MultiplicityGate,
    MultiverseGate,
    PowerGate,
    ReplicationGate,
)
from confirm.multiverse import run_multiverse
from confirm.power import power_check
from confirm.replication import replicate
from confirm.schema import normalize_sex
from confirm.verdict import decide

SEED = 20260619
COHORT = "data/prepared_data/external/NACC.parquet"
CLAIMS = "data/external_benchmark/nacc_claims.csv"
COVARIATES = ["age", "sex", "smri_icv"]
MIN_PER_GROUP = 10
RANDOM_NULL_SEEDS = [101, 202]  # x regions -> external random-label null trials
OUT_DIR = "review-stage/external-nacc"


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
    left: list[str] = []
    right: list[str] = []
    ln = rn = 0
    for s in sites:
        if ln <= rn:
            left.append(s); ln += int(counts[s])
        else:
            right.append(s); rn += int(counts[s])
    disc = df[df["site"].astype(str).isin(left)].copy(); disc["cohort"] = "NACC_DISC"
    rep = df[df["site"].astype(str).isin(right)].copy(); rep["cohort"] = "NACC_REP"
    return disc, rep


def build_contract(claim_id: str, outcome: str, group_var: str, case: str, control: str, direction: str) -> ClaimContract:
    est = Estimand(
        type="group_diff", outcome=outcome, predictor=group_var,
        group=GroupSpec(var=group_var, case=case, control=control),
        direction=direction, unit="scalar",
    )
    gates = Gates(
        multiplicity=MultiplicityGate(method="fdr_bh", alpha=0.05, family_size=1),
        confound=ConfoundGate(require_covariates=COVARIATES, motion_check=False),
        power=PowerGate(min_power=0.8, ref_effect=None),
        multiverse=MultiverseGate(min_fraction_consistent=0.6),
        replication=ReplicationGate(alpha=0.05, require_same_sign=True, require_ci_overlap=False, harmonize="combat"),
    )
    return ClaimContract(
        claim_id=claim_id, question=claim_id, estimand=est,
        covariates=COVARIATES, inclusion=None,
        discovery_cohort="NACC_DISC", replication_cohorts=["NACC_REP"],
        search_provenance={"declared": True, "family_size": 1, "selection": "preregistered"},
        gates=gates, reporting_language_allowed=["confirmed", "non_replicated", "under_powered", "fragile"],
    )


def _prep(df: pd.DataFrame, outcome: str, group_var: str, case: str, control: str) -> pd.DataFrame:
    d = df[df[group_var].isin([case, control])].copy()
    d = d.dropna(subset=[outcome, "age", "sex", "smri_icv", group_var])
    d["sex"] = normalize_sex(d["sex"])
    return d


def _n(d: pd.DataFrame, group_var: str, val: str) -> int:
    return int((d[group_var] == val).sum())


def evaluate(disc_all, rep_all, *, claim_id, label_class, outcome, group_var, case, control, direction) -> dict[str, Any]:
    disc, rep = _prep(disc_all, outcome, group_var, case, control), _prep(rep_all, outcome, group_var, case, control)
    if min(_n(disc, group_var, case), _n(disc, group_var, control),
           _n(rep, group_var, case), _n(rep, group_var, control)) < MIN_PER_GROUP:
        return {"claim_id": claim_id, "label_class": label_class, "outcome": outcome, "final_label": "skipped_n"}
    contract = build_contract(claim_id, outcome, group_var, case, control, direction)
    eff = fit_effect(disc, contract, covariates=COVARIATES, model="ols")
    pw = power_check(eff, contract)
    mv = run_multiverse(disc, contract)
    rp = replicate(eff, disc, [rep], contract)
    verdict = decide(eff, mv, pw, rp, contract)
    return {
        "claim_id": claim_id, "label_class": label_class, "outcome": outcome,
        "contrast": f"{case} vs {control}", "expected_sign": direction,
        "n_disc_case": _n(disc, group_var, case), "n_disc_control": _n(disc, group_var, control),
        "n_rep_case": _n(rep, group_var, case), "n_rep_control": _n(rep, group_var, control),
        "beta": float(eff.beta), "p": float(eff.p), "std_effect": float(eff.standardized_effect),
        "achieved_power": pw.to_dict().get("achieved_power"), "under_powered": bool(pw.under_powered),
        "multiverse_consistent": float(mv.fraction_consistent),
        "replication_passed": bool(rp.passed), "replication_reason": rp.reason,
        "final_label": verdict.label, "confirmation_subtype": verdict.confirmation_subtype,
        "baseline_significant": bool(eff.p <= 0.05 and directionally_consistent(eff.beta, contract)),
        "rationale": verdict.rationale,
    }


def assign_random_group(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["rand_group"] = np.random.default_rng(seed).choice(["case", "control"], size=len(out))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_parquet(COHORT)
    df["site"] = df["site"].astype(str)
    claims = pd.read_csv(CLAIMS)
    regions = [c for c in df.columns if c.startswith("smri_") and c != "smri_icv"]
    disc_all, rep_all = center_split(df, SEED)

    rows: list[dict[str, Any]] = []

    # 1) literature claims (positives, spared controls, fragile) keyed on dx
    for _, c in claims.iterrows():
        rows.append(evaluate(
            disc_all, rep_all, claim_id=str(c["claim_id"]), label_class=str(c["label_class"]),
            outcome=str(c["outcome"]), group_var="dx", case=str(c["case"]), control=str(c["control"]),
            direction=str(c["expected_sign"]),
        ))

    # 2) random-label negative controls within CN (genuine external nulls)
    cn_disc = disc_all[disc_all["dx"] == "CN"]
    cn_rep = rep_all[rep_all["dx"] == "CN"]
    for seed in RANDOM_NULL_SEEDS:
        d_disc = assign_random_group(cn_disc, seed)
        d_rep = assign_random_group(cn_rep, seed + 1)
        for region in regions:
            rows.append(evaluate(
                d_disc, d_rep, claim_id=f"random_{region.replace('smri_','')}_s{seed}",
                label_class="random_null", outcome=region, group_var="rand_group",
                case="case", control="control", direction="two_sided",
            ))

    res = pd.DataFrame(rows)
    scored = res[~res["final_label"].isin(["skipped_n", "error"])]

    def rate(sub: pd.DataFrame) -> dict[str, Any]:
        k, n = int((sub["final_label"] == "confirmed").sum()), len(sub)
        return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": clopper_pearson(k, n)}

    pos = scored[scored["label_class"] == "known_positive"]
    rnull = scored[scored["label_class"] == "random_null"]
    spared = scored[scored["label_class"] == "ad_spared_control"]
    base_fcr_k = int(rnull["baseline_significant"].sum()) if len(rnull) else 0

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cohort": "NACC (unseen)", "n_rows": len(res), "n_scored": len(scored),
        "n_skipped": int((res["final_label"] == "skipped_n").sum()),
        "discovery_centers": sorted(disc_all["site"].unique().tolist()),
        "replication_centers": sorted(rep_all["site"].unique().tolist()),
        "CONFIRM_external": {
            "TPR_known_positive": rate(pos),
            "FCR_random_null": rate(rnull),
        },
        "baseline_significance_only": {
            "FCR_random_null": {"k": base_fcr_k, "n": len(rnull),
                                "rate": (base_fcr_k / len(rnull)) if len(rnull) else None,
                                "ci95": clopper_pearson(base_fcr_k, len(rnull))},
        },
        "ad_spared_control_specificity": {
            **rate(spared),
            "note": "AD-spared primary cortices carry small real global-atrophy effects (|d|~0.3-0.5); "
                    "NOT clean nulls. Reported for specificity, excluded from FCR.",
        },
        "prereg_bar": {
            "FCR_random_null_upper95_below_0.10": clopper_pearson(rate(rnull)["k"], rate(rnull)["n"])[1] < 0.10,
            "TPR_above_0.70": (rate(pos)["rate"] or 0) > 0.70,
        },
        "lockfile": {
            "seed": SEED, "random_null_seeds": RANDOM_NULL_SEEDS, "covariates": COVARIATES,
            "min_per_group": MIN_PER_GROUP, "n_regions": len(regions),
            "gates": "mult fdr_bh a=.05 fam=1 | confound require age,sex,icv | power>=.8 | multiverse>=.6 | replication same-sign+sig a=.05 combat",
            "claims_sha256": hashlib.sha256(Path(CLAIMS).read_bytes()).hexdigest(),
            "cohort_sha256": hashlib.sha256(Path(COHORT).read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path("src/bench/run_nacc_external.py").read_bytes()).hexdigest(),
        },
        "claims": rows,
    }

    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    (out / "nacc_external_results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    res.to_csv(out / "nacc_external_audit.csv", index=False)
    print(json.dumps({k: summary[k] for k in
                      ["CONFIRM_external", "baseline_significance_only", "ad_spared_control_specificity", "prereg_bar", "n_scored", "n_skipped"]},
                     indent=2, default=str))
    print(f"\nwrote {out}/nacc_external_results.json")
    return summary


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run(argparse.Namespace())
