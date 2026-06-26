"""Open-data pilots for CONFIRM — the B0 go/no-go gate, on REAL public data.

P1 (must CONFIRM):     OASIS-1 age -> smri_gm_total (known-positive atrophy).
P2 (must NOT confirm): injected random-label null on OASIS-1 CN subjects.
P3 (must NOT confirm): ABIDE ASD-vs-TC functional connectivity across two sites.

NOTE: P1 uses a split-half of OASIS-1 (same scanner) as the replication cohort — a
within-cohort replication demonstration. TRUE cross-cohort structural replication
(e.g. ADNI -> OASIS-3) awaits the credentialed tables. P2/P3 use independent splits/sites.
Run: python -m pilots.run_open_pilots
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from confirm.analysis import run_primary
from confirm.contract import ClaimContract
from confirm.multiverse import run_multiverse
from confirm.power import power_check
from confirm.replication import replicate
from confirm.verdict import decide

DATA = Path("data/canonical")


def build_contract(claim_id, *, kind, outcome, predictor, direction, covars, disc, rep,
                   group=None, require=None, min_fraction=0.6, min_power=0.8, ref_effect=None):
    require = require if require is not None else list(covars)
    return ClaimContract.model_validate({
        "claim_id": claim_id, "question": claim_id,
        "estimand": {"type": kind, "outcome": outcome, "predictor": predictor,
                     "group": group, "direction": direction},
        "covariates": list(covars), "inclusion": None,
        "discovery_cohort": disc, "replication_cohorts": [rep],
        "gates": {
            "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
            "confound": {"require_covariates": list(require), "motion_check": False},
            "power": {"min_power": min_power, "ref_effect": ref_effect},
            "multiverse": {"min_fraction_consistent": min_fraction},
            "replication": {"alpha": 0.05, "require_same_sign": True,
                            "require_ci_overlap": False, "harmonize": "combat"},
        },
        "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
    })


def run_all(disc, rep, contract):
    primary = run_primary(disc, contract)
    mv = run_multiverse(disc, contract)
    pw = power_check(primary, contract, contract.gates.power.ref_effect)
    rp = replicate(primary, disc, [rep], contract)
    return decide(primary, mv, pw, rp, contract), primary


def split_half(df, a, b, seed=0):
    idx = np.arange(len(df))
    np.random.default_rng(seed).shuffle(idx)
    half = len(idx) // 2
    A = df.iloc[idx[:half]].copy(); A["cohort"] = a
    B = df.iloc[idx[half:]].copy(); B["cohort"] = b
    return A, B


def main(argv=None) -> int:
    oa = pd.read_parquet(DATA / "OASIS1.parquet")
    ab = pd.read_parquet(DATA / "ABIDE.parquet")
    results = []

    # ---- P1: known-positive age -> GM total, OASIS split-half -> CONFIRM ----
    try:
        d = oa.dropna(subset=["age", "smri_nwbv", "sex"]).copy()
        A, B = split_half(d, "OASIS1A", "OASIS1B", seed=1)
        c = build_contract("p1_age_nwbv", kind="association", outcome="smri_nwbv",
                           predictor="age", direction="negative", covars=["sex"],
                           disc="OASIS1A", rep="OASIS1B", require=["sex"])
        v, pr = run_all(A, B, c)
        results.append(("P1 age->nWBV atrophy", "confirmed", v.label,
                        f"beta={pr.beta:.2f} p={pr.p:.1e} n={pr.n}", v.rationale))
    except Exception as exc:
        results.append(("P1 age->GM atrophy", "confirmed", f"ERROR:{exc}", "", ""))

    # ---- P2: injected random-label null on CN subjects -> NOT confirmed ----
    try:
        cn = oa[oa["dx"] == "CN"].dropna(subset=["smri_gm_total", "sex", "eTIV", "age"]).copy()
        cn["arm_code"] = np.random.default_rng(0).choice(["A", "B"], len(cn))
        A, B = split_half(cn, "NULLA", "NULLB", seed=2)
        c = build_contract("p2_null", kind="group_diff", outcome="smri_gm_total",
                           predictor="arm_code", direction="two_sided", covars=["sex", "eTIV"],
                           disc="NULLA", rep="NULLB",
                           group={"var": "arm_code", "case": "A", "control": "B"})
        v, pr = run_all(A, B, c)
        results.append(("P2 injected-null", "!=confirmed", v.label,
                        f"beta={pr.beta:.3f} p={pr.p:.2f} n={pr.n}", v.rationale))
    except Exception as exc:
        results.append(("P2 injected-null", "!=confirmed", f"ERROR:{exc}", "", ""))

    # ---- P3: ASD vs TC FC across two ABIDE sites -> NOT confirmed (fragile) ----
    try:
        sites = ab["site"].value_counts().index.tolist()
        s1, s2 = sites[0], sites[1]
        d1 = ab[ab["site"] == s1].copy(); d1["cohort"] = f"ABIDE_{s1}"
        d2 = ab[ab["site"] == s2].copy(); d2["cohort"] = f"ABIDE_{s2}"
        c = build_contract("p3_asd_fc", kind="group_diff", outcome="fc_mean_abs",
                           predictor="dx", direction="two_sided", covars=["age", "sex"],
                           disc=f"ABIDE_{s1}", rep=f"ABIDE_{s2}",
                           group={"var": "dx", "case": "ASD", "control": "TC"}, require=["age", "sex"])
        v, pr = run_all(d1, d2, c)
        results.append((f"P3 ASD-vs-TC FC ({s1}->{s2})", "!=confirmed", v.label,
                        f"beta={pr.beta:.3f} p={pr.p:.2f} n={pr.n}", v.rationale))
    except Exception as exc:
        results.append(("P3 ASD-vs-TC FC", "!=confirmed", f"ERROR:{exc}", "", ""))

    print("\n==== OPEN-DATA PILOT RESULTS ====")
    all_ok = True
    for name, exp, act, stat, rat in results:
        ok = (act == "confirmed") if exp == "confirmed" else (act != "confirmed" and not act.startswith("ERROR"))
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  (expect {exp})")
        print(f"       verdict={act} | {stat}")
        if rat:
            print(f"       rationale={rat}")
    print(f"\nPILOT SUITE: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
