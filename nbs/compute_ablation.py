"""Baselines + leave-one-out gate ablation.

FCR is reported on BOTH the small MAIN negative set (27) and the larger, diverse
177-claim stress suite, because MAIN is too small to discriminate gates (its FCR
floors at 0 for several configurations). Independent per-gate signals are
recombined; power is recomputed with the new MDE logic (d=0.3).
"""
from __future__ import annotations
import json
import pandas as pd
from statsmodels.stats.power import TTestPower

MAIN_SRC = ["review-stage/round5-combat/expanded_benchmark_results.json",
            "review-stage/round5-combat/multimodal_benchmark_results.json"]
STRESS_AUDIT = "review-stage/negatives-expansion/negatives_expansion_audit.csv"
MDE, ALPHA, MINPOW, MVMIN = 0.3, 0.05, 0.8, 0.6
_tp = TTestPower()


def powpass(n):
    try:
        return float(_tp.power(MDE, int(n), ALPHA)) >= MINPOW
    except Exception:
        return False


# ---- MAIN claims (positives + negatives), independent flags ----
mc = []
for s in MAIN_SRC:
    try:
        mc += json.load(open(s)).get("claims", [])
    except FileNotFoundError:
        pass
mdf = pd.DataFrame(mc)
mdf = mdf[mdf.get("label_authority", "").astype(str).str.lower() == "main"]


def main_flags(c):
    rep = c.get("replication")
    rok = rep.get("passed") if isinstance(rep, dict) else c.get("+replication")
    mv = c.get("multiverse_fraction_consistent")
    return {"exec": bool(c.get("exec_only")), "confound": bool(c.get("confound_valid")),
            "power": powpass(c.get("n_discovery") or 0),
            "multiverse": (mv is not None and float(mv) >= MVMIN), "replication": bool(rok)}


pos = [main_flags(c) for c in mdf[mdf.scoring_label == "known_positive"].to_dict("records")]
negm = [main_flags(c) for c in mdf[mdf.scoring_label.isin(["known_null", "fragile"])].to_dict("records")]

# ---- stress negatives (150 synthetic; new-gate audit) ----
sa = pd.read_csv(STRESS_AUDIT)


def stress_flags(r):
    cc = r.get("confound_completeness_passed")
    if pd.isna(cc):
        cc = str(r.get("confound_completeness_reason")) == "passed"
    mv = r.get("multiverse_fraction_consistent")
    return {"exec": bool(r.get("exec_only")), "confound": bool(cc),
            "power": not bool(r.get("under_powered")),
            "multiverse": (pd.notna(mv) and float(mv) >= MVMIN),
            "replication": bool(r.get("replication_passed"))}


negs = [stress_flags(r) for _, r in sa.iterrows()]

RULES = {
    "Significance + FDR (exec only)": [],
    "  + replication only": ["replication"],
    "  + confound only": ["confound"],
    "  + confound + replication": ["confound", "replication"],
    "CONFIRM - confound": ["power", "multiverse", "replication"],
    "CONFIRM - power": ["confound", "multiverse", "replication"],
    "CONFIRM - multiverse": ["confound", "power", "replication"],
    "CONFIRM - replication": ["confound", "power", "multiverse"],
    "Full CONFIRM": ["confound", "power", "multiverse", "replication"],
}


def conf(f, g):
    return f["exec"] and all(f[x] for x in g)


print(f"positives={len(pos)}  neg_MAIN={len(negm)}  neg_stress={len(negs)}\n")
print(f"{'Rule':32s} {'TPR':>7s} {'FCR-main':>9s} {'FCR-stress':>11s}")
print("-" * 62)
for name, g in RULES.items():
    tp = sum(conf(f, g) for f in pos)
    fm = sum(conf(f, g) for f in negm)
    fs = sum(conf(f, g) for f in negs)
    print(f"{name:32s} {f'{tp}/{len(pos)}':>7s} {f'{fm}/{len(negm)}':>9s} {f'{fs}/{len(negs)}':>11s}")
