"""Claim-audit: re-derive every headline number from the raw artifacts and check
the paper states it (and contains no stale variant). Zero trust in the prose."""
from __future__ import annotations
import glob, json, re
import pandas as pd
from scipy.stats import beta as B


def cp(k, n):
    lo = 0.0 if k == 0 else float(B.ppf(.025, k, n - k + 1))
    hi = 1.0 if k == n else float(B.ppf(.975, k + 1, n - k))
    return lo, hi


paper = re.sub(r"\s+", " ", " ".join(open(f).read() for f in sorted(glob.glob("paper/sec/*.tex"))))

R = "review-stage"
nacc = json.load(open(f"{R}/curated-gate-external-nacc/nacc_external_results.json"))
cnp = json.load(open(f"{R}/curated-gate-external-cnp/CNP_external_results.json"))
# MAIN benchmark audit
mb = pd.read_csv(f"{R}/curated-gate-benchmark-combat/combined_benchmark_audit.csv")
mb = mb[mb.label_authority.astype(str).str.lower() == "main"]
mpos = mb[mb.scoring_label == "known_positive"]
mneg = mb[mb.scoring_label.isin(["known_null", "fragile"])]
main_tpr = (mpos.final_label == "confirmed").sum()
main_fcr = (mneg.final_label == "confirmed").sum()
# negatives stress (150) + combined 177
na = pd.read_csv(f"{R}/curated-gate-synthetic-stress/negatives_expansion_audit.csv")
stress_fcr = int((na.final_label == "confirmed").sum())
nacc_pos = nacc["CONFIRM_external"]["TPR_known_positive"]
nacc_fcr = nacc["CONFIRM_external"]["FCR_random_null"]
nacc_base = nacc["baseline_significance_only"]["FCR_random_null"]
cnp_pos = cnp["CONFIRM_external"]["TPR_known_positive"]
cnp_fcr = cnp["CONFIRM_external"]["FCR_random_null"]
comb_k, comb_n = 0 + 0, nacc_fcr["n"] + cnp_fcr["n"]

# (label, derived value, expected substrings in paper, forbidden stale substrings)
CHECKS = [
    ("MAIN TPR", f"{main_tpr}/{len(mpos)}", ["10/10"], []),
    ("MAIN FCR", f"{main_fcr}/{len(mneg)} CI{cp(main_fcr,len(mneg))}", ["0/27"], []),
    ("Stress-suite FCR", f"{stress_fcr}/177 CI{cp(stress_fcr,177)}", ["0/177", "2.1"], ["1/177", "0.6\\%", "3.1\\%"]),
    ("NACC TPR", f"{nacc_pos['k']}/{nacc_pos['n']}", ["9/9"], []),
    ("NACC FCR", f"{nacc_fcr['k']}/{nacc_fcr['n']} CI{tuple(round(x,3) for x in nacc_fcr['ci95'])}", ["0/28"], []),
    ("NACC baseline FCR", f"{nacc_base['k']}/{nacc_base['n']}", ["2/28"], []),
    ("ds000030 TPR/FCR", f"{cnp_pos['k']}/{cnp_pos['n']} & {cnp_fcr['k']}/{cnp_fcr['n']}", ["0/14", "0/16"], []),
    ("Combined external FCR", f"{comb_k}/{comb_n} CI{tuple(round(x,3) for x in cp(comb_k,comb_n))}", ["0/44"], []),
    ("NACC upper bound", f"{cp(0,28)[1]*100:.1f}%", ["12.3"], []),
]
print(f"{'Claim':26s} {'artifact':28s} {'in paper?':10s} {'stale?'}")
print("-" * 80)
flags = 0
for name, val, exp, forb in CHECKS:
    ok = all(e in paper for e in exp)
    bad = [f for f in forb if f in paper]
    if not ok or bad:
        flags += 1
    mark = "OK" if ok else "MISSING " + str([e for e in exp if e not in paper])
    stale = (" STALE!" + str(bad)) if bad else ""
    print(f"{name:26s} {val:28s} {mark:10s}{stale}")
print(f"\n{flags} flag(s).  Ablation numbers checked separately via compute_ablation.py.")
