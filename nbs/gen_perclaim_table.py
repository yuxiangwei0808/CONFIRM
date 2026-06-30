"""Per-claim provenance/verdict table (supplementary) from the MAIN benchmark source JSONs."""
from __future__ import annotations
import json
from statsmodels.stats.power import TTestPower

SRC = ["review-stage/round5-combat/expanded_benchmark_results.json",
       "review-stage/round5-combat/multimodal_benchmark_results.json"]
_tp = TTestPower()


def powmark(n):
    try:
        return r"\checkmark" if float(_tp.power(0.3, int(n), 0.05)) >= 0.8 else r"$\times$"
    except Exception:
        return "--"


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&")


def short(cid):
    return esc(cid[:26])


rows = []
for s in SRC:
    try:
        d = json.load(open(s))
    except FileNotFoundError:
        continue
    for c in d.get("claims", []):
        if str(c.get("label_authority", "")).lower() != "main":
            continue
        rep = c.get("replication")
        rep_ok = rep.get("passed") if isinstance(rep, dict) else c.get("+replication")
        mv = c.get("multiverse_fraction_consistent")
        p = c.get("best_p")
        d_eff = c.get("best_standardized_effect")
        rows.append({
            "claim": short(c.get("claim_id", "")),
            "mod": esc(c.get("modality", "")),
            "label": esc(str(c.get("scoring_label", "")).replace("known_", "").replace("underpowered_small_positive", "up_small_pos")),
            "ndisc": c.get("n_discovery", ""), "nrep": c.get("n_replication", ""),
            "d": f"{float(d_eff):.2f}" if isinstance(d_eff, (int, float)) else "--",
            "p": f"{float(p):.1e}" if isinstance(p, (int, float)) else "--",
            "pow": powmark(c.get("n_discovery") or 0),
            "mv": r"\checkmark" if (mv is not None and float(mv) >= 0.6) else r"$\times$",
            "rep": r"\checkmark" if rep_ok else r"$\times$",
            "verdict": esc(c.get("final_label", "")),
        })

rows.sort(key=lambda r: (r["label"], r["claim"]))
lines = [
    r"{\scriptsize\setlength{\tabcolsep}{4pt}",
    r"\begin{longtable}{@{}lllcccccc@{}}",
    r"\caption{Per-claim provenance and verdicts on the main benchmark subset. "
    r"Pow/MV/Rep are the power, multiverse, and replication gate outcomes "
    r"(\checkmark{}=pass); power is judged against the $d{=}0.3$ MDE.}\label{tab:perclaim}\\",
    r"\toprule Claim & Mod. & Label & $n_{d}/n_{r}$ & std $d$ & $p$ & Pow & MV & Rep \\ \midrule \endfirsthead",
    r"\toprule Claim & Mod. & Label & $n_{d}/n_{r}$ & std $d$ & $p$ & Pow & MV & Rep \\ \midrule \endhead",
    r"\bottomrule \endfoot",
]
for r in rows:
    lines.append(f"{r['claim']} & {r['mod']} & {r['label']} & {r['ndisc']}/{r['nrep']} & {r['d']} & {r['p']} & {r['pow']} & {r['mv']} & {r['rep']} \\\\")
lines += [r"\end{longtable}", r"}"]

open("paper/figures/tab_perclaim.tex", "w").write("\n".join(lines) + "\n")
print(f"wrote paper/figures/tab_perclaim.tex  ({len(rows)} main claims)")
print("\n--- preview (first 12 rows) ---")
TIMES = r"$\times$"
for r in rows[:12]:
    pw, mvk, rpk = r['pow'] != TIMES, r['mv'] != TIMES, r['rep'] != TIMES
    nd_nr = f"{r['ndisc']}/{r['nrep']}"
    print(f"  {r['claim']:28s} {r['label']:14s} n={nd_nr:11s} d={r['d']:>6} p={r['p']:>9} pow={pw:d} mv={mvk:d} rep={rpk:d} -> {r['verdict']}")
