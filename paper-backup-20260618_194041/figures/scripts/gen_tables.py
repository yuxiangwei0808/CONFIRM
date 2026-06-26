"""Generate LaTeX tables for CONFIRM paper figures."""

from __future__ import annotations

import csv
from pathlib import Path

from bench.plot_coverage_fcr import build_coverage_table

from paper_style import (
    ARTIFACTS,
    FIG_DIR,
    RUNG_LABELS,
    RUNGS,
    format_float,
    latex_escape,
    load_json,
    rate_ci_text,
    stat_count_text,
    stat_rate_ci_only,
    write_text,
)

POSITIVE_CLAIM_IDS = [
    "ad_hippocampal_atrophy_adni_oasis3",
    "ad_entorhinal_atrophy_adni_oasis3",
    "ad_midtemp_atrophy_adni_oasis3",
    "ad_wholebrain_atrophy_adni_oasis3",
    "sz_fc_within_cobre_fbirn",
    "brain_aging_hippocampus_adni_oasis3_cn",
]

NEGATIVE_FAMILY_ORDER = [
    "random_label",
    "site_confound",
    "p_fishing",
    "underpowered",
    "cross_cohort_nonreplication",
]


def _audit_by_claim(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["claim_id"]: row for row in csv.DictReader(handle)}


def _tex_table(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def generate_positives_table() -> Path:
    payload = load_json("combined_results")
    claims = {claim["claim_id"]: claim for claim in payload["claims"]}
    audit = _audit_by_claim(ARTIFACTS["combined_audit"])

    rows = []
    for claim_id in POSITIVE_CLAIM_IDS:
        claim = claims.get(claim_id)
        audit_row = audit.get(claim_id)
        if not claim or not audit_row or audit_row.get("final_label") != "confirmed":
            continue
        meta = claim.get("label_metadata", {})
        effect = (
            "$\\beta$="
            + latex_escape(format_float(audit_row.get("best_beta"), digits=3))
            + "; std="
            + latex_escape(format_float(audit_row.get("best_standardized_effect"), digits=3))
        )
        p_value = latex_escape(format_float(audit_row.get("best_p"), digits=3))
        rows.append(
            [
                latex_escape(meta.get("phenotype", claim_id)),
                latex_escape(meta.get("cohorts", f"{claim.get('discovery_cohort')};{claim.get('replication_cohort')}")),
                latex_escape(audit_row.get("best_region")),
                latex_escape(meta.get("expected_direction", "")),
                effect,
                "$p$=" + p_value,
                latex_escape(audit_row.get("final_label")),
            ]
        )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Confirmed positive controls used to audit CONFIRM. Effect sizes and verdicts are read from the combined benchmark audit; standardized effects are reported when present in the artifact.}",
        r"\label{tab:positives}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}p{0.22\textwidth}p{0.10\textwidth}p{0.14\textwidth}p{0.18\textwidth}p{0.16\textwidth}p{0.09\textwidth}p{0.07\textwidth}@{}}",
        r"\toprule",
        r"Domain & Cohorts & Outcome & Direction & Effect size & Evidence & Verdict \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return write_text(FIG_DIR / "tab_positives.tex", _tex_table(lines))


def generate_gate_ladder_table() -> Path:
    payload = load_json("combined_results")
    table = build_coverage_table(payload)
    ladder = table[table["section"] == "gate_ladder"].copy()

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Gate-ladder operating characteristics for the full benchmark and the main-label subset. Entries show rate [exact 95\% CI] with count/denominator in parentheses. TPR is known-positive recall.}",
        r"\label{tab:gate_ladder}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Subset & Rung & TPR & FCR & Coverage \\",
        r"\midrule",
    ]
    for subset in ["FULL", "MAIN"]:
        part = ladder[ladder["subset"] == subset].sort_values("rung_index")
        for idx, rung in enumerate(RUNGS):
            row = part[part["rung"] == rung]
            if row.empty:
                continue
            row = row.iloc[0]
            prefix = subset if idx == 0 else ""
            lines.append(
                " & ".join(
                    [
                        latex_escape(prefix),
                        latex_escape(RUNG_LABELS[rung]),
                        rate_ci_text(
                            row["known_positive_recall"],
                            row["known_positive_recall_ci_low"],
                            row["known_positive_recall_ci_high"],
                            row["known_positive_recall_count"],
                            row["known_positive_recall_denominator"],
                        ),
                        rate_ci_text(row["FCR"], row["FCR_ci_low"], row["FCR_ci_high"], row["FCR_count"], row["FCR_denominator"]),
                        rate_ci_text(
                            row["coverage"],
                            row["coverage_ci_low"],
                            row["coverage_ci_high"],
                            row["coverage_count"],
                            row["coverage_denominator"],
                        ),
                    ]
                )
                + r" \\"
            )
        if subset == "FULL":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return write_text(FIG_DIR / "tab_gate_ladder.tex", _tex_table(lines))


def _family_label(name: str) -> str:
    return name.replace("_", " ")


def generate_negatives_table() -> Path:
    payload = load_json("negatives_results")
    families = payload["per_family_FCR"]
    combined = payload["combined_main_FCR"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{False-confirm rate on expanded negative controls. Family rows summarize the 150 newly generated negatives; the final rows compare the prior main negatives with the combined full-gate denominator. Entries show FCR [exact 95\% CI] with count/denominator in parentheses.}",
        r"\label{tab:negatives}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}p{0.43\columnwidth}cc@{}}",
        r"\toprule",
        r"Set & Count & FCR [95\% CI] \\",
        r"\midrule",
    ]
    for family in NEGATIVE_FAMILY_ORDER:
        if family not in families:
            continue
        lines.append(
            latex_escape(_family_label(family))
            + " & "
            + stat_count_text(families[family], "FCR")
            + " & "
            + stat_rate_ci_only(families[family], "FCR")
            + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            "Prior main negatives & "
            + stat_count_text(combined["old_main"], "FCR")
            + " & "
            + stat_rate_ci_only(combined["old_main"], "FCR")
            + r" \\",
            "Expanded negatives & "
            + stat_count_text(combined["new_negatives_main"], "FCR")
            + " & "
            + stat_rate_ci_only(combined["new_negatives_main"], "FCR")
            + r" \\",
            "Combined main negatives & "
            + stat_count_text(combined["combined_main"], "FCR")
            + " & "
            + stat_rate_ci_only(combined["combined_main"], "FCR")
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return write_text(FIG_DIR / "tab_negatives.tex", _tex_table(lines))


def generate_latex_includes() -> Path:
    lines = [
        r"% Auto-generated include snippets for CONFIRM WACV figures and tables.",
        r"% Rebuild with: PYTHONPATH=src ./.venv/bin/python paper/figures/scripts/generate_all.py",
        "",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \includegraphics[width=\textwidth]{figures/fig_coverage_fcr.pdf}",
        r"    \caption{Coverage, false-confirm rate (FCR), and known-positive recall (TPR) across progressively stricter CONFIRM gate rungs for the full benchmark and main-label subset. FCR error bars are exact 95\% Clopper--Pearson intervals.}",
        r"    \label{fig:coverage_fcr}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \includegraphics[width=\textwidth]{figures/fig_neuroclaw.pdf}",
        r"    \caption{Head-to-head comparison with NeuroClaw and post-hoc CONFIRM layering. Bars show TPR and FCR with exact 95\% binomial intervals; labels show count/denominator. The CONFIRM layer removes NeuroClaw false confirms while preserving the NeuroClaw positive-recall count.}",
        r"    \label{fig:neuroclaw}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[t]",
        r"    \centering",
        r"    \includegraphics[width=\textwidth]{figures/fig_multillm.pdf}",
        r"    \caption{Agentic multi-LLM sweep across six models. Bars report artifact-derived draft-success, estimand-match, and gate-success rates; the inset summarizes cross-model verdict agreement and anti-hallucination catches.}",
        r"    \label{fig:multillm}",
        r"\end{figure*}",
        "",
        r"\input{figures/tab_positives.tex}",
        "",
        r"\input{figures/tab_gate_ladder.tex}",
        "",
        r"\input{figures/tab_negatives.tex}",
        "",
    ]
    return write_text(FIG_DIR / "latex_includes.tex", _tex_table(lines))


def generate() -> list[Path]:
    return [
        generate_positives_table(),
        generate_gate_ladder_table(),
        generate_negatives_table(),
        generate_latex_includes(),
    ]


def main() -> int:
    for path in generate():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
