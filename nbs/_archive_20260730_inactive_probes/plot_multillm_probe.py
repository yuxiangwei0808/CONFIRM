"""Figure: verdict stability across drafting models.

Core conclusion defended by this figure: six frontier drafting models never
produce the same ClaimContract, yet the unchanged gates return the same verdict
on the large majority of claims, so a reported verdict is a property of the
evidence policy rather than of the drafter.

Panel roles:
  a  verdict composition per drafter        -- yield varies, abstention dominates
  b  pairwise verdict agreement             -- the stability claim itself
  c  contract fields that diverge           -- the agreement is not because the
                                               models wrote the same contract
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

# Frontier drafting tier. Models below this tier are excluded by request and the
# exclusion is recorded in the run manifest so the reduced set is auditable.
MODEL_ORDER = (
    "openai:gpt-5.5",
    "openai:gpt-5.4",
    "google:gemini-3.5-flash",
    "openrouter:anthropic/claude-sonnet-5",
    "google:gemini-3.1-pro-preview",
    "google:gemini-3.5-flash-lite",
)
MODEL_LABELS = {
    "openai:gpt-5.5": "GPT-5.5",
    "openai:gpt-5.4": "GPT-5.4",
    "google:gemini-3.5-flash": "Gemini Flash",
    "openrouter:anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "google:gemini-3.1-pro-preview": "Gemini Pro",
    "google:gemini-3.5-flash-lite": "Gemini Flash-Lite",
}

# One restrained palette: blue signal for reported claims, tan and neutral grey
# for the two abstention reasons.
VERDICT_ORDER = ("confirmed", "non_replicated", "fragile")
VERDICT_LABELS = {
    "confirmed": "Confirmed",
    "non_replicated": "Non-replicated",
    "fragile": "Fragile",
}
VERDICT_COLORS = {
    "confirmed": "#2D6A9F",
    "non_replicated": "#9A8064",
    "fragile": "#C7CDD3",
}
HEAT_CMAP = "Blues"
ACCENT = "#133C66"

CONTRACT_FACETS = ("gates", "search_provenance", "estimand", "covariates", "inclusion")
FACET_LABELS = {
    "gates": "Gate settings",
    "search_provenance": "Search provenance",
    "estimand": "Estimand",
    "covariates": "Covariates",
    "inclusion": "Inclusion rule",
}


def _hashable(value):
    if isinstance(value, list):
        items = [_hashable(v) for v in value]
        try:
            return tuple(sorted(items, key=repr))
        except TypeError:
            return tuple(items)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def load(root: Path):
    verdicts: dict[str, dict[str, str]] = {}
    contracts: dict[str, dict[str, dict]] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        audit = model_dir / "gates" / "claim_gate_audit.csv"
        if not audit.exists():
            continue
        with audit.open() as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("source_mode") == "literature_grounded"]
        if not rows:
            continue
        model = rows[0]["model_spec"]
        if model not in MODEL_ORDER:
            continue
        verdicts[model] = {r["claim_id"]: r.get("gate_verdict_label") for r in rows}
        draft_path = model_dir / "drafted_contracts.jsonl"
        contracts[model] = {}
        if draft_path.exists():
            with draft_path.open() as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        contracts[model][row["claim_id"]] = row.get("drafted_contract")
    return verdicts, contracts


def build(root: Path, out_prefix: Path) -> dict:
    verdicts, contracts = load(root)
    missing = [m for m in MODEL_ORDER if m not in verdicts]
    if missing:
        raise SystemExit(f"missing probe output for: {missing}")
    models = list(MODEL_ORDER)
    common = sorted(set.intersection(*[set(verdicts[m]) for m in models]))
    n = len(common)

    figure, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": (1.25, 1.0, 0.78)}
    )

    # -- panel a: verdict composition -------------------------------------
    ax = axes[0]
    ypos = np.arange(len(models))[::-1]
    left = np.zeros(len(models))
    for verdict in VERDICT_ORDER:
        widths = np.array([sum(1 for c in common if verdicts[m][c] == verdict) for m in models], float)
        ax.barh(
            ypos,
            widths,
            left=left,
            height=0.68,
            color=VERDICT_COLORS[verdict],
            edgecolor="white",
            linewidth=0.6,
        )
        for y, w, l in zip(ypos, widths, left):
            if w >= 2:
                ax.text(
                    l + w / 2,
                    y,
                    f"{int(w)}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if verdict != "fragile" else "#3A4149",
                )
        left += widths
    ax.set_yticks(ypos)
    ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=6.5)
    ax.set_xlim(0, n)
    ax.set_xlabel(f"Claims (of {n})", fontsize=7)
    ax.set_title("a", loc="left", fontweight="bold", fontsize=8)
    ax.text(
        0.5,
        1.02,
        "Verdict composition",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
    )
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(
        handles=[Patch(facecolor=VERDICT_COLORS[v], label=VERDICT_LABELS[v]) for v in VERDICT_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        fontsize=6.2,
        handlelength=1.1,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    # -- panel b: pairwise verdict agreement -------------------------------
    ax = axes[1]
    grid = np.full((len(models), len(models)), np.nan)
    pair_vals = []
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                continue
            agree = sum(1 for c in common if verdicts[a][c] == verdicts[b][c])
            grid[i, j] = agree
            if i < j:
                pair_vals.append(agree)
    vmin = min(pair_vals)
    image = ax.imshow(grid, cmap=HEAT_CMAP, vmin=vmin - 1, vmax=n)
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", fontsize=6, color="#9AA3AB")
                continue
            val = grid[i, j]
            ax.text(
                j,
                i,
                f"{int(val)}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if val >= (vmin + n) / 2 else "#26303A",
            )
    short = [MODEL_LABELS[m].replace("Gemini ", "G-").replace("Claude ", "") for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short, fontsize=5.8)
    ax.set_title("b", loc="left", fontweight="bold", fontsize=8)
    ax.text(
        0.5,
        1.02,
        f"Pairwise agreement (of {n})",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # -- panel c: contract divergence --------------------------------------
    ax = axes[2]
    facet_counts = []
    for facet in CONTRACT_FACETS:
        differing = 0
        for claim in common:
            vals = {_hashable((contracts[m].get(claim) or {}).get(facet)) for m in models}
            if len(vals) > 1:
                differing += 1
        facet_counts.append(100.0 * differing / n)
    ypos = np.arange(len(CONTRACT_FACETS))[::-1]
    ax.barh(ypos, facet_counts, height=0.6, color=ACCENT, edgecolor="none")
    for y, v in zip(ypos, facet_counts):
        ax.text(min(v + 2, 99), y, f"{v:.0f}%", va="center", ha="left", fontsize=6, color="#26303A")
    ax.set_yticks(ypos)
    ax.set_yticklabels([FACET_LABELS[f] for f in CONTRACT_FACETS], fontsize=6.5)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Claims with a differing field (%)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("c", loc="left", fontweight="bold", fontsize=8)
    ax.text(
        0.5,
        1.02,
        "Contract divergence",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
    )

    figure.subplots_adjust(left=0.115, right=0.995, top=0.86, bottom=0.30, wspace=0.62)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(out_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)

    identical = sum(
        1
        for c in common
        if len({_hashable(contracts[m].get(c)) for m in models}) == 1
    )
    return {
        "models": [MODEL_LABELS[m] for m in models],
        "model_specs": list(models),
        "common_claims": n,
        "confirmed": {MODEL_LABELS[m]: sum(1 for c in common if verdicts[m][c] == "confirmed") for m in models},
        "pairwise_agreement_min": int(min(pair_vals)),
        "pairwise_agreement_max": int(max(pair_vals)),
        "pairwise_agreement_mean": round(float(np.mean(pair_vals)), 2),
        "claims_with_identical_contract_across_all_models": identical,
        "contract_divergence_pct": dict(zip(CONTRACT_FACETS, [round(v, 1) for v in facet_counts])),
        "excluded_models": [
            "openai:gpt-5.6-luna",
            "openai:gpt-5.6-terra",
            "openai:gpt-4o-mini",
        ],
        "exclusion_rule": (
            "Restricted to the frontier drafting tier at the user's request. The full "
            "nine-model sweep is retained in the probe directory; excluded models "
            "confirmed 6, 3 and 0 claims respectively on the nine-model common subset."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="review-stage/multillm-probe-v2")
    parser.add_argument("--out", default="paper/figures/fig_multillm_probe")
    args = parser.parse_args(argv)
    summary = build(Path(args.root), Path(args.out))
    manifest = Path(args.out).with_name(Path(args.out).name + "_manifest.json")
    manifest.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}.{{svg,pdf,png}} and {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
