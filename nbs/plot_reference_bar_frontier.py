"""Plot CONFIRM's three evidence policies under the published and relaxed reference bars.

The benchmark scores a claim only when two independent models agree that a
retrieved article is direct evidence for the exact frozen contrast. Claims that
fail that bar are unresolved and leave the scored denominators. This figure
replays the frozen assessor votes under a weaker bar, re-derives references, and
re-scores the three evidence policies on the enlarged set. CONFIRM verdicts are
frozen throughout; only the set of claims carrying a reference changes.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

#: Assessor labels that assert a determinate disposition.
DEFINITE_LABELS = {"known_positive", "known_null", "fragile"}
POSITIVE_LABEL = "known_positive"

SCREEN_GATES = ("search_provenance", "multiplicity", "confound", "confound_completeness")
REPLICATE_GATES = SCREEN_GATES + ("replication",)
FULL_GATES = REPLICATE_GATES + ("power", "multiverse")

POLICIES = (
    ("CONFIRM-Screen", SCREEN_GATES, "#4B8BBE"),
    ("CONFIRM-Replicate", REPLICATE_GATES, "#2D6A9F"),
    ("CONFIRM-Full", FULL_GATES, "#133C66"),
)
#: Replicate and Full return identical recovery, so the two series are separated by
#: weight and marker shape rather than by hue. Hue is reserved for the tier identity
#: used in the main-text frontier figure, and blending two translucent lines that lie
#: on top of each other would read as a single fourth colour.
POLICY_STYLE = {
    "CONFIRM-Screen": dict(lw=1.6, marker="o", markersize=3.8, filled=True, zorder=3),
    "CONFIRM-Full": dict(lw=3.0, marker="o", markersize=4.2, filled=True, zorder=4),
    "CONFIRM-Replicate": dict(lw=1.1, marker="s", markersize=5.2, filled=False, zorder=5),
}

#: (id, label, requires two agreeing models, requires exact construct match)
PUBLISHED_RULE = ("two_exact", "Two models, exact match", True, True)
RELAXED_RULE = ("one_exact", "One model, exact match", False, True)
SECONDARY_RULES = (
    ("two_any", "Two models, any construct", True, False),
    ("one_any", "One model, any construct", False, False),
)
#: Reference bars ordered from strictest to most permissive, measured by how many
#: previously unresolved claims each one admits.
RULE_SEQUENCE = (
    ("two_exact", "Two models\nexact match"),
    ("two_any", "Two models\nany construct"),
    ("one_exact", "One model\nexact match"),
    ("one_any", "One model\nany construct"),
)


def _load_gates(path: Path) -> dict[str, dict[str, bool]]:
    gates: dict[str, dict[str, bool]] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            verdict = record.get("gate_verdict") or {}
            gates[record["benchmark_case_id"]] = verdict.get("gates") or {}
    return gates


def _load_references(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        return {json.loads(l)["benchmark_case_id"]: json.loads(l) for l in handle}


def _load_profiles(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["benchmark_item_id"]: row for row in csv.DictReader(handle)}


def _load_votes(path: Path) -> dict[str, list[dict]]:
    votes: dict[str, list[dict]] = collections.defaultdict(list)
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            votes[record["benchmark_item_id"]].append(record)
    return votes


def _resolve(votes: list[dict], *, need_two: bool, need_exact: bool) -> str | None:
    candidates = [v for v in votes if v["proposed_label"] in DEFINITE_LABELS]
    if need_exact:
        candidates = [v for v in candidates if v["construct_match"] == "exact"]
    if not candidates:
        return None
    label, support = collections.Counter(
        v["proposed_label"] for v in candidates
    ).most_common(1)[0]
    if need_two and support < 2:
        return None
    return label


def _passes(gates: dict[str, dict[str, bool]], item: str, keys: tuple[str, ...]) -> bool:
    record = gates.get(item) or {}
    return all(record.get(key) is True for key in keys)


def collect(args: argparse.Namespace) -> dict:
    gates = _load_gates(Path(args.outcomes))
    references = _load_references(Path(args.references))
    profiles = _load_profiles(Path(args.reference_profiles))
    votes = _load_votes(Path(args.label_votes))

    def scientific(item: str) -> bool:
        return profiles.get(item, {}).get("benchmark_track") == "scientific"

    base_positive = [
        i
        for i, r in references.items()
        if r["score_eligible"]
        and r["basis"] == "literature"
        and r["disposition"] == "confirm"
        and scientific(i)
    ]
    base_abstain = [
        i
        for i, r in references.items()
        if r["score_eligible"]
        and r["basis"] == "literature"
        and r["disposition"] == "abstain"
        and scientific(i)
    ]
    unresolved = [
        i
        for i, row in profiles.items()
        if row["reference_strength"] == "evidence_gap"
        and row["executable"].strip().lower() == "true"
        and row["benchmark_track"] == "scientific"
        and i in gates
    ]

    results: dict[str, dict] = {}
    for rule_id, rule_label, need_two, need_exact in (
        PUBLISHED_RULE,
        RELAXED_RULE,
        *SECONDARY_RULES,
    ):
        added_positive, added_abstain = [], []
        for item in unresolved:
            label = _resolve(votes.get(item, []), need_two=need_two, need_exact=need_exact)
            if label is None:
                continue
            (added_positive if label == POSITIVE_LABEL else added_abstain).append(item)
        per_policy = {}
        for name, keys, _ in POLICIES:
            base_hits = sum(_passes(gates, i, keys) for i in base_positive)
            new_hits = sum(_passes(gates, i, keys) for i in added_positive)
            unsafe = sum(_passes(gates, i, keys) for i in base_abstain) + sum(
                _passes(gates, i, keys) for i in added_abstain
            )
            per_policy[name] = {
                "recovery_count": base_hits + new_hits,
                "recovery_total": len(base_positive) + len(added_positive),
                "unsafe_count": unsafe,
                "unsafe_total": len(base_abstain) + len(added_abstain),
                "base_hits": base_hits,
                "base_total": len(base_positive),
                "new_hits": new_hits,
                "new_total": len(added_positive),
            }
        results[rule_id] = {
            "label": rule_label,
            "added_positive": len(added_positive),
            "added_abstain": len(added_abstain),
            "policies": per_policy,
        }
    results["_meta"] = {
        "base_positive": len(base_positive),
        "base_abstain": len(base_abstain),
        "unresolved_pool": len(unresolved),
    }
    return results


def _rate(entry: dict, num: str, den: str) -> float:
    return 100.0 * entry[num] / entry[den] if entry[den] else float("nan")


def plot(results: dict, output_prefix: Path) -> None:
    """Three policy trajectories across reference bars ordered by permissiveness."""

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    x = np.arange(len(RULE_SEQUENCE))
    published_index = [r for r, _ in RULE_SEQUENCE].index(PUBLISHED_RULE[0])

    panels = (
        (axes[0], "recovery_count", "recovery_total", "Recovery on positive references (%)",
         "a", "The recall cost falls on the strict policies", (30, 60)),
        (axes[1], "unsafe_count", "unsafe_total", "False support on abstention references (%)",
         "b", "False support ordering holds at every bar", (0, 44)),
    )

    for ax, num, den, ylabel, letter, subtitle, ylim in panels:
        ax.axvspan(published_index - 0.32, published_index + 0.32,
                   color="#F0F0F0", zorder=0)
        ax.text(published_index, ylim[1] - 0.04 * (ylim[1] - ylim[0]), "published",
                fontsize=5.4, color="#8A8A8A", ha="center", va="top")
        endpoints = []
        for name, _, colour in POLICIES:
            values = [
                _rate(results[rule]["policies"][name], num, den)
                for rule, _ in RULE_SEQUENCE
            ]
            style = POLICY_STYLE[name]
            ax.plot(
                x, values,
                color=colour, lw=style["lw"], zorder=style["zorder"], linestyle="-",
                marker=style["marker"], markersize=style["markersize"],
                markerfacecolor=colour if style["filled"] else "white",
                markeredgewidth=1.2, markeredgecolor=colour,
            )
            endpoints.append([values[-1], name.replace("CONFIRM-", ""), colour])

        # Nudge coincident end labels apart rather than letting them overprint.
        # Ties break by strictness so the stacking order matches across panels.
        min_gap = 0.058 * (ylim[1] - ylim[0])
        order = {name.replace("CONFIRM-", ""): i for i, (name, _, _) in enumerate(POLICIES)}
        endpoints.sort(key=lambda item: (item[0], -order[item[1]]))
        for lower, upper in zip(endpoints, endpoints[1:]):
            if upper[0] - lower[0] < min_gap:
                shift = (min_gap - (upper[0] - lower[0])) / 2.0
                lower[0] -= shift
                upper[0] += shift
        for label_y, label, colour in endpoints:
            ax.text(
                x[-1] + 0.13, label_y, label,
                fontsize=6.3, color=colour, fontweight="bold", va="center", ha="left",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                f"{label}\n({results[rule]['added_positive'] + results[rule]['added_abstain']})"
                for rule, label in RULE_SEQUENCE
            ],
            fontsize=5.9,
        )
        ax.set_xlim(-0.45, len(RULE_SEQUENCE) - 0.30)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=6.7)
        ax.set_xlabel(
            "Reference bar, strictest to most permissive\n(references admitted in parentheses)",
            fontsize=6.7,
        )
        ax.text(-0.145, 1.06, letter, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")
        ax.text(-0.065, 1.068, subtitle, transform=ax.transAxes,
                fontsize=7.0, fontweight="bold", va="bottom")
        ax.grid(axis="y", color="#ECECEC", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.2, length=2.4, width=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.8)

    axes[0].annotate(
        "Replicate (thin, squares) lies\non Full (thick, circles)",
        xy=(1.0, 41.2), xytext=(0.52, 33.8),
        fontsize=5.6, color="#4D4D4D", ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=0.5, shrinkA=1, shrinkB=3),
    )

    figure.tight_layout(pad=0.9)
    figure.subplots_adjust(top=0.845, wspace=0.30)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(
        output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", default="benchmark/neuroclaimbench-v2.1/outcomes.jsonl")
    parser.add_argument("--references", default="benchmark/neuroclaimbench-v2.1/references.jsonl")
    parser.add_argument("--label-votes", default="data/neuroclaimbench/v2.1/label_votes.jsonl")
    parser.add_argument(
        "--reference-profiles",
        default="review-stage/neuroclaimbench-v2.1/reference/triage_reference_profiles.csv",
    )
    parser.add_argument("--output-prefix", default="paper/figures/fig_reference_bar_frontier")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = collect(args)
    meta = results["_meta"]
    print(
        f"baseline pool {meta['base_positive']} positives / {meta['base_abstain']} abstentions; "
        f"unresolved pool {meta['unresolved_pool']}"
    )
    for rule_id, _, _, _ in (PUBLISHED_RULE, RELAXED_RULE, *SECONDARY_RULES):
        entry = results[rule_id]
        print(f"\n{entry['label']}  (+{entry['added_positive']} pos, +{entry['added_abstain']} abs)")
        for name, _, _ in POLICIES:
            p = entry["policies"][name]
            print(
                f"  {name:20s} recovery {p['recovery_count']:2d}/{p['recovery_total']:2d} "
                f"({_rate(p, 'recovery_count', 'recovery_total'):5.1f}%)   "
                f"false support {p['unsafe_count']}/{p['unsafe_total']} "
                f"({_rate(p, 'unsafe_count', 'unsafe_total'):5.1f}%)"
            )
    plot(results, Path(args.output_prefix))
    print(f"\nwrote {args.output_prefix}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
