"""Test whether the unresolved claims could be labelled at a weaker evidence bar.

A reference is derived only when two independent models agree that a retrieved
article is direct evidence for the exact frozen contrast. Claims that fail this
bar are unresolved and leave the scored denominators. This replay relaxes the
bar, re-derives references from the frozen assessor votes, and re-scores CONFIRM
on the enlarged set. It quantifies how much the scored subset differs from the
claims that could not be labelled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "neuroclaimbench-v2.1-reference-bar-sensitivity-v1"

#: Assessor labels that assert a determinate disposition.
DEFINITE_LABELS = {"known_positive", "known_null", "fragile"}
#: Labels mapped onto the reference disposition used for scoring.
POSITIVE_LABEL = "known_positive"

RULES: tuple[tuple[str, str, bool, bool], ...] = (
    ("two_exact", "Two models agree, exact construct match", True, True),
    ("two_any", "Two models agree, any construct match", True, False),
    ("one_exact", "One model, exact construct match", False, True),
    ("one_any", "One model, any construct match", False, False),
)


def _load_votes(path: Path) -> dict[str, list[dict[str, Any]]]:
    votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            votes[record["benchmark_item_id"]].append(record)
    return votes


def _load_profiles(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["benchmark_item_id"]: row for row in csv.DictReader(handle)}


def _load_verdicts(path: Path) -> dict[str, tuple[str, str]]:
    """Map benchmark item to (task id, CONFIRM verdict)."""

    verdicts: dict[str, tuple[str, str]] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            verdicts[record["benchmark_item_id"]] = (
                record["task_id"],
                record["confirm_outcome"],
            )
    return verdicts


def _resolve(
    votes: list[dict[str, Any]],
    *,
    need_two: bool,
    need_exact: bool,
) -> str | None:
    candidates = [vote for vote in votes if vote["proposed_label"] in DEFINITE_LABELS]
    if need_exact:
        candidates = [vote for vote in candidates if vote["construct_match"] == "exact"]
    if not candidates:
        return None
    counts = Counter(vote["proposed_label"] for vote in candidates)
    label, support = counts.most_common(1)[0]
    if need_two and support < 2:
        return None
    return label


def run(args: argparse.Namespace) -> None:
    votes = _load_votes(Path(args.label_votes))
    profiles = _load_profiles(Path(args.reference_profiles))
    verdicts = _load_verdicts(Path(args.task_outcomes))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Executable scientific claims that carry no reference under the frozen bar.
    unresolved = [
        item
        for item, row in profiles.items()
        if row["reference_strength"] == "evidence_gap"
        and row["executable"].strip().lower() == "true"
        and row["benchmark_track"] == "scientific"
        and item in verdicts
    ]

    baseline_recovery = (args.baseline_recovery_count, args.baseline_recovery_denominator)
    baseline_unsafe = (args.baseline_unsafe_count, args.baseline_unsafe_denominator)

    rows: list[dict[str, Any]] = []
    for rule_id, rule_label, need_two, need_exact in RULES:
        added_positive = added_abstain = 0
        confirmed_positive = confirmed_abstain = 0
        for item in unresolved:
            label = _resolve(votes.get(item, []), need_two=need_two, need_exact=need_exact)
            if label is None:
                continue
            confirmed = verdicts[item][1] == "confirmed"
            if label == POSITIVE_LABEL:
                added_positive += 1
                confirmed_positive += int(confirmed)
            else:
                added_abstain += 1
                confirmed_abstain += int(confirmed)
        recovery_count = baseline_recovery[0] + confirmed_positive
        recovery_denominator = baseline_recovery[1] + added_positive
        unsafe_count = baseline_unsafe[0] + confirmed_abstain
        unsafe_denominator = baseline_unsafe[1] + added_abstain
        rows.append(
            {
                "rule_id": rule_id,
                "rule_label": rule_label,
                "requires_two_models": need_two,
                "requires_exact_match": need_exact,
                "is_published_bar": rule_id == "two_exact",
                "unresolved_pool": len(unresolved),
                "newly_labelled": added_positive + added_abstain,
                "added_positive_references": added_positive,
                "added_abstention_references": added_abstain,
                "recovery_count": recovery_count,
                "recovery_denominator": recovery_denominator,
                "recovery_rate": recovery_count / recovery_denominator,
                "unsafe_count": unsafe_count,
                "unsafe_denominator": unsafe_denominator,
                "unsafe_rate": unsafe_count / unsafe_denominator,
                "recovery_on_new_positives": (
                    confirmed_positive / added_positive if added_positive else None
                ),
            }
        )

    with (output_dir / "reference_bar_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "label_votes": str(args.label_votes),
        "label_votes_sha256": hashlib.sha256(Path(args.label_votes).read_bytes()).hexdigest(),
        "reference_profiles": str(args.reference_profiles),
        "unresolved_scientific_executable": len(unresolved),
        "baseline_recovery": list(baseline_recovery),
        "baseline_unsafe": list(baseline_unsafe),
        "interpretation_restrictions": [
            "Deterministic replay of frozen assessor votes; no new retrieval or model call.",
            "Relaxed bars are diagnostics, not proposed reference tiers.",
            "A single-model rule has no cross-model agreement and is weaker than the provisional tier.",
        ],
    }
    (output_dir / "reference_bar_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )

    print(f"unresolved executable scientific claims: {len(unresolved)}")
    for row in rows:
        marker = " (published bar)" if row["is_published_bar"] else ""
        rate = row["recovery_on_new_positives"]
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        print(
            f"  {row['rule_label']:42}{marker}\n"
            f"      labels +{row['newly_labelled']:3d}  "
            f"recovery {row['recovery_count']}/{row['recovery_denominator']} "
            f"({row['recovery_rate']:.1%})  "
            f"recovery on new positives {rate_text}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-votes", default="data/neuroclaimbench/v2.1/label_votes.jsonl")
    parser.add_argument(
        "--reference-profiles",
        default="review-stage/neuroclaimbench-v2.1/reference/triage_reference_profiles.csv",
    )
    parser.add_argument(
        "--task-outcomes",
        default="review-stage/neuroclaimbench-v2.1/results/task_outcomes.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="review-stage/neuroclaimbench-v2.1/reference-bar-sensitivity-v1",
    )
    parser.add_argument("--baseline-recovery-count", type=int, default=21)
    parser.add_argument("--baseline-recovery-denominator", type=int, default=51)
    parser.add_argument("--baseline-unsafe-count", type=int, default=2)
    parser.add_argument("--baseline-unsafe-denominator", type=int, default=19)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
