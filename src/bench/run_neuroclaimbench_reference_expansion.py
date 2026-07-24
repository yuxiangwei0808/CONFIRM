"""Derive full-corpus evidence-triage references from frozen adjudication artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench.neuroclaimbench_v21_compat import (
    BenchmarkItem,
    LabelVote,
    TriageReferenceProfile,
    derive_triage_reference,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("claims", "results"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_mode(item: BenchmarkItem) -> str:
    modes = sorted({reference.source_mode or "unknown" for reference in item.source_references})
    return "+".join(modes) if modes else "unknown"


def _observed_by_item(
    items: list[BenchmarkItem],
    result_paths: list[Path],
) -> dict[str, str]:
    by_source: dict[str, str] = {}
    for path in result_paths:
        for row in _read_results(path):
            source_id = str(row.get("claim_id") or "")
            label = str(row.get("final_label") or row.get("gate_verdict_label") or "")
            if not source_id or not label:
                continue
            previous = by_source.get(source_id)
            if previous is not None and previous != label:
                raise ValueError(f"Conflicting observed labels for {source_id}: {previous} vs {label}")
            by_source[source_id] = label

    observed: dict[str, str] = {}
    for item in items:
        labels = {
            by_source[reference.source_id]
            for reference in item.source_references
            if reference.source_id in by_source
        }
        if len(labels) > 1:
            raise ValueError(
                f"Exact item aliases have conflicting outcomes: "
                f"{item.benchmark_item_id}: {sorted(labels)}"
            )
        if labels:
            observed[item.benchmark_item_id] = next(iter(labels))
    return observed


def _group_summary(
    profiles: list[TriageReferenceProfile],
    observed: dict[str, str],
    field: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in sorted({str(getattr(profile, field)) for profile in profiles}):
        rows = [profile for profile in profiles if str(getattr(profile, field)) == value]
        evaluated = [profile for profile in rows if profile.benchmark_item_id in observed]
        confirmed = sum(observed[profile.benchmark_item_id] == "confirmed" for profile in evaluated)
        output[value] = {
            "n_items": len(rows),
            "n_executable": sum(profile.executable for profile in rows),
            "n_observed": len(evaluated),
            "confirmed_count": confirmed,
            "confirmation_rate": confirmed / len(evaluated) if evaluated else None,
        }
    return output


def _summary(
    items: list[BenchmarkItem],
    profiles: list[TriageReferenceProfile],
    observed: dict[str, str],
) -> dict[str, Any]:
    executable = [profile for profile in profiles if profile.executable]
    scored_references = [
        profile for profile in executable if profile.reference_strength in {"strict", "provisional"}
    ]
    return {
        "inventory": {
            "n_items": len(profiles),
            "n_executable": len(executable),
            "triage_label_counts": dict(Counter(profile.triage_label for profile in profiles)),
            "triage_disposition_counts": dict(
                Counter(profile.triage_disposition for profile in profiles)
            ),
            "reference_strength_counts": dict(
                Counter(profile.reference_strength for profile in profiles)
            ),
            "agreement_pattern_counts": dict(
                Counter(profile.agreement_pattern for profile in profiles)
            ),
            "benchmark_track_counts": dict(Counter(profile.benchmark_track for profile in profiles)),
            "target_family_counts": dict(Counter(profile.target_family for profile in profiles)),
            "evidence_triage_coverage": len(profiles) / len(items) if items else None,
            "scored_reference_count": len(scored_references),
            "scored_reference_coverage_among_executable": (
                len(scored_references) / len(executable) if executable else None
            ),
        },
        "observed_gate_coverage": {
            "n_observed": len(observed),
            "n_missing": len(profiles) - len(observed),
        },
        "metrics_by_triage_label": _group_summary(
            profiles, observed, "triage_label"
        ),
        "metrics_by_reference_strength": _group_summary(
            profiles, observed, "reference_strength"
        ),
        "metrics_by_agreement_pattern": _group_summary(
            profiles, observed, "agreement_pattern"
        ),
        "metrics_by_benchmark_track": _group_summary(
            profiles, observed, "benchmark_track"
        ),
        "metrics_by_target_family": _group_summary(
            profiles, observed, "target_family"
        ),
        "interpretation_restrictions": [
            "provisional references are separate from strict literature adjudications",
            "insufficient_evidence is not a null or negative label",
            "constructed, scientific, and external-transfer references are not pooled for headline accuracy",
            "observed CONFIRM labels do not determine triage reference derivation",
        ],
    }


def _write_csv(
    path: Path,
    items: dict[str, BenchmarkItem],
    profiles: list[TriageReferenceProfile],
    observed: dict[str, str],
) -> None:
    fieldnames = [
        "benchmark_item_id",
        "benchmark_track",
        "target_family",
        "source_mode",
        "source_label",
        "source_adjudication_status",
        "scientific_question_sha256",
        "triage_label",
        "triage_disposition",
        "reference_basis",
        "reference_strength",
        "derivation_rule",
        "executable",
        "agreeing_model_count",
        "agreement_pattern",
        "supporting_evidence_count",
        "score_tracks",
        "observed_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            item = items[profile.benchmark_item_id]
            writer.writerow(
                {
                    "benchmark_item_id": profile.benchmark_item_id,
                    "benchmark_track": profile.benchmark_track,
                    "target_family": profile.target_family,
                    "source_mode": _source_mode(item),
                    "source_label": profile.source_label,
                    "source_adjudication_status": profile.source_adjudication_status,
                    "scientific_question_sha256": profile.scientific_question_sha256,
                    "triage_label": profile.triage_label,
                    "triage_disposition": profile.triage_disposition,
                    "reference_basis": profile.reference_basis,
                    "reference_strength": profile.reference_strength,
                    "derivation_rule": profile.derivation_rule,
                    "executable": str(profile.executable).lower(),
                    "agreeing_model_count": len(profile.agreeing_models),
                    "agreement_pattern": profile.agreement_pattern,
                    "supporting_evidence_count": len(profile.supporting_evidence_ids),
                    "score_tracks": ";".join(profile.score_tracks),
                    "observed_label": observed.get(profile.benchmark_item_id, ""),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(args.package_dir)
    item_path = package / "benchmark_items.jsonl"
    vote_path = package / "label_votes.jsonl"
    items = [
        BenchmarkItem.model_validate(row)
        for row in _read_jsonl(item_path)
    ]
    item_by_id = {item.benchmark_item_id: item for item in items}
    if len(item_by_id) != len(items):
        raise ValueError("Duplicate benchmark_item_id values in benchmark_items.jsonl")

    votes_by_item: dict[str, list[LabelVote]] = defaultdict(list)
    for row in _read_jsonl(vote_path):
        vote = LabelVote.model_validate(row)
        if vote.benchmark_item_id not in item_by_id:
            raise ValueError(f"Vote references unknown benchmark item: {vote.benchmark_item_id}")
        votes_by_item[vote.benchmark_item_id].append(vote)

    profiles = [
        derive_triage_reference(item, votes_by_item.get(item.benchmark_item_id, []))
        for item in sorted(items, key=lambda row: row.benchmark_item_id)
    ]
    observed = _observed_by_item(items, [Path(path) for path in args.results])
    summary = _summary(items, profiles, observed)
    summary["provenance"] = {
        "benchmark_items_sha256": _file_sha256(item_path),
        "label_votes_sha256": _file_sha256(vote_path),
        "input_results": [str(path) for path in args.results],
        "derivation_policy": "neuroclaimbench-triage-reference-v1",
        "uses_llm_calls": False,
        "uses_pubmed_calls": False,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "triage_reference_profiles.jsonl"
    profile_path.write_text(
        "".join(profile.model_dump_json() + "\n" for profile in profiles),
        encoding="utf-8",
    )
    _write_csv(
        out_dir / "triage_reference_profiles.csv",
        item_by_id,
        profiles,
        observed,
    )
    (out_dir / "triage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default="data/neuroclaimbench/v2.1")
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/reference",
    )
    parser.add_argument(
        "--results",
        action="append",
        default=None,
        help="Repeatable CONFIRM result JSON used only for observed-decision summaries.",
    )
    parser.add_argument(
        "--no-observed-results",
        action="store_true",
        help="Derive references without attaching any prior CONFIRM outcomes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_observed_results:
        args.results = []
    elif args.results is None:
        args.results = [
            "review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json",
            "review-stage/claim-search-safety-gpt55-r10-c10-v7/gates/known_negative_results.json",
        ]
    summary = run(args)
    print(json.dumps({"status": "completed", "inventory": summary["inventory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
