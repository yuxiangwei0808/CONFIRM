"""Generate NeuroClaimBench v2.1 audit tables after the frozen task rerun."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench.benchmark import BenchmarkDataset


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_row(stratum: str, metrics: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    return {
        "stratum": stratum,
        **metadata,
        "n_claims": metrics.get("n_claims"),
        "n_score_eligible": metrics.get("n_score_eligible"),
        "n_evaluated": metrics.get("n_evaluated"),
        "n_errors": metrics.get("n_errors"),
        "confirmable_count": metrics.get("confirmable_claim_recall_count"),
        "confirmable_denominator": metrics.get("confirmable_claim_recall_denominator"),
        "confirmable_recall": metrics.get("confirmable_claim_recall"),
        "confirmable_wilson_95": json.dumps(
            metrics.get("confirmable_claim_recall_wilson_95")
        ),
        "confirmable_cluster_bootstrap_95": json.dumps(
            metrics.get("confirmable_claim_recall_cluster_bootstrap_95")
        ),
        "unsafe_count": metrics.get("unsafe_confirmation_count"),
        "unsafe_denominator": metrics.get("unsafe_confirmation_denominator"),
        "unsafe_rate": metrics.get("unsafe_confirmation_rate"),
        "unsafe_wilson_95": json.dumps(metrics.get("unsafe_confirmation_wilson_95")),
        "unsafe_cluster_bootstrap_95": json.dumps(
            metrics.get("unsafe_confirmation_cluster_bootstrap_95")
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(args.package_dir)
    results = Path(args.results_dir)
    reference = Path(args.reference_dir)
    out_dir = Path(args.out_dir)
    compact = BenchmarkDataset(args.compact_dir)
    items = compact.cases
    tasks = compact.tasks
    outcomes = compact.outcomes
    profiles = compact.references
    item_by_id = {item.benchmark_case_id: item for item in items}
    profile_by_id = {
        profile.benchmark_case_id: profile for profile in profiles
    }
    outcome_by_task = {outcome.task_id: outcome for outcome in outcomes}
    if len(outcome_by_task) != len(outcomes):
        raise ValueError("Duplicate task outcomes")
    missing_outcomes = {task.task_id for task in tasks} - set(outcome_by_task)
    extra_outcomes = set(outcome_by_task) - {task.task_id for task in tasks}
    if missing_outcomes or extra_outcomes:
        raise ValueError(
            f"Task/result reconciliation failed: missing={len(missing_outcomes)} extra={len(extra_outcomes)}"
        )
    missing_partition_hashes = [
        (task.task_id, cohort)
        for task in tasks
        for cohort in [task.discovery_cohort, *task.replication_cohorts]
        if not task.partition_hashes.get(cohort)
    ]
    if missing_partition_hashes:
        raise ValueError(f"Missing frozen partition hashes: {missing_partition_hashes[:5]}")
    errors = [outcome for outcome in outcomes if outcome.status == "error"]
    if errors:
        raise ValueError(f"v2.1 task rerun contains {len(errors)} execution errors")

    scored_alignment_mismatches = []
    for profile in profiles:
        item = item_by_id[profile.benchmark_case_id]
        if profile.strength in {"strict", "provisional", "constructed"} and (
            item.migration_status != "ready"
            or item.alignment_disposition
            not in {"aligned", "aligned_with_safety_augmentation", "repairable_contract"}
        ):
            scored_alignment_mismatches.append(item.benchmark_case_id)
    if scored_alignment_mismatches:
        raise ValueError(
            f"Scored items contain unresolved question-contract mismatches: {scored_alignment_mismatches[:5]}"
        )

    agreement_groups: dict[tuple[str, str, str], int] = Counter(
        (
            profile.basis,
            profile.strength,
            profile.agreement_pattern,
        )
        for profile in profiles
    )
    agreement_rows = [
        {
            "reference_basis": basis,
            "reference_strength": strength,
            "agreement_pattern": pattern,
            "claim_count": count,
        }
        for (basis, strength, pattern), count in sorted(agreement_groups.items())
    ]

    unresolved_rows: list[dict[str, Any]] = []
    for profile in profiles:
        if profile.disposition != "unresolved":
            continue
        item = item_by_id[profile.benchmark_case_id]
        task_outcomes = [
            outcome_by_task[task_id].confirm_outcome
            for task_id in item.task_ids
            if task_id in outcome_by_task
        ]
        unresolved_rows.append(
            {
                "benchmark_item_id": item.benchmark_case_id,
                "benchmark_track": item.benchmark_track,
                "target_family": item.target_family,
                "modality": item.modality,
                "alignment_disposition": item.alignment_disposition or "",
                "agreement_pattern": profile.agreement_pattern,
                "vote_count": sum(profile.vote_counts.values()),
                "confirm_outcome": (
                    "confirmed"
                    if "confirmed" in task_outcomes
                    else ("abstained" if "abstained" in task_outcomes else "not_evaluated")
                ),
            }
        )

    cache_manifest_path = Path(args.pubmed_cache_dir) / "cache_manifest.json"
    cache_audit: dict[str, Any] = {"available": cache_manifest_path.exists()}
    if cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_audit.update(
            {
                "cache_version": cache_manifest.get("cache_version"),
                "status": cache_manifest.get("status"),
                "counts": cache_manifest.get("counts"),
                "parameters": cache_manifest.get("parameters"),
                "assessor_models": cache_manifest.get("assessor_models"),
            }
        )
    alignment_manifest = json.loads(
        Path(args.alignment_manifest).read_text(encoding="utf-8")
    )
    reference_summary = json.loads(
        (reference / "triage_summary.json").read_text(encoding="utf-8")
    )
    benchmark_summary = json.loads(
        (results / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    strata_rows = [
        _metric_row(
            "scientific_literature_combined",
            benchmark_summary["primary_scientific_literature"],
            benchmark_split="scientific",
            reference_basis="literature",
            reference_strength="strict+provisional",
            dataset_id="",
        )
    ]
    for strength, metrics in benchmark_summary["scientific_literature_sensitivity"].items():
        strata_rows.append(
            _metric_row(
                f"scientific_literature_{strength}",
                metrics,
                benchmark_split="scientific",
                reference_basis="literature",
                reference_strength=strength,
                dataset_id="",
            )
        )
    for dataset, tiers in benchmark_summary["external_literature_by_dataset"].items():
        for strength, metrics in tiers.items():
            strata_rows.append(
                _metric_row(
                    f"external_literature_{dataset}_{strength}",
                    metrics,
                    benchmark_split="external_transfer",
                    reference_basis="literature",
                    reference_strength=(
                        "strict+provisional"
                        if strength == "combined_primary"
                        else strength
                    ),
                    dataset_id=dataset,
                )
            )
    for dataset, metrics in benchmark_summary[
        "external_constructed_controls_by_dataset"
    ].items():
        strata_rows.append(
            _metric_row(
                f"external_constructed_{dataset}",
                metrics,
                benchmark_split="external_transfer",
                reference_basis="constructed_control",
                reference_strength="constructed",
                dataset_id=dataset,
            )
        )
    strata_rows.append(
        _metric_row(
            "synthetic_constructed",
            benchmark_summary["synthetic_constructed_controls"],
            benchmark_split="synthetic_safety",
            reference_basis="constructed_control",
            reference_strength="constructed",
            dataset_id="",
        )
    )
    target_rows: list[dict[str, Any]] = []
    target_metrics = benchmark_summary[
        "metrics_by_split_target_reference_basis_and_strength"
    ]
    for split, targets in target_metrics.items():
        for target, bases in targets.items():
            for basis, strengths in bases.items():
                for strength, metrics in strengths.items():
                    target_rows.append(
                        _metric_row(
                            f"{split}:{target}:{basis}:{strength}",
                            metrics,
                            benchmark_split=split,
                            target_family=target,
                            reference_basis=basis,
                            reference_strength=strength,
                        )
                    )
    unresolved_distribution = {
        "by_track": {
            track: dict(
                Counter(
                    row["confirm_outcome"]
                    for row in unresolved_rows
                    if row["benchmark_track"] == track
                )
            )
            for track in sorted({row["benchmark_track"] for row in unresolved_rows})
        },
        "by_target_family": {
            target: dict(
                Counter(
                    row["confirm_outcome"]
                    for row in unresolved_rows
                    if row["target_family"] == target
                )
            )
            for target in sorted({row["target_family"] for row in unresolved_rows})
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "reference_agreement_audit.csv", agreement_rows)
    _write_csv(out_dir / "benchmark_strata_summary.csv", strata_rows)
    _write_csv(out_dir / "target_reference_summary.csv", target_rows)
    _write_csv(
        out_dir / "unresolved_case_summary.csv",
        unresolved_rows,
        fieldnames=[
            "benchmark_item_id",
            "benchmark_track",
            "target_family",
            "modality",
            "alignment_disposition",
            "agreement_pattern",
            "vote_count",
            "confirm_outcome",
        ],
    )
    audit = {
        "version": "neuroclaimbench-v2.1-analysis-v1",
        "acceptance": {
            "benchmark_items": len(items),
            "evaluation_tasks": len(tasks),
            "task_outcomes": len(outcomes),
            "missing_partition_hash_count": len(missing_partition_hashes),
            "scored_alignment_mismatch_count": len(scored_alignment_mismatches),
            "execution_error_count": len(errors),
            "task_result_reconciled": not missing_outcomes and not extra_outcomes,
        },
        "alignment_audit": alignment_manifest,
        "reference_expansion_audit": reference_summary,
        "pubmed_retrieval_audit": cache_audit,
        "unresolved_case_audit": {
            "count": len(unresolved_rows),
            "verdict_distribution": unresolved_distribution,
        },
        "benchmark_metrics": benchmark_summary,
        "interpretation_restrictions": [
            "v2.1 is retrospective, not prospective validation",
            "unresolved claims are excluded from accuracy denominators",
            "constructed controls are separate from literature references",
            "external NACC and CNP results remain separate",
        ],
    }
    (out_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, payload in (
        ("alignment_audit.json", alignment_manifest),
        ("reference_expansion_audit.json", reference_summary),
        ("pubmed_retrieval_audit.json", cache_audit),
    ):
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    input_paths = {
        "cases.jsonl": Path(args.compact_dir) / "cases.jsonl",
        "references.jsonl": Path(args.compact_dir) / "references.jsonl",
        "tasks.jsonl": Path(args.compact_dir) / "tasks.jsonl",
        "outcomes.jsonl": Path(args.compact_dir) / "outcomes.jsonl",
        "benchmark_summary.json": results / "benchmark_summary.json",
        "alignment_manifest.json": Path(args.alignment_manifest),
    }
    output_paths = {
        name: out_dir / name
        for name in (
            "reference_agreement_audit.csv",
            "benchmark_strata_summary.csv",
            "target_reference_summary.csv",
            "unresolved_case_summary.csv",
            "analysis_audit.json",
            "alignment_audit.json",
            "reference_expansion_audit.json",
            "pubmed_retrieval_audit.json",
        )
    }
    analysis_manifest = {
        "version": "neuroclaimbench-v2.1-analysis-v1",
        "input_sha256": {
            name: _sha256_file(path) for name, path in input_paths.items()
        },
        "output_sha256": {
            name: _sha256_file(path) for name, path in output_paths.items()
        },
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 20260723,
        "outcome_blind_reference_construction": True,
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default="data/neuroclaimbench/v2.1")
    parser.add_argument(
        "--compact-dir",
        default="benchmark/neuroclaimbench-v2.1/compact",
    )
    parser.add_argument("--results-dir", default="review-stage/neuroclaimbench-v2.1/results")
    parser.add_argument("--reference-dir", default="review-stage/neuroclaimbench-v2.1/reference")
    parser.add_argument(
        "--alignment-manifest",
        default="review-stage/neuroclaimbench-v2.1/alignment/alignment_manifest.json",
    )
    parser.add_argument(
        "--pubmed-cache-dir",
        default="data/neuroclaimbench/pubmed-cache-v2.1",
    )
    parser.add_argument("--out-dir", default="review-stage/neuroclaimbench-v2.1/analysis")
    return parser


def main(argv: list[str] | None = None) -> int:
    audit = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", "acceptance": audit["acceptance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
