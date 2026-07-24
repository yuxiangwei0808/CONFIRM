"""Build a compact, result-preserving NeuroClaimBench v2.1 release view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from bench.benchmark import (
    BenchmarkCase,
    BenchmarkEvaluationTask,
    BenchmarkReference,
    TaskOutcome,
)
from bench.io import atomic_text, read_jsonl, write_jsonl


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _disposition(value: str) -> str:
    return "unresolved" if value == "request_evidence" else value


def compact_benchmark(
    *,
    package_dir: Path,
    reference_dir: Path,
    results_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    items = read_jsonl(package_dir / "benchmark_items.jsonl")
    profiles = read_jsonl(
        reference_dir / "triage_reference_profiles.jsonl"
    )
    tasks = read_jsonl(package_dir / "evaluation_tasks.jsonl")
    outcomes = read_jsonl(results_dir / "task_outcomes.jsonl")
    profile_by_id = {
        row["benchmark_item_id"]: row for row in profiles
    }
    if len(profile_by_id) != len(items):
        raise ValueError("Every benchmark item must have one reference profile")

    cases: list[BenchmarkCase] = []
    references: list[BenchmarkReference] = []
    for item in items:
        item_id = str(item["benchmark_item_id"])
        profile = profile_by_id[item_id]
        disposition = _disposition(
            str(profile["triage_disposition"])
        )
        score_eligible = bool(
            disposition != "unresolved"
            and profile.get("executable")
        )
        case = BenchmarkCase(
            benchmark_case_id=item_id,
            claim_uid=item["claim_uid"],
            semantic_cluster_id=item["semantic_cluster_id"],
            benchmark_track=item["benchmark_track"],
            target_family=item["target_family"],
            modality=item["modality"],
            question=item["question"],
            question_sha256=item["scientific_question_sha256"],
            contract=item.get("contract"),
            contract_sha256=item.get("exact_contract_sha256"),
            scientific_core_sha256=item.get(
                "scientific_core_sha256", ""
            ),
            provenance=list(item.get("source_references") or []),
            aliases=list(item.get("aliases") or []),
            reference_id=item_id,
            task_ids=list(item.get("evaluation_task_ids") or []),
            migration_status=item["migration_status"],
            alignment_disposition=item.get("alignment_disposition"),
            pre_v2_contract_sha256=item.get(
                "pre_v2_contract_sha256"
            ),
        )
        reference = BenchmarkReference(
            reference_id=item_id,
            benchmark_case_id=item_id,
            disposition=disposition,
            basis=profile["reference_basis"],
            strength=profile["reference_strength"],
            score_eligible=score_eligible,
            evidence_ids=list(
                profile.get("supporting_evidence_ids") or []
            ),
            derivation_rule=profile["derivation_rule"],
            source_label=profile["source_label"],
            source_adjudication_status=profile[
                "source_adjudication_status"
            ],
            agreeing_models=list(
                profile.get("agreeing_models") or []
            ),
            agreement_pattern=str(
                profile.get("agreement_pattern") or ""
            ),
            vote_counts={
                str(key): int(value)
                for key, value in (
                    profile.get("vote_counts") or {}
                ).items()
            },
        )
        cases.append(case)
        references.append(reference)

    compact_tasks = [
        BenchmarkEvaluationTask(
            task_id=row["task_id"],
            benchmark_case_id=row["benchmark_item_id"],
            dataset_id=row["dataset_id"],
            contract=row["contract"],
            contract_sha256=row["executable_contract_sha256"],
            discovery_cohort=row["discovery_cohort"],
            replication_cohorts=list(row["replication_cohorts"]),
            evidence_role=row["evidence_role"],
            evidence_freshness=row["evidence_freshness"],
            partition_paths=list(row["partition_paths"]),
            partition_hashes=dict(row["partition_hashes"]),
            generator_spec=row.get("generator_spec"),
            generator_artifact_hashes=dict(
                row.get("generator_artifact_hashes") or {}
            ),
            question_sha256=row["scientific_question_sha256"],
            scientific_core_sha256=row["scientific_core_sha256"],
            code_sha=row["code_sha"],
            schema_version=row["schema_version"],
            gate_policy_version=row["gate_policy_version"],
        )
        for row in tasks
    ]
    compact_outcomes = [
        TaskOutcome(
            task_id=row["task_id"],
            benchmark_case_id=row["benchmark_item_id"],
            benchmark_claim_id=row.get("benchmark_claim_id"),
            status=row["status"],
            confirm_outcome=row.get("confirm_outcome"),
            raw_final_label=row.get("raw_final_label"),
            gate_verdict=row.get("gate_verdict"),
            error=row.get("error"),
            task_fingerprint=row["task_fingerprint"],
            result_source=row["result_source"],
            cohort_paths=[
                (
                    os.path.relpath(path, Path.cwd())
                    if Path(path).is_absolute()
                    else path
                )
                for path in row.get("cohort_paths") or []
            ],
            cohort_content_hashes=dict(
                row.get("cohort_content_hashes") or {}
            ),
            generated_cohort_hashes=dict(
                row.get("generated_cohort_hashes") or {}
            ),
            detailed_result_sha256=_json_sha256(
                row.get("gate_results")
            ),
        )
        for row in outcomes
    ]

    task_ids = {task.task_id for task in compact_tasks}
    outcome_task_ids = {outcome.task_id for outcome in compact_outcomes}
    case_ids = {case.benchmark_case_id for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("Duplicate compact benchmark case IDs")
    if len(task_ids) != len(compact_tasks):
        raise ValueError("Duplicate compact evaluation task IDs")
    if outcome_task_ids != task_ids:
        raise ValueError(
            "Compact task/outcome reconciliation failed: "
            f"missing={sorted(task_ids - outcome_task_ids)[:5]} "
            f"extra={sorted(outcome_task_ids - task_ids)[:5]}"
        )
    if any(task.benchmark_case_id not in case_ids for task in compact_tasks):
        raise ValueError("Compact task references an unknown case")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases": out_dir / "cases.jsonl",
        "references": out_dir / "references.jsonl",
        "tasks": out_dir / "tasks.jsonl",
        "outcomes": out_dir / "outcomes.jsonl",
    }
    write_jsonl(paths["cases"], cases)
    write_jsonl(paths["references"], references)
    write_jsonl(paths["tasks"], compact_tasks)
    write_jsonl(paths["outcomes"], compact_outcomes)
    manifest = {
        "benchmark_version": "v2.1",
        "release_schema_version": 2,
        "scientific_results_changed": False,
        "source_package": str(package_dir),
        "source_reference_dir": str(reference_dir),
        "source_results_dir": str(results_dir),
        "counts": {
            "cases": len(cases),
            "references": len(references),
            "tasks": len(compact_tasks),
            "outcomes": len(compact_outcomes),
            "score_eligible": sum(
                reference.score_eligible
                for reference in references
            ),
            "reference_dispositions": dict(
                Counter(
                    reference.disposition
                    for reference in references
                )
            ),
        },
        "source_hashes": {
            "benchmark_items.jsonl": _file_sha256(
                package_dir / "benchmark_items.jsonl"
            ),
            "reference_profiles.jsonl": _file_sha256(
                reference_dir / "triage_reference_profiles.jsonl"
            ),
            "evaluation_tasks.jsonl": _file_sha256(
                package_dir / "evaluation_tasks.jsonl"
            ),
            "task_outcomes.jsonl": _file_sha256(
                results_dir / "task_outcomes.jsonl"
            ),
            "benchmark_summary.json": _file_sha256(
                results_dir / "benchmark_summary.json"
            ),
        },
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for name, path in paths.items()
        },
    }
    atomic_text(
        out_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        default="data/neuroclaimbench/v2.1",
    )
    parser.add_argument(
        "--reference-dir",
        default="review-stage/neuroclaimbench-v2.1/reference",
    )
    parser.add_argument(
        "--results-dir",
        default="review-stage/neuroclaimbench-v2.1/results",
    )
    parser.add_argument(
        "--out",
        default="review-stage/neuroclaimbench-v2.1/compact",
    )
    args = parser.parse_args()
    compact_benchmark(
        package_dir=Path(args.package_dir),
        reference_dir=Path(args.reference_dir),
        results_dir=Path(args.results_dir),
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()
