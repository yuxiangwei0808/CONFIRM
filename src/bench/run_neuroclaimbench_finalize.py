"""Evaluate the frozen NeuroClaimBench package and summarize its results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bench.neuroclaimbench_v21_compat import (
    BenchmarkItem,
    BenchmarkTaskOutcome,
    EvaluationTask,
    SimplifiedBenchmarkClaim,
    TriageReferenceProfile,
    canonical_json,
    exact_contract_hash,
    semantic_contract_hash,
    sha256_payload,
)
from bench.progress import iter_progress
from confirm.execution import evaluate_contract


DEFAULT_RESULTS = [
    "review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json",
    "review-stage/claim-search-safety-gpt55-r10-c10-v7/gates/known_negative_results.json",
]
DEFAULT_DATA_ROOTS = [
    "data/prepared_data/evidence_partitions/benchmark_ready/cohorts",
    "data/prepared_data/evidence_partitions/cohorts",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(text)
        temp = Path(handle.name)
    os.replace(temp, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    payload = "".join(
        canonical_json(row.model_dump(mode="json") if hasattr(row, "model_dump") else row) + "\n"
        for row in rows
    )
    _atomic_text(path, payload)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if value.__class__.__module__.startswith("numpy") and hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _split_for_item(item: BenchmarkItem) -> str:
    return {
        "scientific": "scientific",
        "external_transfer": "external_transfer",
        "synthetic_stress": "synthetic_safety",
    }[item.benchmark_track]


def _reference_basis(item: BenchmarkItem, profile: TriageReferenceProfile) -> str:
    return profile.reference_basis


def build_simplified_claims(package_dir: Path, profile_path: Path) -> tuple[list[SimplifiedBenchmarkClaim], dict[str, Any]]:
    items = [BenchmarkItem.model_validate(row) for row in _read_jsonl(package_dir / "benchmark_items.jsonl")]
    tasks = [EvaluationTask.model_validate(row) for row in _read_jsonl(package_dir / "evaluation_tasks.jsonl")]
    profiles = {
        row.benchmark_item_id: row
        for row in (
            TriageReferenceProfile.model_validate(payload)
            for payload in _read_jsonl(profile_path)
        )
    }
    task_by_id = {task.task_id: task for task in tasks}
    missing_profiles = sorted(item.benchmark_item_id for item in items if item.benchmark_item_id not in profiles)
    if missing_profiles:
        raise ValueError(f"Missing reference profiles for {len(missing_profiles)} items")

    grouped: dict[str, list[tuple[BenchmarkItem, EvaluationTask, TriageReferenceProfile]]] = defaultdict(list)
    excluded_from_executable_benchmark: list[str] = []
    exclusion_status_counts: Counter[str] = Counter()
    for item in items:
        if item.contract is None or item.migration_status != "ready":
            excluded_from_executable_benchmark.append(item.benchmark_item_id)
            exclusion_status_counts[item.migration_status] += 1
            continue
        if not item.evaluation_task_ids:
            raise ValueError(f"Executable item has no evaluation task: {item.benchmark_item_id}")
        for task_id in item.evaluation_task_ids:
            task = task_by_id.get(task_id)
            if task is None:
                raise ValueError(f"Missing evaluation task {task_id} for {item.benchmark_item_id}")
            identity = sha256_payload(
                {
                    "contract": exact_contract_hash(task.contract),
                    "dataset_id": task.dataset_id,
                    "evidence_role": task.evidence_role,
                    "partition_hashes": task.partition_hashes,
                    "scientific_core_sha256": item.scientific_core_sha256,
                    "generator_spec": task.generator_spec,
                    "generator_artifact_hashes": task.generator_artifact_hashes,
                }
            )
            grouped[identity].append((item, task, profiles[item.benchmark_item_id]))

    claims: list[SimplifiedBenchmarkClaim] = []
    for identity, rows in sorted(grouped.items()):
        reference_labels = {
            {
                "confirm": "confirm",
                "abstain": "abstain",
                "request_evidence": "unresolved",
            }[profile.triage_disposition]
            for _, _, profile in rows
        }
        strengths = {profile.reference_strength for _, _, profile in rows}
        if len(reference_labels) > 1:
            reference_label = "unresolved"
            reference_strength = "evidence_gap"
            reference_basis = "literature"
        else:
            reference_label = next(iter(reference_labels))
            reference_strength = (
                next(iter(strengths))
                if len(strengths) == 1
                else ("strict" if "strict" in strengths else "provisional")
            )
            bases = {_reference_basis(item, profile) for item, _, profile in rows}
            if len(bases) != 1:
                raise ValueError(f"Execution identity mixes literature and constructed references: {identity}")
            reference_basis = next(iter(bases))
        representative, task, _ = rows[0]
        semantic_cluster_id = representative.semantic_cluster_id
        if reference_basis == "constructed_control":
            base_contract = (
                task.generator_spec.get("parent_external_claim_id")
                if task.generator_spec
                else semantic_contract_hash(task.contract)
            ) or semantic_contract_hash(task.contract)
            semantic_cluster_id = "ncb-constructed-" + sha256_payload(
                {
                    "base_contract": base_contract,
                    "dataset_id": task.dataset_id,
                }
            )[:16]
        task_ids = sorted({row_task.task_id for _, row_task, _ in rows})
        item_ids = sorted({item.benchmark_item_id for item, _, _ in rows})
        claims.append(
            SimplifiedBenchmarkClaim(
                benchmark_claim_id=f"ncb-case-{identity[:16]}",
                benchmark_item_ids=item_ids,
                benchmark_split=_split_for_item(representative),
                target_family=representative.target_family,
                modality=representative.modality,
                question=representative.question,
                scientific_question_sha256=representative.scientific_question_sha256,
                semantic_cluster_id=semantic_cluster_id,
                contract=task.contract,
                contract_sha256=exact_contract_hash(task.contract),
                execution_identity_sha256=identity,
                reference_label=reference_label,
                score_eligible=reference_label != "unresolved",
                reference_strength=reference_strength,
                reference_basis=reference_basis,
                evaluation_task_ids=task_ids,
                dataset_ids=sorted({row_task.dataset_id for _, row_task, _ in rows}),
                aliases=sorted(
                    {
                        alias
                        for item, _, _ in rows
                        for alias in (
                            list(item.aliases)
                            + [reference.source_id for reference in item.source_references]
                        )
                    }
                ),
            )
        )

    task_ids = [task_id for claim in claims for task_id in claim.evaluation_task_ids]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("An evaluation task maps to more than one simplified benchmark claim")
    manifest = {
        "benchmark_name": "NeuroClaimBench",
        "version": "2.1-paper",
        "benchmark_claim_count": len(claims),
        "score_eligible_count": sum(claim.score_eligible for claim in claims),
        "unresolved_count": sum(not claim.score_eligible for claim in claims),
        "reference_label_counts": dict(Counter(claim.reference_label for claim in claims)),
        "split_counts": dict(Counter(claim.benchmark_split for claim in claims)),
        "excluded_from_executable_benchmark_count": len(
            excluded_from_executable_benchmark
        ),
        "excluded_from_executable_benchmark_item_ids": (
            excluded_from_executable_benchmark
        ),
        "exclusion_status_counts": dict(exclusion_status_counts),
        "source_item_count": len(items),
        "source_task_count": len(tasks),
        "identity_rule": (
            "contract + scientific core + dataset + evidence role + partition hashes "
            "+ generator specification and artifacts"
        ),
    }
    return claims, manifest


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("claims", "results"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    return []


def _existing_outcomes(
    claims: list[SimplifiedBenchmarkClaim],
    items: list[BenchmarkItem],
    tasks: list[EvaluationTask],
    result_paths: list[Path],
) -> dict[str, BenchmarkTaskOutcome]:
    by_source: dict[str, tuple[str, dict[str, Any], str]] = {}
    for path in result_paths:
        for row in _read_result_rows(path):
            source_id = str(row.get("claim_id") or "")
            raw_label = str(row.get("final_label") or row.get("gate_verdict_label") or "")
            if not source_id or not raw_label:
                continue
            previous = by_source.get(source_id)
            if previous is not None and previous[0] != raw_label:
                raise ValueError(f"Conflicting existing outcomes for {source_id}")
            by_source[source_id] = (raw_label, row, str(path))

    claim_by_task = {
        task_id: claim
        for claim in claims
        for task_id in claim.evaluation_task_ids
    }
    task_by_id = {task.task_id: task for task in tasks}
    outcomes: dict[str, BenchmarkTaskOutcome] = {}
    for item in items:
        matches = [
            by_source[reference.source_id]
            for reference in item.source_references
            if reference.source_id in by_source
        ]
        if len({label for label, _, _ in matches}) > 1:
            raise ValueError(f"Conflicting alias outcomes for {item.benchmark_item_id}")
        if not matches:
            continue
        raw_label, row, source = matches[0]
        for task_id in item.evaluation_task_ids:
            claim = claim_by_task.get(task_id)
            task = task_by_id.get(task_id)
            if claim is None or task is None:
                continue
            outcome = BenchmarkTaskOutcome(
                task_id=task_id,
                benchmark_claim_id=claim.benchmark_claim_id,
                benchmark_item_id=task.benchmark_item_id,
                status="completed",
                confirm_outcome="confirmed" if raw_label == "confirmed" else "abstained",
                raw_final_label=raw_label,
                result_source=source,
                gate_verdict=_json_safe(row.get("gate_verdict") or {}),
                gate_results=_json_safe(row.get("gate_results") or {}),
                cohort_paths=[str(value) for value in row.get("cohort_paths") or []],
            )
            previous = outcomes.get(task_id)
            if previous is not None and previous.raw_final_label != raw_label:
                raise ValueError(f"Conflicting task outcomes for {task_id}")
            outcomes[task_id] = outcome
    return outcomes


def _execution_root(task: EvaluationTask, roots: list[Path]) -> Path:
    cohorts = [task.discovery_cohort, *task.replication_cohorts]
    for root in roots:
        if all((root / f"{cohort}.parquet").exists() for cohort in cohorts):
            return root
    raise FileNotFoundError(f"No data root contains all cohorts for {task.task_id}: {cohorts}")


def _task_fingerprint(task: EvaluationTask) -> str:
    return sha256_payload(
        {
            "task": task.model_dump(mode="json"),
            "execution_policy": "neuroclaimbench-v2.1-full-rerun",
        }
    )


def _verify_source_partitions(task: EvaluationTask, source_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for cohort in [task.discovery_cohort, *task.replication_cohorts]:
        path = source_root / f"{cohort}.parquet"
        digest = _file_sha256(path)
        expected = task.partition_hashes.get(cohort)
        if not expected and task.schema_version.startswith("2.1"):
            raise ValueError(f"Task {task.task_id} has no frozen partition hash for {cohort}")
        if expected and digest != expected:
            raise ValueError(
                f"Task {task.task_id} partition hash mismatch for {cohort}: {digest} != {expected}"
            )
        observed[cohort] = digest
    return observed


def _randomized_root(
    task: EvaluationTask,
    source_root: Path,
    work_root: Path,
) -> tuple[Path, dict[str, str]]:
    spec = task.generator_spec or {}
    assignment_path = Path(str(spec.get("assignment_path") or ""))
    if not assignment_path.exists():
        raise FileNotFoundError(f"Missing random-control assignments: {assignment_path}")
    assignment_file_sha256 = _file_sha256(assignment_path)
    expected_file_sha256 = task.generator_artifact_hashes.get("assignment_file_sha256")
    if task.schema_version.startswith("2.1") and (
        not expected_file_sha256 or assignment_file_sha256 != expected_file_sha256
    ):
        raise ValueError(f"Random-control assignment file hash mismatch for {task.task_id}")
    assignments = pd.read_parquet(assignment_path)
    assignments = assignments[assignments["control_id"].astype(str) == task.benchmark_item_id].copy()
    if assignments.empty:
        raise ValueError(f"No random assignments for {task.benchmark_item_id}")
    selected = assignments.sort_values(list(assignments.columns)).reset_index(drop=True)
    selected_sha256 = sha256_payload(
        selected.astype(object).where(pd.notna(selected), None).to_dict(orient="records")
    )
    expected_selected_sha256 = task.generator_artifact_hashes.get(
        "selected_assignment_rows_sha256"
    )
    if task.schema_version.startswith("2.1") and (
        not expected_selected_sha256 or selected_sha256 != expected_selected_sha256
    ):
        raise ValueError(f"Random-control selected-row hash mismatch for {task.task_id}")

    task_root = work_root / task.task_id
    if task_root.exists():
        shutil.rmtree(task_root)
    task_root.mkdir(parents=True)
    generated_hashes: dict[str, str] = {}
    for cohort in [task.discovery_cohort, *task.replication_cohorts]:
        frame = pd.read_parquet(source_root / f"{cohort}.parquet")
        cohort_assignments = assignments[assignments["cohort"].astype(str) == cohort][
            ["subject_id", str(spec.get("group_column") or "ncb_random_group")]
        ].copy()
        cohort_assignments["subject_id"] = cohort_assignments["subject_id"].astype(str)
        frame["subject_id"] = frame["subject_id"].astype(str)
        merged = frame.merge(
            cohort_assignments,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )
        group_column = str(spec.get("group_column") or "ncb_random_group")
        if merged[group_column].isna().any():
            raise ValueError(f"Incomplete random assignments for {task.benchmark_item_id}:{cohort}")
        generated_path = task_root / f"{cohort}.parquet"
        merged.to_parquet(generated_path, index=False)
        generated_hashes[cohort] = _file_sha256(generated_path)
    return task_root, generated_hashes


def _evaluate_task(payload: dict[str, Any], roots: list[str], work_root: str, claim_id: str) -> dict[str, Any]:
    task = EvaluationTask.model_validate(payload)
    fingerprint = _task_fingerprint(task)
    try:
        root = _execution_root(task, [Path(value) for value in roots])
        cohort_hashes = _verify_source_partitions(task, root)
        generated_hashes: dict[str, str] = {}
        if task.generator_spec:
            execution_root, generated_hashes = _randomized_root(
                task, root, Path(work_root)
            )
        else:
            execution_root = root
        verdict, results, cohort_paths = evaluate_contract(
            task.contract,
            execution_root,
            ref_effect=task.contract.gates.power.ref_effect,
        )
        outcome = BenchmarkTaskOutcome(
            task_id=task.task_id,
            benchmark_claim_id=claim_id,
            benchmark_item_id=task.benchmark_item_id,
            status="completed",
            confirm_outcome="confirmed" if verdict.label == "confirmed" else "abstained",
            raw_final_label=verdict.label,
            result_source="neuroclaimbench_v2.1_task_evaluation",
            task_fingerprint=fingerprint,
            gate_verdict=_json_safe(verdict.to_dict()),
            gate_results=_json_safe(results),
            cohort_paths=[str(path) for path in cohort_paths],
            cohort_content_hashes=cohort_hashes,
            generated_cohort_hashes=generated_hashes,
        )
    except Exception as exc:
        outcome = BenchmarkTaskOutcome(
            task_id=task.task_id,
            benchmark_claim_id=claim_id,
            benchmark_item_id=task.benchmark_item_id,
            status="error",
            confirm_outcome="error",
            result_source="neuroclaimbench_v2.1_task_evaluation",
            task_fingerprint=fingerprint,
            error=str(exc),
        )
    return outcome.model_dump(mode="json")


def evaluate_missing_tasks(
    claims: list[SimplifiedBenchmarkClaim],
    package_dir: Path,
    out_dir: Path,
    result_paths: list[Path],
    data_roots: list[Path],
    *,
    max_workers: int,
    backend: str,
    progress: bool,
    reuse_existing_results: bool = False,
) -> list[BenchmarkTaskOutcome]:
    items = [BenchmarkItem.model_validate(row) for row in _read_jsonl(package_dir / "benchmark_items.jsonl")]
    tasks = [EvaluationTask.model_validate(row) for row in _read_jsonl(package_dir / "evaluation_tasks.jsonl")]
    task_by_id = {task.task_id: task for task in tasks}
    claim_by_task = {
        task_id: claim
        for claim in claims
        for task_id in claim.evaluation_task_ids
    }
    completed = (
        _existing_outcomes(claims, items, tasks, result_paths)
        if reuse_existing_results
        else {}
    )
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(checkpoint_dir.glob("*.json")):
        outcome = BenchmarkTaskOutcome.model_validate_json(path.read_text(encoding="utf-8"))
        task = task_by_id.get(outcome.task_id)
        if task is None:
            continue
        expected_fingerprint = _task_fingerprint(task)
        if outcome.task_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Stale task checkpoint fingerprint for {outcome.task_id}; "
                "use a new output directory or remove the stale checkpoint."
            )
        if outcome.status == "completed":
            completed[outcome.task_id] = outcome

    pending = [
        task
        for task in tasks
        if task.task_id in claim_by_task and task.task_id not in completed
    ]
    work_root = out_dir / ".work"
    work_root.mkdir(parents=True, exist_ok=True)

    def record(payload: dict[str, Any]) -> None:
        outcome = BenchmarkTaskOutcome.model_validate(payload)
        _atomic_text(
            checkpoint_dir / f"{outcome.task_id}.json",
            outcome.model_dump_json(indent=2) + "\n",
        )
        completed[outcome.task_id] = outcome

    root_values = [str(path) for path in data_roots]
    if max_workers <= 1:
        for task in iter_progress(
            pending,
            total=len(pending),
            desc="NeuroClaimBench evaluation",
            enabled=progress,
            unit="task",
        ):
            record(
                _evaluate_task(
                    task.model_dump(mode="json"),
                    root_values,
                    str(work_root),
                    claim_by_task[task.task_id].benchmark_claim_id,
                )
            )
    else:
        executor_class = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        with executor_class(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_task,
                    task.model_dump(mode="json"),
                    root_values,
                    str(work_root),
                    claim_by_task[task.task_id].benchmark_claim_id,
                ): task.task_id
                for task in pending
            }
            for future in iter_progress(
                as_completed(futures),
                total=len(futures),
                desc="NeuroClaimBench evaluation",
                enabled=progress,
                unit="task",
            ):
                record(future.result())

    expected = {task_id for claim in claims for task_id in claim.evaluation_task_ids}
    missing = expected - set(completed)
    if missing:
        raise ValueError(f"Missing outcomes for {len(missing)} evaluation tasks")
    ordered = [completed[task_id] for task_id in sorted(expected)]
    _write_jsonl(out_dir / "task_outcomes.jsonl", ordered)
    shutil.rmtree(work_root, ignore_errors=True)
    return ordered


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _wilson_interval(count: int, denominator: int, z: float = 1.959963984540054) -> list[float] | None:
    if denominator == 0:
        return None
    proportion = count / denominator
    scale = 1.0 + z * z / denominator
    center = (proportion + z * z / (2.0 * denominator)) / scale
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / denominator
            + z * z / (4.0 * denominator * denominator)
        )
        / scale
    )
    lower = 0.0 if count == 0 else max(0.0, center - radius)
    upper = 1.0 if count == denominator else min(1.0, center + radius)
    return [lower, upper]


def _cluster_bootstrap_interval(
    claims: list[SimplifiedBenchmarkClaim],
    outcomes_by_claim: dict[str, str],
    *,
    reference_label: str,
    resamples: int = 2000,
    seed: int = 20260723,
) -> list[float] | None:
    eligible = [
        claim
        for claim in claims
        if claim.reference_label == reference_label
        and outcomes_by_claim.get(claim.benchmark_claim_id) in {"confirmed", "abstained"}
    ]
    clusters: dict[str, list[SimplifiedBenchmarkClaim]] = defaultdict(list)
    for claim in eligible:
        clusters[claim.semantic_cluster_id or claim.benchmark_claim_id].append(claim)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return None
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        denominator = sum(len(clusters[cluster_id]) for cluster_id in sampled)
        numerator = sum(
            outcomes_by_claim[claim.benchmark_claim_id] == "confirmed"
            for cluster_id in sampled
            for claim in clusters[cluster_id]
        )
        if denominator:
            rates.append(numerator / denominator)
    if not rates:
        return None
    rates.sort()
    lower = rates[int(0.025 * (len(rates) - 1))]
    upper = rates[int(0.975 * (len(rates) - 1))]
    return [lower, upper]


def _decision_metrics(
    claims: list[SimplifiedBenchmarkClaim],
    outcomes_by_claim: dict[str, str],
) -> dict[str, Any]:
    scored = [claim for claim in claims if claim.score_eligible]
    evaluated = [
        claim
        for claim in scored
        if outcomes_by_claim.get(claim.benchmark_claim_id) in {"confirmed", "abstained"}
    ]
    confirmable = [claim for claim in evaluated if claim.reference_label == "confirm"]
    abstain = [claim for claim in evaluated if claim.reference_label == "abstain"]
    recall = sum(outcomes_by_claim[claim.benchmark_claim_id] == "confirmed" for claim in confirmable)
    unsafe = sum(outcomes_by_claim[claim.benchmark_claim_id] == "confirmed" for claim in abstain)
    return {
        "n_claims": len(claims),
        "n_score_eligible": len(scored),
        "n_evaluated": len(evaluated),
        "n_errors": sum(outcomes_by_claim.get(claim.benchmark_claim_id) == "error" for claim in scored),
        "n_not_evaluated": sum(
            outcomes_by_claim.get(claim.benchmark_claim_id, "not_evaluated") == "not_evaluated"
            for claim in scored
        ),
        "confirmable_claim_recall_count": recall,
        "confirmable_claim_recall_denominator": len(confirmable),
        "confirmable_claim_recall": _rate(recall, len(confirmable)),
        "confirmable_claim_recall_wilson_95": _wilson_interval(recall, len(confirmable)),
        "confirmable_claim_recall_cluster_bootstrap_95": _cluster_bootstrap_interval(
            claims,
            outcomes_by_claim,
            reference_label="confirm",
        ),
        "unsafe_confirmation_count": unsafe,
        "unsafe_confirmation_denominator": len(abstain),
        "unsafe_confirmation_rate": _rate(unsafe, len(abstain)),
        "unsafe_confirmation_wilson_95": _wilson_interval(unsafe, len(abstain)),
        "unsafe_confirmation_cluster_bootstrap_95": _cluster_bootstrap_interval(
            claims,
            outcomes_by_claim,
            reference_label="abstain",
        ),
        "cluster_bootstrap_resamples": 2000,
        "cluster_bootstrap_seed": 20260723,
    }


def summarize(
    claims: list[SimplifiedBenchmarkClaim],
    outcomes: list[BenchmarkTaskOutcome],
    out_dir: Path,
) -> dict[str, Any]:
    task_outcomes: dict[str, list[BenchmarkTaskOutcome]] = defaultdict(list)
    for outcome in outcomes:
        task_outcomes[outcome.benchmark_claim_id].append(outcome)

    claim_outcomes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for claim in claims:
        task_rows = task_outcomes.get(claim.benchmark_claim_id, [])
        completed = {row.confirm_outcome for row in task_rows if row.status == "completed"}
        if not task_rows:
            confirm_outcome = "not_evaluated"
        elif not completed:
            confirm_outcome = "error"
        elif completed == {"confirmed"}:
            confirm_outcome = "confirmed"
        elif "confirmed" in completed:
            confirm_outcome = "confirmed"
        else:
            confirm_outcome = "abstained"
        claim_outcomes[claim.benchmark_claim_id] = confirm_outcome
        rows.append(
            {
                "benchmark_claim_id": claim.benchmark_claim_id,
                "benchmark_split": claim.benchmark_split,
                "target_family": claim.target_family,
                "modality": claim.modality,
                "reference_label": claim.reference_label,
                "score_eligible": claim.score_eligible,
                "confirm_outcome": confirm_outcome,
                "reference_strength": claim.reference_strength,
                "reference_basis": claim.reference_basis,
                "scientific_question_sha256": claim.scientific_question_sha256,
                "semantic_cluster_id": claim.semantic_cluster_id,
                "dataset_ids": ",".join(claim.dataset_ids),
                "contract_sha256": claim.contract_sha256,
                "evaluation_task_count": len(claim.evaluation_task_ids),
                "task_error_count": sum(row.status == "error" for row in task_rows),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scientific_literature = [
        claim
        for claim in claims
        if claim.benchmark_split == "scientific"
        and claim.reference_basis == "literature"
    ]
    scientific_primary = [
        claim
        for claim in scientific_literature
        if claim.reference_strength in {"strict", "provisional"}
    ]
    scientific_by_strength = {
        strength: _decision_metrics(
            [
                claim
                for claim in scientific_literature
                if claim.reference_strength == strength
            ],
            claim_outcomes,
        )
        for strength in ("strict", "provisional")
    }
    synthetic_controls = [
        claim
        for claim in claims
        if claim.benchmark_split == "synthetic_safety"
        and claim.reference_basis == "constructed_control"
    ]
    by_split_target_basis = {
        split: {
            target: {
                basis: _decision_metrics(
                    [
                        claim
                        for claim in claims
                        if claim.benchmark_split == split
                        and claim.target_family == target
                        and claim.reference_basis == basis
                    ],
                    claim_outcomes,
                )
                for basis in sorted(
                    {
                        claim.reference_basis
                        for claim in claims
                        if claim.benchmark_split == split
                        and claim.target_family == target
                    }
                )
            }
            for target in sorted(
                {
                    claim.target_family
                    for claim in claims
                    if claim.benchmark_split == split
                }
            )
        }
        for split in ("scientific", "external_transfer", "synthetic_safety")
    }
    by_split_target_basis_strength = {
        split: {
            target: {
                basis: {
                    strength: _decision_metrics(
                        [
                            claim
                            for claim in claims
                            if claim.benchmark_split == split
                            and claim.target_family == target
                            and claim.reference_basis == basis
                            and claim.reference_strength == strength
                        ],
                        claim_outcomes,
                    )
                    for strength in sorted(
                        {
                            claim.reference_strength
                            for claim in claims
                            if claim.benchmark_split == split
                            and claim.target_family == target
                            and claim.reference_basis == basis
                        }
                    )
                }
                for basis in sorted(
                    {
                        claim.reference_basis
                        for claim in claims
                        if claim.benchmark_split == split
                        and claim.target_family == target
                    }
                )
            }
            for target in sorted(
                {
                    claim.target_family
                    for claim in claims
                    if claim.benchmark_split == split
                }
            )
        }
        for split in ("scientific", "external_transfer", "synthetic_safety")
    }
    external_literature_by_dataset = {
        dataset: {
            "combined_primary": _decision_metrics(
                [
                    claim
                    for claim in claims
                    if claim.benchmark_split == "external_transfer"
                    and claim.reference_basis == "literature"
                    and claim.reference_strength in {"strict", "provisional"}
                    and dataset in claim.dataset_ids
                ],
                claim_outcomes,
            ),
            "strict": _decision_metrics(
                [
                    claim
                    for claim in claims
                    if claim.benchmark_split == "external_transfer"
                    and claim.reference_basis == "literature"
                    and claim.reference_strength == "strict"
                    and dataset in claim.dataset_ids
                ],
                claim_outcomes,
            ),
            "provisional": _decision_metrics(
                [
                    claim
                    for claim in claims
                    if claim.benchmark_split == "external_transfer"
                    and claim.reference_basis == "literature"
                    and claim.reference_strength == "provisional"
                    and dataset in claim.dataset_ids
                ],
                claim_outcomes,
            ),
        }
        for dataset in sorted(
            {
                dataset
                for claim in claims
                if claim.benchmark_split == "external_transfer"
                and claim.reference_basis == "literature"
                for dataset in claim.dataset_ids
            }
        )
    }
    external_controls_by_dataset = {
        dataset: _decision_metrics(
            [
                claim
                for claim in claims
                if claim.benchmark_split == "external_transfer"
                and claim.reference_basis == "constructed_control"
                and dataset in claim.dataset_ids
            ],
            claim_outcomes,
        )
        for dataset in sorted(
            {
                dataset
                for claim in claims
                if claim.benchmark_split == "external_transfer"
                and claim.reference_basis == "constructed_control"
                for dataset in claim.dataset_ids
            }
        )
    }
    literature_by_split_and_strength = {
        split: {
            strength: _decision_metrics(
                [
                    claim
                    for claim in claims
                    if claim.benchmark_split == split
                    and claim.reference_basis == "literature"
                    and claim.reference_strength == strength
                ],
                claim_outcomes,
            )
            for strength in ("strict", "provisional")
        }
        for split in ("scientific", "external_transfer")
    }
    unresolved_verdicts = {
        split: dict(
            Counter(
                claim_outcomes[claim.benchmark_claim_id]
                for claim in claims
                if claim.benchmark_split == split and not claim.score_eligible
            )
        )
        for split in ("scientific", "external_transfer", "synthetic_safety")
    }
    summary = {
        "inventory": {
            "benchmark_claim_count": len(claims),
            "score_eligible_count": sum(claim.score_eligible for claim in claims),
            "unresolved_count": sum(not claim.score_eligible for claim in claims),
            "reference_label_counts": dict(Counter(claim.reference_label for claim in claims)),
            "split_counts": dict(Counter(claim.benchmark_split for claim in claims)),
        },
        "outcome_coverage": {
            "completed_claim_count": sum(value in {"confirmed", "abstained"} for value in claim_outcomes.values()),
            "error_claim_count": sum(value == "error" for value in claim_outcomes.values()),
            "not_evaluated_claim_count": sum(value == "not_evaluated" for value in claim_outcomes.values()),
            "confirm_outcome_counts": dict(Counter(claim_outcomes.values())),
        },
        "primary_scientific_literature": _decision_metrics(
            scientific_primary,
            claim_outcomes,
        ),
        "scientific_literature_sensitivity": scientific_by_strength,
        "external_literature_by_dataset": external_literature_by_dataset,
        "external_constructed_controls_by_dataset": external_controls_by_dataset,
        "synthetic_constructed_controls": _decision_metrics(
            synthetic_controls,
            claim_outcomes,
        ),
        "unresolved_verdict_distribution": unresolved_verdicts,
        "metrics_by_split_target_and_reference_basis": by_split_target_basis,
        "metrics_by_split_target_reference_basis_and_strength": (
            by_split_target_basis_strength
        ),
        "literature_sensitivity_by_split_and_reference_strength": literature_by_split_and_strength,
        "provenance": {
            "benchmark_claims_payload_sha256": sha256_payload(
                [claim.model_dump(mode="json") for claim in claims]
            ),
            "task_outcomes_payload_sha256": sha256_payload(
                [outcome.model_dump(mode="json") for outcome in outcomes]
            ),
        },
        "interpretation_restrictions": [
            "scientific, external-transfer, and synthetic-safety metrics use separate denominators",
            "external literature references and random controls are never pooled",
            "unresolved references are retained but excluded from scored performance",
            "strict plus provisional literature references are the retrospective primary operational result",
            "constructed-control cluster intervals are conditional stress-test uncertainty, not population FCR",
            "multi-model literature references are not human expert ground truth",
            "v2.1 policy was frozen before the v2.1 rerun but after v2 outcomes were visible",
        ],
    }
    sensitivity_rows: list[dict[str, Any]] = []

    def add_sensitivity_row(stratum: str, metrics: dict[str, Any]) -> None:
        for endpoint in ("confirmable_claim_recall", "unsafe_confirmation"):
            sensitivity_rows.append(
                {
                    "stratum": stratum,
                    "endpoint": endpoint,
                    "count": metrics.get(f"{endpoint}_count"),
                    "denominator": metrics.get(f"{endpoint}_denominator"),
                    "rate": metrics.get(endpoint),
                    "wilson_95": json.dumps(metrics.get(f"{endpoint}_wilson_95")),
                    "semantic_cluster_bootstrap_95": json.dumps(
                        metrics.get(f"{endpoint}_cluster_bootstrap_95")
                    ),
                    "bootstrap_resamples": 2000,
                    "bootstrap_seed": 20260723,
                }
            )

    add_sensitivity_row("scientific_literature_combined", summary["primary_scientific_literature"])
    for strength, metrics in scientific_by_strength.items():
        add_sensitivity_row(f"scientific_literature_{strength}", metrics)
    for dataset, strata in external_literature_by_dataset.items():
        for strength, metrics in strata.items():
            add_sensitivity_row(f"external_literature_{dataset}_{strength}", metrics)
    for dataset, metrics in external_controls_by_dataset.items():
        add_sensitivity_row(f"external_constructed_{dataset}", metrics)
    add_sensitivity_row("synthetic_constructed", summary["synthetic_constructed_controls"])
    with (out_dir / "cluster_bootstrap_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    _atomic_text(
        out_dir / "benchmark_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = Path(args.package_dir)
    out_dir = Path(args.out_dir)
    claims_path = package_dir / "benchmark_claims.jsonl"
    profile_path = Path(args.reference_profiles)

    if args.phase in {"build", "all"}:
        claims, manifest = build_simplified_claims(package_dir, profile_path)
        _write_jsonl(claims_path, claims)
        crosswalk_rows = [
            {
                "benchmark_item_id": item_id,
                "benchmark_claim_id": claim.benchmark_claim_id,
                "benchmark_split": claim.benchmark_split,
                "reference_label": claim.reference_label,
                "score_eligible": str(claim.score_eligible).lower(),
            }
            for claim in claims
            for item_id in claim.benchmark_item_ids
        ]
        crosswalk_path = package_dir / "paper_benchmark_crosswalk.csv"
        with crosswalk_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(crosswalk_rows[0]))
            writer.writeheader()
            writer.writerows(crosswalk_rows)
        manifest["input_sha256"] = {
            "benchmark_items.jsonl": _file_sha256(package_dir / "benchmark_items.jsonl"),
            "evaluation_tasks.jsonl": _file_sha256(package_dir / "evaluation_tasks.jsonl"),
            "reference_profiles.jsonl": _file_sha256(profile_path),
        }
        manifest["output_sha256"] = {
            "benchmark_claims.jsonl": _file_sha256(claims_path),
            "paper_benchmark_crosswalk.csv": _file_sha256(crosswalk_path),
        }
        _atomic_text(
            package_dir / "paper_benchmark_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    else:
        claims = [
            SimplifiedBenchmarkClaim.model_validate(row)
            for row in _read_jsonl(claims_path)
        ]

    if args.phase in {"evaluate", "all"}:
        outcomes = evaluate_missing_tasks(
            claims,
            package_dir,
            out_dir,
            [Path(value) for value in args.results],
            [Path(value) for value in args.data_root],
            max_workers=args.max_workers,
            backend=args.parallel_backend,
            progress=not args.no_progress,
            reuse_existing_results=args.reuse_existing_results,
        )
    else:
        outcomes = [
            BenchmarkTaskOutcome.model_validate(row)
            for row in _read_jsonl(out_dir / "task_outcomes.jsonl")
        ]

    if args.phase in {"summarize", "all"}:
        return summarize(claims, outcomes, out_dir)
    return {
        "benchmark_claim_count": len(claims),
        "task_outcome_count": len(outcomes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["build", "evaluate", "summarize", "all"], default="all")
    parser.add_argument("--package-dir", default="data/neuroclaimbench/v2.1")
    parser.add_argument(
        "--reference-profiles",
        default="review-stage/neuroclaimbench-v2.1/reference/triage_reference_profiles.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/results",
    )
    parser.add_argument("--results", action="append", default=None)
    parser.add_argument("--data-root", action="append", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--reuse-existing-results", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.results is None:
        args.results = list(DEFAULT_RESULTS)
    if args.data_root is None:
        args.data_root = list(DEFAULT_DATA_ROOTS)
    result = run(args)
    print(json.dumps({"status": "completed", "phase": args.phase, "summary": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
