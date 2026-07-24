"""Build NeuroClaimBench v2.1 from its frozen source snapshot and alignment audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bench.neuroclaimbench_v21_compat import (
    AdjudicationRecord,
    BenchmarkItem,
    EvaluationTask,
    EvidenceRecord,
    LabelVote,
    QuestionContractAlignment,
    exact_contract_hash,
    scientific_core_hash,
    scientific_question_hash,
    semantic_contract_hash,
    sha256_payload,
)
from confirm.candidate_preflight import CandidatePreflightContext
from confirm.provenance import git_sha

SCHEMA_VERSION = "2.1.0"
DEFAULT_DATA_ROOTS = [
    "data/prepared_data/evidence_partitions/benchmark_ready/cohorts",
    "data/prepared_data/evidence_partitions/cohorts",
    "review-stage/claim-search-safety-gpt55-r10-c10-v7/data/cohorts",
]
GATE_POLICY_FILES = [
    "src/confirm/analysis.py",
    "src/confirm/agent.py",
    "src/confirm/brainwide.py",
    "src/confirm/confounds.py",
    "src/confirm/multiplicity.py",
    "src/confirm/multiverse.py",
    "src/confirm/power.py",
    "src/confirm/replication.py",
    "src/confirm/verdict.py",
]


def _stable_benchmark_item_id(item_id: str) -> str:
    """Replace the legacy status-bearing ID while preserving its hash suffix."""
    if item_id.startswith("ncb-pending-"):
        return f"ncb-source-{item_id.removeprefix('ncb-pending-')}"
    return item_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp = Path(handle.name)
    os.replace(temp, path)


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(
                row.model_dump(mode="json") if hasattr(row, "model_dump") else row,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), Path.cwd().resolve())


def _gate_policy_version() -> str:
    payload = {
        path: _sha256_file(Path(path))
        for path in GATE_POLICY_FILES
        if Path(path).exists()
    }
    return sha256_payload(payload)


def _generator_hashes(
    task: EvaluationTask,
    assignment_path: Path | None,
) -> dict[str, str]:
    if not task.generator_spec:
        return {}
    spec = dict(task.generator_spec)
    hashes = {
        "generator_parameters_sha256": sha256_payload(
            {
                key: value
                for key, value in spec.items()
                if key not in {"assignment_path", "assignment_sha256", "assignment_count"}
            }
        )
    }
    if assignment_path is None or not assignment_path.exists():
        raise FileNotFoundError(f"Missing assignment artifact for generated control {task.task_id}")
    hashes["assignment_file_sha256"] = _sha256_file(assignment_path)
    frame = pd.read_parquet(assignment_path)
    selected = frame[frame["control_id"].astype(str) == task.benchmark_item_id].copy()
    if selected.empty:
        raise ValueError(f"No selected assignment rows for {task.benchmark_item_id}")
    selected = selected.sort_values(list(selected.columns)).reset_index(drop=True)
    hashes["selected_assignment_rows_sha256"] = sha256_payload(
        selected.astype(object).where(pd.notna(selected), None).to_dict(orient="records")
    )
    return hashes


def _task_for_item(
    item: BenchmarkItem,
    old_task: EvaluationTask,
    *,
    context: CandidatePreflightContext,
    assignment_path: Path | None,
    code_sha: str,
    gate_policy_version: str,
) -> EvaluationTask:
    assert item.contract is not None
    cohorts = [item.contract.discovery_cohort, *item.contract.replication_cohorts]
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for cohort in cohorts:
        info = context.resolve(cohort)
        if info is None:
            raise FileNotFoundError(f"Cannot resolve v2.1 evidence partition {cohort!r}")
        path = Path(info.path)
        paths[cohort] = _relative(path)
        hashes[cohort] = _sha256_file(path)
    generator_hashes = _generator_hashes(old_task, assignment_path)
    generator_spec = dict(old_task.generator_spec or {}) or None
    if generator_spec is not None and assignment_path is not None:
        generator_spec["assignment_path"] = _relative(assignment_path)
        generator_spec["assignment_file_sha256"] = generator_hashes["assignment_file_sha256"]
        generator_spec["selected_assignment_rows_sha256"] = generator_hashes[
            "selected_assignment_rows_sha256"
        ]
    identity = {
        "benchmark_item_id": item.benchmark_item_id,
        "contract": exact_contract_hash(item.contract),
        "question": item.scientific_question_sha256,
        "role": old_task.evidence_role,
        "dataset": old_task.dataset_id,
        "partition_hashes": hashes,
        "generator_artifact_hashes": generator_hashes,
        "schema_version": SCHEMA_VERSION,
        "gate_policy_version": gate_policy_version,
    }
    return EvaluationTask(
        task_id=f"ncb21-task-{sha256_payload(identity)[:16]}",
        benchmark_item_id=item.benchmark_item_id,
        contract=item.contract,
        evidence_role=old_task.evidence_role,
        dataset_id=old_task.dataset_id,
        discovery_cohort=item.contract.discovery_cohort,
        replication_cohorts=list(item.contract.replication_cohorts),
        partition_paths=paths,
        partition_hashes=hashes,
        executable_contract_sha256=exact_contract_hash(item.contract),
        scientific_question_sha256=item.scientific_question_sha256,
        scientific_core_sha256=item.scientific_core_sha256,
        evidence_freshness=old_task.evidence_freshness,
        generator_spec=generator_spec,
        generator_artifact_hashes=generator_hashes,
        code_sha=code_sha,
        schema_version=SCHEMA_VERSION,
        gate_policy_version=gate_policy_version,
    )


def _migrated_item(
    item: BenchmarkItem,
    alignment: QuestionContractAlignment,
) -> tuple[BenchmarkItem, bool, bool]:
    old_contract_hash = exact_contract_hash(item.contract) if item.contract is not None else None
    old_question_hash = item.scientific_question_sha256
    resolution = alignment.final_outcome_blind_resolution
    executable = resolution in {
        "aligned",
        "aligned_with_safety_augmentation",
        "repairable_contract",
    }
    contract = alignment.repaired_contract if executable else item.contract
    if contract is not None:
        contract = contract.model_copy(update={"question": alignment.canonical_question})
    contract_changed = (
        contract is not None
        and old_contract_hash is not None
        and exact_contract_hash(contract) != old_contract_hash
    )
    identity_changed = (
        contract_changed
        or alignment.canonical_question_sha256 != old_question_hash
    )
    migration_status = (
        "ready"
        if executable
        else ("non_executable" if resolution == "non_executable" else "ambiguous_unresolved")
    )
    update: dict[str, Any] = {
        "question": alignment.canonical_question,
        "scientific_question_sha256": alignment.canonical_question_sha256,
        "contract": contract,
        "exact_contract_sha256": exact_contract_hash(contract) if contract is not None else None,
        "semantic_claim_sha256": (
            semantic_contract_hash(contract) if contract is not None else item.semantic_claim_sha256
        ),
        "scientific_core_sha256": scientific_core_hash(contract) if contract is not None else "",
        "migration_status": migration_status,
        "alignment_disposition": resolution,
        "alignment_policy_version": alignment.policy_version,
        "pre_v2_contract_sha256": old_contract_hash,
        "evaluation_task_ids": [],
    }
    if (
        (identity_changed or not executable)
        and item.adjudication_status != "construction_derived"
    ):
        update.update(
            {
                "label_class": "candidate_unknown",
                "reference_disposition": "unresolved",
                "adjudication_status": "pending" if executable else "unresolved",
                "score_eligible": False,
            }
        )
    return item.model_copy(update=update), contract_changed, identity_changed


def _portable_alignment_record(
    record: QuestionContractAlignment,
    *,
    assignment_path: Path | None = None,
) -> dict[str, Any]:
    payload = record.model_dump(mode="json")
    preflight = payload.get("deterministic_preflight") or {}
    preflight["resolved_data_paths"] = {
        cohort: _relative(Path(path))
        for cohort, path in (preflight.get("resolved_data_paths") or {}).items()
    }
    generator = (preflight.get("design_diagnostics") or {}).get("generator")
    if generator and assignment_path is not None and generator.get("assignment_path"):
        generator["assignment_path"] = _relative(assignment_path)
    payload["deterministic_preflight"] = preflight
    return payload


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_package)
    out_dir = Path(args.out_dir)
    alignment_path = Path(args.alignment_records)
    items = [BenchmarkItem.model_validate(row) for row in _read_jsonl(source / "benchmark_items.jsonl")]
    old_tasks = [EvaluationTask.model_validate(row) for row in _read_jsonl(source / "evaluation_tasks.jsonl")]
    alignments = {
        row.benchmark_item_id: row
        for row in (
            QuestionContractAlignment.model_validate(payload)
            for payload in _read_jsonl(alignment_path)
        )
    }
    if set(alignments) != {item.benchmark_item_id for item in items}:
        missing = {item.benchmark_item_id for item in items} - set(alignments)
        extra = set(alignments) - {item.benchmark_item_id for item in items}
        raise ValueError(f"Alignment coverage mismatch: missing={len(missing)} extra={len(extra)}")
    missing_advisory = [
        item.benchmark_item_id
        for item in items
        if item.contract is not None
        and item.adjudication_status != "construction_derived"
        and alignments[item.benchmark_item_id].gemini_assessment is None
    ]
    if missing_advisory:
        raise ValueError(
            "v2.1 build requires the frozen Gemini advisory audit for every contracted "
            f"literature-derived item; missing {len(missing_advisory)} (first: {missing_advisory[:5]})."
        )
    pending_source_adjudication = [
        item.benchmark_item_id
        for item in items
        if item.contract is not None
        and item.adjudication_status == "pending"
        and item.benchmark_track in {"scientific", "external_transfer"}
    ]
    if pending_source_adjudication:
        raise ValueError(
            "v2.1 build requires literature adjudication to be finalized in the source "
            f"snapshot; {len(pending_source_adjudication)} contracted items remain pending "
            f"(first: {pending_source_adjudication[:5]})."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    alignment_policy_source = alignment_path.parent / "alignment_policy.json"
    if not alignment_policy_source.exists():
        raise FileNotFoundError(
            "The frozen alignment policy must accompany alignment_records.jsonl"
        )
    shutil.copy2(alignment_policy_source, out_dir / "alignment_policy.json")
    source_assignment = source / "external_random_control_assignments.parquet"
    assignment_path: Path | None = None
    if source_assignment.exists():
        assignment_path = out_dir / source_assignment.name
        shutil.copy2(source_assignment, assignment_path)

    task_by_item = {task.benchmark_item_id: task for task in old_tasks}
    context = CandidatePreflightContext.from_roots(args.data_root)
    code_sha = git_sha() or "unknown"
    gate_policy_version = _gate_policy_version()
    migrated: list[BenchmarkItem] = []
    next_item_by_source: dict[str, BenchmarkItem] = {}
    source_to_migrated_id: dict[str, str] = {}
    changed_ids: set[str] = set()
    contract_repair_ids: set[str] = set()
    for item in items:
        source_item_id = item.benchmark_item_id
        next_item, contract_changed, identity_changed = _migrated_item(
            item,
            alignments[source_item_id],
        )
        stable_item_id = _stable_benchmark_item_id(source_item_id)
        if stable_item_id != source_item_id:
            next_item = next_item.model_copy(
                update={"benchmark_item_id": stable_item_id}
            )
        migrated.append(next_item)
        next_item_by_source[source_item_id] = next_item
        source_to_migrated_id[source_item_id] = stable_item_id
        if contract_changed:
            contract_repair_ids.add(stable_item_id)
        if identity_changed:
            changed_ids.add(stable_item_id)

    canonical_by_key: dict[tuple[str, str, str], BenchmarkItem] = {}
    alias_to_canonical: dict[str, str] = {}
    for item in sorted(migrated, key=lambda row: row.benchmark_item_id):
        if item.contract is None or item.migration_status != "ready":
            canonical_by_key[(item.benchmark_track, item.benchmark_item_id, item.benchmark_item_id)] = item
            alias_to_canonical[item.benchmark_item_id] = item.benchmark_item_id
            continue
        key = (item.benchmark_track, exact_contract_hash(item.contract), item.scientific_core_sha256)
        canonical = canonical_by_key.get(key)
        if canonical is None:
            canonical_by_key[key] = item
            alias_to_canonical[item.benchmark_item_id] = item.benchmark_item_id
            continue
        alias_to_canonical[item.benchmark_item_id] = canonical.benchmark_item_id
        canonical.aliases = sorted(set(canonical.aliases + item.aliases + [item.benchmark_item_id]))
        canonical.question_aliases = sorted(
            set(canonical.question_aliases + item.question_aliases + [item.question])
            - {canonical.question}
        )
        known_sources = {(row.source_collection, row.source_id) for row in canonical.source_references}
        canonical.source_references.extend(
            row
            for row in item.source_references
            if (row.source_collection, row.source_id) not in known_sources
        )
        if item.benchmark_item_id in changed_ids:
            changed_ids.add(canonical.benchmark_item_id)
        if item.benchmark_item_id in contract_repair_ids:
            contract_repair_ids.add(canonical.benchmark_item_id)
    for source_item_id, migrated_item_id in source_to_migrated_id.items():
        alias_to_canonical[source_item_id] = alias_to_canonical.get(
            migrated_item_id,
            migrated_item_id,
        )

    canonical_items = sorted(canonical_by_key.values(), key=lambda row: row.benchmark_item_id)
    new_tasks: list[EvaluationTask] = []
    for item in canonical_items:
        if item.migration_status != "ready" or item.contract is None:
            continue
        source_ids = [item.benchmark_item_id, *item.aliases]
        old_task = next((task_by_item[source_id] for source_id in source_ids if source_id in task_by_item), None)
        if old_task is None:
            raise ValueError(f"Ready v2.1 item has no v2 task: {item.benchmark_item_id}")
        task = _task_for_item(
            item,
            old_task,
            context=context,
            assignment_path=assignment_path,
            code_sha=code_sha,
            gate_policy_version=gate_policy_version,
        )
        item.evaluation_task_ids = [task.task_id]
        new_tasks.append(task)

    item_by_id = {item.benchmark_item_id: item for item in canonical_items}

    def remap(item_id: str) -> str:
        return alias_to_canonical.get(item_id, item_id)

    evidence: list[EvidenceRecord] = []
    for payload in _read_jsonl(source / "evidence_records.jsonl"):
        old = EvidenceRecord.model_validate(payload)
        item_id = remap(old.benchmark_item_id)
        if item_id in changed_ids or item_id not in item_by_id:
            continue
        evidence.append(
            old.model_copy(
                update={
                    "benchmark_item_id": item_id,
                    "scientific_question_sha256": item_by_id[item_id].scientific_question_sha256,
                }
            )
        )
    votes: list[LabelVote] = []
    seen_votes: set[tuple[str, str]] = set()
    for payload in _read_jsonl(source / "label_votes.jsonl"):
        old = LabelVote.model_validate(payload)
        item_id = remap(old.benchmark_item_id)
        key = (item_id, old.model_spec)
        if item_id in changed_ids or item_id not in item_by_id or key in seen_votes:
            continue
        seen_votes.add(key)
        votes.append(
            old.model_copy(
                update={
                    "benchmark_item_id": item_id,
                    "scientific_question_sha256": item_by_id[item_id].scientific_question_sha256,
                }
            )
        )
    adjudications: list[AdjudicationRecord] = []
    seen_adjudications: set[str] = set()
    for payload in _read_jsonl(source / "adjudications.jsonl"):
        old = AdjudicationRecord.model_validate(payload)
        item_id = remap(old.benchmark_item_id)
        item = item_by_id.get(item_id)
        if item is None or item_id in seen_adjudications:
            continue
        if item_id in changed_ids and item.adjudication_status != "construction_derived":
            continue
        seen_adjudications.add(item_id)
        adjudications.append(
            old.model_copy(
                update={
                    "benchmark_item_id": item_id,
                    "scientific_question_sha256": item.scientific_question_sha256,
                }
            )
        )

    _write_jsonl(out_dir / "benchmark_items.jsonl", canonical_items)
    _write_jsonl(out_dir / "evaluation_tasks.jsonl", sorted(new_tasks, key=lambda row: row.task_id))
    _write_jsonl(out_dir / "evidence_records.jsonl", sorted(evidence, key=lambda row: row.evidence_id))
    _write_jsonl(
        out_dir / "label_votes.jsonl",
        sorted(votes, key=lambda row: (row.benchmark_item_id, row.model_spec)),
    )
    _write_jsonl(
        out_dir / "adjudications.jsonl",
        sorted(adjudications, key=lambda row: row.benchmark_item_id),
    )
    _write_jsonl(
        out_dir / "alignment_records.jsonl",
        (
            _portable_alignment_record(
                row.model_copy(
                    update={
                        "benchmark_item_id": source_to_migrated_id[row.benchmark_item_id],
                    }
                ),
                assignment_path=assignment_path,
            )
            for row in sorted(alignments.values(), key=lambda value: value.benchmark_item_id)
        ),
    )

    crosswalk_path = out_dir / "v2_to_v2.1_crosswalk.csv"
    with crosswalk_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "v2_benchmark_item_id",
                "v2.1_benchmark_item_id",
                "pre_v2_contract_sha256",
                "v2.1_contract_sha256",
                "scientific_question_sha256",
                "alignment_disposition",
                "adjudication_invalidated",
            ],
        )
        writer.writeheader()
        for source_item in sorted(items, key=lambda row: row.benchmark_item_id):
            item = next_item_by_source[source_item.benchmark_item_id]
            canonical = item_by_id[remap(source_item.benchmark_item_id)]
            writer.writerow(
                {
                    "v2_benchmark_item_id": source_item.benchmark_item_id,
                    "v2.1_benchmark_item_id": canonical.benchmark_item_id,
                    "pre_v2_contract_sha256": item.pre_v2_contract_sha256 or "",
                    "v2.1_contract_sha256": canonical.exact_contract_sha256 or "",
                    "scientific_question_sha256": item.scientific_question_sha256,
                    "alignment_disposition": item.alignment_disposition or "",
                    "adjudication_invalidated": str(
                        (
                            item.benchmark_item_id in changed_ids
                            or item.migration_status != "ready"
                        )
                        and item.adjudication_status != "construction_derived"
                    ).lower(),
                }
            )

    splits = {
        "version": SCHEMA_VERSION,
        "tracks": {
            track: [
                item.benchmark_item_id
                for item in canonical_items
                if item.benchmark_track == track
            ]
            for track in ("scientific", "synthetic_stress", "external_transfer")
        },
        "adjudication_candidates": [
            item.benchmark_item_id
            for item in canonical_items
            if item.migration_status == "ready"
            and item.benchmark_item_id in changed_ids
            and item.adjudication_status != "construction_derived"
            and item.benchmark_track in {"scientific", "external_transfer"}
        ],
        "non_executable": [
            item.benchmark_item_id
            for item in canonical_items
            if item.migration_status == "non_executable"
        ],
        "ambiguous_unresolved": [
            item.benchmark_item_id
            for item in canonical_items
            if item.migration_status == "ambiguous_unresolved"
        ],
    }
    _atomic_text(out_dir / "benchmark_splits.json", json.dumps(splits, indent=2, sort_keys=True) + "\n")

    repair_rows = [
        {
            "benchmark_item_id": source_to_migrated_id[row.benchmark_item_id],
            "canonical_question_sha256": row.canonical_question_sha256,
            "pre_repair_contract_sha256": row.executable_contract_sha256,
            "post_repair_contract_sha256": row.repaired_contract_sha256,
            "deterministic_repairs": [
                repair.model_dump(mode="json") for repair in row.deterministic_repairs
            ],
            "safety_augmentations": [
                repair.model_dump(mode="json") for repair in row.safety_augmentations
            ],
            "final_outcome_blind_resolution": row.final_outcome_blind_resolution,
            "policy_version": row.policy_version,
        }
        for row in sorted(alignments.values(), key=lambda value: value.benchmark_item_id)
    ]
    repair_manifest = {
        "version": SCHEMA_VERSION,
        "outcome_blind": True,
        "alignment_policy_sha256": _sha256_file(out_dir / "alignment_policy.json"),
        "alignment_records_sha256": _sha256_file(alignment_path),
        "repair_record_count": len(repair_rows),
        "contract_repair_count": sum(bool(row["deterministic_repairs"]) for row in repair_rows),
        "safety_augmentation_count": sum(bool(row["safety_augmentations"]) for row in repair_rows),
        "records": repair_rows,
        "forbidden_inputs_used": [],
    }
    _atomic_text(
        out_dir / "repair_manifest.json",
        json.dumps(repair_manifest, indent=2, sort_keys=True) + "\n",
    )

    partition_rows = [
        {
            "task_id": task.task_id,
            "benchmark_item_id": task.benchmark_item_id,
            "partition_paths": task.partition_paths,
            "partition_hashes": task.partition_hashes,
            "generator_artifact_hashes": task.generator_artifact_hashes,
        }
        for task in sorted(new_tasks, key=lambda row: row.task_id)
    ]
    _write_jsonl(out_dir / "task_evidence_hashes.jsonl", partition_rows)

    output_names = [
        "benchmark_items.jsonl",
        "evaluation_tasks.jsonl",
        "evidence_records.jsonl",
        "label_votes.jsonl",
        "adjudications.jsonl",
        "alignment_records.jsonl",
        "alignment_policy.json",
        "repair_manifest.json",
        "v2_to_v2.1_crosswalk.csv",
        "benchmark_splits.json",
        "task_evidence_hashes.jsonl",
    ]
    if assignment_path is not None:
        output_names.append(assignment_path.name)
    manifest = {
        "benchmark_name": "NeuroClaimBench",
        "version": SCHEMA_VERSION,
        "source_package": _relative(source),
        "source_package_hashes": {
            name: _sha256_file(source / name)
            for name in (
                "benchmark_items.jsonl",
                "evaluation_tasks.jsonl",
                "evidence_records.jsonl",
                "label_votes.jsonl",
                "adjudications.jsonl",
            )
        },
        "alignment_records": {
            "path": _relative(alignment_path),
            "sha256": _sha256_file(alignment_path),
        },
        "code_sha": code_sha,
        "gate_policy_version": gate_policy_version,
        "counts": {
            "source_items": len(items),
            "benchmark_items": len(canonical_items),
            "evaluation_tasks": len(new_tasks),
            "contract_repairs": len(contract_repair_ids),
            "adjudication_identity_changes": len(changed_ids),
            "alignment_dispositions": dict(
                Counter(item.alignment_disposition for item in canonical_items)
            ),
            "reference_labels_invalidated": sum(
                item.adjudication_status != "construction_derived"
                and (
                    item.benchmark_item_id in changed_ids
                    or item.migration_status != "ready"
                )
                for item in canonical_items
            ),
        },
        "output_files": {
            name: {"path": _relative(out_dir / name), "sha256": _sha256_file(out_dir / name)}
            for name in output_names
        },
        "interpretation_restrictions": [
            "v2.1 is a retrospective benchmark revision frozen before its v2.1 gate rerun",
            "constructed controls are not literature references",
            "low-powered identifiable claims remain executable",
            "alignment and repair used no gate outcomes, labels, p-values, or feedback results",
        ],
    }
    _atomic_text(out_dir / "build_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", default="data/neuroclaimbench/v2.1-source")
    parser.add_argument(
        "--alignment-records",
        default="review-stage/neuroclaimbench-v2.1/alignment/alignment_records.jsonl",
    )
    parser.add_argument("--out-dir", default="data/neuroclaimbench/v2.1")
    parser.add_argument("--data-root", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is None:
        args.data_root = list(DEFAULT_DATA_ROOTS)
    manifest = build(args)
    print(json.dumps({"status": "completed", "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
