#!/usr/bin/env python3
"""Run receipt-backed FastSurfer/FreeSurfer jobs from external subject manifests."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any

from nbs_data.external_dataset_registry import SubjectManifestRow, load_registry
from nbs_data.freesurfer.external_stats import (
    CompletionReceipt,
    adopt_legacy_completion,
    completion_check,
    required_artifact_check,
    write_completion_receipt,
)


def validate_freesurfer_runtime(freesurfer_home: str | Path, *, expected_version: str = "7.4.1") -> str:
    home = Path(freesurfer_home)
    binary = home / "bin/recon-all"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"FREESURFER_HOME does not contain executable bin/recon-all: {home}")
    process = subprocess.run([str(binary), "-version"], check=False, capture_output=True, text=True)
    output = f"{process.stdout}\n{process.stderr}".strip()
    if process.returncode != 0 or re.search(rf"(?<!\d){re.escape(expected_version)}(?!\d)", output) is None:
        raise ValueError(f"FreeSurfer {expected_version} is required; recon-all reported: {output!r}")
    return expected_version


def validate_fastsurfer_runtime(
    fastsurfer_home: str | Path,
    fastsurfer_python: str | Path,
) -> str:
    home = Path(fastsurfer_home)
    runner = home / "run_fastsurfer.sh"
    python = Path(fastsurfer_python)
    python3 = python.parent / "python3"
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"FASTSURFER_HOME does not contain executable run_fastsurfer.sh: {home}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"FastSurfer Python is not executable: {python}")
    if not python3.is_file() or not os.access(python3, os.X_OK):
        raise ValueError(f"FastSurfer requires an executable python3 beside its Python: {python3}")
    process = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import sys; "
                "assert sys.version_info >= (3, 10); "
                "import nibabel, torch, yacs; "
                "print('.'.join(map(str, sys.version_info[:3])))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = f"{process.stdout}\n{process.stderr}".strip()
    if process.returncode != 0:
        raise ValueError(
            "FastSurfer Python must be >=3.10 and import nibabel, torch, and yacs; "
            f"preflight reported: {output!r}"
        )
    return process.stdout.strip().splitlines()[-1]


def run(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.config)
    datasets = registry.selected(args.datasets)
    freesurfer_home = args.freesurfer_home or os.environ.get("FREESURFER_HOME")
    if not freesurfer_home:
        raise ValueError("FREESURFER_HOME must be supplied explicitly; module fallback is disabled")
    freesurfer_version = validate_freesurfer_runtime(
        freesurfer_home,
        expected_version=args.expected_freesurfer_version,
    )
    fastsurfer_home = args.fastsurfer_home or os.environ.get("FASTSURFER_HOME")
    fastsurfer_python = args.fastsurfer_python or os.environ.get("FASTSURFER_PYTHON")
    fastsurfer_python_version = None
    if args.engine == "fastsurfer":
        if not fastsurfer_home:
            raise ValueError("FASTSURFER_HOME is required for engine=fastsurfer")
        fastsurfer_python = fastsurfer_python or str(Path(fastsurfer_home) / ".venv/bin/python")
        fastsurfer_python_version = validate_fastsurfer_runtime(fastsurfer_home, fastsurfer_python)
        args.fastsurfer_python = str(fastsurfer_python)
    fs_license = (
        args.fs_license
        or os.environ.get("FS_LICENSE")
        or str(Path(freesurfer_home) / "license.txt")
    )
    if not Path(fs_license).is_file():
        raise ValueError(f"FreeSurfer license does not exist: {fs_license}")

    out_root = Path(args.out_root)
    subjects_root = Path(args.subjects_root)
    status_path = out_root / "audits" / "freesurfer_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if args.engine == "fastsurfer" and not gpu_ids:
        raise ValueError("At least one GPU ID is required for FastSurfer")
    if args.engine == "recon-all":
        gpu_ids = [""]

    tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset.structural is None:
            continue
        manifest_path = out_root / "manifests" / f"{dataset.dataset_id}_subjects.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest {manifest_path}; run the manifest stage first")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [SubjectManifestRow.model_validate(item) for item in payload.get("selected_rows", [])]
        if args.limit is not None:
            rows = rows[: args.limit]
        for row in rows:
            tasks.append(
                {
                    "dataset": dataset,
                    "row": row,
                    "subjects_root": subjects_root,
                    "out_root": out_root,
                    "engine": args.engine,
                    "threads": args.threads,
                    "freesurfer_home": str(freesurfer_home),
                    "freesurfer_version": freesurfer_version,
                    "fastsurfer_home": str(fastsurfer_home) if fastsurfer_home else None,
                    "fastsurfer_version": _fastsurfer_version(fastsurfer_home),
                    "fastsurfer_python": str(fastsurfer_python) if fastsurfer_python else None,
                    "fastsurfer_python_version": fastsurfer_python_version,
                    "fs_license": fs_license,
                    "retry_failed": args.retry_failed,
                }
            )

    results: list[dict[str, Any]] = []
    _checkpoint(status_path, args, results, len(tasks))
    workers = min(len(gpu_ids), len(tasks)) if tasks else 1
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            result = _run_one(task, gpu_ids[(index - 1) % len(gpu_ids)])
            results.append(result)
            _checkpoint(status_path, args, results, len(tasks))
            print(_progress_message(index, len(tasks), result), flush=True)
    else:
        result_queue: Queue[dict[str, Any]] = Queue()
        task_queues = _gpu_task_queues(tasks, gpu_ids)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_gpu_queue, gpu_tasks, gpu_id, result_queue)
                for gpu_id, gpu_tasks in task_queues
            ]
            for index in range(1, len(tasks) + 1):
                result = result_queue.get()
                results.append(result)
                _checkpoint(status_path, args, results, len(tasks))
                print(_progress_message(index, len(tasks), result), flush=True)
            for future in futures:
                future.result()

    has_failures = any(result.get("status") == "failed" for result in results)
    final_status = "completed_with_failures" if has_failures else "completed"
    output = _checkpoint(status_path, args, results, len(tasks), status=final_status)
    print(f"wrote {status_path}")
    return output


def _gpu_task_queues(
    tasks: list[dict[str, Any]],
    gpu_ids: list[str],
) -> list[tuple[str, list[dict[str, Any]]]]:
    return [(gpu_id, tasks[index:: len(gpu_ids)]) for index, gpu_id in enumerate(gpu_ids)]


def _run_gpu_queue(
    tasks: list[dict[str, Any]],
    gpu_id: str,
    result_queue: Queue[dict[str, Any]],
) -> None:
    for task in tasks:
        try:
            result = _run_one(task, gpu_id)
        except Exception as exc:  # noqa: BLE001
            row: SubjectManifestRow = task["row"]
            result = {
                "dataset_id": row.dataset_id,
                "subject_id": row.subject_id,
                "status": "failed",
                "error": str(exc),
                "gpu_id": gpu_id,
            }
        result_queue.put(result)


def _run_one(task: dict[str, Any], gpu_id: str) -> dict[str, Any]:
    dataset = task["dataset"]
    row: SubjectManifestRow = task["row"]
    subjects_root: Path = task["subjects_root"]
    out_root: Path = task["out_root"]
    subject_dir = subjects_root / dataset.dataset_id / row.subject_id

    if dataset.structural.existing_subjects_dir:
        legacy_dir = Path(dataset.structural.existing_subjects_dir) / row.subject_id
        legacy_check = completion_check(legacy_dir)
        if not legacy_check.complete:
            legacy_check = completion_check(legacy_dir, allow_legacy=True)
            if legacy_check.complete:
                adopt_legacy_completion(
                    legacy_dir,
                    subject_id=row.subject_id,
                    t1_path=row.t1_path,
                    t1_sha256=row.t1_sha256,
                )
        if legacy_check.complete:
            return {
                "dataset_id": dataset.dataset_id,
                "subject_id": row.subject_id,
                "status": "quarantined" if dataset.quarantine else "complete",
                "processing_status": "complete",
                "source": "adopted_legacy_output",
                "subject_dir": str(legacy_dir),
                "gpu_id": gpu_id,
            }

    existing = completion_check(subject_dir)
    if existing.complete:
        return {
            "dataset_id": dataset.dataset_id,
            "subject_id": row.subject_id,
            "status": "quarantined" if dataset.quarantine else "complete",
            "processing_status": "complete",
            "source": "existing_receipt",
            "subject_dir": str(subject_dir),
            "gpu_id": gpu_id,
        }
    if subject_dir.exists() and any(subject_dir.iterdir()):
        if not task["retry_failed"]:
            return {
                "dataset_id": dataset.dataset_id,
                "subject_id": row.subject_id,
                "status": "failed",
                "error": "partial output preserved; rerun with --retry-failed",
                "subject_dir": str(subject_dir),
                "gpu_id": gpu_id,
            }
        archive = out_root / "failed_attempts" / dataset.dataset_id / row.subject_id / datetime.now().strftime("%Y%m%dT%H%M%S")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(subject_dir), str(archive))

    dataset_root = subjects_root / dataset.dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "logs" / "freesurfer" / dataset.dataset_id / f"{row.subject_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    command = _command(task, row, dataset_root)
    environment = os.environ.copy()
    path_entries = [str(Path(task["freesurfer_home"]) / "bin")]
    if task["engine"] == "fastsurfer":
        fastsurfer_bin = str(Path(task["fastsurfer_python"]).parent)
        path_entries.insert(0, fastsurfer_bin)
        environment["VIRTUAL_ENV"] = str(Path(fastsurfer_bin).parent)
    environment.update(
        {
            "FREESURFER_HOME": task["freesurfer_home"],
            "FS_LICENSE": task["fs_license"],
            "SUBJECTS_DIR": str(dataset_root),
            "PATH": ":".join([*path_entries, environment.get("PATH", "")]),
        }
    )
    if gpu_id:
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"command={json.dumps(command)}\n")
        log.flush()
        process = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT, env=environment)
    finished = datetime.now().isoformat(timespec="seconds")
    artifacts = required_artifact_check(subject_dir)
    processing_status = "complete" if process.returncode == 0 and artifacts.complete else "failed"
    status = "quarantined" if processing_status == "complete" and dataset.quarantine else processing_status
    receipt = CompletionReceipt(
        subject_id=row.subject_id,
        status=processing_status,
        exit_code=process.returncode,
        engine=task["engine"],
        freesurfer_version=task["freesurfer_version"],
        fastsurfer_version=task["fastsurfer_version"] if task["engine"] == "fastsurfer" else None,
        fastsurfer_python=task["fastsurfer_python"] if task["engine"] == "fastsurfer" else None,
        fastsurfer_python_version=(
            task["fastsurfer_python_version"] if task["engine"] == "fastsurfer" else None
        ),
        t1_path=row.t1_path,
        t1_sha256=row.t1_sha256,
        started_at=started,
        finished_at=finished,
        required_artifacts=artifacts.required_artifacts,
    )
    subject_dir.mkdir(parents=True, exist_ok=True)
    write_completion_receipt(subject_dir, receipt)
    return {
        "dataset_id": dataset.dataset_id,
        "subject_id": row.subject_id,
        "status": status,
        "processing_status": processing_status,
        "exit_code": process.returncode,
        "missing_artifacts": artifacts.missing_artifacts,
        "error": (
            None
            if processing_status == "complete"
            else f"process exit={process.returncode}; missing required artifacts={artifacts.missing_artifacts}"
        ),
        "subject_dir": str(subject_dir),
        "log_path": str(log_path),
        "gpu_id": gpu_id,
    }


def _command(task: dict[str, Any], row: SubjectManifestRow, dataset_root: Path) -> list[str]:
    if task["engine"] == "fastsurfer":
        return [
            str(Path(task["fastsurfer_home"]) / "run_fastsurfer.sh"),
            "--t1",
            row.t1_path,
            "--sid",
            row.subject_id,
            "--sd",
            str(dataset_root),
            "--threads",
            str(task["threads"]),
            "--device",
            "cuda",
        ]
    return [
        str(Path(task["freesurfer_home"]) / "bin/recon-all"),
        "-subjid",
        row.subject_id,
        "-i",
        row.t1_path,
        "-all",
        "-sd",
        str(dataset_root),
        "-no-isrunning",
        "-parallel",
        "-openmp",
        str(task["threads"]),
    ]


def _fastsurfer_version(home: str | None) -> str | None:
    if not home:
        return None
    root = Path(home)
    process = subprocess.run(
        ["git", "-C", str(root), "describe", "--always", "--dirty"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = process.stdout.strip()
    return value or "unknown"


def _checkpoint(
    path: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    task_count: int,
    *,
    status: str = "running",
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        key = str(result.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    output = {
        "status": status,
        "task_count": task_count,
        "completed_task_count": len(results),
        "pending_task_count": max(task_count - len(results), 0),
        "status_counts": counts,
        "engine": args.engine,
        "expected_freesurfer_version": args.expected_freesurfer_version,
        "fastsurfer_python": getattr(args, "fastsurfer_python", None),
        "results": sorted(
            results,
            key=lambda item: (str(item.get("dataset_id")), str(item.get("subject_id"))),
        ),
    }
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _progress_message(index: int, total: int, result: dict[str, Any]) -> str:
    message = (
        f"[freesurfer {index}/{total}] dataset={result.get('dataset_id')} "
        f"subject={result.get('subject_id')} status={result.get('status')}"
    )
    if result.get("error"):
        message += f" error={result['error']}"
    return message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/external_datasets.yml")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset IDs or 'all'.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--subjects-root", default="/data/users1/ywei/confirm_external_prep/subjects")
    parser.add_argument("--engine", choices=["fastsurfer", "recon-all"], default="fastsurfer")
    parser.add_argument("--freesurfer-home", default=None)
    parser.add_argument("--expected-freesurfer-version", default="7.4.1")
    parser.add_argument("--fastsurfer-home", default=None)
    parser.add_argument("--fastsurfer-python", default=None)
    parser.add_argument("--fs-license", default=None)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Canary limit applied separately to each dataset.")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    output = run(build_parser().parse_args(argv))
    return 1 if output.get("status_counts", {}).get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
