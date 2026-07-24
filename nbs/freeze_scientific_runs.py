"""Register immutable scientific inputs before result-preserving refactors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _optional_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [_record(path) for path in paths if path.is_file()]


def build_manifest(
    *,
    sweep_dir: Path,
    benchmark_dir: Path,
    benchmark_results_dir: Path,
    paper_dir: Path,
    source_snapshot_dir: Path,
) -> dict[str, Any]:
    matrix_path = sweep_dir / "matrix_summary.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    arm_rows = matrix.get("rows", [])
    if len(arm_rows) != 12:
        raise ValueError(f"Expected 12 frozen sweep arms, found {len(arm_rows)}")

    arms: list[dict[str, Any]] = []
    for row in sorted(
        arm_rows,
        key=lambda item: (
            int(item["max_rounds"]),
            int(item["max_candidates_per_round"]),
        ),
    ):
        artifact = Path(str(row["artifact"]))
        run_dir = artifact.parent
        provenance_path = run_dir / "run_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        arms.append(
            {
                "max_rounds": int(row["max_rounds"]),
                "max_candidates_per_round": int(
                    row["max_candidates_per_round"]
                ),
                "result": _record(artifact),
                "provenance": _record(provenance_path),
                "source_sha256": provenance["source"]["sha256"],
                "prompt_sha256": provenance["prompt_sha256"],
                "schema_sha256": provenance["schema_sha256"],
                "evidence_manifest_sha256": provenance[
                    "evidence_manifest"
                ]["sha256"],
                "partition_hashes_sha256": provenance[
                    "partition_hashes_sha256"
                ],
                "implementation_hashes": provenance[
                    "implementation_hashes"
                ],
            }
        )

    benchmark_files = [
        benchmark_dir / "benchmark_items.jsonl",
        benchmark_dir / "benchmark_claims.jsonl",
        benchmark_dir / "reference_profiles.jsonl",
        benchmark_dir / "evaluation_tasks.jsonl",
        benchmark_dir / "normalized_task_outcomes.jsonl",
        benchmark_dir / "benchmark_summary.json",
        benchmark_dir / "build_manifest.json",
        benchmark_dir / "analysis_manifest.json",
        benchmark_dir / "paper_benchmark_manifest.json",
        benchmark_dir / "SHA256SUMS",
    ]
    result_files = [
        benchmark_results_dir / "benchmark_summary.json",
        benchmark_results_dir / "task_outcomes.jsonl",
    ]
    paper_files = [
        paper_dir / "figures/tab_neuroclaimbench_v21.tex",
        paper_dir / "figures/tab_neuroclaimbench_v21_sensitivity.tex",
        paper_dir / "figures/tab_feedback_forced_choice.tex",
        paper_dir / "sec/04_benchmark.tex",
        paper_dir / "sec/05_experiments.tex",
    ]
    snapshot_files = [
        source_snapshot_dir / "source_snapshot.tar.gz",
        source_snapshot_dir / "workspace_from_head.patch",
        source_snapshot_dir / "paper_from_head.patch",
        source_snapshot_dir / "SHA256SUMS",
    ]

    return {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "neuroclaimbench_version": "v2.1",
            "claim_search_version": "v7",
            "scientific_results_frozen": True,
            "api_reruns_permitted": False,
            "gemini_role": "outcome_blind_eligibility_adjudicator",
            "gemini_ambiguous_exclusion_count": 40,
        },
        "claim_search": {
            "root": str(sweep_dir),
            "matrix_summary": _record(matrix_path),
            "source": _record(
                sweep_dir / "source/claim_search_source.json"
            ),
            "arms": arms,
        },
        "neuroclaimbench": {
            "root": str(benchmark_dir),
            "files": _optional_records(benchmark_files),
            "result_files": _optional_records(result_files),
        },
        "paper_tables_and_sources": _optional_records(paper_files),
        "experiment_source_snapshot": {
            "root": str(source_snapshot_dir),
            "files": _optional_records(snapshot_files),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        default="review-stage/claim-search-gpt55-sweep-v7",
    )
    parser.add_argument(
        "--benchmark",
        default="benchmark/neuroclaimbench-v2.1",
    )
    parser.add_argument(
        "--benchmark-results",
        default="review-stage/neuroclaimbench-v2.1/results",
    )
    parser.add_argument("--paper", default="paper")
    parser.add_argument(
        "--source-snapshot",
        default="review-stage/_archive_20260723_simplification_source",
    )
    parser.add_argument(
        "--out",
        default="benchmark/FROZEN_RUNS.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(
        sweep_dir=Path(args.sweep),
        benchmark_dir=Path(args.benchmark),
        benchmark_results_dir=Path(args.benchmark_results),
        paper_dir=Path(args.paper),
        source_snapshot_dir=Path(args.source_snapshot),
    )
    _atomic_json(Path(args.out), manifest)


if __name__ == "__main__":
    main()
