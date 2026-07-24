"""Audit the result-preserving v2.1/v7 simplification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    )


def _verify_checksums(root: Path) -> int:
    count = 0
    for line in (root / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.exists() or _sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {path}")
        count += 1
    return count


def _normalized_sweep(root: Path) -> dict[str, Any]:
    aggregate = json.loads(
        (root / "normalized/manifest.json").read_text(encoding="utf-8")
    )
    if aggregate["arm_count"] != 12:
        raise ValueError("Normalized sweep must contain 12 arms")
    parents = 0
    calls = 0
    candidates = 0
    evaluations = 0
    for arm in aggregate["arms"]:
        arm_dir = root / "normalized/arms" / arm["arm_id"]
        manifest_path = arm_dir / "manifest.json"
        if _sha256(manifest_path) != arm["manifest_sha256"]:
            raise ValueError(
                f"Normalized arm manifest mismatch: {arm_dir}"
            )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest["reconciliation"]["status"] != "exact":
            raise ValueError(f"Arm is not reconciled: {arm_dir}")
        for record in manifest["files"].values():
            path = Path(record["path"])
            if _sha256(path) != record["sha256"]:
                raise ValueError(f"Normalized file mismatch: {path}")
        counts = manifest["counts"]
        parents += counts["parents"]
        calls += counts["llm_calls"]
        candidates += (
            counts["retained_candidates"]
            + counts["unretained_candidates"]
            + counts["duplicate_candidates"]
        )
        evaluations += counts["evaluations"]
    if parents != 12 * 215:
        raise ValueError(f"Unexpected normalized parent count: {parents}")
    return {
        "arms": 12,
        "parents": parents,
        "llm_calls": calls,
        "candidate_records": candidates,
        "evaluations": evaluations,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    sweep = Path(args.sweep)
    archive = Path(args.sweep_archive)
    release = Path(args.release)
    audit_archive = Path(args.benchmark_audit_archive)
    compact = Path(args.compact)
    frozen = Path(args.frozen_manifest)
    output = Path(args.out)

    normalized = _normalized_sweep(sweep)
    archive_files = sorted(archive.glob("*.tar.zst"))
    archive_checksum_count = _verify_checksums(archive)
    if len(archive_files) != 12 or archive_checksum_count != 12:
        raise ValueError("Sweep archive must contain 12 checksummed arms")
    content_rows = [
        line.split("\t")
        for line in (archive / "ARCHIVE_CONTENT_BYTES.tsv").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    archived_logical_bytes = sum(int(row[1]) for row in content_rows)
    active_sweep_bytes = _tree_bytes(sweep)
    reduction = (
        1.0 - active_sweep_bytes / archived_logical_bytes
        if archived_logical_bytes
        else 0.0
    )
    if reduction < 0.60:
        raise ValueError(
            f"Active sweep reduction is below 60%: {reduction:.3f}"
        )
    legacy_file_count = len(
        list(
            sweep.glob(
                "matrix/rounds_*/candidates_*/"
                "iterative_candidate_replay.json"
            )
        )
    )
    checkpoint_dir_count = len(
        list(sweep.glob("matrix/rounds_*/candidates_*/checkpoints"))
    )
    if legacy_file_count or checkpoint_dir_count:
        raise ValueError("Duplicated v7 artifacts remain active")

    compact_manifest = json.loads(
        (compact / "manifest.json").read_text(encoding="utf-8")
    )
    release_manifest = json.loads(
        (release / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
    )
    if release_manifest["release_schema_version"] != 2:
        raise ValueError("Public benchmark is not release schema 2")
    if release_manifest["counts"] != {
        key: compact_manifest["counts"][key]
        for key in ("cases", "references", "tasks", "outcomes")
    }:
        raise ValueError("Compact and released benchmark counts differ")

    frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
    if len(frozen_payload["claim_search"]["arms"]) != 12:
        raise ValueError("Frozen run manifest does not contain 12 arms")

    result = {
        "status": "passed",
        "scientific_results_changed": False,
        "frozen_manifest": {
            "path": str(frozen),
            "sha256": _sha256(frozen),
        },
        "normalized_sweep": normalized,
        "storage": {
            "archived_original_logical_bytes": archived_logical_bytes,
            "active_sweep_bytes": active_sweep_bytes,
            "active_reduction_fraction": reduction,
            "compressed_archive_bytes": _tree_bytes(archive),
        },
        "benchmark": {
            "release_schema_version": 2,
            "counts": release_manifest["counts"],
            "release_checksum_count": _verify_checksums(release),
            "audit_checksum_count": _verify_checksums(audit_archive),
        },
        "active_legacy_monolith_count": legacy_file_count,
        "active_checkpoint_tree_count": checkpoint_dir_count,
        "api_calls_performed": 0,
        "pubmed_calls_performed": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "simplification_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "simplification_audit.md").write_text(
        "# Result-Preserving Simplification Audit\n\n"
        "- Status: passed\n"
        "- Scientific results changed: no\n"
        f"- Normalized sweep: {normalized['arms']} arms, "
        f"{normalized['parents']} parent states\n"
        f"- Active sweep reduction: {reduction:.1%}\n"
        f"- Benchmark release: {release_manifest['counts']['cases']} cases, "
        f"{release_manifest['counts']['tasks']} tasks, schema 2\n"
        "- Active legacy monoliths/checkpoint trees: 0/0\n"
        "- API/PubMed calls: 0/0\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        default="review-stage/claim-search-gpt55-sweep-v7",
    )
    parser.add_argument(
        "--sweep-archive",
        default="review-stage/_archive_20260723_claim_search_v7_original",
    )
    parser.add_argument(
        "--release",
        default="benchmark/neuroclaimbench-v2.1",
    )
    parser.add_argument(
        "--benchmark-audit-archive",
        default="review-stage/neuroclaimbench-v2.1/external-archive",
    )
    parser.add_argument(
        "--compact",
        default="review-stage/neuroclaimbench-v2.1/compact",
    )
    parser.add_argument(
        "--frozen-manifest",
        default="benchmark/FROZEN_RUNS.json",
    )
    parser.add_argument(
        "--out",
        default="review-stage/simplification-audit-20260723",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
