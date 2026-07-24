"""Package the compact NeuroClaimBench v2.1 release and audit archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from bench.benchmark import BenchmarkDataset
from bench.io import read_jsonl, write_jsonl


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _assert_empty(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _relative_path(value: str) -> str:
    path = Path(value)
    return os.path.relpath(path, Path.cwd()) if path.is_absolute() else value


def _normalize_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        payload["cohort_paths"] = [
            _relative_path(str(path))
            for path in payload.get("cohort_paths") or []
        ]
        normalized.append(payload)
    return normalized


def _assert_no_private_absolute_paths(path: Path) -> None:
    text_suffixes = {".json", ".jsonl", ".csv", ".md", ".txt"}
    for file in path.rglob("*"):
        if not file.is_file() or file.suffix not in text_suffixes:
            continue
        text = file.read_text(encoding="utf-8", errors="ignore")
        if "/Users/" in text or str(Path.home()) in text:
            raise ValueError(
                f"Private absolute path found in release artifact: {file}"
            )


def _checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }


def _write_checksums(root: Path) -> None:
    hashes = _checksums(root)
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in hashes.items()
        ),
        encoding="utf-8",
    )


def _release_readme() -> str:
    return """# NeuroClaimBench v2.1

This is the compact, metadata-only NeuroClaimBench v2.1 release using
`release_schema_version=2`.

- `cases.jsonl` stores canonical questions and frozen contracts.
- `references.jsonl` stores literature or constructed-control dispositions.
- `tasks.jsonl` stores exact evidence identities and executable contracts.
- `outcomes.jsonl` stores compact CONFIRM verdicts. Detailed gate bundles are
  retained in the checksummed audit archive.
- Constructed controls are never described as literature references.
- Low-powered but identifiable claims remain executable and are handled by the
  unchanged CONFIRM power gate.
- v2.1 is a retrospective benchmark revision. Its repair and eligibility
  policy was frozen before the v2.1 rerun.

Raw cohort data and the local PubMed abstract cache are not redistributed.
"""


def _data_dictionary() -> str:
    return """# Data Dictionary

| File | Purpose |
|---|---|
| `cases.jsonl` | Canonical questions, contracts, provenance, aliases, references, and task IDs |
| `references.jsonl` | Reference disposition, basis, strength, evidence IDs, and eligibility |
| `tasks.jsonl` | Frozen evidence identity, executable contract, and partition hashes |
| `outcomes.jsonl` | Compact CONFIRM verdicts and hashes of detailed audit results |
| `benchmark_splits.json` | Frozen benchmark tracks and adjudication subset |
| `benchmark_summary.json` | Tiered benchmark metrics and uncertainty |
| `benchmark_strata_summary.csv` | Combined, tiered, external, and safety results |
| `target_reference_summary.csv` | Target-by-reference basis and strength results |
| `feedback_parent_crosswalk.csv` | Exact pre-v2 feedback-parent mapping |
| `benchmark_manifest.json` | Compact-schema source hashes and reconciled counts |
| `RELEASE_MANIFEST.json` | Release identity and packaging restrictions |
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    compact = Path(args.compact_dir)
    package = Path(args.package_dir)
    results = Path(args.results_dir)
    analysis = Path(args.analysis_dir)
    crosswalk = Path(args.feedback_crosswalk_dir)
    release = Path(args.release_dir)
    archive = Path(args.archive_dir)
    _assert_empty(release, "Release")
    _assert_empty(archive, "Archive")

    dataset = BenchmarkDataset(compact)
    compact_manifest = json.loads(
        (compact / "manifest.json").read_text(encoding="utf-8")
    )
    expected = compact_manifest["counts"]
    observed = {
        "cases": len(dataset.cases),
        "references": len(dataset.references),
        "tasks": len(dataset.tasks),
        "outcomes": len(dataset.outcomes),
    }
    if any(observed[key] != int(expected[key]) for key in observed):
        raise ValueError(
            f"Compact benchmark count mismatch: {observed} != {expected}"
        )

    release_sources = {
        "cases.jsonl": compact / "cases.jsonl",
        "references.jsonl": compact / "references.jsonl",
        "tasks.jsonl": compact / "tasks.jsonl",
        "outcomes.jsonl": compact / "outcomes.jsonl",
        "benchmark_manifest.json": compact / "manifest.json",
        "benchmark_splits.json": package / "benchmark_splits.json",
        "v2_to_v2.1_crosswalk.csv": package
        / "v2_to_v2.1_crosswalk.csv",
        "alignment_policy.json": package / "alignment_policy.json",
        "benchmark_summary.json": results / "benchmark_summary.json",
        "benchmark_results.csv": results / "benchmark_results.csv",
        "cluster_bootstrap_sensitivity.csv": results
        / "cluster_bootstrap_sensitivity.csv",
        "benchmark_strata_summary.csv": analysis
        / "benchmark_strata_summary.csv",
        "target_reference_summary.csv": analysis
        / "target_reference_summary.csv",
        "reference_agreement_audit.csv": analysis
        / "reference_agreement_audit.csv",
        "unresolved_case_summary.csv": analysis
        / "unresolved_case_summary.csv",
        "analysis_audit.json": analysis / "analysis_audit.json",
        "analysis_manifest.json": analysis / "analysis_manifest.json",
        "feedback_parent_crosswalk.csv": crosswalk
        / "feedback_parent_crosswalk.csv",
        "feedback_reference_summary.csv": crosswalk
        / "feedback_reference_summary.csv",
        "feedback_crosswalk_manifest.json": crosswalk
        / "feedback_crosswalk_manifest.json",
    }
    for name, source in release_sources.items():
        _copy(source, release / name)

    audit_sources = {
        "benchmark_items_v2.1.jsonl": package
        / "benchmark_items.jsonl",
        "evaluation_tasks_v2.1.jsonl": package
        / "evaluation_tasks.jsonl",
        "reference_profiles_v2.1.jsonl": Path(args.reference_dir)
        / "triage_reference_profiles.jsonl",
        "alignment_records.jsonl": package / "alignment_records.jsonl",
        "repair_manifest.json": package / "repair_manifest.json",
        "build_manifest.json": package / "build_manifest.json",
        "label_votes.jsonl": package / "label_votes.jsonl",
        "adjudications.jsonl": package / "adjudications.jsonl",
    }
    for name, source in audit_sources.items():
        _copy(source, archive / name)

    detailed_outcomes = _normalize_outcomes(
        read_jsonl(results / "task_outcomes.jsonl")
    )
    write_jsonl(
        archive / "detailed_task_outcomes.jsonl",
        detailed_outcomes,
    )
    evidence_index = [
        {
            key: row.get(key)
            for key in (
                "evidence_id",
                "benchmark_item_id",
                "scientific_question_sha256",
                "pmid",
                "doi",
                "title",
                "journal",
                "year",
                "query",
            )
        }
        for row in read_jsonl(package / "evidence_records.jsonl")
    ]
    write_jsonl(archive / "pubmed_evidence_index.jsonl", evidence_index)

    trace_index: list[dict[str, Any]] = []
    checkpoints = Path(args.adjudication_dir) / "checkpoints"
    for path in sorted(checkpoints.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence = payload.get("evidence_freeze") or {}
        trace_index.append(
            {
                "benchmark_item_id": payload.get("benchmark_item_id"),
                "run_fingerprint": payload.get("run_fingerprint"),
                "status": payload.get("status"),
                "completed_at": payload.get("completed_at"),
                "evidence_sha256": evidence.get("evidence_sha256"),
                "pubmed_cache_packet_sha256": evidence.get(
                    "pubmed_cache_packet_sha256"
                ),
                "vote_prompt_sha256s": [
                    vote.get("prompt_sha256")
                    for vote in payload.get("votes") or []
                ],
                "vote_response_sha256s": [
                    vote.get("response_sha256")
                    for vote in payload.get("votes") or []
                ],
            }
        )
    write_jsonl(archive / "adjudication_trace_index.jsonl", trace_index)

    (release / "README.md").write_text(
        _release_readme(),
        encoding="utf-8",
    )
    (release / "DATA_DICTIONARY.md").write_text(
        _data_dictionary(),
        encoding="utf-8",
    )
    (archive / "README.md").write_text(
        "Checksummed NeuroClaimBench v2.1 audit payload. It excludes "
        "PubMed abstracts, raw cohort data, and private filesystem paths.\n",
        encoding="utf-8",
    )
    manifest = {
        "benchmark_version": "v2.1",
        "release_schema_version": 2,
        "scientific_results_changed": False,
        "counts": observed,
        "pubmed_abstracts_released": False,
        "raw_cohort_data_released": False,
        "full_audit_trace_location": str(archive),
        "doi": None,
        "doi_status": "pending_external_deposit",
    }
    manifest["release_file_count"] = (
        sum(path.is_file() for path in release.rglob("*")) + 2
    )
    manifest["archive_file_count"] = (
        sum(path.is_file() for path in archive.rglob("*")) + 1
    )
    (release / "RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _assert_no_private_absolute_paths(release)
    _assert_no_private_absolute_paths(archive)
    _write_checksums(release)
    _write_checksums(archive)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact-dir",
        default="review-stage/neuroclaimbench-v2.1/compact",
    )
    parser.add_argument(
        "--package-dir",
        default="data/neuroclaimbench/v2.1",
    )
    parser.add_argument(
        "--results-dir",
        default="review-stage/neuroclaimbench-v2.1/results",
    )
    parser.add_argument(
        "--reference-dir",
        default="review-stage/neuroclaimbench-v2.1/reference",
    )
    parser.add_argument(
        "--analysis-dir",
        default="review-stage/neuroclaimbench-v2.1/analysis",
    )
    parser.add_argument(
        "--feedback-crosswalk-dir",
        default="review-stage/neuroclaimbench-v2.1/feedback-crosswalk",
    )
    parser.add_argument(
        "--adjudication-dir",
        default="review-stage/neuroclaimbench-v2.1/adjudication",
    )
    parser.add_argument(
        "--release-dir",
        default="benchmark/neuroclaimbench-v2.1",
    )
    parser.add_argument(
        "--archive-dir",
        default="review-stage/neuroclaimbench-v2.1/external-archive",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    manifest = run(build_parser().parse_args(argv))
    print(
        json.dumps(
            {"status": "completed", "manifest": manifest},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
