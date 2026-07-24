"""Join frozen v7 feedback parents to NeuroClaimBench v2.1 references outcome-blindly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bench.benchmark import BenchmarkCase, BenchmarkDataset
from bench.neuroclaimbench import exact_contract_hash, sha256_payload
from confirm.contract import ClaimContract


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(paths: list[Path], root: Path) -> str:
    return sha256_payload(
        {
            path.relative_to(root).as_posix(): _file_sha256(path)
            for path in sorted(paths)
        }
    )


def _source_ids(item: BenchmarkCase) -> set[str]:
    return set(item.aliases) | {
        str(row.get("source_id") or "") for row in item.provenance
    }


def _arm_paths(sweep: Path) -> list[tuple[int, int, Path]]:
    rows: list[tuple[int, int, Path]] = []
    normalized = sweep / "normalized" / "arms"
    if normalized.exists():
        for arm in sorted(normalized.glob("r*_c*")):
            rounds_text, candidates_text = arm.name.split("_", 1)
            rows.append(
                (
                    int(rounds_text[1:]),
                    int(candidates_text[1:]),
                    arm,
                )
            )
        if rows:
            return rows
    for rounds_dir in sorted((sweep / "matrix").glob("rounds_*")):
        rounds = int(rounds_dir.name.split("_", 1)[1])
        for candidates_dir in sorted(rounds_dir.glob("candidates_*")):
            candidates = int(candidates_dir.name.split("_", 1)[1])
            rows.append((rounds, candidates, candidates_dir))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(args.package_dir)
    sweep = Path(args.sweep)
    out_dir = Path(args.out_dir)
    sweep_source_path = sweep / "source" / "claim_search_source.json"
    matrix_summary_path = sweep / "matrix_summary.json"
    if not sweep_source_path.exists():
        raise FileNotFoundError(sweep_source_path)
    if not matrix_summary_path.exists():
        raise FileNotFoundError(matrix_summary_path)
    sweep_source_sha256 = _file_sha256(sweep_source_path)
    dataset = BenchmarkDataset(package)
    items = dataset.cases
    profiles = {
        row.benchmark_case_id: row for row in dataset.references
    }
    source_to_item: dict[str, BenchmarkCase] = {}
    for item in items:
        for source_id in _source_ids(item):
            previous = source_to_item.get(source_id)
            if (
                previous is not None
                and previous.benchmark_case_id != item.benchmark_case_id
            ):
                raise ValueError(f"Source ID maps to multiple v2.1 items: {source_id}")
            source_to_item[source_id] = item

    arms = _arm_paths(sweep)
    if not arms:
        raise ValueError(f"No sweep arms found under {sweep}")
    crosswalk_rows: list[dict[str, Any]] = []
    summary_groups: dict[tuple[Any, ...], dict[str, int]] = defaultdict(
        lambda: {"parent_count": 0, "supported_parent_count": 0}
    )
    arm_run_provenance_sha256: dict[str, str] = {}
    arm_parent_checkpoint_tree_sha256: dict[str, str] = {}
    expected_parent_ids: set[str] | None = None
    for rounds, candidates, arm in arms:
        arm_key = f"R{rounds}C{candidates}"
        normalized_manifest_path = arm / "manifest.json"
        normalized_parent_path = arm / "parents.jsonl"
        if normalized_manifest_path.exists() and normalized_parent_path.exists():
            normalized_manifest = json.loads(
                normalized_manifest_path.read_text(encoding="utf-8")
            )
            recorded_source_sha256 = str(
                normalized_manifest["source"][
                    "claim_search_source_sha256"
                ]
            )
            arm_run_provenance_sha256[arm_key] = str(
                normalized_manifest["source"][
                    "run_provenance_sha256"
                ]
            )
            normalized_files = sorted(arm.glob("*.json*"))
            arm_parent_checkpoint_tree_sha256[arm_key] = _tree_sha256(
                normalized_files,
                arm,
            )
            parent_payloads = _read_jsonl(normalized_parent_path)
        else:
            run_provenance_path = arm / "run_provenance.json"
            if not run_provenance_path.exists():
                raise FileNotFoundError(run_provenance_path)
            run_provenance = json.loads(
                run_provenance_path.read_text(encoding="utf-8")
            )
            recorded_source_sha256 = str(
                (run_provenance.get("source") or {}).get("sha256") or ""
            )
            arm_run_provenance_sha256[arm_key] = _file_sha256(
                run_provenance_path
            )
            checkpoints = sorted(
                (arm / "checkpoints" / "parents").glob("*.json")
            )
            if len(checkpoints) != 215:
                raise ValueError(
                    "Expected 215 parent checkpoints for "
                    f"R{rounds}/C{candidates}, got {len(checkpoints)}"
                )
            arm_parent_checkpoint_tree_sha256[arm_key] = _tree_sha256(
                checkpoints,
                arm,
            )
            parent_payloads = [
                {
                    "claim_id": payload["claim_id"],
                    "row": payload["row"],
                    "original_claim": payload["state"][
                        "original_claim"
                    ],
                }
                for payload in (
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in checkpoints
                )
            ]
        if recorded_source_sha256 != sweep_source_sha256:
            raise ValueError(
                f"Frozen source hash mismatch for {arm_key}: "
                f"{recorded_source_sha256} != {sweep_source_sha256}"
            )
        if len(parent_payloads) != 215:
            raise ValueError(
                f"Expected 215 parents for R{rounds}/C{candidates}, "
                f"got {len(parent_payloads)}"
            )
        arm_parent_ids: set[str] = set()
        for payload in parent_payloads:
            claim_id = str(payload["claim_id"])
            row = payload["row"]
            parent = ClaimContract.model_validate(
                payload["original_claim"]
            )
            pre_v2_hash = exact_contract_hash(parent)
            item = source_to_item.get(claim_id)
            if item is None:
                raise ValueError(f"No v2.1 item maps feedback parent {claim_id}")
            arm_parent_ids.add(claim_id)
            exact_match = (
                item.migration_status == "ready"
                and item.contract is not None
                and item.pre_v2_contract_sha256 == pre_v2_hash
                and exact_contract_hash(item.contract) == pre_v2_hash
            )
            match_status = "exact_v2.1_reference" if exact_match else "pre_v2_contract_only"
            profile = profiles.get(item.benchmark_case_id)
            if profile is None:
                raise ValueError(
                    "Missing v2.1 reference profile for "
                    f"{item.benchmark_case_id}"
                )
            reference_basis = profile.basis if exact_match else ""
            reference_strength = profile.strength if exact_match else ""
            reference_disposition = (
                profile.disposition if exact_match else ""
            )
            supported = bool(row.get("parent_with_internal_support"))
            crosswalk_rows.append(
                {
                    "rounds": rounds,
                    "candidates": candidates,
                    "claim_id": claim_id,
                    "benchmark_item_id": item.benchmark_case_id,
                    "pre_v2_contract_sha256": pre_v2_hash,
                    "v2.1_contract_sha256": item.contract_sha256 or "",
                    "match_status": match_status,
                    "alignment_disposition": item.alignment_disposition or "",
                    "target_family": str(row.get("target_family") or ""),
                    "source_mode": str(row.get("source_mode") or ""),
                    "reference_basis": reference_basis,
                    "reference_strength": reference_strength,
                    "reference_disposition": reference_disposition,
                    "parent_with_internal_support": str(supported).lower(),
                    "supported_candidate_count": int(row.get("supported_candidate_count") or 0),
                }
            )
            key = (
                rounds,
                candidates,
                match_status,
                str(row.get("target_family") or ""),
                str(row.get("source_mode") or ""),
                reference_basis,
                reference_strength,
                reference_disposition,
            )
            summary_groups[key]["parent_count"] += 1
            summary_groups[key]["supported_parent_count"] += int(supported)
        if expected_parent_ids is None:
            expected_parent_ids = arm_parent_ids
        elif arm_parent_ids != expected_parent_ids:
            raise ValueError(f"Parent lineage set differs for R{rounds}/C{candidates}")

    summary_rows: list[dict[str, Any]] = []
    for key, counts in sorted(summary_groups.items()):
        (
            rounds,
            candidates,
            match_status,
            target_family,
            source_mode,
            basis,
            strength,
            disposition,
        ) = key
        denominator = counts["parent_count"]
        summary_rows.append(
            {
                "rounds": rounds,
                "candidates": candidates,
                "match_status": match_status,
                "target_family": target_family,
                "source_mode": source_mode,
                "reference_basis": basis,
                "reference_strength": strength,
                "reference_disposition": disposition,
                "parent_count": denominator,
                "supported_parent_count": counts["supported_parent_count"],
                "supported_parent_rate": (
                    counts["supported_parent_count"] / denominator if denominator else None
                ),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    parent_crosswalk_path = out_dir / "feedback_parent_crosswalk.csv"
    reference_summary_path = out_dir / "feedback_reference_summary.csv"
    _write_csv(parent_crosswalk_path, crosswalk_rows)
    _write_csv(reference_summary_path, summary_rows)
    manifest = {
        "version": "neuroclaimbench-v2.1-feedback-crosswalk-v2",
        "arm_count": len(arms),
        "parent_lineage_count": len(expected_parent_ids or set()),
        "row_count": len(crosswalk_rows),
        "match_status_counts": dict(Counter(row["match_status"] for row in crosswalk_rows)),
        "uses_feedback_outcomes_for_matching": False,
        "matching_rule": "source claim ID plus exact pre-v2 executable contract hash",
        "input_sha256": {
            "cases.jsonl": _file_sha256(package / "cases.jsonl"),
            "references.jsonl": _file_sha256(
                package / "references.jsonl"
            ),
            "claim_search_source.json": sweep_source_sha256,
            "matrix_summary.json": _file_sha256(matrix_summary_path),
        },
        "source_code_sha256": _file_sha256(Path(__file__)),
        "arm_run_provenance_sha256": dict(sorted(arm_run_provenance_sha256.items())),
        "arm_parent_checkpoint_tree_sha256": dict(
            sorted(arm_parent_checkpoint_tree_sha256.items())
        ),
        "parent_checkpoint_tree_sha256": sha256_payload(
            dict(sorted(arm_parent_checkpoint_tree_sha256.items()))
        ),
        "parent_checkpoint_count": len(crosswalk_rows),
        "output_sha256": {
            parent_crosswalk_path.name: _file_sha256(parent_crosswalk_path),
            reference_summary_path.name: _file_sha256(reference_summary_path),
        },
        "interpretation_restrictions": [
            "repaired or semantically changed parents are pre_v2_contract_only",
            "only exact_v2.1_reference rows enter reference-stratified summaries",
            "the frozen feedback sweep was not rerun or modified",
        ],
    }
    (out_dir / "feedback_crosswalk_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        default="benchmark/neuroclaimbench-v2.1/compact",
    )
    parser.add_argument("--sweep", default="review-stage/claim-search-gpt55-sweep-v7")
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/feedback-crosswalk",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", "manifest": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
