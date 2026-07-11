#!/usr/bin/env python3
"""Prepare, audit, and summarize external CONFIRM evidence datasets."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from confirm.contract import ClaimContract
from confirm.derived_columns import columns_with_virtuals
from confirm.evidence_partitions import load_evidence_manifest
from confirm.schema import columns_with_canonical_aliases, validate_canonical
from nbs_data.external_dataset_registry import (
    ExternalDatasetSpec,
    SubjectManifestRow,
    build_subject_manifest,
    load_registry,
    registry_audit,
    write_subject_manifest,
)
from nbs_data.external_fmri import prepare_fmri_dataset
from nbs_data.external_metadata import load_metadata
from nbs_data.freesurfer.external_stats import CompletionReceipt, canonical_features, completion_check


DEFAULT_CONFIG = "configs/external_datasets.yml"
DEFAULT_OUT_ROOT = "/data/users1/ywei/confirm_external_prep/runs/manual"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.config)
    datasets = registry.selected(args.datasets)
    rows: list[dict[str, object]] = []
    for index, dataset in enumerate(datasets, start=1):
        print(f"[audit {index}/{len(datasets)}] dataset={dataset.dataset_id}", flush=True)
        rows.extend(registry_audit(registry, [dataset]))
    by_id = {dataset.dataset_id: dataset for dataset in datasets}
    for row in rows:
        dataset = by_id[str(row["dataset_id"])]
        try:
            metadata = load_metadata(dataset)
            row["canonical_metadata_rows"] = int(len(metadata))
            row["canonical_metadata_columns"] = "|".join(map(str, metadata.columns))
            row["canonical_metadata_schema_sha256"] = _schema_hash(metadata.columns)
            row["metadata_age_missing"] = int(metadata["age"].isna().sum()) if "age" in metadata else len(metadata)
            row["metadata_sex_missing"] = int(metadata["sex"].isna().sum()) if "sex" in metadata else len(metadata)
            row["metadata_dx_missing"] = int(metadata["dx"].isna().sum()) if "dx" in metadata else len(metadata)
            row["metadata_join_status"] = "available" if not metadata.empty else "unavailable"
        except Exception as exc:  # noqa: BLE001
            row["canonical_metadata_rows"] = 0
            row["canonical_metadata_columns"] = ""
            row["canonical_metadata_schema_sha256"] = ""
            row["metadata_age_missing"] = 0
            row["metadata_sex_missing"] = 0
            row["metadata_dx_missing"] = 0
            row["metadata_join_status"] = f"error: {exc}"
    out_root = Path(args.out_root)
    audit_dir = out_root / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audit_dir / "dataset_audit.csv"
    json_path = audit_dir / "dataset_audit.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    result = {"datasets": rows, "dataset_count": len(rows), "config": str(args.config)}
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return result


def run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.config)
    datasets = registry.selected(args.datasets)
    manifest_dir = Path(args.out_root) / "manifests"
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset.structural is None:
            continue
        try:
            metadata = load_metadata(dataset)
        except Exception as exc:  # noqa: BLE001
            metadata = pd.DataFrame()
            print(f"[{dataset.dataset_id}] metadata unavailable for scan selection: {exc}", flush=True)
        result = build_subject_manifest(dataset, metadata=metadata, progress=True)
        json_path, tsv_path = write_subject_manifest(result, manifest_dir)
        summary = {
            "dataset_id": dataset.dataset_id,
            "candidates_found": result.candidates_found,
            "subjects_found": result.subjects_found,
            "selected_subjects": len(result.selected_rows),
            "imaging_qc_failure_count": len(result.imaging_qc_failures),
            "selection_exclusion_count": len(result.selection_exclusions),
            "quarantine": dataset.quarantine,
            "json_path": str(json_path),
            "tsv_path": str(tsv_path),
        }
        summaries.append(summary)
        print(
            f"[{dataset.dataset_id}] candidates={result.candidates_found} subjects={result.subjects_found} "
            f"selected={len(result.selected_rows)} qc_failures={len(result.imaging_qc_failures)}",
            flush=True,
        )
    output = {"datasets": summaries}
    summary_path = manifest_dir / "structural_manifest_summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    return output


def run_fmri(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.config)
    datasets = registry.selected(args.datasets)
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset.fmri is None:
            continue
        print(f"[{dataset.dataset_id}] preparing fMRI backend={dataset.fmri.backend}", flush=True)
        try:
            results.append(prepare_fmri_dataset(dataset, args.out_root, max_workers=args.max_workers))
        except Exception as exc:  # noqa: BLE001
            results.append({"dataset_id": dataset.dataset_id, "status": "failed", "error": str(exc)})
            print(f"[{dataset.dataset_id}] failed: {exc}", flush=True)
    output = {"datasets": results}
    audit_dir = Path(args.out_root) / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "fmri_preparation.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return output


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.config)
    datasets = registry.selected(args.datasets)
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset.structural is None:
            continue
        try:
            result = _aggregate_dataset(dataset, Path(args.out_root), Path(args.subjects_root))
        except Exception as exc:  # noqa: BLE001
            result = {"dataset_id": dataset.dataset_id, "status": "failed", "error": str(exc)}
            print(f"[{dataset.dataset_id}] aggregation failed: {exc}", flush=True)
        results.append(result)
    audit_dir = Path(args.out_root) / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "smri_aggregation.json"
    path.write_text(json.dumps({"datasets": results}, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return {"datasets": results}


def run_coverage(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.config)
    selected = {dataset.dataset_id: dataset for dataset in registry.selected(args.datasets)}
    results_payload = json.loads(Path(args.stage2_results).read_text(encoding="utf-8"))
    claims = results_payload.get("claims", [])
    canonical_root = Path(args.out_root) / "canonical"
    dataset_rows: list[dict[str, Any]] = []
    for path in sorted(canonical_root.glob("*.parquet")):
        dataset = selected.get(_prepared_base_dataset(path.stem))
        if dataset is None:
            continue
        columns = columns_with_canonical_aliases(pd.read_parquet(path, engine="pyarrow").columns)
        columns = columns_with_virtuals(dataset.dataset_id, columns)
        for target_family in dataset.target_families:
            target_claims = [claim for claim in claims if str(claim.get("target_family")) == target_family]
            compatible = 0
            reasons: dict[str, int] = {}
            for row in target_claims:
                contract_payload = row.get("contract") or row.get("drafted_contract")
                try:
                    contract = ClaimContract.model_validate(contract_payload)
                    ok, reason = _schema_supports_contract(columns, contract)
                except Exception:
                    ok, reason = False, "invalid_contract"
                if ok:
                    compatible += 1
                else:
                    reasons[reason] = reasons.get(reason, 0) + 1
            dataset_rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "target_family": target_family,
                    "stage2_claim_count": len(target_claims),
                    "schema_compatible_claim_count": compatible,
                    "schema_compatible_rate": compatible / len(target_claims) if target_claims else 0.0,
                    "incompatibility_reasons": json.dumps(reasons, sort_keys=True),
                    "schema_sha256": _schema_hash(columns),
                }
            )
    manifest = load_evidence_manifest(args.evidence_manifest)
    claim_rows: list[dict[str, Any]] = []
    for row in claims:
        contract_payload = row.get("contract") or row.get("drafted_contract")
        compatible_ids: list[str] = []
        primary_id = None
        error = None
        try:
            contract = ClaimContract.model_validate(contract_payload)
            if manifest is not None:
                compatible = manifest.external_sets_for_contract(contract)
                compatible_ids = [item.evidence_set_id for item in compatible]
                primary = manifest.primary_external_set_for_contract(contract)
                primary_id = primary.evidence_set_id if primary is not None else None
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        claim_rows.append(
            {
                "claim_id": row.get("claim_id"),
                "target_family": row.get("target_family"),
                "source_mode": row.get("source_mode"),
                "external_primary_available": primary_id is not None,
                "primary_external_evidence_set_id": primary_id,
                "compatible_external_evidence_set_ids": "|".join(compatible_ids),
                "coverage_error": error,
            }
        )
    by_target: dict[str, dict[str, int]] = {}
    for row in claim_rows:
        target = str(row.get("target_family") or "unknown")
        target_summary = by_target.setdefault(target, {"contracts": 0, "primary_external_covered": 0})
        target_summary["contracts"] += 1
        target_summary["primary_external_covered"] += int(bool(row["external_primary_available"]))

    audit_dir = Path(args.coverage_out_dir or (Path(args.out_root) / "audits"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    csv_path = audit_dir / "stage2_dataset_contract_coverage.csv"
    claims_csv_path = audit_dir / "stage2_external_contract_coverage.csv"
    json_path = audit_dir / "stage2_contract_coverage.json"
    pd.DataFrame(dataset_rows).to_csv(csv_path, index=False)
    pd.DataFrame(claim_rows).to_csv(claims_csv_path, index=False)
    output = {
        "stage2_results": str(args.stage2_results),
        "stage2_claim_count": len(claims),
        "claim_rows_written": len(claim_rows),
        "primary_external_covered_count": sum(
            int(bool(row["external_primary_available"])) for row in claim_rows
        ),
        "by_target_family": by_target,
        "evidence_manifest": str(args.evidence_manifest),
        "datasets": dataset_rows,
    }
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {claims_csv_path}")
    return output


def _aggregate_dataset(dataset: ExternalDatasetSpec, out_root: Path, subjects_root: Path) -> dict[str, Any]:
    manifest_path = out_root / "manifests" / f"{dataset.dataset_id}_subjects.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing structural manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = [SubjectManifestRow.model_validate(row) for row in payload.get("selected_rows", [])]
    features: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    complete_receipt_count = 0
    for index, row in enumerate(manifest_rows, start=1):
        subject_dir = _subject_dir(dataset, row, subjects_root)
        check = completion_check(subject_dir)
        if not check.complete:
            incomplete.append({"subject_id": row.subject_id, "subject_dir": str(subject_dir), "reason": check.reason})
            continue
        complete_receipt_count += 1
        receipt_path = subject_dir / ".confirm_complete.json"
        receipt = CompletionReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        output_dataset_id = f"{dataset.dataset_id}_sMRI"
        feature_row = {
            "subject_id": row.subject_id,
            "source_subject_id": row.source_subject_id,
            "session": row.session,
            "cohort": output_dataset_id,
            "site": row.site,
            "field_strength": pd.NA,
            "fs_version": receipt.freesurfer_version,
            "recon_engine": receipt.engine,
            "parcellation": "DKTatlas",
            "completion_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "source_t1_sha256": receipt.t1_sha256,
            "adopted_legacy_output": receipt.adopted_legacy_output,
            **canonical_features(subject_dir),
        }
        features.append(feature_row)
        if index == len(manifest_rows) or index % max(1, len(manifest_rows) // 20) == 0:
            print(f"[{dataset.dataset_id}] aggregate {index}/{len(manifest_rows)}", flush=True)

    feature_frame = pd.DataFrame(features)
    if len(feature_frame) != complete_receipt_count:
        raise ValueError(
            f"Aggregation row count {len(feature_frame)} does not match strict completion receipts "
            f"{complete_receipt_count}"
        )
    if feature_frame.empty:
        raise ValueError("No complete FreeSurfer subjects were available")
    metadata = load_metadata(dataset)
    if metadata.empty:
        merged = feature_frame.copy()
        merged["age"] = pd.NA
        merged["sex"] = pd.NA
        merged["dx"] = pd.NA
    else:
        merged = feature_frame.merge(metadata, on=["subject_id", "session"], how="left", suffixes=("", "_metadata"))
        for column in ("site",):
            metadata_column = f"{column}_metadata"
            if metadata_column in merged:
                merged[column] = merged[metadata_column].combine_first(merged[column])
                merged = merged.drop(columns=[metadata_column])

    destination = out_root / ("quarantine" if dataset.quarantine else "canonical")
    destination.mkdir(parents=True, exist_ok=True)
    if dataset.quarantine:
        prepared = merged
        status = "quarantined"
    else:
        missing_metadata = merged[["age", "sex"]].isna().any(axis=1)
        if missing_metadata.any():
            raise ValueError(f"{int(missing_metadata.sum())} complete subjects lack required age/sex metadata")
        if len(merged) != len(feature_frame):
            raise ValueError("metadata join changed the completed-subject row count")
        prepared = validate_canonical(merged)
        status = "ready"
    output_dataset_id = f"{dataset.dataset_id}_sMRI"
    output_path = destination / f"{output_dataset_id}.parquet"
    prepared.to_parquet(output_path, index=False)
    feature_manifest = {
        "dataset_id": dataset.dataset_id,
        "output_dataset_id": output_dataset_id,
        "status": status,
        "manifest_subject_count": len(manifest_rows),
        "complete_subject_count": len(feature_frame),
        "completion_receipt_count": complete_receipt_count,
        "output_row_count": len(prepared),
        "incomplete_subject_count": len(incomplete),
        "incomplete_subjects": incomplete,
        "volume_unit": "mm3",
        "thickness_unit": "mm",
        "canonical_volume_features": [
            "eTIV",
            "smri_hippocampus",
            "smri_entorhinal",
            "smri_fusiform",
            "smri_midtemp",
            "smri_ventricles",
            "smri_wholebrain",
        ],
        "schema_sha256": _schema_hash(prepared.columns),
        "output_path": str(output_path),
    }
    (destination / f"{output_dataset_id}.features.json").write_text(
        json.dumps(feature_manifest, indent=2), encoding="utf-8"
    )
    return feature_manifest


def _subject_dir(dataset: ExternalDatasetSpec, row: SubjectManifestRow, subjects_root: Path) -> Path:
    assert dataset.structural is not None
    if dataset.structural.existing_subjects_dir:
        legacy = Path(dataset.structural.existing_subjects_dir) / row.subject_id
        if completion_check(legacy, allow_legacy=True).complete:
            return legacy
    return subjects_root / dataset.dataset_id / row.subject_id


def _schema_supports_contract(columns: list[str], contract: ClaimContract) -> tuple[bool, str]:
    available = set(columns)
    required = {contract.estimand.predictor, *contract.covariates}
    if contract.estimand.group is not None:
        required.add(contract.estimand.group.var)
    if not required.issubset(available):
        return False, "missing_analysis_columns"
    outcomes = contract.estimand.outcome if isinstance(contract.estimand.outcome, list) else [contract.estimand.outcome]
    for outcome in outcomes:
        if any(token in outcome for token in "*?[") or outcome.endswith("_"):
            pattern = outcome if any(token in outcome for token in "*?[") else f"{outcome}*"
            if not any(fnmatch.fnmatch(column, pattern) for column in available):
                return False, "missing_outcome_family"
        elif outcome not in available:
            return False, "missing_outcome"
    return True, "supported"


def _schema_hash(columns: Any) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, columns))).encode("utf-8")).hexdigest()


def _prepared_base_dataset(dataset_id: str) -> str:
    for suffix in ("_fMRI", "_sMRI"):
        if dataset_id.endswith(suffix):
            return dataset_id[: -len(suffix)]
    return dataset_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "manifest", "fmri", "aggregate", "coverage"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset IDs or 'all'.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--subjects-root", default="/data/users1/ywei/confirm_external_prep/subjects")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--stage2-results",
        default="review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json",
    )
    parser.add_argument(
        "--evidence-manifest",
        default="data/prepared_data/evidence_partitions/manifest.json",
    )
    parser.add_argument("--coverage-out-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "audit": run_audit,
        "manifest": run_manifest,
        "fmri": run_fmri,
        "aggregate": run_aggregate,
        "coverage": run_coverage,
    }
    handlers[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
