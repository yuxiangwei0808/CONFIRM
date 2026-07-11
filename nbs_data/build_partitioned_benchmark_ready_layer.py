"""Build a benchmark-ready view that uses materialized evidence partitions."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from confirm.evidence_partitions import (
    EvidencePartitionManifest,
    canonical_base_cohort,
    load_evidence_manifest,
)


def _target_family_for_row(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("claim_id", "discovery_cohort", "replication_cohort", "outcome_family", "predictor_or_group")
    ).lower()
    if "adhd" in text:
        return "adhd"
    if "asd" in text or "abide" in text:
        return "asd"
    if any(term in text for term in ("adni", "oasis", "dementia", "hippocamp", "entorhinal", "gsp")):
        return "ad_aging"
    if any(term in text for term in ("sz", "cobre", "fbirn", "bsnip", "psychosis")):
        return "psychosis"
    return "normative_fmri"


def _record_for(manifest: EvidencePartitionManifest, cohort: str, target_family: str, role: str):
    base = canonical_base_cohort(cohort)
    for record in manifest.records:
        if record.base_dataset == base and record.target_family == target_family and record.role == role:
            return record
    return None


def _map_claim_inventory(inventory: pd.DataFrame, manifest: EvidencePartitionManifest) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for _, row in inventory.iterrows():
        target_family = _target_family_for_row(row)
        discovery = _record_for(manifest, str(row["discovery_cohort"]), target_family, "discovery")
        replication = _record_for(manifest, str(row["replication_cohort"]), target_family, "replication")
        if discovery is None or replication is None:
            skipped.append(
                {
                    "claim_id": str(row.get("claim_id")),
                    "reason": "missing discovery/replication partition",
                    "target_family": target_family,
                    "discovery_cohort": str(row.get("discovery_cohort")),
                    "replication_cohort": str(row.get("replication_cohort")),
                }
            )
            continue
        item = row.to_dict()
        item["discovery_cohort"] = discovery.partition_id
        item["replication_cohort"] = replication.partition_id
        item["evidence_target_family"] = target_family
        item["source_discovery_cohort"] = str(row["discovery_cohort"])
        item["source_replication_cohort"] = str(row["replication_cohort"])
        rows.append(item)
    return pd.DataFrame(rows), skipped


def _write_feature_dictionary(feature_dictionary: pd.DataFrame, manifest: EvidencePartitionManifest, out_path: Path) -> None:
    records = [record for record in manifest.records if record.role in {"discovery", "replication"}]
    rows: list[pd.DataFrame] = []
    for record in records:
        subset = feature_dictionary[feature_dictionary["cohort"].astype(str) == record.base_dataset].copy()
        if subset.empty:
            continue
        subset["cohort"] = record.partition_id
        subset["source_cohort"] = record.base_dataset
        rows.append(subset)
    payload = pd.concat(rows, ignore_index=True, sort=False) if rows else feature_dictionary.iloc[0:0].copy()
    payload.to_csv(out_path, index=False)


def _copy_partition_cohorts(manifest: EvidencePartitionManifest, out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for record in manifest.records:
        if record.role not in {"discovery", "replication"}:
            continue
        source = Path(record.path)
        target = out_dir / f"{record.partition_id}.parquet"
        shutil.copyfile(source, target)
        copied.append(
            {
                "partition_id": record.partition_id,
                "role": record.role,
                "target_family": record.target_family,
                "path": str(target),
            }
        )
    return copied


def run(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_root = Path(args.benchmark_root)
    out_root = Path(args.out_root)
    manifest = load_evidence_manifest(args.evidence_manifest)
    if manifest is None:
        raise FileNotFoundError(f"Evidence manifest not found: {args.evidence_manifest}")

    out_root.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(benchmark_root / "claim_inventory_ready.csv")
    feature_dictionary = pd.read_csv(benchmark_root / "feature_dictionary.csv")

    mapped_inventory, skipped = _map_claim_inventory(inventory, manifest)
    mapped_inventory.to_csv(out_root / "claim_inventory_ready.csv", index=False)
    _write_feature_dictionary(feature_dictionary, manifest, out_root / "feature_dictionary.csv")
    copied = _copy_partition_cohorts(manifest, out_root / "cohorts")

    for optional_name in ("README.md", "cohort_manifest.csv", "misc_table_manifest.csv"):
        source = benchmark_root / optional_name
        if source.exists():
            shutil.copyfile(source, out_root / optional_name)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_root": str(benchmark_root),
        "evidence_manifest": str(args.evidence_manifest),
        "out_root": str(out_root),
        "claim_count": int(len(mapped_inventory)),
        "skipped_claim_count": len(skipped),
        "copied_partition_count": len(copied),
        "skipped": skipped,
        "copied_partitions": copied,
    }
    (out_root / "partitioned_layer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", default="data/prepared_data/benchmark_ready")
    parser.add_argument("--evidence-manifest", default="data/prepared_data/evidence_partitions/manifest.json")
    parser.add_argument("--out-root", default="data/prepared_data/evidence_partitions/benchmark_ready")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
