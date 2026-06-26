"""Combine benchmark result JSON files into one label-aware summary."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bench.labels import claim_label_for_claim, label_authority
from bench.metrics import summarize_rows
from bench.run_benchmark_ready import RUNGS, _json_safe


def _ensure_label_authority(row: dict[str, Any]) -> None:
    value = str(row.get("label_authority", "") or "").strip().lower()
    if value in {"main", "supplementary"}:
        row["label_authority"] = value
        return
    metadata = row.get("label_metadata")
    if isinstance(metadata, dict) and metadata:
        row["label_authority"] = label_authority({key: str(value) for key, value in metadata.items()})
        return
    label_row = claim_label_for_claim({"claim_id": row.get("claim_id", "")}) or {}
    row["label_authority"] = label_row.get("label_authority", "supplementary")


def _read_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("claims", [])
    for row in rows:
        _ensure_label_authority(row)
        row["source_results"] = str(path)
    payload["claims"] = rows
    return payload


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"combined_benchmark_results_{timestamp}.json"
    audit_path = out_dir / f"combined_benchmark_audit_{timestamp}.csv"
    risk_path = out_dir / f"combined_benchmark_risk_coverage_{timestamp}.csv"
    latest_json = out_dir / "combined_benchmark_results.json"
    latest_audit = out_dir / "combined_benchmark_audit.csv"
    latest_risk = out_dir / "combined_benchmark_risk_coverage.csv"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    audit_cols = [
        "claim_id",
        "modality",
        "scoring_label",
        "scoring_bucket",
        "label_authority",
        "final_label",
        "confirmation_subtype",
        "heterogeneity_i2",
        *RUNGS,
        "n_discovery",
        "n_replication",
        "n_features",
        "best_region",
        "best_beta",
        "best_p",
        "best_standardized_effect",
        "multiverse_fraction_consistent",
        "source_results",
        "rationale",
    ]
    audit = pd.DataFrame(payload["claims"])
    for col in audit_cols:
        if col not in audit.columns:
            audit[col] = None
    audit[audit_cols].to_csv(audit_path, index=False)
    pd.DataFrame(payload.get("risk_coverage", [])).to_csv(risk_path, index=False)
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(audit_path, latest_audit)
    shutil.copyfile(risk_path, latest_risk)
    return json_path, audit_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    sources = [Path(path) for path in args.input]
    for source in sources:
        payload = _read_payload(source)
        rows.extend(payload.get("claims", []))
        errors.extend(payload.get("errors", []))
        skipped.extend(payload.get("skipped", []))
    metrics = summarize_rows(rows, RUNGS)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [str(path) for path in sources],
        **metrics,
        "metrics_exact_ci": metrics.get("summary", {}),
        "claims": rows,
        "errors": errors,
        "skipped": skipped,
    }
    json_path, audit_path = write_outputs(Path(args.out_dir), payload)
    print(f"wrote {json_path}")
    print(f"wrote {audit_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Benchmark result JSON. May repeat.")
    parser.add_argument("--out-dir", default="review-stage/combined-label-aware")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
