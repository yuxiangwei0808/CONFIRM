"""Evaluate LLM-drafted CONFIRM claim contracts with unchanged gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.progress import iter_progress
from confirm.agent import _execute_contract
from confirm.contract import ClaimContract
from confirm.evidence_partitions import is_excluded_evidence_cohort
from confirm.verdict import Verdict

DEFAULT_DATA_ROOTS = (
    Path("data/prepared_data/evidence_partitions/benchmark_ready/cohorts"),
)


def _json_safe(data: Any) -> Any:
    if hasattr(data, "to_dict"):
        return _json_safe(data.to_dict())
    if hasattr(data, "model_dump"):
        return _json_safe(data.model_dump(mode="json"))
    if data.__class__.__module__.startswith("numpy") and hasattr(data, "item"):
        return _json_safe(data.item())
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_safe(value) for value in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def _execution_root(contract: ClaimContract, data_roots: list[Path]) -> Path:
    needed = [contract.discovery_cohort, *contract.replication_cohorts]
    for root in data_roots:
        if all((root / f"{cohort}.parquet").exists() for cohort in needed):
            return root
    raise FileNotFoundError(f"No data root contains all contract cohorts: {needed}")


def _gate_row(source_row: dict[str, Any], contract: ClaimContract, verdict: Verdict, results: dict[str, Any], cohort_paths: list[Path]) -> dict[str, Any]:
    gate_results = _json_safe(results)
    return {
        "claim_id": source_row.get("claim_id") or contract.claim_id,
        "target_family": source_row.get("target_family"),
        "source_mode": source_row.get("source_mode"),
        "model_spec": source_row.get("model_spec"),
        "question": source_row.get("question") or contract.question,
        "label_class": source_row.get("label_class"),
        "label_basis": source_row.get("label_basis"),
        "ground_truth": source_row.get("label_class"),
        "scoring_label": source_row.get("label_class"),
        "source_citation": source_row.get("source_citation"),
        "notes": source_row.get("notes"),
        "draft_success": True,
        "estimand_match": True,
        "gate_success": True,
        "gate_verdict_label": verdict.label,
        "gate_verdict": verdict.to_dict(),
        "final_label": verdict.label,
        "abstained": verdict.abstained,
        "rationale": verdict.rationale,
        "contract": contract.model_dump(mode="json"),
        "drafted_contract": contract.model_dump(mode="json"),
        "gate_results": gate_results,
        "source_results": gate_results,
        "cohort_paths": [str(path) for path in cohort_paths],
    }


def _error_row(source_row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "claim_id": source_row.get("claim_id"),
        "target_family": source_row.get("target_family"),
        "source_mode": source_row.get("source_mode"),
        "model_spec": source_row.get("model_spec"),
        "question": source_row.get("question"),
        "draft_success": bool(source_row.get("draft_success")),
        "gate_success": False,
        "error_stage": "gate_execution",
        "error": str(exc),
    }


def _evaluate_one(source_row: dict[str, Any], data_roots: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        payload = source_row.get("drafted_contract")
        if not isinstance(payload, dict):
            raise ValueError("row has no drafted_contract object")
        contract = ClaimContract.model_validate(payload)
        excluded_cohorts = [
            cohort
            for cohort in [contract.discovery_cohort, *contract.replication_cohorts]
            if is_excluded_evidence_cohort(cohort)
        ]
        if excluded_cohorts:
            raise ValueError(f"Stage 2 cannot evaluate excluded evidence cohorts: {excluded_cohorts}")
        roots = [Path(item) for item in data_roots]
        root = _execution_root(contract, roots)
        verdict, results, cohort_paths = _execute_contract(contract, root, ref_effect=contract.gates.power.ref_effect)
        return _gate_row(source_row, contract, verdict, results, cohort_paths), None
    except Exception as exc:
        return None, _error_row(source_row, exc)


def _run_serial(rows: list[dict[str, Any]], data_roots: list[str], progress: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in iter_progress(rows, total=len(rows), desc="gate evaluation", enabled=progress, unit="claim"):
        claim, error = _evaluate_one(row, data_roots)
        if claim is not None:
            claims.append(claim)
        if error is not None:
            errors.append(error)
    return claims, errors


def _run_parallel(
    rows: list[dict[str, Any]],
    data_roots: list[str],
    *,
    max_workers: int,
    backend: str,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    claims: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with executor_cls(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_one, row, data_roots): i for i, row in enumerate(rows)}
        ordered: dict[int, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
        for future in iter_progress(as_completed(futures), total=len(futures), desc="gate evaluation", enabled=progress, unit="claim"):
            ordered[futures[future]] = future.result()
    for index in range(len(rows)):
        claim, error = ordered[index]
        if claim is not None:
            claims.append(claim)
        if error is not None:
            errors.append(error)
    return claims, errors


def _summary(claims: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("final_label")) for row in claims)
    source_modes = Counter(str(row.get("source_mode")) for row in claims)
    target_families = Counter(str(row.get("target_family")) for row in claims)
    return {
        "n_claims": len(claims),
        "n_errors": len(errors),
        "final_label_counts": dict(labels),
        "source_mode_counts": dict(source_modes),
        "target_family_counts": dict(target_families),
    }


def _write_audit_csv(path: Path, claims: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    fieldnames = [
        "claim_id",
        "target_family",
        "source_mode",
        "model_spec",
        "final_label",
        "gate_verdict_label",
        "draft_success",
        "gate_success",
        "error_stage",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in claims:
            writer.writerow({key: row.get(key) for key in fieldnames})
        for row in errors:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_claims_csv(path: Path, claims: list[dict[str, Any]]) -> None:
    fieldnames = ["claim_id", "target_family", "source_mode", "model_spec", "question", "final_label", "label_class", "label_basis"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in claims:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(Path(args.contracts))
    if args.limit is not None:
        rows = rows[: args.limit]
    data_roots = [str(Path(item)) for item in (args.data_root or [str(path) for path in DEFAULT_DATA_ROOTS])]
    if args.max_workers <= 1:
        claims, errors = _run_serial(rows, data_roots, not args.no_progress)
    else:
        claims, errors = _run_parallel(
            rows,
            data_roots,
            max_workers=args.max_workers,
            backend=args.parallel_backend,
            progress=not args.no_progress,
        )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [{"path": str(args.contracts), "kind": "drafted_contracts_jsonl"}],
        "command": {
            "contracts": str(args.contracts),
            "out_dir": str(args.out_dir),
            "data_root": data_roots,
            "max_workers": args.max_workers,
            "parallel_backend": args.parallel_backend,
        },
        **_summary(claims, errors),
        "claims": claims,
        "errors": errors,
        "skipped": [],
    }
    (out_dir / "combined_benchmark_results.json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(_summary(claims, errors)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_audit_csv(out_dir / "claim_gate_audit.csv", claims, errors)
    _write_claims_csv(out_dir / "claims.csv", claims)
    print(f"wrote {out_dir / 'combined_benchmark_results.json'}")
    print(f"wrote {out_dir / 'claim_gate_audit.csv'}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contracts", default="review-stage/initial-claims-all-gpt55/drafted_contracts.jsonl")
    parser.add_argument("--out-dir", default="review-stage/confirm-gates-all-gpt55")
    parser.add_argument("--data-root", action="append", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
