"""Summarize claim-search matrix replay artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_artifact(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    skipped = payload.get("skipped") if isinstance(payload.get("skipped"), list) else []
    n_searches = int(summary.get("n_searches", 0) or 0)
    searchable_claim_count = int(
        payload.get("searchable_claim_count")
        or summary.get("searchable_claim_count")
        or (n_searches + len(skipped))
    )
    valid_connected_lineage_count = int(summary.get("valid_connected_lineage_count", 0) or 0)
    source = (payload.get("provenance") or {}).get("source") or {}
    return {
        "artifact": str(path),
        "status": payload.get("status"),
        "source_sha256": source.get("sha256"),
        "max_rounds": config.get("max_rounds"),
        "max_candidates_per_round": config.get("max_candidates_per_round"),
        "llm_schema_retries": config.get("llm_schema_retries"),
        "llm_model": payload.get("llm_model"),
        "searchable_claim_count": searchable_claim_count,
        "completed_search_count": int(payload.get("completed_search_count") or n_searches),
        "skipped_search_count": int(payload.get("skipped_search_count") or len(skipped)),
        "n_searches": n_searches,
        "candidate_count": summary.get("candidate_count", 0),
        "generated_candidate_count": summary.get("generated_candidate_count", 0),
        "unretained_generated_candidate_count": summary.get("unretained_generated_candidate_count", 0),
        "unique_candidate_count": summary.get("unique_candidate_count", 0),
        "duplicate_candidate_count": summary.get("duplicate_candidate_count", 0),
        "valid_connected_candidate_count": summary.get("valid_connected_candidate_count", 0),
        "valid_connected_candidate_rate": summary.get("valid_connected_candidate_rate", 0.0),
        "valid_connected_executable_candidate_count": summary.get(
            "valid_connected_executable_candidate_count",
            0,
        ),
        "valid_connected_executable_candidate_rate": summary.get(
            "valid_connected_executable_candidate_rate",
            0.0,
        ),
        "valid_connected_lineage_count": valid_connected_lineage_count,
        "valid_connected_completed_lineage_rate": summary.get("valid_connected_lineage_rate", 0.0),
        "valid_connected_lineage_rate": (
            valid_connected_lineage_count / searchable_claim_count
            if searchable_claim_count
            else 0.0
        ),
        "preflight_pass_candidate_count": summary.get("preflight_pass_candidate_count", 0),
        "preflight_pass_candidate_rate": summary.get("preflight_pass_candidate_rate", 0.0),
        "preflight_block_count": summary.get("preflight_block_count", 0),
        "admissible_evaluation_count": summary.get("admissible_evaluation_count", 0),
        "current_data_evaluated_count": summary.get("current_data_evaluated_count", 0),
        "exploratory_confirmed_count": summary.get("exploratory_confirmed_count", 0),
        "same_data_exploratory_confirmed_count": summary.get("same_data_exploratory_confirmed_count", 0),
        "confirmed_count": summary.get("confirmed_count", 0),
        "final_confirmed_count": summary.get("final_confirmed_count", summary.get("confirmed_count", 0)),
        "supported_candidate_count": summary.get("supported_candidate_count", summary.get("any_supported_candidate_count", 0)),
        "any_supported_candidate_count": summary.get("any_supported_candidate_count", 0),
        "contract_repair_confirmed_count": summary.get("contract_repair_confirmed_count", 0),
        "holdout_confirmed_count": summary.get("holdout_confirmed_count", 0),
        "external_confirmed_count": summary.get("external_confirmed_count", 0),
        "confirmed_on_external_evidence_count": summary.get("confirmed_on_external_evidence_count", 0),
        "confirmed_on_excluded_evidence_count": summary.get("confirmed_on_excluded_evidence_count", 0),
        "false_current_data_confirmation_count": summary.get("false_current_data_confirmation_count", 0),
        "known_negative_or_fragile_search_count": summary.get("known_negative_or_fragile_search_count", 0),
        "known_negative_or_fragile_exploratory_confirmed_count": summary.get(
            "known_negative_or_fragile_exploratory_confirmed_count",
            0,
        ),
        "known_negative_same_data_exploratory_confirmed_count": summary.get(
            "known_negative_same_data_exploratory_confirmed_count",
            0,
        ),
        "known_negative_exploratory_risk_rate": summary.get("known_negative_exploratory_risk_rate", 0.0),
        "hacking_block_count": summary.get("hacking_block_count", 0),
        "no_holdout_abstention_count": summary.get("no_holdout_abstention_count", 0),
        "execution_error_count": summary.get("execution_error_count", 0),
        "execution_error_type_counts": json.dumps(summary.get("execution_error_type_counts", {}), sort_keys=True),
        "excluded_evidence_error_count": summary.get("excluded_evidence_error_count", 0),
        "excluded_evidence_unavailable_count": summary.get("excluded_evidence_unavailable_count", 0),
        "excluded_evidence_query_count": summary.get("excluded_evidence_query_count", 0),
        "analysis_non_identifiable_count": summary.get("analysis_non_identifiable_count", 0),
        "excluded_evidence_error_type_counts": json.dumps(
            summary.get("excluded_evidence_error_type_counts", {}),
            sort_keys=True,
        ),
        "raw_final_label_counts": json.dumps(summary.get("raw_final_label_counts", {}), sort_keys=True),
        "effective_final_label_counts": json.dumps(summary.get("effective_final_label_counts", {}), sort_keys=True),
        "final_label_counts": json.dumps(summary.get("final_label_counts", {}), sort_keys=True),
        "stopped_reason_counts": json.dumps(summary.get("stopped_reason_counts", {}), sort_keys=True),
        "searches_by_target_family": json.dumps(summary.get("searches_by_target_family", {}), sort_keys=True),
        "searches_by_source_mode": json.dumps(summary.get("searches_by_source_mode", {}), sort_keys=True),
        "raw_candidate_final_label_counts_by_target_family": json.dumps(
            summary.get("raw_candidate_final_label_counts_by_target_family", {}),
            sort_keys=True,
        ),
        "effective_candidate_final_label_counts_by_target_family": json.dumps(
            summary.get("effective_candidate_final_label_counts_by_target_family", {}),
            sort_keys=True,
        ),
        "candidate_final_label_counts_by_target_family": json.dumps(
            summary.get("candidate_final_label_counts_by_target_family", {}),
            sort_keys=True,
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.out_root)
    artifacts = sorted(root.glob("matrix/rounds_*/candidates_*/iterative_candidate_replay.json"))
    rows = [_row_from_artifact(path) for path in artifacts]
    _validate_matrix_rows(
        rows,
        expected_rounds=getattr(args, "expected_rounds", None),
        expected_candidates=getattr(args, "expected_candidates", None),
    )
    queried_arms = [row for row in rows if int(row.get("excluded_evidence_query_count") or 0) != 0]
    if queried_arms and not args.allow_excluded_queries:
        raise ValueError("Sweep artifacts queried excluded evidence; canonical selection is invalid.")
    selected = _select_canonical_config(rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(root),
        "artifact_count": len(artifacts),
        "interpretation": "One stochastic descriptive realization; not a causal hyperparameter comparison.",
        "selection_rule": (
            "Maximum valid-connected-executable lineage coverage; choose the smallest rounds*candidates "
            "arm within one percentage point, then fewer rounds, then fewer candidates."
        ),
        "selected_config": selected,
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "matrix_summary.json"
    csv_path = root / "matrix_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    selected_path = root / "selected_config.json"
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {selected_path}")
    return summary


def _parse_expected_values(value: str | None) -> list[int]:
    if not value:
        return []
    return sorted({int(item) for item in value.replace(",", " ").split()})


def _validate_matrix_rows(
    rows: list[dict[str, Any]],
    *,
    expected_rounds: str | None,
    expected_candidates: str | None,
) -> None:
    if not rows:
        raise ValueError("No completed matrix artifacts were found.")
    incomplete = [row["artifact"] for row in rows if row.get("status") != "completed"]
    if incomplete:
        raise ValueError(f"Matrix contains incomplete artifacts: {incomplete}")
    configurations = [
        (int(row["max_rounds"]), int(row["max_candidates_per_round"]))
        for row in rows
    ]
    if len(configurations) != len(set(configurations)):
        raise ValueError("Matrix contains duplicate hyperparameter configurations.")
    source_hashes = {row.get("source_sha256") for row in rows}
    if None in source_hashes or len(source_hashes) != 1:
        raise ValueError("Matrix arms must record and share one source SHA-256 hash.")
    source_counts = {int(row.get("searchable_claim_count") or 0) for row in rows}
    if 0 in source_counts or len(source_counts) != 1:
        raise ValueError("Matrix arms must use the same nonzero searchable claim count.")
    models = {row.get("llm_model") for row in rows}
    if None in models or len(models) != 1:
        raise ValueError("Matrix arms must use one recorded LLM model.")

    rounds = _parse_expected_values(expected_rounds)
    candidates = _parse_expected_values(expected_candidates)
    if bool(rounds) != bool(candidates):
        raise ValueError("Expected rounds and candidates must be supplied together.")
    if rounds:
        expected = {(round_count, candidate_count) for round_count in rounds for candidate_count in candidates}
        observed = set(configurations)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                f"Matrix grid is incomplete or unexpected: missing={missing}, unexpected={unexpected}"
            )


def _select_canonical_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No completed matrix artifacts were found.")
    maximum = max(float(row.get("valid_connected_lineage_rate") or 0.0) for row in rows)
    threshold = max(0.0, maximum - 0.01)
    eligible = [
        row
        for row in rows
        if float(row.get("valid_connected_lineage_rate") or 0.0) >= threshold
    ]
    chosen = min(
        eligible,
        key=lambda row: (
            int(row["max_rounds"]) * int(row["max_candidates_per_round"]),
            int(row["max_rounds"]),
            int(row["max_candidates_per_round"]),
        ),
    )
    return {
        "max_rounds": int(chosen["max_rounds"]),
        "max_candidates_per_round": int(chosen["max_candidates_per_round"]),
        "valid_connected_executable_lineage_coverage": float(chosen["valid_connected_lineage_rate"]),
        "maximum_coverage": maximum,
        "within_one_percentage_point_threshold": threshold,
        "selected_artifact": chosen["artifact"],
        "selection_used_support_counts": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--allow-excluded-queries",
        action="store_true",
        help="Allow canonical-run artifacts that intentionally queried excluded evidence.",
    )
    parser.add_argument("--expected-rounds", help="Whitespace- or comma-separated expected max-round values.")
    parser.add_argument("--expected-candidates", help="Whitespace- or comma-separated expected candidate-count values.")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
