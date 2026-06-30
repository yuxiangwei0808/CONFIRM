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
    return {
        "artifact": str(path),
        "max_rounds": config.get("max_rounds"),
        "max_candidates_per_round": config.get("max_candidates_per_round"),
        "llm_schema_retries": config.get("llm_schema_retries"),
        "llm_model": payload.get("llm_model"),
        "n_searches": summary.get("n_searches", 0),
        "candidate_count": summary.get("candidate_count", 0),
        "valid_connected_candidate_count": summary.get("valid_connected_candidate_count", 0),
        "valid_connected_candidate_rate": summary.get("valid_connected_candidate_rate", 0.0),
        "preflight_pass_candidate_count": summary.get("preflight_pass_candidate_count", 0),
        "preflight_pass_candidate_rate": summary.get("preflight_pass_candidate_rate", 0.0),
        "preflight_block_count": summary.get("preflight_block_count", 0),
        "admissible_evaluation_count": summary.get("admissible_evaluation_count", 0),
        "exploratory_confirmed_count": summary.get("exploratory_confirmed_count", 0),
        "confirmed_count": summary.get("confirmed_count", 0),
        "confirmed_on_external_evidence_count": summary.get("confirmed_on_external_evidence_count", 0),
        "confirmed_on_excluded_evidence_count": summary.get("confirmed_on_excluded_evidence_count", 0),
        "false_current_data_confirmation_count": summary.get("false_current_data_confirmation_count", 0),
        "known_negative_or_fragile_search_count": summary.get("known_negative_or_fragile_search_count", 0),
        "known_negative_or_fragile_exploratory_confirmed_count": summary.get(
            "known_negative_or_fragile_exploratory_confirmed_count",
            0,
        ),
        "hacking_block_count": summary.get("hacking_block_count", 0),
        "no_holdout_abstention_count": summary.get("no_holdout_abstention_count", 0),
        "execution_error_count": summary.get("execution_error_count", 0),
        "final_label_counts": json.dumps(summary.get("final_label_counts", {}), sort_keys=True),
        "stopped_reason_counts": json.dumps(summary.get("stopped_reason_counts", {}), sort_keys=True),
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
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(root),
        "artifact_count": len(artifacts),
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "matrix_summary.json"
    csv_path = root / "matrix_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
