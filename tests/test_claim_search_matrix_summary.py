from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "nbs" / "summarize_claim_search_matrix.py"
_SPEC = importlib.util.spec_from_file_location("claim_search_matrix_summary", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _artifact(root: Path, rounds: int, candidates: int, coverage: float, excluded_queries: int = 0) -> None:
    path = root / "matrix" / f"rounds_{rounds}" / f"candidates_{candidates}" / "iterative_candidate_replay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "config": {"max_rounds": rounds, "max_candidates_per_round": candidates},
                "llm_model": "openai:gpt-5.5",
                "searchable_claim_count": 1000,
                "completed_search_count": 1000,
                "skipped_search_count": 0,
                "provenance": {
                    "source": {"sha256": "same-source"},
                    "prompt_sha256": "prompt",
                    "schema_sha256": "schema",
                    "implementation_hashes": {"src/example.py": "implementation"},
                    "evidence_manifest": {"sha256": "manifest"},
                    "partition_hashes_sha256": "partitions",
                },
                "summary": {
                    "n_searches": 1000,
                    "valid_connected_lineage_count": int(coverage * 1000),
                    "valid_connected_lineage_rate": coverage,
                    "proposals_returned_count": 1200,
                    "schema_valid_candidate_count": 1190,
                    "policy_valid_candidate_count": 900,
                    "unique_source_tested_count": 850,
                    "execution_complete_candidate_count": 840,
                    "unique_internally_supported_contract_count": 12,
                    "contract_repair_supported_count": 3,
                    "contract_repair_confirmed_count": 3,
                    "excluded_evidence_query_count": excluded_queries,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_matrix_summary_reports_arms_without_selecting_a_winner(tmp_path):
    _artifact(tmp_path, 1, 2, 0.80)
    _artifact(tmp_path, 3, 2, 0.895)
    _artifact(tmp_path, 1, 10, 0.90)

    result = _MODULE.run(
        Namespace(
            out_root=str(tmp_path),
            allow_excluded_queries=False,
            expected_rounds=None,
            expected_candidates=None,
        )
    )
    assert result["selection_rule"] is None
    assert len(result["rows"]) == 3
    assert result["rows"][0]["proposals_returned_count"] == 1200
    assert result["rows"][0]["unique_source_tested_count"] == 850
    assert result["rows"][0]["contract_repair_supported_count"] == 3
    assert result["deprecated_metric_aliases"]["contract_repair_confirmed_count"] == (
        "contract_repair_supported_count"
    )
    assert not (tmp_path / "selected_config.json").exists()


def test_sweep_summary_rejects_any_excluded_query(tmp_path):
    _artifact(tmp_path, 1, 2, 0.8, excluded_queries=1)

    with pytest.raises(ValueError, match="queried excluded evidence"):
        _MODULE.run(
            Namespace(
                out_root=str(tmp_path),
                allow_excluded_queries=False,
                expected_rounds=None,
                expected_candidates=None,
            )
        )


def test_matrix_summary_rejects_missing_grid_cell(tmp_path):
    _artifact(tmp_path, 1, 2, 0.8)

    with pytest.raises(ValueError, match="grid is incomplete"):
        _MODULE.run(
            Namespace(
                out_root=str(tmp_path),
                allow_excluded_queries=False,
                expected_rounds="1 3",
                expected_candidates="2",
            )
        )


def test_matrix_coverage_denominator_includes_skipped_lineages(tmp_path):
    _artifact(tmp_path, 1, 2, 0.9)
    path = tmp_path / "matrix" / "rounds_1" / "candidates_2" / "iterative_candidate_replay.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completed_search_count"] = 900
    payload["skipped_search_count"] = 100
    payload["summary"]["n_searches"] = 900
    payload["summary"]["valid_connected_lineage_count"] = 810
    payload["summary"]["valid_connected_lineage_rate"] = 0.9
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    result = _MODULE.run(
        Namespace(
            out_root=str(tmp_path),
            allow_excluded_queries=False,
            expected_rounds="1",
            expected_candidates="2",
        )
    )

    assert result["rows"][0]["valid_connected_completed_lineage_rate"] == 0.9
    assert result["rows"][0]["valid_connected_lineage_rate"] == 0.81


def test_matrix_artifact_reader_does_not_parse_large_result_tail(tmp_path):
    path = tmp_path / "iterative_candidate_replay.json"
    path.write_text(
        """{
  \"status\": \"completed\",
  \"llm_model\": \"openai:gpt-5.5\",
  \"config\": {\"max_rounds\": 1, \"max_candidates_per_round\": 2},
  \"provenance\": {\"source\": {\"sha256\": \"source\"}, \"prompt_sha256\": \"prompt\", \"schema_sha256\": \"schema\", \"implementation_hashes\": {\"x\": \"y\"}, \"evidence_manifest\": {\"sha256\": \"manifest\"}, \"partition_hashes_sha256\": \"partitions\"},
  \"completed_search_count\": 215,
  \"summary\": {\"n_searches\": 215, \"excluded_evidence_query_count\": 0},
  \"rows\": [THIS TAIL IS INTENTIONALLY NOT JSON
""",
        encoding="utf-8",
    )

    row = _MODULE._row_from_artifact(path)

    assert row["completed_search_count"] == 215
    assert row["max_candidates_per_round"] == 2
