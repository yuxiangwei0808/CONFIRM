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
                "provenance": {"source": {"sha256": "same-source"}},
                "summary": {
                    "n_searches": 1000,
                    "valid_connected_lineage_count": int(coverage * 1000),
                    "valid_connected_lineage_rate": coverage,
                    "excluded_evidence_query_count": excluded_queries,
                },
            }
        ),
        encoding="utf-8",
    )


def test_matrix_selection_uses_smallest_arm_within_one_percentage_point(tmp_path):
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
    selected = result["selected_config"]

    assert selected["maximum_coverage"] == 0.90
    assert selected["max_rounds"] == 3
    assert selected["max_candidates_per_round"] == 2
    assert selected["selection_used_support_counts"] is False
    assert json.loads((tmp_path / "selected_config.json").read_text())["max_rounds"] == 3


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
    path.write_text(json.dumps(payload), encoding="utf-8")

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
