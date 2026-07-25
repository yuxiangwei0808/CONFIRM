from __future__ import annotations

from nbs.analyze_feedback_method_baselines import (
    _evidence_parent_cells,
    _evidence_summary,
    _paired_cells,
    _summary_from_rows,
)


def test_parent_pair_cells_use_all_matched_lineages() -> None:
    reference = [
        {"claim_id": "a", "parent_with_internal_support": True},
        {"claim_id": "b", "parent_with_internal_support": False},
        {"claim_id": "c", "parent_with_internal_support": True},
    ]
    comparison = [
        {"claim_id": "a", "parent_with_internal_support": True},
        {"claim_id": "b", "parent_with_internal_support": True},
        {"claim_id": "c", "parent_with_internal_support": False},
    ]
    result = _paired_cells(
        reference,
        comparison,
        comparison_method="self_refine",
    )
    assert result["paired_parent_count"] == 3
    assert result["both"] == 1
    assert result["failure_specific_only"] == 1
    assert result["comparison_only"] == 1


def test_evidence_summary_keeps_external_sets_separate() -> None:
    rows = [
        {
            "method": "self_refine",
            "track": "scientific",
            "parent_claim_id": "a",
            "evidence_kind": "external",
            "evidence_set_id": "NACC",
            "compatible": 1,
            "evaluated": 1,
            "supported": 1,
            "execution_error": 0,
        },
        {
            "method": "self_refine",
            "track": "scientific",
            "parent_claim_id": "b",
            "evidence_kind": "external",
            "evidence_set_id": "CNP",
            "compatible": 1,
            "evaluated": 1,
            "supported": 0,
            "execution_error": 0,
        },
    ]
    summary = _evidence_summary(rows)
    assert {row["evidence_set_id"] for row in summary} == {"NACC", "CNP"}
    assert len(summary) == 2


def test_unavailable_external_rows_are_not_labeled_internal_holdout() -> None:
    summary = _evidence_summary(
        [
            {
                "method": "self_refine",
                "track": "scientific",
                "parent_claim_id": "a",
                "evidence_kind": "external",
                "evidence_set_id": None,
                "compatible": 0,
                "evaluated": 0,
                "supported": 0,
                "execution_error": 0,
            }
        ]
    )
    assert summary[0]["evidence_set_id"] == "external_unavailable"


def test_evidence_parent_cells_preserve_full_parent_denominator() -> None:
    reference = [
        {
            "parent_claim_id": "a",
            "evidence_kind": "holdout",
            "evidence_set_id": None,
            "evaluated": 1,
            "supported": 1,
        }
    ]
    comparison = [
        {
            "parent_claim_id": "b",
            "evidence_kind": "holdout",
            "evidence_set_id": None,
            "evaluated": 1,
            "supported": 1,
        }
    ]
    result = _evidence_parent_cells(
        reference,
        comparison,
        comparison_method="failure_blind",
        evidence_kind="holdout",
        evidence_set_id="internal_holdout",
        parent_ids={"a", "b", "c"},
    )
    assert result["paired_parent_count"] == 3
    assert result["failure_specific_only"] == 1
    assert result["comparison_only"] == 1
    assert result["neither"] == 1
    assert result["both_evaluated_parent_count"] == 0


def test_feedback_summary_distinguishes_retry_attempts_from_retained_candidates() -> None:
    summary = _summary_from_rows(
        method="self_refine",
        track="scientific",
        rows=[
            {
                "generated_candidate_count": 10,
                "unique_candidate_count": 4,
                "valid_candidate_count": 3,
                "current_data_evaluated_count": 3,
                "supported_candidate_count": 1,
                "parent_with_internal_support": True,
            }
        ],
        llm_calls=4,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        reported_cost=None,
        budget="R3/C5",
    )
    assert summary["returned_proposal_attempt_count"] == 10
    assert summary["retained_unique_candidate_count"] == 4
