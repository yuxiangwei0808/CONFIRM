from __future__ import annotations

import json

import pytest

from nbs.summarize_claim_search_control import preflight_current_implementation, run


def _write_arm(
    root,
    mode: str,
    *,
    source_hash: str = "source",
    search_hash: str = "same",
    freezer_hash: str = "same-freezer",
    evidence_partition_hash: str = "same-partitions-code",
    search_implementation_hash: str | None = None,
) -> None:
    arm = root / mode
    arm.mkdir(parents=True)
    payload = {
        "status": "completed",
        "llm_model": "openai:gpt-5.5",
        "completed_search_count": 2,
        "config": {
            "max_rounds": 3,
            "max_candidates_per_round": 5,
            "llm_schema_retries": 2,
            "feedback_mode": mode,
        },
        "provenance": {
            "source": {"sha256": source_hash},
            "schema_sha256": "schema",
            "implementation_hashes": {
                "src/example.py": search_hash,
                "src/confirm/frozen_evidence.py": freezer_hash,
                "src/confirm/evidence_partitions.py": evidence_partition_hash,
            },
            "search_implementation_hashes_sha256": search_implementation_hash,
            "partition_hashes_sha256": "partitions",
            "evidence_manifest": {"sha256": "manifest"},
        },
        "summary": {"n_searches": 2},
        "rows": [
            {
                "claim_id": "claim_a",
                "target_family": "adhd",
                "source_mode": "llm_proposed",
                "generated_candidate_count": 3 if mode == "structured_diagnosis" else 2,
                "schema_valid_candidate_count": 3 if mode == "structured_diagnosis" else 2,
                "valid_candidate_count": 2,
                "current_data_evaluated_count": 2,
                "provisional_internal_pass_count": 1,
                "final_multiplicity_adjusted_internal_pass_count": 1,
                "multiplicity_retraction_count": 0,
                "execution_error_count": 0,
            },
            {
                "claim_id": "claim_b",
                "target_family": "asd",
                "source_mode": "literature_grounded",
            },
        ],
    }
    (arm / "iterative_candidate_replay.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def test_control_summary_reconciles_matched_parent_rows(tmp_path):
    _write_arm(tmp_path, "structured_diagnosis")
    _write_arm(tmp_path, "generic_retry")

    summary = run(tmp_path, expected_parent_count=2)

    assert summary["parent_count"] == 2
    assert summary["paired_difference_totals"]["generated_candidate_count"] == 1
    assert summary["paired_parent_support_cells"] == {
        "both": 1,
        "structured_only": 0,
        "generic_only": 0,
        "neither": 1,
    }
    assert summary["parent_support_by_stratum"]["target_family"]["adhd"] == {
        "parent_count": 1,
        "structured_supported_parent_count": 1,
        "generic_supported_parent_count": 1,
    }
    assert summary["arms"]["structured_diagnosis"]["llm_call_count"] == 0
    assert summary["arms"]["structured_diagnosis"]["completed_trace_llm_call_count"] == 0
    assert summary["arms"]["structured_diagnosis"]["superseded_transient_attempt_count"] == 0
    assert summary["causal_interpretation_allowed"] is False
    assert (tmp_path / "control_summary.json").exists()
    assert (tmp_path / "control_parent_pairs.csv").exists()


def test_control_summary_separates_completed_and_transient_attempts(tmp_path):
    _write_arm(tmp_path, "structured_diagnosis")
    _write_arm(tmp_path, "generic_retry")
    provenance = {
        "rendered_prompt_record_count": 5,
        "superseded_transient_prompt_record_count": 3,
        "total_prompt_attempt_record_count": 8,
    }
    (tmp_path / "generic_retry" / "run_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )

    summary = run(tmp_path, expected_parent_count=2)

    arm = summary["arms"]["generic_retry"]
    assert arm["completed_trace_llm_call_count"] == 5
    assert arm["superseded_transient_attempt_count"] == 3
    assert arm["total_llm_attempt_count"] == 8
    assert arm["llm_call_count"] == 8


def test_control_summary_rejects_mixed_sources(tmp_path):
    _write_arm(tmp_path, "structured_diagnosis", source_hash="source-a")
    _write_arm(tmp_path, "generic_retry", source_hash="source-b")

    with pytest.raises(ValueError, match="source hash"):
        run(tmp_path, expected_parent_count=2)


def test_control_summary_accepts_explicit_artifact_paths(tmp_path):
    _write_arm(tmp_path, "structured_diagnosis")
    _write_arm(tmp_path, "generic_retry")

    summary = run(
        tmp_path / "summary-output",
        structured_artifact=tmp_path / "structured_diagnosis" / "iterative_candidate_replay.json",
        generic_artifact=tmp_path / "generic_retry" / "iterative_candidate_replay.json",
        expected_parent_count=2,
    )

    assert summary["parent_count"] == 2
    assert (tmp_path / "summary-output" / "control_summary.json").exists()


def test_control_summary_preserves_full_legacy_hashes_for_audit(tmp_path):
    _write_arm(tmp_path, "structured_diagnosis", freezer_hash="old")
    _write_arm(tmp_path, "generic_retry", freezer_hash="new")

    summary = run(tmp_path, expected_parent_count=2)

    assert len(set(summary["full_implementation_hashes_sha256_by_arm"].values())) == 2


def test_control_summary_rejects_search_implementation_drift(tmp_path):
    _write_arm(
        tmp_path,
        "structured_diagnosis",
        search_implementation_hash="old",
    )
    _write_arm(
        tmp_path,
        "generic_retry",
        search_implementation_hash="new",
    )

    with pytest.raises(ValueError, match="implementation"):
        run(tmp_path, expected_parent_count=2)


def test_control_summary_records_legacy_provenance(tmp_path):
    _write_arm(
        tmp_path,
        "structured_diagnosis",
        evidence_partition_hash="old",
    )
    _write_arm(
        tmp_path,
        "generic_retry",
        evidence_partition_hash="new",
    )

    summary = run(tmp_path, expected_parent_count=2)

    assert summary["implementation_compatibility_status"] == "legacy_search_fingerprint_unavailable"
    assert summary["search_implementation_hashes_sha256"] is None


def test_control_preflight_accepts_legacy_artifact(tmp_path):
    _write_arm(tmp_path, "structured_diagnosis")

    result = preflight_current_implementation(
        tmp_path / "structured_diagnosis" / "iterative_candidate_replay.json"
    )

    assert result["status"] == "legacy_search_fingerprint_unavailable"


def test_control_preflight_rejects_current_implementation_drift(tmp_path):
    _write_arm(
        tmp_path,
        "structured_diagnosis",
        search_implementation_hash="different",
    )

    with pytest.raises(ValueError, match="different claim-search code"):
        preflight_current_implementation(
            tmp_path / "structured_diagnosis" / "iterative_candidate_replay.json"
        )
