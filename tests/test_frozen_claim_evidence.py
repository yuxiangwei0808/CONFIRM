from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from confirm.contract import ClaimContract
from confirm.evidence_partitions import (
    EvidencePartitionManifest,
    EvidencePartitionRecord,
    ExternalEvidenceSetRecord,
)
from confirm import frozen_evidence as frozen


def _contract(
    *,
    claim_id: str,
    family_size: int = 1,
    outcome: str = "smri_hippocampus",
    discovery_cohort: str = "ADNI_DISC",
    replication_cohorts: list[str] | None = None,
) -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": claim_id,
            "question": f"Is age associated with hippocampal volume for {claim_id}?",
            "estimand": {
                "type": "association",
                "outcome": outcome,
                "predictor": "age",
                "group": None,
                "direction": "two_sided",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["sex", "site"],
            "inclusion": None,
            "discovery_cohort": discovery_cohort,
            "replication_cohorts": replication_cohorts or ["ADNI_REP"],
            "search_provenance": {
                "declared": True,
                "family_size": family_size,
                "selection": "discovery_only",
            },
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": family_size},
                "confound": {"require_covariates": ["sex", "site"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "none",
                    "pattern_corr_min": 0.0,
                    "region_replication_frac_min": 0.0,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def _write_sweep(
    root: Path,
    *,
    source_on_holdout: bool = False,
    known_negative: bool = False,
) -> tuple[ClaimContract, ClaimContract]:
    source = (
        {"discovery_cohort": "ADNI_HOLDOUT_DISC", "replication_cohorts": ["ADNI_HOLDOUT_REP"]}
        if source_on_holdout
        else {}
    )
    parent = _contract(claim_id="parent", family_size=1, **source)
    candidate = _contract(
        claim_id="candidate", family_size=2, outcome="smri_entorhinal", **source
    )
    candidate_id = "parent_r1_c1_narrower_outcome_family"
    raw_contract = _contract(
        claim_id="candidate", family_size=1, outcome="smri_entorhinal", **source
    )
    raw = {
        "proposal_type": "exploratory_followup_claim",
        "transform_type": "narrower_outcome_family",
        "proposed_question": "Does age relate to entorhinal volume?",
        "proposed_contract": raw_contract.model_dump(mode="json"),
        "connection_rationale": "Same modality and predictor, narrower outcome.",
        "provenance": "post_hoc_followup",
        "requires_new_evidence": False,
        "can_confirm_on_current_data": True,
        "validation_split": "current_data_adaptive",
        "supported_by_evidence": [],
    }
    normalized = {
        **raw,
        "candidate_id": candidate_id,
        "parent_claim_id": "parent",
        "round_index": 1,
        "proposed_contract": candidate.model_dump(mode="json"),
        "declared_transform": "narrower_outcome_family",
        "inferred_transform": "narrower_outcome_family",
        "transform_match": True,
        "executable_contract_delta": {
            "estimand.outcome": {
                "parent": "smri_hippocampus",
                "candidate": "smri_entorhinal",
            }
        },
    }
    evaluation = {
        "candidate_id": candidate_id,
        "proposal": normalized,
        "validation": {"ok": True, "violations": [], "warnings": []},
        "eligible_for_confirmation": True,
        "evaluated": True,
        "final_label": "exploratory_confirmed",
        "current_data_supported": True,
        "provisional_supported": True,
        "final_family_size": 2,
        "multiplicity_retracted": False,
        "effective_family_size": 2,
    }
    state = {
        "original_claim": parent.model_dump(mode="json"),
        "source_metadata": {
            "target_family": "ad_aging",
            "source_mode": "synthetic_stress" if known_negative else "llm_proposed",
            "source_scoring_label": "known_null" if known_negative else None,
            "synthetic_failure_family": "random_label" if known_negative else None,
        },
        "failure_localization": {"failure_kind": "evidence_failure"},
        "candidate_history": [normalized],
        "duplicate_candidates": [],
        "evaluations": [evaluation],
        "internally_supported_candidate_ids": [candidate_id],
        "provisional_supported_candidate_ids": [candidate_id],
        "final_search_family_size": 2,
        "selected_candidate_id": None,
        "selection_reason": None,
        "generated_candidate_count": 1,
        "schema_valid_candidate_count": 1,
        "valid_candidate_count": 1,
        "current_data_evaluated_count": 1,
        "llm_candidate_responses": [
            {
                "round_index": 1,
                "parent_claim_id": "parent",
                "attempt_index": 0,
                "raw_response": json.dumps({"candidates": [raw]}),
                "candidate_count": 1,
                "parse_error": None,
            }
        ],
        "stopped_reason": "all_candidates_supported",
    }
    artifact = {
        "status": "completed",
        "llm_model": "openai:gpt-5.5",
        "config": {"max_rounds": 1, "max_candidates_per_round": 1},
        "provenance": {
            "source": {"sha256": "source-hash"},
            "resume_identity_sha256": "resume-hash",
            "prompt_sha256": "prompt-hash",
            "schema_sha256": "schema-hash",
        },
        "completed_search_count": 1,
        "summary": {
            "n_searches": 1,
            "generated_candidate_count": 1,
            "schema_valid_candidate_count": 1,
            "policy_valid_candidate_count": 1,
            "candidate_count": 1,
            "current_data_evaluated_count": 1,
            "execution_complete_candidate_count": 1,
            "provisional_internal_pass_count": 1,
            "duplicate_candidate_count": 0,
            "unretained_generated_candidate_count": 0,
            "exploratory_confirmed_count": 1,
            "final_multiplicity_adjusted_internal_pass_count": 1,
            "excluded_evidence_query_count": 0,
        },
        "states": [state],
    }
    artifact_path = root / "matrix" / "rounds_1" / "candidates_1" / "iterative_candidate_replay.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return parent, candidate


def _materialize_parent_checkpoint(root: Path) -> Path:
    arm = root / "matrix" / "rounds_1" / "candidates_1"
    artifact_path = arm / "iterative_candidate_replay.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    state = artifact["states"][0]
    body = {
        "resume_identity_sha256": "resume-hash",
        "index": 1,
        "claim_id": "parent",
        "row": {"claim_id": "parent"},
        "state": state,
    }
    checkpoint = {**body, "checkpoint_sha256": frozen.sha256_json(body)}
    checkpoint_path = arm / "checkpoints" / "parents" / "parent_0001.json"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    (arm / "run_provenance.json").write_text(
        json.dumps(artifact["provenance"], indent=2), encoding="utf-8"
    )
    artifact["states"] = [{"this_tail_must_not_be_used": True}]
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return checkpoint_path


def test_freeze_disposition_maps_duplicate_from_superseded_retry():
    signature = "scientific-signature"
    duplicates = frozen.Counter({("candidate", signature): 1})

    status = frozen._generation_status(
        is_used_response=False,
        candidate_id="candidate",
        scientific_signature=signature,
        history_by_id={},
        duplicate_occurrences=duplicates,
    )

    assert status == "duplicate"
    assert sum(duplicates.values()) == 0


def test_freeze_disposition_prioritizes_retained_used_response():
    signature = "scientific-signature"
    duplicates = frozen.Counter({("candidate", signature): 1})

    status = frozen._generation_status(
        is_used_response=True,
        candidate_id="candidate",
        scientific_signature=signature,
        history_by_id={"candidate": {}},
        duplicate_occurrences=duplicates,
    )

    assert status == "retained"
    assert sum(duplicates.values()) == 1


def test_freeze_uses_atomic_parent_checkpoints_instead_of_monolithic_states(tmp_path):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    checkpoint = _materialize_parent_checkpoint(sweep)

    summary = frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)

    assert summary["observed_counts"]["searched_lineage_events"] == 1
    assert summary["run_provenance_hashes"] == {
        "r1_c1": frozen.sha256_file(checkpoint.parents[2] / "run_provenance.json")
    }


def test_freeze_accepts_compatible_legacy_checkpoint_identity(tmp_path):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    checkpoint = _materialize_parent_checkpoint(sweep)
    arm = checkpoint.parents[2]

    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    checkpoint_payload.pop("checkpoint_sha256")
    checkpoint_payload["resume_identity_sha256"] = "legacy-resume-hash"
    checkpoint_payload["checkpoint_sha256"] = frozen.sha256_json(checkpoint_payload)
    checkpoint.write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    for provenance_path in (
        arm / "run_provenance.json",
        arm / "iterative_candidate_replay.json",
    ):
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance = payload["provenance"] if "provenance" in payload else payload
        provenance["compatible_resume_identity_sha256s"] = ["legacy-resume-hash"]
        provenance_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)

    assert summary["observed_counts"]["searched_lineage_events"] == 1


def test_freeze_rejects_tampered_parent_checkpoint(tmp_path):
    sweep = tmp_path / "sweep"
    _write_sweep(sweep)
    checkpoint = _materialize_parent_checkpoint(sweep)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["claim_id"] = "tampered"
    checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint hash"):
        frozen.freeze_sweep(sweep, tmp_path / "audit", enforce_reference_counts=False)


def _frame(prefix: str, n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"{prefix}-{index}" for index in range(n)],
            "cohort": ["ADNI"] * n,
            "site": [f"site-{index % 3}" for index in range(n)],
            "age": [60.0 + index / 5 for index in range(n)],
            "sex": ["F", "M"] * (n // 2),
            "smri_hippocampus": [3000.0 - index * 2 for index in range(n)],
            "smri_entorhinal": [1800.0 - index for index in range(n)],
        }
    )


def _record(path: Path, partition_id: str, role: str, evaluation_role: str) -> EvidencePartitionRecord:
    return EvidencePartitionRecord(
        partition_id=partition_id,
        base_dataset="ADNI",
        target_family="ad_aging",
        role=role,
        evaluation_role=evaluation_role,
        path=str(path),
        source_path=str(path),
        split_method="fixture",
        seed=7,
        n_rows=50,
        site_count=2,
        exclusion_role="excluded_evaluation" if role == "holdout" else "claim_source",
        subject_id_sha256=frozen.sha256_json(sorted(_frame(partition_id)["subject_id"])),
        source_row_count=50,
        columns=list(_frame(partition_id).columns),
        schema_sha256="schema",
        content_sha256=frozen.sha256_file(path),
        modality="sMRI",
        feature_families=["regional_volume"],
        units={"smri_*": "mm3"},
    )


def _write_evidence(root: Path, *, external_set_count: int = 0) -> Path:
    cohort_root = root / "cohorts"
    benchmark_root = root / "benchmark_ready" / "cohorts"
    cohort_root.mkdir(parents=True)
    benchmark_root.mkdir(parents=True)
    records = []
    for partition_id, role, evaluation_role, parent in (
        ("ADNI_DISC", "discovery", "discovery", benchmark_root),
        ("ADNI_REP", "replication", "replication", benchmark_root),
        ("ADNI_HOLDOUT_DISC", "holdout", "discovery", cohort_root),
        ("ADNI_HOLDOUT_REP", "holdout", "replication", cohort_root),
    ):
        path = parent / f"{partition_id}.parquet"
        _frame(partition_id).to_parquet(path, index=False)
        records.append(_record(path, partition_id, role, evaluation_role))
    external_sets = []
    for index in range(external_set_count):
        base_dataset = f"EXTERNAL_{index + 1}"
        discovery_id = f"{base_dataset}_DISC"
        replication_id = f"{base_dataset}_REP"
        for partition_id in (discovery_id, replication_id):
            path = cohort_root / f"{partition_id}.parquet"
            frame = _frame(partition_id)
            frame["cohort"] = base_dataset
            frame.to_parquet(path, index=False)
            records.append(
                EvidencePartitionRecord(
                    partition_id=partition_id,
                    base_dataset=base_dataset,
                    target_family="ad_aging",
                    role="external_eval",
                    evaluation_role="external",
                    path=str(path),
                    source_path=str(path),
                    split_method="fixture",
                    seed=7,
                    n_rows=len(frame),
                    site_count=int(frame["site"].nunique()),
                    exclusion_role="excluded_external_evaluation",
                    subject_id_sha256=frozen.sha256_json(sorted(frame["subject_id"])),
                    source_row_count=len(frame),
                    columns=list(frame.columns),
                    schema_sha256="schema",
                    content_sha256=frozen.sha256_file(path),
                    modality="sMRI",
                    feature_families=["regional_volume"],
                    units={"smri_*": "mm3"},
                )
            )
        external_sets.append(
            ExternalEvidenceSetRecord(
                evidence_set_id=f"external_set_{index + 1}",
                target_family="ad_aging",
                modality="sMRI",
                feature_family="regional_volume",
                discovery_partition_id=discovery_id,
                replication_partition_ids=[replication_id],
                supported_predictors=["age"],
                priority=index,
                confirmation_role="primary" if index == 0 else "secondary",
                units={"smri_*": "mm3"},
            )
        )
    manifest = EvidencePartitionManifest(
        seed=7,
        records=records,
        external_evidence_sets=external_sets,
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest_path


def test_freeze_preflight_evaluate_and_summarize_are_hash_bound(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    manifest_path = _write_evidence(tmp_path / "evidence")

    freeze_summary = frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    assert freeze_summary["observed_counts"]["generated_candidate_count"] == 1
    assert freeze_summary["observed_counts"]["final_internal_supported_count"] == 1
    inventory = frozen.read_jsonl(output / "frozen_search_inventory.jsonl")
    assert inventory[0]["parsed_proposal"]["validation_split"] == "current_data_adaptive"
    assert inventory[0]["normalized_proposal"]["validation_split"] == "current_data_adaptive"
    assert inventory[0]["final_internal_supported"] is True
    responses = frozen.read_jsonl(output / "frozen_llm_responses.jsonl")
    assert responses[0]["raw_response"]
    assert responses[0]["raw_response_sha256"]
    assert inventory[0]["response_id"] == responses[0]["response_id"]

    preflight = frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
    )
    assert preflight["outcome_blind"] is True
    assert preflight["deduplicated_query_task_count"] == 1
    preflight_rows = frozen.read_jsonl(output / "evidence_preflight.jsonl")
    assert {
        row["interpretation_label"]
        for row in preflight_rows
        if row["evidence_kind"] == "holdout"
    } == {"excluded_evidence_compatible"}
    query_tasks = frozen.read_jsonl(output / "evidence_query_plan.jsonl")
    assert all(task["outcome_blind"] is True for task in query_tasks)
    assert {task["implementation_sha256"] for task in query_tasks} == {
        preflight["implementation_sha256"]
    }
    assert {
        task["mapped_contract"]["gates"]["multiplicity"]["family_size"]
        for task in query_tasks
    } == {2}
    assert not (output / "evidence_evaluations.jsonl").exists()

    calls = []

    def fake_execute(contract, roots, **kwargs):
        calls.append(contract.claim_id)
        return {"final_label": "confirmed", "gate_results": {"contract": contract.model_dump(mode="json")}}

    monkeypatch.setattr(frozen, "execute_contract", fake_execute)
    first = frozen.evaluate_query_plan(output, max_workers=1, progress=False)
    assert first["newly_executed_count"] == 1
    assert len(calls) == 1
    assert first["final_confirmation_eligible"] is False

    second = frozen.evaluate_query_plan(output, max_workers=1, progress=False)
    assert second["newly_executed_count"] == 0
    assert len(calls) == 1

    summary = frozen.summarize_evidence_audit(output)
    assert summary["primary_configuration"] is None
    assert summary["final_internal_supported_candidate_count"] == 1
    assert summary["arm_summary"][0]["parent_lineage_count"] == 1
    assert summary["arm_summary"][0]["holdout_conditional_survival_rate"] == 1.0
    assert summary["arm_summary"][0]["holdout_compatible_candidate_pair_count"] == 1
    assert summary["arm_summary"][0]["generated_candidate_count"] == 1
    assert summary["arm_summary"][0]["llm_response_count"] == 1
    assert summary["final_confirmation_eligible"] is False
    assert (output / "transform_audit.csv").exists()


def test_evaluation_refuses_a_modified_query_plan(tmp_path):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    manifest_path = _write_evidence(tmp_path / "evidence")
    frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
    )
    query_path = output / "evidence_query_plan.jsonl"
    query_path.write_text(query_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    try:
        frozen.evaluate_query_plan(output, progress=False)
    except ValueError as exc:
        assert "query plan changed" in str(exc)
    else:
        raise AssertionError("Modified query plan was accepted")


def test_preflight_blocks_incompatible_outcome_units(tmp_path):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    manifest_path = _write_evidence(tmp_path / "evidence")
    manifest = EvidencePartitionManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    for record in manifest.records:
        if record.role == "holdout":
            record.units = {"smri_*": "cm3"}
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    summary = frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
    )

    records = frozen.read_jsonl(output / "evidence_preflight.jsonl")
    holdout = [record for record in records if record["evidence_kind"] == "holdout"]
    assert holdout
    assert {record["reason_code"] for record in holdout} == {"unit_incompatible"}
    assert summary["deduplicated_query_task_count"] == 0


def test_source_holdout_partitions_cannot_be_reused_as_excluded_evidence(tmp_path):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep, source_on_holdout=True)
    manifest_path = _write_evidence(tmp_path / "evidence")

    frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    summary = frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[
            manifest_path.parent / "benchmark_ready" / "cohorts",
            manifest_path.parent / "cohorts",
        ],
    )

    records = frozen.read_jsonl(output / "evidence_preflight.jsonl")
    holdout = [record for record in records if record["evidence_kind"] == "holdout"]
    assert holdout
    assert all(record["status"] == "unavailable" for record in holdout)
    assert summary["deduplicated_query_task_count"] == 0


def test_every_compatible_external_evidence_set_is_scheduled_independently(tmp_path):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    manifest_path = _write_evidence(tmp_path / "evidence", external_set_count=2)

    frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    summary = frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
    )

    tasks = frozen.read_jsonl(output / "evidence_query_plan.jsonl")
    external_tasks = [task for task in tasks if task["evidence_kind"] == "external"]
    assert summary["deduplicated_query_task_count"] == 3
    assert {task["evidence_set_id"] for task in external_tasks} == {
        "external_set_1",
        "external_set_2",
    }
    assert len(external_tasks) == 2


def test_frozen_known_negative_summary_preserves_safety_denominators(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep, known_negative=True)
    manifest_path = _write_evidence(tmp_path / "evidence")

    frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
    )
    monkeypatch.setattr(
        frozen,
        "execute_contract",
        lambda contract, roots, **kwargs: {
            "final_label": "confirmed",
            "gate_results": {"contract": contract.model_dump(mode="json")},
        },
    )
    frozen.evaluate_query_plan(output, max_workers=1, progress=False)
    summary = frozen.summarize_evidence_audit(output)

    assert summary["known_negative_safety_audit"] is True
    assert summary["known_negative_lineage_count"] == 1
    arm = summary["arm_summary"][0]
    assert arm["known_negative_parent_lineage_count"] == 1
    assert arm["known_negative_internal_supported_candidate_count"] == 1
    assert arm["known_negative_holdout_supported_candidate_pair_count"] == 1
    assert arm["known_negative_any_support_risk_rate"] == 1.0
    strata = pd.read_csv(output / "stratified_summary.csv")
    synthetic = strata[
        (strata["dimension"] == "synthetic_failure_family")
        & (strata["dimension_value"] == "random_label")
    ].iloc[0]
    assert synthetic["parent_lineage_count"] == 1
    assert synthetic["proposals_returned_count"] == 1


def test_evaluation_refuses_implementation_hash_drift(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    manifest_path = _write_evidence(tmp_path / "evidence")
    frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
    )
    monkeypatch.setattr(frozen, "implementation_hashes", lambda: {"changed.py": "different"})

    try:
        frozen.evaluate_query_plan(output, progress=False)
    except ValueError as exc:
        assert "implementation changed" in str(exc)
    else:
        raise AssertionError("Implementation drift was accepted")


def test_evaluation_refuses_a_tampered_checkpoint(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    output = tmp_path / "audit"
    _write_sweep(sweep)
    manifest_path = _write_evidence(tmp_path / "evidence")
    frozen.freeze_sweep(sweep, output, enforce_reference_counts=False)
    frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
    )

    monkeypatch.setattr(
        frozen,
        "execute_contract",
        lambda contract, roots, **kwargs: {
            "final_label": "confirmed",
            "gate_results": {"contract": contract.model_dump(mode="json")},
        },
    )
    frozen.evaluate_query_plan(output, max_workers=1, progress=False)
    checkpoint = next((output / "checkpoints" / "evidence").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["raw_gate_label"] = "fragile"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    try:
        frozen.evaluate_query_plan(output, max_workers=1, progress=False)
    except ValueError as exc:
        assert "result hash is invalid" in str(exc)
    else:
        raise AssertionError("Tampered checkpoint was accepted")


def test_initial_claims_are_frozen_and_all_scheduled_for_holdout(tmp_path, monkeypatch):
    output = tmp_path / "initial-audit"
    manifest_path = _write_evidence(tmp_path / "evidence")
    confirmed = _contract(claim_id="initial-confirmed", family_size=1)
    fragile = _contract(claim_id="initial-fragile", family_size=2)

    def result_row(contract: ClaimContract, label: str) -> dict:
        payload = contract.model_dump(mode="json")
        return {
            "claim_id": contract.claim_id,
            "contract": payload,
            "drafted_contract": payload,
            "gate_results": {"contract": payload},
            "gate_success": True,
            "final_label": label,
            "gate_verdict_label": label,
            "target_family": "ad_aging",
            "source_mode": "llm_proposed",
            "ground_truth": None,
            "label_class": None,
            "scoring_label": None,
            "source_citation": None,
            "model_spec": "stub",
        }

    initial_results = tmp_path / "initial_results.json"
    initial_results.write_text(
        json.dumps(
            {
                "claims": [
                    result_row(confirmed, "confirmed"),
                    result_row(fragile, "fragile"),
                ]
            }
        ),
        encoding="utf-8",
    )

    freeze_summary = frozen.freeze_initial_claims(initial_results, output)
    assert freeze_summary["observed_counts"]["initial_claim_count"] == 2
    assert freeze_summary["observed_counts"]["initial_confirmed_count"] == 1

    preflight = frozen.build_evidence_preflight(
        output,
        manifest_path,
        evidence_roots=[manifest_path.parent / "cohorts"],
        source_roots=[manifest_path.parent / "benchmark_ready" / "cohorts"],
        schedule_all_parents=True,
    )
    assert preflight["schedule_all_parents"] is True
    assert preflight["deduplicated_query_task_count"] == 2
    tasks = frozen.read_jsonl(output / "evidence_query_plan.jsonl")
    assert {reference["role"] for task in tasks for reference in task["references"]} == {
        "parent"
    }

    def fake_execute(contract, roots, **kwargs):
        label = "confirmed" if contract.gates.multiplicity.family_size == 1 else "fragile"
        return {
            "final_label": label,
            "gate_results": {"contract": contract.model_dump(mode="json")},
        }

    monkeypatch.setattr(frozen, "execute_contract", fake_execute)
    evaluation = frozen.evaluate_query_plan(output, max_workers=1, progress=False)
    assert evaluation["newly_executed_count"] == 2

    summary = frozen.summarize_initial_claim_evidence(output)
    assert summary["overall"]["initial_claim_count"] == 2
    assert summary["overall"]["initial_confirmed_count"] == 1
    assert summary["overall"]["holdout_evaluated_count"] == 2
    assert summary["overall"]["holdout_supported_count"] == 1
    assert summary["overall"]["holdout_support_ci_high"] > 0
    assert summary["matched_internal_holdout_counts"] == {
        "internal_0_holdout_0": 1,
        "internal_1_holdout_1": 1,
    }
    assert summary["overall"]["external_evaluated_count"] == 0
    assert (output / "initial_claim_evidence.csv").exists()
