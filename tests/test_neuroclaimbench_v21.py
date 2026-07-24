from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from bench.neuroclaimbench import (
    BenchmarkItem,
    EvaluationTask,
    FieldAlignment,
    GeminiAlignmentAssessment,
    SimplifiedBenchmarkClaim,
    derive_triage_reference,
    exact_contract_hash,
    scientific_question_hash,
    semantic_contract_hash,
)
from bench.run_neuroclaimbench_alignment import (
    _deterministic_record,
    _finalize_with_gemini,
    _gemini_prompt,
    _standard_confirm_dx_contrast,
)
from bench.run_neuroclaimbench_finalize import (
    _cluster_bootstrap_interval,
    _task_fingerprint,
)
from bench.run_neuroclaimbench_pubmed_cache import _plan_fingerprint
from bench.run_neuroclaimbench_v21_build import build as build_v21
from bench.run_neuroclaimbench_v21_build import (
    _migrated_item,
    _portable_alignment_record,
    _stable_benchmark_item_id,
)
from bench.run_neuroclaimbench_v21_release import run as build_release
from confirm.candidate_preflight import CandidatePreflightContext
from confirm.contract import ClaimContract


def _contract(*, direction: str = "negative", case: str = "case", control: str = "control") -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "adhd_subtype_claim",
            "question": "Is inattentive ADHD lower than combined ADHD on smri_metric?",
            "estimand": {
                "type": "group_diff",
                "outcome": "smri_metric",
                "predictor": "confirm_dx",
                "group": {"var": "confirm_dx", "case": case, "control": control},
                "direction": direction,
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["age"],
            "inclusion": None,
            "discovery_cohort": "ADHD200_DISC",
            "replication_cohorts": ["ADHD200_REP"],
            "search_provenance": {
                "declared": True,
                "family_size": 1,
                "selection": "preregistered",
            },
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["age"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "combat",
                    "pattern_corr_min": 0.5,
                    "region_replication_frac_min": 0.5,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": [
                "confirmed",
                "non_replicated",
                "under_powered",
                "fragile",
            ],
        }
    )


def _item(contract: ClaimContract, *, item_id: str = "ncb-adhd") -> BenchmarkItem:
    return BenchmarkItem(
        benchmark_item_id=item_id,
        claim_uid=f"claim-{item_id}",
        semantic_cluster_id=f"cluster-{item_id}",
        benchmark_track="scientific",
        target_family="adhd",
        modality="sMRI",
        question="Is inattentive ADHD lower than combined ADHD on smri_metric?",
        contract=contract,
        exact_contract_sha256=exact_contract_hash(contract),
        semantic_claim_sha256=semantic_contract_hash(contract),
        source_references=[],
        evaluation_task_ids=["task"],
    )


def test_legacy_pending_id_migrates_to_stable_source_id():
    assert _stable_benchmark_item_id("ncb-pending-0123456789abcdef") == (
        "ncb-source-0123456789abcdef"
    )
    assert _stable_benchmark_item_id("ncb-adhd") == "ncb-adhd"


def _adhd_context(tmp_path: Path, *, include_inattentive: bool = True) -> CandidatePreflightContext:
    root = tmp_path / "cohorts"
    root.mkdir(parents=True)
    dx = ["1", "1", "1", "1", "1", "2" if include_inattentive else "3"]
    for cohort, offset in (("ADHD200_DISC", 0), ("ADHD200_REP", 10)):
        pd.DataFrame(
            {
                "subject_id": [f"{cohort}-{index}" for index in range(6)],
                "dx": dx,
                "age": [8 + offset, 9 + offset, 10 + offset, 11 + offset, 12 + offset, 14 + offset],
                "sex": ["M", "F", "M", "F", "M", "F"],
                "smri_metric": [0.1, 0.2, 0.0, 0.3, 0.1, -0.1],
            }
        ).to_parquet(root / f"{cohort}.parquet", index=False)
    return CandidatePreflightContext.from_roots([root])


def test_small_but_identifiable_adhd_subtype_is_repaired_not_excluded(tmp_path: Path):
    contract = _contract()
    record = _deterministic_record(_item(contract), _adhd_context(tmp_path))
    assert record.final_outcome_blind_resolution == "repairable_contract"
    assert record.repaired_contract is not None
    assert record.repaired_contract.estimand.group is not None
    assert record.repaired_contract.estimand.group.var == "dx"
    assert record.repaired_contract.estimand.group.case == "2"
    assert record.repaired_contract.estimand.group.control == "1"
    diagnostics = record.deterministic_preflight["design_diagnostics"]
    assert diagnostics["ADHD200_DISC"]["complete_group_counts"]["2"] == 1


def test_absent_requested_adhd_subtype_is_non_executable(tmp_path: Path):
    record = _deterministic_record(
        _item(_contract()),
        _adhd_context(tmp_path, include_inattentive=False),
    )
    assert record.final_outcome_blind_resolution == "non_executable"
    assert "missing group levels" in record.resolution_reason


def test_direction_mismatch_is_not_silently_repaired(tmp_path: Path):
    record = _deterministic_record(
        _item(_contract(direction="positive")),
        _adhd_context(tmp_path),
    )
    assert record.final_outcome_blind_resolution == "ambiguous_unresolved"
    assert "direction" in record.resolution_reason


def test_constructed_control_has_constructed_basis_and_strength():
    contract = _contract()
    item = _item(contract, item_id="constructed")
    item.benchmark_track = "synthetic_stress"
    item.label_class = "known_null"
    item.reference_disposition = "abstain"
    item.adjudication_status = "construction_derived"
    item.score_eligible = True
    profile = derive_triage_reference(item, [])
    assert profile.reference_basis == "constructed_control"
    assert profile.reference_strength == "constructed"


def test_question_hash_changes_cache_and_task_fingerprints():
    contract = _contract()
    first = _item(contract)
    second = _item(contract, item_id="ncb-adhd-2")
    second.question = "Does combined ADHD differ from controls on smri_metric?"
    second.scientific_question_sha256 = scientific_question_hash(second.question)
    common = {
        "package_sha256": "a" * 64,
        "assessor_models": ["openai:gpt-5.5", "google:gemini-3.5-flash"],
        "prompt_sha256": "b" * 64,
    }
    assert _plan_fingerprint(first, **common) != _plan_fingerprint(second, **common)

    task = EvaluationTask(
        task_id="task",
        benchmark_item_id=first.benchmark_item_id,
        contract=contract,
        evidence_role="source",
        dataset_id="ADHD200",
        discovery_cohort=contract.discovery_cohort,
        replication_cohorts=contract.replication_cohorts,
        partition_hashes={"ADHD200_DISC": "1", "ADHD200_REP": "2"},
        scientific_question_sha256=first.scientific_question_sha256,
        schema_version="2.1.0",
    )
    changed = task.model_copy(
        update={"scientific_question_sha256": second.scientific_question_sha256}
    )
    assert _task_fingerprint(task) != _task_fingerprint(changed)


def test_changed_canonical_question_invalidates_adjudication_identity(tmp_path: Path):
    context = _adhd_context(tmp_path)
    initial = _deterministic_record(_item(_contract()), context)
    assert initial.repaired_contract is not None
    item = _item(initial.repaired_contract)
    item.label_class = "known_positive"
    item.reference_disposition = "confirm"
    item.adjudication_status = "multi_model_consensus"
    item.score_eligible = True
    record = _deterministic_record(item, context)
    changed_question = "Is inattentive ADHD different from combined ADHD on smri_metric?"
    record = record.model_copy(
        update={
            "canonical_question": changed_question,
            "canonical_question_sha256": scientific_question_hash(changed_question),
        }
    )
    migrated, contract_changed, identity_changed = _migrated_item(item, record)
    assert not contract_changed
    assert identity_changed
    assert migrated.adjudication_status == "pending"
    assert migrated.reference_disposition == "unresolved"


def test_nonready_literature_item_cannot_retain_scored_reference(tmp_path: Path):
    context = _adhd_context(tmp_path)
    item = _item(_contract())
    item.label_class = "known_positive"
    item.reference_disposition = "confirm"
    item.adjudication_status = "multi_model_consensus"
    item.score_eligible = True
    record = _deterministic_record(item, context).model_copy(
        update={
            "final_outcome_blind_resolution": "ambiguous_unresolved",
            "resolution_reason": "Substantive mismatch.",
        }
    )
    migrated, _, _ = _migrated_item(item, record)
    assert migrated.migration_status == "ambiguous_unresolved"
    assert migrated.label_class == "candidate_unknown"
    assert migrated.reference_disposition == "unresolved"
    assert migrated.adjudication_status == "unresolved"
    assert not migrated.score_eligible


def test_packaged_alignment_paths_are_relative(tmp_path: Path):
    record = _deterministic_record(_item(_contract()), _adhd_context(tmp_path))
    preflight = dict(record.deterministic_preflight)
    preflight["design_diagnostics"] = {
        "generator": {
            "assignment_path": "obsolete/assignments.parquet"
        }
    }
    record = record.model_copy(update={"deterministic_preflight": preflight})
    assignment_path = Path(
        "data/neuroclaimbench/v2.1/external_random_control_assignments.parquet"
    )
    portable = _portable_alignment_record(record, assignment_path=assignment_path)
    paths = portable["deterministic_preflight"]["resolved_data_paths"]
    assert paths
    assert all(not Path(path).is_absolute() for path in paths.values())
    assert (
        portable["deterministic_preflight"]["design_diagnostics"]["generator"][
            "assignment_path"
        ]
        == str(assignment_path)
    )


def test_stale_question_hash_is_rejected():
    with pytest.raises(ValueError, match="scientific question hash is stale"):
        BenchmarkItem.model_validate(
            {
                **_item(_contract()).model_dump(mode="json"),
                "scientific_question_sha256": "stale",
            }
        )


def test_semantic_cluster_bootstrap_is_deterministic():
    contract = _contract()
    claims = [
        SimplifiedBenchmarkClaim(
            benchmark_claim_id=f"claim-{index}",
            benchmark_item_ids=[f"item-{index}"],
            benchmark_split="scientific",
            target_family="adhd",
            modality="sMRI",
            question=contract.question,
            scientific_question_sha256=scientific_question_hash(contract.question),
            semantic_cluster_id="cluster-a" if index < 2 else "cluster-b",
            contract=contract,
            contract_sha256=exact_contract_hash(contract),
            execution_identity_sha256=str(index),
            reference_label="confirm",
            score_eligible=True,
            reference_strength="strict",
            reference_basis="literature",
            evaluation_task_ids=[f"task-{index}"],
            dataset_ids=["ADHD200"],
        )
        for index in range(3)
    ]
    outcomes = {"claim-0": "confirmed", "claim-1": "confirmed", "claim-2": "abstained"}
    first = _cluster_bootstrap_interval(claims, outcomes, reference_label="confirm")
    second = _cluster_bootstrap_interval(claims, outcomes, reference_label="confirm")
    assert first == second
    assert first is not None


def test_gemini_alignment_prompt_contains_no_outcomes_or_reference_labels(tmp_path: Path):
    item = _item(_contract())
    context = _adhd_context(tmp_path)
    record = _deterministic_record(item, context)
    prompt = _gemini_prompt(item, record, context)
    for forbidden in (
        "gate_verdict",
        "p_value",
        "effect_estimate",
        "label_class",
        "reference_disposition",
        "feedback_result",
    ):
        assert forbidden not in prompt


def test_gemini_cannot_reject_policy_authorized_safety_covariates(tmp_path: Path):
    record = _deterministic_record(_item(_contract()), _adhd_context(tmp_path))
    record = record.model_copy(
        update={
            "canonical_question": record.canonical_question
            + ", with no additional covariates.",
        }
    )
    assessment = GeminiAlignmentAssessment(
        aligned=False,
        field_assessments=[
            FieldAlignment(
                field="covariates",
                status="mismatch",
                question_value="no additional covariates",
                contract_value=["age"],
                reason="The question requests no covariates.",
            )
        ],
        recommended_disposition="repairable_contract",
        rationale="Remove the covariates.",
    )
    finalized = _finalize_with_gemini(record, assessment)
    assert finalized.final_outcome_blind_resolution == "repairable_contract"
    assert "frozen policy" in finalized.resolution_reason


def test_substantive_gemini_mismatch_remains_unresolved(tmp_path: Path):
    record = _deterministic_record(_item(_contract()), _adhd_context(tmp_path))
    assessment = GeminiAlignmentAssessment(
        aligned=False,
        field_assessments=[
            FieldAlignment(
                field="outcome",
                status="mismatch",
                question_value="different_outcome",
                contract_value="smri_metric",
                reason="The outcomes differ.",
            )
        ],
        recommended_disposition="repairable_contract",
        rationale="The outcome does not match.",
    )
    finalized = _finalize_with_gemini(record, assessment)
    assert finalized.final_outcome_blind_resolution == "ambiguous_unresolved"


def test_confirm_dx_is_only_equivalent_for_standard_disease_control_contrasts():
    contract = _contract()
    psychosis = _item(contract).model_copy(
        update={
            "target_family": "psychosis",
            "question": "Do SZ and HC differ in smri_metric using dx?",
        }
    )
    assert _standard_confirm_dx_contrast(psychosis, contract)
    mci = _item(contract).model_copy(
        update={
            "target_family": "ad_aging",
            "question": "Is MCI lower than CN in smri_metric using dx?",
        }
    )
    assert not _standard_confirm_dx_contrast(mci, contract)


def test_gemini_cannot_reject_documented_confirm_dx_normalization(tmp_path: Path):
    item = _item(_contract()).model_copy(
        update={
            "target_family": "psychosis",
            "question": "Do SZ and HC differ in smri_metric using dx?",
        }
    )
    record = _deterministic_record(item, _adhd_context(tmp_path))
    assessment = GeminiAlignmentAssessment(
        aligned=False,
        field_assessments=[
            FieldAlignment(
                field="predictor",
                status="mismatch",
                question_value="dx",
                contract_value="confirm_dx",
                reason="The columns have different names.",
            )
        ],
        recommended_disposition="repairable_contract",
        rationale="Use dx instead of confirm_dx.",
    )
    finalized = _finalize_with_gemini(record, assessment)
    assert finalized.final_outcome_blind_resolution == record.mismatch_disposition


def test_v21_build_requires_gemini_advisory_for_literature_items(tmp_path: Path):
    source = tmp_path / "v2"
    source.mkdir()
    item = _item(_contract())
    task = EvaluationTask(
        task_id="task",
        benchmark_item_id=item.benchmark_item_id,
        contract=item.contract,
        evidence_role="source",
        dataset_id="ADHD200",
        discovery_cohort="ADHD200_DISC",
        replication_cohorts=["ADHD200_REP"],
    )
    (source / "benchmark_items.jsonl").write_text(
        item.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (source / "evaluation_tasks.jsonl").write_text(
        task.model_dump_json() + "\n",
        encoding="utf-8",
    )
    alignment = _deterministic_record(item, _adhd_context(tmp_path / "alignment-data"))
    alignment_path = tmp_path / "alignment.jsonl"
    alignment_path.write_text(alignment.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires the frozen Gemini advisory"):
        build_v21(
            argparse.Namespace(
                source_package=str(source),
                alignment_records=str(alignment_path),
                out_dir=str(tmp_path / "v2.1"),
                data_root=[],
            )
        )


def test_v21_build_rejects_pending_source_adjudication(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    item = _item(_contract())
    task = EvaluationTask(
        task_id="task",
        benchmark_item_id=item.benchmark_item_id,
        contract=item.contract,
        evidence_role="source",
        dataset_id="ADHD200",
        discovery_cohort="ADHD200_DISC",
        replication_cohorts=["ADHD200_REP"],
    )
    (source / "benchmark_items.jsonl").write_text(
        item.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (source / "evaluation_tasks.jsonl").write_text(
        task.model_dump_json() + "\n",
        encoding="utf-8",
    )
    alignment = _deterministic_record(item, _adhd_context(tmp_path / "alignment-data"))
    alignment = _finalize_with_gemini(
        alignment,
        GeminiAlignmentAssessment(
            aligned=True,
            field_assessments=[],
            recommended_disposition="aligned",
            rationale="The question and contract are aligned.",
        ),
    )
    alignment_dir = tmp_path / "alignment"
    alignment_dir.mkdir()
    alignment_path = alignment_dir / "alignment_records.jsonl"
    alignment_path.write_text(alignment.model_dump_json() + "\n", encoding="utf-8")
    (alignment_dir / "alignment_policy.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="adjudication to be finalized"):
        build_v21(
            argparse.Namespace(
                source_package=str(source),
                alignment_records=str(alignment_path),
                out_dir=str(tmp_path / "v2.1"),
                data_root=[],
            )
        )


def test_release_checksum_includes_release_manifest(tmp_path: Path):
    compact = tmp_path / "compact"
    package = tmp_path / "package"
    results = tmp_path / "results"
    reference = tmp_path / "reference"
    analysis = tmp_path / "analysis"
    crosswalk = tmp_path / "crosswalk"
    adjudication = tmp_path / "adjudication"
    for directory in (
        compact,
        package,
        results,
        reference,
        analysis,
        crosswalk,
        adjudication,
    ):
        directory.mkdir()
    for name in ("cases.jsonl", "references.jsonl", "tasks.jsonl", "outcomes.jsonl"):
        (compact / name).write_text("", encoding="utf-8")
    (compact / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {
                    "cases": 0,
                    "references": 0,
                    "tasks": 0,
                    "outcomes": 0,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "benchmark_items.jsonl",
        "evaluation_tasks.jsonl",
        "benchmark_claims.jsonl",
        "alignment_records.jsonl",
        "evidence_records.jsonl",
        "label_votes.jsonl",
        "adjudications.jsonl",
    ):
        (package / name).write_text("", encoding="utf-8")
    for name in (
        "benchmark_splits.json",
        "alignment_policy.json",
        "repair_manifest.json",
        "build_manifest.json",
        "paper_benchmark_manifest.json",
    ):
        (package / name).write_text("{}\n", encoding="utf-8")
    (package / "v2_to_v2.1_crosswalk.csv").write_text("column\n", encoding="utf-8")
    for name in ("benchmark_summary.json",):
        (results / name).write_text("{}\n", encoding="utf-8")
    for name in ("benchmark_results.csv", "cluster_bootstrap_sensitivity.csv"):
        (results / name).write_text("column\n", encoding="utf-8")
    (results / "task_outcomes.jsonl").write_text("", encoding="utf-8")
    (reference / "triage_reference_profiles.jsonl").write_text("", encoding="utf-8")
    for name in (
        "benchmark_strata_summary.csv",
        "target_reference_summary.csv",
        "reference_agreement_audit.csv",
        "unresolved_case_summary.csv",
    ):
        (analysis / name).write_text("column\n", encoding="utf-8")
    for name in ("analysis_audit.json", "analysis_manifest.json"):
        (analysis / name).write_text("{}\n", encoding="utf-8")
    for name in ("feedback_parent_crosswalk.csv", "feedback_reference_summary.csv"):
        (crosswalk / name).write_text("column\n", encoding="utf-8")
    (crosswalk / "feedback_crosswalk_manifest.json").write_text("{}\n", encoding="utf-8")

    release = tmp_path / "release"
    archive = tmp_path / "archive"
    manifest = build_release(
        argparse.Namespace(
            compact_dir=str(compact),
            package_dir=str(package),
            results_dir=str(results),
            reference_dir=str(reference),
            analysis_dir=str(analysis),
            feedback_crosswalk_dir=str(crosswalk),
            adjudication_dir=str(adjudication),
            release_dir=str(release),
            archive_dir=str(archive),
        )
    )
    checksummed_names = {
        line.split("  ", 1)[1]
        for line in (release / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert "RELEASE_MANIFEST.json" in checksummed_names
    assert manifest["release_schema_version"] == 2
    assert manifest["counts"] == {
        "cases": 0,
        "references": 0,
        "tasks": 0,
        "outcomes": 0,
    }
    assert manifest["release_file_count"] == sum(
        path.is_file() for path in release.rglob("*")
    )
    assert manifest["archive_file_count"] == sum(
        path.is_file() for path in archive.rglob("*")
    )
