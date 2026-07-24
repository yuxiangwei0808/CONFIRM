from __future__ import annotations

from pathlib import Path

from bench.neuroclaimbench import (
    BenchmarkItem,
    EvaluationTask,
    SimplifiedBenchmarkClaim,
    TriageReferenceProfile,
    exact_contract_hash,
)
from bench.run_neuroclaimbench_finalize import build_simplified_claims
from confirm.contract import ClaimContract


def _contract(claim_id: str = "claim") -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": claim_id,
            "question": "Is age positively associated with outcome?",
            "estimand": {
                "type": "association",
                "outcome": "outcome",
                "predictor": "age",
                "direction": "positive",
                "unit": "scalar",
                "group": None,
                "region_set": None,
            },
            "covariates": ["sex"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
            "search_provenance": {
                "declared": True,
                "family_size": 1,
                "selection": "preregistered",
            },
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["sex"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "none",
                    "pattern_corr_min": 0.0,
                    "region_replication_frac_min": 1.0,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "fragile"],
        }
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def _item(item_id: str, task_id: str, contract: ClaimContract) -> BenchmarkItem:
    return BenchmarkItem.model_validate(
        {
            "benchmark_item_id": item_id,
            "claim_uid": f"uid-{item_id}",
            "semantic_cluster_id": f"sem-{item_id}",
            "benchmark_track": "scientific",
            "target_family": "normative_fmri",
            "modality": "other",
            "question": contract.question,
            "contract": contract.model_dump(mode="json"),
            "exact_contract_sha256": exact_contract_hash(contract),
            "semantic_claim_sha256": f"semantic-{item_id}",
            "source_references": [
                {
                    "source_collection": "test",
                    "source_id": item_id,
                    "source_path": "test.json",
                }
            ],
            "aliases": [item_id],
            "evaluation_task_ids": [task_id],
        }
    )


def _task(
    task_id: str,
    item_id: str,
    contract: ClaimContract,
    generator_spec: dict | None = None,
) -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        benchmark_item_id=item_id,
        contract=contract,
        evidence_role="source" if generator_spec is None else "synthetic_control",
        dataset_id="test",
        discovery_cohort="DISC",
        replication_cohorts=["REP"],
        generator_spec=generator_spec,
    )


def _profile(item_id: str, disposition: str = "confirm") -> TriageReferenceProfile:
    return TriageReferenceProfile.model_validate(
        {
            "benchmark_item_id": item_id,
            "benchmark_track": "scientific",
            "target_family": "normative_fmri",
            "source_label": "candidate_unknown",
            "source_adjudication_status": "unresolved",
            "triage_label": "supported" if disposition == "confirm" else "insufficient_evidence",
            "triage_disposition": disposition,
            "reference_strength": "provisional" if disposition == "confirm" else "evidence_gap",
            "derivation_rule": "test",
            "executable": True,
        }
    )


def test_build_collapses_exact_duplicate_tasks(tmp_path: Path) -> None:
    contract = _contract()
    items = [_item("item-a", "task-a", contract), _item("item-b", "task-b", contract)]
    tasks = [_task("task-a", "item-a", contract), _task("task-b", "item-b", contract)]
    package = tmp_path / "package"
    package.mkdir()
    _write_jsonl(package / "benchmark_items.jsonl", items)
    _write_jsonl(package / "evaluation_tasks.jsonl", tasks)
    profiles = tmp_path / "profiles.jsonl"
    _write_jsonl(profiles, [_profile("item-a"), _profile("item-b")])

    claims, manifest = build_simplified_claims(package, profiles)

    assert len(claims) == 1
    assert claims[0].reference_label == "confirm"
    assert claims[0].evaluation_task_ids == ["task-a", "task-b"]
    assert claims[0].dataset_ids == ["test"]
    assert manifest["benchmark_claim_count"] == 1


def test_generator_spec_keeps_randomized_controls_distinct(tmp_path: Path) -> None:
    contract = _contract()
    items = [_item("item-a", "task-a", contract), _item("item-b", "task-b", contract)]
    tasks = [
        _task("task-a", "item-a", contract, {"type": "random", "seed": 1}),
        _task("task-b", "item-b", contract, {"type": "random", "seed": 2}),
    ]
    package = tmp_path / "package"
    package.mkdir()
    _write_jsonl(package / "benchmark_items.jsonl", items)
    _write_jsonl(package / "evaluation_tasks.jsonl", tasks)
    profiles = tmp_path / "profiles.jsonl"
    _write_jsonl(profiles, [_profile("item-a"), _profile("item-b")])

    claims, _ = build_simplified_claims(package, profiles)

    assert len(claims) == 2
    assert len({claim.execution_identity_sha256 for claim in claims}) == 2


def test_unresolved_reference_is_not_score_eligible(tmp_path: Path) -> None:
    contract = _contract()
    package = tmp_path / "package"
    package.mkdir()
    _write_jsonl(package / "benchmark_items.jsonl", [_item("item-a", "task-a", contract)])
    _write_jsonl(package / "evaluation_tasks.jsonl", [_task("task-a", "item-a", contract)])
    profiles = tmp_path / "profiles.jsonl"
    _write_jsonl(profiles, [_profile("item-a", "request_evidence")])

    claims, _ = build_simplified_claims(package, profiles)

    assert claims[0].reference_label == "unresolved"
    assert claims[0].score_eligible is False


def test_manifest_separates_ambiguous_and_non_executable_exclusions(
    tmp_path: Path,
) -> None:
    contract = _contract()
    ready = _item("item-ready", "task-ready", contract)
    ambiguous = _item("item-ambiguous", "task-ambiguous", contract).model_copy(
        update={"migration_status": "ambiguous_unresolved"}
    )
    non_executable = _item(
        "item-non-executable", "task-non-executable", contract
    ).model_copy(update={"migration_status": "non_executable"})
    package = tmp_path / "package"
    package.mkdir()
    _write_jsonl(
        package / "benchmark_items.jsonl",
        [ready, ambiguous, non_executable],
    )
    _write_jsonl(
        package / "evaluation_tasks.jsonl",
        [_task("task-ready", "item-ready", contract)],
    )
    profiles = tmp_path / "profiles.jsonl"
    _write_jsonl(
        profiles,
        [
            _profile("item-ready"),
            _profile("item-ambiguous", "request_evidence"),
            _profile("item-non-executable", "request_evidence"),
        ],
    )

    claims, manifest = build_simplified_claims(package, profiles)

    assert len(claims) == 1
    assert manifest["excluded_from_executable_benchmark_count"] == 2
    assert manifest["exclusion_status_counts"] == {
        "ambiguous_unresolved": 1,
        "non_executable": 1,
    }
    assert "excluded_non_executable_count" not in manifest
