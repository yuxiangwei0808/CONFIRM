from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from bench import run_iterative_claim_search_replay as replay
from bench.run_iterative_claim_search_replay import _known_negative_or_fragile_source, _replay_specific_summary
from confirm.candidate_preflight import CandidatePreflightContext
from confirm.contract import ClaimContract
from confirm.evidence_partitions import build_evidence_partitions

_SOURCE_BUILDER_PATH = Path(__file__).resolve().parents[1] / "nbs" / "build_claim_search_source_from_results.py"
_SOURCE_BUILDER_SPEC = importlib.util.spec_from_file_location("claim_search_source_builder", _SOURCE_BUILDER_PATH)
assert _SOURCE_BUILDER_SPEC is not None and _SOURCE_BUILDER_SPEC.loader is not None
_SOURCE_BUILDER = importlib.util.module_from_spec(_SOURCE_BUILDER_SPEC)
_SOURCE_BUILDER_SPEC.loader.exec_module(_SOURCE_BUILDER)
_source_row = _SOURCE_BUILDER._source_row


def _contract(**overrides) -> ClaimContract:
    data = {
        "claim_id": "external_claim",
        "question": "External replay contract.",
        "estimand": {
            "type": "group_diff",
            "outcome": "smri_hippocampus",
            "predictor": "dx",
            "group": {"var": "dx", "case": "AD", "control": "CN"},
            "direction": "negative",
            "unit": "scalar",
            "region_set": None,
        },
        "covariates": ["age", "sex", "smri_icv"],
        "inclusion": None,
        "discovery_cohort": "ds000030_DISC",
        "replication_cohorts": ["ds000030_REP"],
        "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
        "gates": {
            "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
            "confound": {"require_covariates": ["age", "sex", "smri_icv"], "motion_check": False},
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
        "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
    }
    data.update(overrides)
    return ClaimContract.model_validate(data)


def test_external_result_row_with_embedded_contract_becomes_executable_source_row():
    contract = _contract()
    row = {
        "claim_id": "external_claim",
        "label_class": "random_null",
        "ground_truth": "random_null",
        "scoring_label": "random_null",
        "label_authority": "external_cnp",
        "final_label": "fragile",
        "rationale": "Failed gates: multiplicity",
        "gate_state": {"multiplicity": False, "confound": True, "power": True, "multiverse": True, "replication": True},
        "gate_results": {"contract": contract.model_dump(mode="json"), "primary": {"p": 0.5}},
        "contract": contract.model_dump(mode="json"),
    }

    converted = _source_row(row, Path("review-stage/curated-gate-external-cnp/CNP_external_results.json"))

    assert converted is not None
    assert converted["drafted_contract"]["claim_id"] == "external_claim"
    assert converted["gate_results"]["contract"]["claim_id"] == "external_claim"
    assert converted["source_scoring_label"] == "random_null"
    assert converted["source_ground_truth"] == "random_null"


def test_cnp_style_split_aliases_resolve_to_external_parquet(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    df = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(30)],
            "cohort": ["ds000030"] * 30,
            "site": ["site1"] * 30,
            "age": [65 + idx % 5 for idx in range(30)],
            "sex": ["F", "M"] * 15,
            "dx": ["AD"] * 15 + ["CN"] * 15,
            "smri_icv": [1000.0 + idx for idx in range(30)],
            "smri_hippocampus": [1.0 + idx for idx in range(30)],
        }
    )
    df.to_parquet(root / "ds000030.parquet")
    context = CandidatePreflightContext.from_roots([root])

    result = context.validate_contract(_contract())

    assert result.ok
    assert result.resolved_data_paths["ds000030_DISC"].endswith("ds000030.parquet")
    assert result.resolved_data_paths["ds000030_REP"].endswith("ds000030.parquet")


def test_known_negative_uses_source_labels_not_gate_verdict():
    assert not _known_negative_or_fragile_source({"claim_id": "real_claim", "gate_verdict_label": "fragile"})
    assert _known_negative_or_fragile_source({"claim_id": "real_claim", "source_scoring_label": "fragile"})
    assert _known_negative_or_fragile_source({"claim_id": "real_claim", "label_class": "random_null"})
    assert _known_negative_or_fragile_source({"claim_id": "neg_synthetic", "gate_verdict_label": "confirmed"})


def test_replay_summary_counts_known_negative_same_data_risk():
    summary = _replay_specific_summary(
        [
            {
                "known_negative_or_fragile_source": True,
                "exploratory_confirmed_count": 1,
                "same_data_exploratory_confirmed_count": 1,
            },
            {
                "known_negative_or_fragile_source": False,
                "exploratory_confirmed_count": 1,
                "same_data_exploratory_confirmed_count": 1,
            },
        ]
    )

    assert summary["known_negative_or_fragile_search_count"] == 1
    assert summary["known_negative_same_data_exploratory_confirmed_count"] == 1
    assert summary["known_negative_exploratory_risk_rate"] == 1.0


def test_source_builder_keeps_same_base_asd_claim_when_holdout_pair_exists(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    df = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx}" for idx in range(600)],
            "cohort": ["ABIDE1"] * 600,
            "site": [f"site{idx % 6}" for idx in range(600)],
            "age": [12 + idx % 20 for idx in range(600)],
            "sex": ["F", "M"] * 300,
            "dx": ["ASD"] * 300 + ["HC"] * 300,
            "fc_mean_abs": [float(idx % 20) for idx in range(600)],
        }
    )
    df.to_parquet(source_root / "ABIDE1.parquet")
    config = {
        "seed": 11,
        "output_root": str(tmp_path / "parts"),
        "default_split": {"discovery": 0.6, "replication": 0.2, "holdout": 0.2},
        "min_rows": {"default_partition_rows": 20, "continuous_rows": 20},
        "datasets": [
            {
                "dataset": "ABIDE1",
                "source": str(source_root / "ABIDE1.parquet"),
                "target_families": ["asd"],
                "split_method": "site_or_stratified",
            }
        ],
    }
    config_path = tmp_path / "evidence.yml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = build_evidence_partitions(config_path, tmp_path / "parts")
    contract = ClaimContract.model_validate(
        {
            "claim_id": "asd_same_base",
            "question": "Is ASD associated with fc_mean_abs in ABIDE1?",
            "estimand": {
                "type": "group_diff",
                "outcome": "fc_mean_abs",
                "predictor": "dx",
                "group": {"var": "dx", "case": "ASD", "control": "HC"},
                "direction": "two_sided",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["age", "sex", "site"],
            "inclusion": None,
            "discovery_cohort": "ABIDE1_DISC",
            "replication_cohorts": ["ABIDE1_REP"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["age", "sex", "site"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "combat",
                    "pattern_corr_min": 0.0,
                    "region_replication_frac_min": 0.0,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "asd_same_base",
                        "target_family": "asd",
                        "source_mode": "llm_proposed",
                        "model_spec": "benchmark/test",
                        "final_label": "fragile",
                        "rationale": "Failed gates: replication",
                        "contract": contract.model_dump(mode="json"),
                        "gate_results": {"contract": contract.model_dump(mode="json")},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output = _SOURCE_BUILDER.run(
        Namespace(
            input=[str(result_path)],
            out=str(tmp_path / "claim_search_source.json"),
            model_spec="benchmark/test",
            evidence_manifest=str(tmp_path / "parts" / "manifest.json"),
            require_excluded_evidence=True,
            exclude_external_only_sources=True,
        )
    )

    rows = output["source"]["models"][0]["initial_claims"]
    assert manifest.has_excluded_evidence_for_contract(contract)
    assert len(rows) == 1
    assert rows[0]["target_family"] == "asd"
    assert rows[0]["source_mode"] == "llm_proposed"
    assert output["summary"]["eligible_claims_by_target_family"]["asd"] == 1
    assert output["summary"]["excluded_by_evidence_policy"] == 0


def test_holdout_evaluator_falls_back_to_external_pair_when_holdout_pair_is_unusable(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    small = pd.DataFrame(
        {
            "subject_id": [f"adni-{idx}" for idx in range(60)],
            "cohort": ["ADNI"] * 60,
            "site": [f"site{idx % 3}" for idx in range(60)],
            "age": [70 + idx % 5 for idx in range(60)],
            "sex": ["F", "M"] * 30,
            "dx": ["AD", "CN"] * 30,
            "smri_icv": [1000.0 + idx for idx in range(60)],
            "smri_hippocampus": [1.0 + idx for idx in range(60)],
        }
    )
    external = pd.DataFrame(
        {
            "subject_id": [f"nacc-{idx}" for idx in range(200)],
            "cohort": ["NACC"] * 200,
            "site": [f"site{idx % 5}" for idx in range(200)],
            "age": [70 + idx % 5 for idx in range(200)],
            "sex": ["F", "M"] * 100,
            "dx": ["AD", "CN"] * 100,
            "smri_icv": [1000.0 + idx for idx in range(200)],
            "smri_hippocampus": [1.0 + idx for idx in range(200)],
        }
    )
    small.to_parquet(source_root / "ADNI.parquet")
    external.to_parquet(source_root / "NACC.parquet")
    config = {
        "seed": 13,
        "output_root": str(tmp_path / "parts"),
        "default_split": {"discovery": 0.6, "replication": 0.2, "holdout": 0.2},
        "min_rows": {"default_partition_rows": 20, "continuous_rows": 20},
        "datasets": [
            {
                "dataset": "ADNI",
                "source": str(source_root / "ADNI.parquet"),
                "target_families": ["ad_aging"],
                "split_method": "site_or_stratified",
            },
            {
                "dataset": "NACC",
                "source": str(source_root / "NACC.parquet"),
                "target_families": ["ad_aging"],
                "external_eval": True,
                "split_method": "site_or_stratified",
            },
        ],
        "external_evidence_sets": [
            {
                "evidence_set_id": "nacc_primary",
                "target_family": "ad_aging",
                "modality": "sMRI",
                "feature_family": "regional_volume",
                "discovery_partition_id": "NACC_EXTERNAL_DISC",
                "replication_partition_ids": ["NACC_EXTERNAL_REP"],
                "supported_predictors": ["dx"],
                "supported_group_vars": ["dx"],
                "priority": 1,
                "confirmation_role": "primary",
            }
        ],
    }
    config_path = tmp_path / "evidence.yml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = build_evidence_partitions(config_path, tmp_path / "parts")
    contract = _contract(
        discovery_cohort="ADNI_DISC",
        replication_cohorts=["ADNI_REP"],
        covariates=["age", "sex", "smri_icv"],
    )
    calls = []

    def fake_execute(contract, data_roots, *, evidence_scope="current", target_family=None, source_contract=None):
        calls.append((contract, evidence_scope, target_family, source_contract))
        return {
            "final_label": "confirmed",
            "gate_results": {
                "evidence_scope": {"scope": evidence_scope, "target_family": target_family},
                "data_paths": {"discovery": "external-disc", "replication": ["external-rep"]},
            },
        }

    monkeypatch.setattr(replay, "_execute_candidate_contract", fake_execute)
    evaluator = replay._candidate_evaluator(
        [tmp_path / "parts" / "cohorts"],
        evidence_manifest=manifest,
        evidence_kind="holdout",
    )

    class Candidate:
        proposed_contract = contract

    result = evaluator(Candidate())

    assert result["final_label"] == "confirmed"
    assert calls
    mapped_contract, evidence_scope, target_family, source_contract = calls[0]
    assert evidence_scope == "external"
    assert mapped_contract.discovery_cohort == "NACC_EXTERNAL_DISC"
    assert mapped_contract.replication_cohorts == ["NACC_EXTERNAL_REP"]
    assert target_family == "ad_aging"
    assert source_contract.discovery_cohort == "ADNI_DISC"
