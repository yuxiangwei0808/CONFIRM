from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from confirm.candidate_preflight import CandidatePreflightContext
from confirm.agent import _contract_prompt, _execute_contract
from confirm.contract import ClaimContract
from confirm.verdict import Verdict, classify_support

from bench import run_drafted_contract_gates as gates
from bench import run_initial_claim_drafting as drafting


def _contract_payload(claim_id: str = "test_claim") -> dict:
    return {
        "claim_id": claim_id,
        "question": "Is age associated with lower smri_hippocampus?",
        "estimand": {
            "type": "association",
            "outcome": "smri_hippocampus",
            "predictor": "age",
            "group": None,
            "direction": "negative",
            "unit": "scalar",
            "region_set": None,
        },
        "covariates": ["sex", "eTIV"],
        "inclusion": None,
        "discovery_cohort": "ADNI_DISC",
        "replication_cohorts": ["OASIS3_REP"],
        "gates": {
            "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
            "confound": {"require_covariates": ["sex", "eTIV"], "motion_check": False},
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


def _fc_contract_payload(claim_id: str = "fc_claim") -> dict:
    payload = _contract_payload(claim_id)
    payload["question"] = "Is age associated with resting-state functional connectivity?"
    payload["estimand"] = {
        "type": "association",
        "outcome": "fc_fc_",
        "predictor": "age",
        "group": None,
        "direction": "two_sided",
        "unit": "brainwide",
        "region_set": None,
    }
    payload["covariates"] = ["sex", "site"]
    payload["discovery_cohort"] = "ABCD_DISC"
    payload["replication_cohorts"] = ["ADHD200_REP"]
    payload["gates"]["multiplicity"]["family_size"] = 2
    payload["gates"]["confound"]["require_covariates"] = ["sex", "site"]
    return payload


def _write_sources(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "claim_id",
        "target_family",
        "source_mode",
        "question",
        "label_class",
        "label_basis",
        "source_citation",
        "notes",
        "include_in_main",
        "discovery_cohort",
        "replication_cohorts",
        "allowed_covariates",
        "shared_outcome_columns_sample",
        "shared_outcome_prefixes",
        "group_var",
        "case_label",
        "control_label",
        "source_pmid",
        "source_seed_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def test_initial_catalog_and_stage2_exclude_reserved_evaluation_cohorts(tmp_path: Path) -> None:
    root = tmp_path / "cohorts"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "cohort": ["ADNI_DISC", "ADNI_DISC"],
            "site": ["a", "b"],
            "age": [60, 70],
            "sex": ["F", "M"],
            "eTIV": [1_400_000.0, 1_500_000.0],
            "smri_hippocampus": [7_000.0, 6_000.0],
        }
    )
    frame.to_parquet(root / "ADNI_DISC.parquet", index=False)
    frame.assign(cohort="NACC_EXTERNAL_DISC").to_parquet(
        root / "NACC_EXTERNAL_DISC.parquet",
        index=False,
    )
    frame.assign(cohort="ADNI_HOLDOUT_DISC").to_parquet(
        root / "ADNI_HOLDOUT_DISC.parquet",
        index=False,
    )

    catalog = drafting._merge_catalogs([root])
    assert [item["cohort"] for item in catalog["cohorts"]] == ["ADNI_DISC"]
    assert drafting.DEFAULT_DATA_ROOTS == (
        Path("data/prepared_data/evidence_partitions/benchmark_ready/cohorts"),
    )

    payload = _contract_payload("external_not_allowed")
    payload["discovery_cohort"] = "NACC_EXTERNAL_DISC"
    payload["replication_cohorts"] = ["NACC_EXTERNAL_REP"]
    claim, error = gates._evaluate_one(
        {"claim_id": "external_not_allowed", "drafted_contract": payload},
        [str(root)],
    )
    assert claim is None
    assert "Stage 2 cannot evaluate excluded evidence cohorts" in error["error"]


def test_fixed_sources_keep_synthetic_separate(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.csv"
    synthetic = tmp_path / "synthetic.csv"
    _write_sources(
        fixed,
        [
            {
                "claim_id": "lit",
                "target_family": "ad_aging",
                "source_mode": "literature",
                "question": "literature question",
                "label_class": "known_positive",
                "label_basis": "canonical_literature",
                "source_citation": "citation",
                "notes": "",
                "include_in_main": "true",
            },
            {
                "claim_id": "litg",
                "target_family": "ad_aging",
                "source_mode": "literature_grounded",
                "question": "pubmed-grounded question",
                "label_class": "candidate_unknown",
                "label_basis": "pubmed_literature",
                "source_citation": "PMID:1",
                "notes": "",
                "include_in_main": "true",
            },
            {
                "claim_id": "inv",
                "target_family": "asd",
                "source_mode": "inventory",
                "question": "inventory question",
                "label_class": "fragile",
                "label_basis": "internal_split_half",
                "source_citation": "citation",
                "notes": "",
                "include_in_main": "true",
            },
        ],
    )
    _write_sources(
        synthetic,
        [
            {
                "claim_id": "stress",
                "target_family": "normative_fmri",
                "source_mode": "synthetic_stress",
                "question": "stress question",
                "label_class": "known_null",
                "label_basis": "synthetic_stress",
                "source_citation": "synthetic",
                "notes": "",
                "include_in_main": "false",
            }
        ],
    )

    selected = drafting.load_claim_questions("all", fixed, synthetic)
    assert [row.claim_id for row in selected] == ["lit", "litg", "inv"]

    selected_with_stress = drafting.load_claim_questions("all", fixed, synthetic, include_synthetic_stress=True)
    assert [row.claim_id for row in selected_with_stress] == ["lit", "litg", "inv", "stress"]
    assert selected_with_stress[-1].include_in_main is False


class StubLLM:
    model = "stub"

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        if response_model is drafting.QuestionGenerationResponse:
            payload = json.loads(user)
            target = payload["target_family"]
            count = payload["exact_questions"]
            return json.dumps(
                {
                    "questions": [
                        {
                            "claim_id": f"{target}_q{i}",
                            "target_family": target,
                            "question": f"Question {i} for {target}",
                            "scientific_rationale": "connected target rationale",
                            "expected_modality": "sMRI",
                            "suggested_cohort_family": "ADNI/OASIS3",
                        }
                        for i in range(count)
                    ]
                }
            )
        return json.dumps(_contract_payload("drafted"))

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_llm_generation_quota_is_per_target(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(drafting, "make_llm", lambda spec: StubLLM())
    monkeypatch.setattr(
        drafting,
        "_merge_catalogs",
        lambda roots: {
            "data_roots": [],
            "cohorts": [
                {
                    "cohort": "ADNI_DISC",
                    "n": 100,
                    "idps": ["smri_hippocampus"],
                    "dx_levels": ["CN", "Dementia"],
                },
                {
                    "cohort": "OASIS3_REP",
                    "n": 100,
                    "idps": ["smri_hippocampus"],
                    "dx_levels": ["CN", "Dementia"],
                },
            ],
        },
    )
    args = argparse.Namespace(
        mode="llm_proposed",
        model="stub",
        num_claims_per_target=2,
        target_family=["ad_aging", "asd"],
        out_dir=str(tmp_path),
        fixed_claims=str(tmp_path / "missing_fixed.csv"),
        synthetic_claims=str(tmp_path / "missing_synthetic.csv"),
        include_synthetic_stress=False,
        schema_retries=0,
        llm_max_tokens=8192,
        max_workers=2,
        parallel_backend="thread",
        no_progress=True,
        limit=None,
        data_root=[],
    )

    summary = drafting.run(args)

    assert summary["n_questions"] == 4
    assert summary["question_counts_by_source_mode"] == {"llm_proposed": 4}
    assert summary["question_counts_by_target_family"] == {"ad_aging": 2, "asd": 2}
    assert len((tmp_path / "drafted_contracts.jsonl").read_text().splitlines()) == 4


class ShortThenExactQuestionLLM:
    model = "short-then-exact"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls += 1
        payload = json.loads(user)
        target = payload["target_family"]
        count = payload["exact_questions"]
        returned_count = count - 1 if self.calls == 1 else count
        return json.dumps(
            {
                "questions": [
                    {
                        "claim_id": f"{target}_q{i}",
                        "target_family": target,
                        "question": f"Question {i} for {target}",
                        "scientific_rationale": "connected target rationale",
                        "expected_modality": "sMRI",
                        "suggested_cohort_family": "ADNI/OASIS3",
                    }
                    for i in range(returned_count)
                ]
            }
        )

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, drafting.QuestionGenerationResponse)


def test_llm_question_generation_retries_until_exact_quota() -> None:
    llm = ShortThenExactQuestionLLM()

    questions, prompts, responses = drafting.generate_llm_questions(
        llm,
        {"cohorts": []},
        "ad_aging",
        count=2,
        schema_retries=1,
    )

    assert len(questions) == 2
    assert len(prompts) == 2
    assert "Expected exactly 2 questions" in responses[0]["schema_error"]
    assert "fix_previous_validation_error" in prompts[1]["user"]


class RetryLLM:
    model = "retry"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls += 1
        if self.calls == 1:
            return '{"claim_id": "bad"}'
        return json.dumps(_contract_payload("valid_after_retry"))

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_contract_drafting_retries_schema_failures() -> None:
    question = drafting.ClaimQuestion(
        claim_id="source_id",
        target_family="ad_aging",
        source_mode="literature",
        question="Is age associated with lower smri_hippocampus?",
    )
    llm = RetryLLM()

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        {"cohorts": []},
        llm,
        schema_retries=1,
    )

    assert contract.claim_id == "source_id"
    assert len(prompts) == 2
    assert "schema_error" in responses[0]


class SourceDriftLLM:
    model = "source-drift"

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        return json.dumps(_contract_payload("drifted"))

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_fixed_source_claims_reject_explicit_cohort_drift() -> None:
    question = drafting.ClaimQuestion(
        claim_id="aibl_ad_smri",
        target_family="ad_aging",
        source_mode="literature",
        question=(
            "For AD structural marker in AIBL in AIBL/AD, using AIBL as discovery and "
            "ADNI/OASIS3 as replication, test the claim with expected direction negative."
        ),
    )

    with pytest.raises(drafting.DraftContractError) as err:
        drafting.draft_contract_with_trace(
            question,
            {"cohorts": []},
            SourceDriftLLM(),
            schema_retries=0,
        )

    assert "source_preservation_error" in str(err.value)
    assert "discovery_cohort" in err.value.responses[0]["source_preservation_error"]


class InclusionRetryLLM:
    model = "inclusion-retry"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls += 1
        payload = _contract_payload("draft_with_inclusion")
        payload["inclusion"] = "male Alzheimer's participants" if self.calls == 1 else 'sex == "M"'
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def _write_preflight_cohorts(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    n = 60
    table = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx:03d}" for idx in range(n)],
            "age": [60 + idx for idx in range(n)],
            "sex": ["M" if idx % 4 < 2 else "F" for idx in range(n)],
            "dx": ["CN" for _ in range(n)],
            "eTIV": [1450.0 + ((idx * 37) % 211) for idx in range(n)],
            "smri_hippocampus": [
                4.0 - idx * 0.006 + ((idx * 11) % 7) * 0.003
                for idx in range(n)
            ],
        }
    )
    for cohort in ("ADNI_DISC", "OASIS3_REP"):
        table.to_parquet(root / f"{cohort}.parquet", index=False)
    cohort_details = [
        {
            "cohort": cohort,
            "n": n,
            "columns": list(table.columns),
            "idps": ["smri_hippocampus"],
            "dx_levels": ["CN"],
        }
        for cohort in ("ADNI_DISC", "OASIS3_REP")
    ]
    return {"data_roots": [str(root)], "cohorts": cohort_details}


def _write_stage1_fc_cohorts(root: Path, *, dx_levels: list[str] | None = None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    n = 60
    levels = dx_levels or ["HC"] * n
    table = pd.DataFrame(
        {
            "subject_id": [f"sub-{idx:03d}" for idx in range(n)],
            "cohort": ["cohort"] * n,
            "site": [f"site{idx % 3 + 1}" for idx in range(n)],
            "age": [10 + idx for idx in range(n)],
            "sex": ["M" if idx % 4 < 2 else "F" for idx in range(n)],
            "dx": [levels[idx % len(levels)] for idx in range(n)],
            "fc_fc_Default_Default": [
                0.1 + idx * 0.001 + ((idx * 7) % 11) * 0.0003
                for idx in range(n)
            ],
            "fc_fc_Default_DorsAttn": [
                0.2 + idx * 0.0007 + ((idx * 13) % 17) * 0.0002
                for idx in range(n)
            ],
        }
    )
    cohorts = []
    for cohort in ("ABCD_DISC", "ADHD200_REP"):
        cohort_table = table.copy()
        cohort_table["cohort"] = cohort
        cohort_table.to_parquet(root / f"{cohort}.parquet", index=False)
        cohorts.append(
            {
                "cohort": cohort,
                "n": n,
                "columns": list(cohort_table.columns),
                "idps": ["fc_fc_Default_Default", "fc_fc_Default_DorsAttn"],
                "dx_levels": sorted(set(cohort_table["dx"])),
                "data_root": str(root),
            }
        )
    return {"data_roots": [str(root)], "cohorts": cohorts}


def _fc_literature_question() -> drafting.ClaimQuestion:
    return drafting.ClaimQuestion(
        claim_id="pubmed_fc",
        target_family="adhd",
        source_mode="literature_grounded",
        question="For literature-grounded age effect on functional connectivity, use fixed ADHD evidence.",
        label_basis="pubmed_literature",
        source_citation="PMID:1",
        discovery_cohort="ABCD_DISC",
        replication_cohorts="ADHD200_REP",
        allowed_covariates="age;sex;site",
        shared_outcome_columns_sample="fc_fc_Default_Default;fc_fc_Default_DorsAttn",
        shared_outcome_prefixes="fc_fc_",
        source_pmid="1",
        source_seed_id="seed_fc",
    )


def test_contract_drafting_retries_inclusion_preflight_failures(tmp_path: Path) -> None:
    question = drafting.ClaimQuestion(
        claim_id="source_id",
        target_family="ad_aging",
        source_mode="literature",
        question="In male participants, is age associated with lower smri_hippocampus?",
    )
    catalog = _write_preflight_cohorts(tmp_path / "cohorts")
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])
    llm = InclusionRetryLLM()

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        catalog,
        llm,
        schema_retries=1,
        preflight_context=context,
    )

    assert contract.inclusion == 'sex == "M"'
    assert len(prompts) == 2
    assert "preflight_error" in responses[0]
    assert "invalid inclusion query" in responses[0]["preflight_error"]
    assert "semantic_preflight_error" in prompts[1]["user"]


class MissingFieldRetryLLM:
    model = "missing-field-retry"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls += 1
        payload = _contract_payload("draft_with_missing_field")
        if self.calls == 1:
            payload["estimand"]["predictor"] = "cognition"
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_contract_drafting_retries_missing_predictor_preflight_failures(tmp_path: Path) -> None:
    question = drafting.ClaimQuestion(
        claim_id="source_id",
        target_family="ad_aging",
        source_mode="literature",
        question="Is age associated with lower smri_hippocampus?",
    )
    catalog = _write_preflight_cohorts(tmp_path / "cohorts")
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])
    llm = MissingFieldRetryLLM()

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        catalog,
        llm,
        schema_retries=1,
        preflight_context=context,
    )

    assert contract.estimand.predictor == "age"
    assert len(prompts) == 2
    assert "missing analysis columns" in responses[0]["preflight_error"]
    assert "semantic_preflight_error" in prompts[1]["user"]


def test_literature_grounded_prompt_is_pair_specific_and_forbids_unshared_covariates(tmp_path: Path) -> None:
    catalog = _write_stage1_fc_cohorts(tmp_path / "cohorts")
    question = _fc_literature_question()

    prompt = drafting._contract_prompt_for_question(question, catalog)

    assert "ABCD_DISC" in prompt
    assert "ADHD200_REP" in prompt
    assert "ADNI_DISC" not in prompt
    assert "eTIV" in prompt
    assert "FORBIDDEN_COVARIATES" in prompt
    assert "fc_fc_" in prompt
    assert "fc_fc_*_z" not in prompt


class BadThenFixedFCLLM:
    model = "bad-then-fixed-fc"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls += 1
        payload = _fc_contract_payload("fc_retry")
        if self.calls == 1:
            payload["estimand"]["outcome"] = "fc_fc_*_z"
            payload["covariates"] = ["sex", "site", "eTIV"]
            payload["gates"]["confound"]["require_covariates"] = ["sex", "site", "eTIV"]
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_literature_grounded_drafting_retries_unsupported_covariate_and_fc_suffix(tmp_path: Path) -> None:
    catalog = _write_stage1_fc_cohorts(tmp_path / "cohorts")
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])
    question = _fc_literature_question()

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        catalog,
        BadThenFixedFCLLM(),
        schema_retries=1,
        preflight_context=context,
    )

    assert contract.estimand.outcome == "fc_fc_"
    assert "eTIV" not in contract.covariates
    assert len(prompts) == 2
    assert "missing outcome columns" in responses[0]["preflight_error"]
    assert "missing analysis columns" not in responses[0]["preflight_error"]
    assert responses[0]["retry_hints"]["allowed_covariates"] == ["age", "sex", "site"]
    assert responses[0]["retry_hints"]["shared_outcome_prefixes"] == ["fc_fc_"]


def test_literature_grounded_source_drift_gets_exact_fixed_cohort_hint() -> None:
    question = _fc_literature_question()

    with pytest.raises(drafting.DraftContractError) as err:
        drafting.draft_contract_with_trace(
            question,
            {"cohorts": []},
            SourceDriftLLM(),
            schema_retries=0,
        )

    hints = err.value.responses[0]["retry_hints"]
    assert hints["fixed_discovery_cohort"] == "ABCD_DISC"
    assert hints["fixed_replication_cohorts"] == ["ADHD200_REP"]
    assert "Use exactly these fixed literature-grounded cohorts" in hints["instruction"]


def test_confirm_dx_virtual_column_is_available_to_preflight_and_execution(tmp_path: Path) -> None:
    catalog = _write_stage1_fc_cohorts(tmp_path / "cohorts", dx_levels=["0.0", "1.0"])
    payload = _fc_contract_payload("confirm_dx_contract")
    payload["estimand"] = {
        "type": "group_diff",
        "outcome": "fc_fc_",
        "predictor": "confirm_dx",
        "group": {"var": "confirm_dx", "case": "case", "control": "control"},
        "direction": "two_sided",
        "unit": "brainwide",
        "region_set": None,
    }
    payload["covariates"] = ["age", "sex", "site"]
    payload["gates"]["confound"]["require_covariates"] = ["age", "sex", "site"]
    contract = ClaimContract.model_validate(payload)
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])

    preflight = context.validate_contract(contract)
    verdict, results, paths = _execute_contract(contract, Path(catalog["data_roots"][0]))

    assert preflight.ok, preflight.violations
    assert "confirm_dx" in context.cohorts["ABCD_DISC"].columns
    assert len(paths) == 2
    assert verdict.label in {"confirmed", "fragile", "non_replicated", "under_powered"}
    assert results["contract"]["estimand"]["group"]["var"] == "confirm_dx"


def test_literature_grounded_missing_shared_outcomes_skips_before_llm(monkeypatch, tmp_path: Path) -> None:
    catalog = _write_stage1_fc_cohorts(tmp_path / "cohorts")
    question = _fc_literature_question().model_copy(update={"shared_outcome_columns_sample": "", "shared_outcome_prefixes": ""})
    monkeypatch.setattr(drafting, "_make_llm", lambda model_spec, max_tokens: BadThenFixedFCLLM())

    _, draft, prompts, responses, validation = drafting._draft_question_worker(
        (0, question.model_dump(mode="json"), catalog, "stub", 1, 8192)
    )

    assert draft is None
    assert prompts == []
    assert responses == []
    assert validation["error_stage"] == "semantic_preflight"
    assert validation["draft_disposition"] == "unsupported_local_columns"
    assert "no shared outcome columns" in validation["error"]


def test_accepted_contract_audit_flags_unsupported_contract_terms() -> None:
    row = {
        "claim_id": "bad_supported_claim",
        "target_family": "psychosis",
        "question": "Schizophrenia changes effective connectivity.",
        "drafted_contract": _fc_contract_payload("bad_supported_claim"),
    }
    row["drafted_contract"]["question"] = "Schizophrenia changes effective connectivity."

    audit = drafting._accepted_contract_audit([row])

    assert audit["unsupported_contract_hit_count"] == 1
    assert audit["unsupported_contract_hits"][0]["claim_id"] == "bad_supported_claim"


class InvalidGroupFCLLM:
    model = "invalid-group-fc"

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        payload = _fc_contract_payload("bad_group")
        payload["estimand"] = {
            "type": "group_diff",
            "outcome": "fc_fc_",
            "predictor": "dx",
            "group": {"var": "dx", "case": "ASD", "control": "HC"},
            "direction": "two_sided",
            "unit": "brainwide",
            "region_set": None,
        }
        payload["covariates"] = ["age", "sex", "site"]
        payload["gates"]["confound"]["require_covariates"] = ["age", "sex", "site"]
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_literature_grounded_invalid_group_is_semantic_preflight_disposition(monkeypatch, tmp_path: Path) -> None:
    catalog = _write_stage1_fc_cohorts(tmp_path / "cohorts", dx_levels=["HC"])
    question = _fc_literature_question().model_copy(update={"group_var": "dx", "case_label": "ASD", "control_label": "HC"})
    monkeypatch.setattr(drafting, "_make_llm", lambda model_spec, max_tokens: InvalidGroupFCLLM())

    _, draft, _, responses, validation = drafting._draft_question_worker(
        (0, question.model_dump(mode="json"), catalog, "stub", 0, 8192)
    )

    assert draft is None
    assert validation["error_stage"] == "semantic_preflight"
    assert validation["draft_disposition"] == "invalid_group_contrast"
    assert "missing group levels" in responses[0]["preflight_error"]


class CategoricalPredictorRetryLLM:
    model = "categorical-predictor-retry"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls += 1
        payload = _contract_payload("categorical_predictor")
        if self.calls == 1:
            payload["estimand"]["predictor"] = "sex"
            payload["estimand"]["direction"] = "two_sided"
            payload["covariates"] = ["age", "eTIV"]
            payload["gates"]["confound"]["require_covariates"] = ["age", "eTIV"]
        else:
            payload["estimand"] = {
                "type": "group_diff",
                "outcome": "smri_hippocampus",
                "predictor": "sex",
                "group": {"var": "sex", "case": "F", "control": "M"},
                "direction": "two_sided",
                "unit": "scalar",
                "region_set": None,
            }
            payload["covariates"] = ["age", "eTIV"]
            payload["gates"]["confound"]["require_covariates"] = ["age", "eTIV"]
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_contract_drafting_retries_categorical_association_predictors(tmp_path: Path) -> None:
    question = drafting.ClaimQuestion(
        claim_id="source_id",
        target_family="ad_aging",
        source_mode="literature",
        question="Is sex associated with smri_hippocampus?",
    )
    catalog = _write_preflight_cohorts(tmp_path / "cohorts")
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])
    llm = CategoricalPredictorRetryLLM()

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        catalog,
        llm,
        schema_retries=1,
        preflight_context=context,
    )

    assert contract.estimand.type == "group_diff"
    assert contract.estimand.group is not None
    assert "association predictor 'sex'" in responses[0]["preflight_error"]
    assert "semantic_preflight_error" in prompts[1]["user"]


class TautologyInclusionRetryLLM:
    model = "tautology-inclusion-retry"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        self.calls += 1
        payload = _contract_payload("tautology_inclusion")
        payload["inclusion"] = "dx == dx" if self.calls == 1 else None
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_contract_drafting_retries_tautological_inclusion(tmp_path: Path) -> None:
    question = drafting.ClaimQuestion(
        claim_id="source_id",
        target_family="ad_aging",
        source_mode="literature",
        question="Is age associated with lower smri_hippocampus?",
    )
    catalog = _write_preflight_cohorts(tmp_path / "cohorts")
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])
    llm = TautologyInclusionRetryLLM()

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        catalog,
        llm,
        schema_retries=1,
        preflight_context=context,
    )

    assert contract.inclusion is None
    assert "tautological inclusion comparison" in responses[0]["preflight_error"]
    assert "semantic_preflight_error" in prompts[1]["user"]


class RawDxDiseaseContrastLLM:
    model = "raw-dx-disease-contrast"

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        payload = _fc_contract_payload("raw_dx")
        payload["question"] = "Is ADHD diagnosis associated with resting-state FC?"
        payload["estimand"] = {
            "type": "group_diff",
            "outcome": "fc_fc_",
            "predictor": "dx",
            "group": {"var": "dx", "case": "1.0", "control": "0.0"},
            "direction": "two_sided",
            "unit": "brainwide",
            "region_set": None,
        }
        payload["covariates"] = ["age", "sex", "site"]
        payload["gates"]["confound"]["require_covariates"] = ["age", "sex", "site"]
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_llm_proposed_disease_contrast_uses_confirm_dx_before_preflight(tmp_path: Path) -> None:
    catalog = _write_stage1_fc_cohorts(tmp_path / "cohorts", dx_levels=["0.0", "1.0"])
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])
    question = drafting.ClaimQuestion(
        claim_id="adhd_llm",
        target_family="adhd",
        source_mode="llm_proposed",
        question="Does ADHD diagnosis differ from controls in fMRI connectivity?",
    )

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        catalog,
        RawDxDiseaseContrastLLM(),
        schema_retries=0,
        preflight_context=context,
    )

    assert contract.estimand.predictor == "confirm_dx"
    assert contract.estimand.group is not None
    assert contract.estimand.group.var == "confirm_dx"
    assert contract.estimand.group.case == "case"
    assert contract.estimand.group.control == "control"
    assert responses[0]["preflight"]["ok"] is True
    assert "confirm_dx" in prompts[0]["user"]


class EtivFCLLM:
    model = "etiv-fc"

    def complete_structured(self, system: str, user: str, response_model: type) -> str:
        payload = _fc_contract_payload("etiv_fc")
        payload["covariates"] = ["sex", "site", "eTIV"]
        payload["gates"]["confound"]["require_covariates"] = ["sex", "site", "eTIV"]
        return json.dumps(payload)

    def complete(self, system: str, user: str) -> str:
        return self.complete_structured(system, user, ClaimContract)


def test_llm_proposed_fc_contract_drops_etiv_before_preflight(tmp_path: Path) -> None:
    catalog = _write_stage1_fc_cohorts(tmp_path / "cohorts")
    context = CandidatePreflightContext.from_roots(catalog["data_roots"])
    question = drafting.ClaimQuestion(
        claim_id="fc_llm",
        target_family="normative_fmri",
        source_mode="llm_proposed",
        question="Is age associated with fMRI functional connectivity?",
    )

    contract, prompts, responses = drafting.draft_contract_with_trace(
        question,
        catalog,
        EtivFCLLM(),
        schema_retries=0,
        preflight_context=context,
    )

    assert "eTIV" not in contract.covariates
    assert "eTIV" not in contract.gates.confound.require_covariates
    assert responses[0]["preflight"]["ok"] is True
    assert "do not include eTIV" in prompts[0]["user"]


def test_contract_prompt_and_schema_describe_high_risk_fields() -> None:
    catalog = {
        "cohorts": [
            {
                "cohort": "ADNI_DISC",
                "n": 100,
                "columns": ["age", "sex", "dx", "eTIV", "smri_hippocampus"],
                "idps": ["smri_hippocampus"],
                "dx_levels": ["CN", "Dementia"],
            }
        ]
    }
    prompt = _contract_prompt("Question", catalog)
    schema_text = json.dumps(ClaimContract.model_json_schema())

    assert "Use estimand.type=''group_diff''" in prompt
    assert "sex, diagnosis/dx" in prompt
    assert "confirm_dx" in prompt
    assert "not use eTIV for fMRI/FC claims" in prompt
    assert "age:sex" in prompt
    assert "dx == dx" in prompt
    assert "fc_fc_*_z" not in prompt
    assert "fc_fc_*_z" not in schema_text
    assert "Use 'association' only for numeric continuous predictors" in schema_text
    assert "Do not use prose, tautologies like" in schema_text


def test_gate_runner_writes_feedback_compatible_rows(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "drafted_contracts.jsonl"
    row = {
        "claim_id": "source_id",
        "target_family": "ad_aging",
        "source_mode": "literature",
        "model_spec": "stub",
        "question": "Question",
        "draft_success": True,
        "label_class": "known_positive",
        "label_basis": "canonical_literature",
        "drafted_contract": _contract_payload("source_id"),
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        gates,
        "resolve_execution_root",
        lambda contract, roots: tmp_path,
    )

    def fake_execute(
        contract: ClaimContract,
        root: Path,
        ref_effect=None,
        minimum_evidence_tier="confirmed",
    ):
        gate_state = {
            "search_provenance": True,
            "confound": True,
            "confound_completeness": True,
            "multiplicity": True,
            "power": True,
            "multiverse": True,
            "replication": True,
        }
        verdict = Verdict(
            label="confirmed",
            abstained=False,
            rationale="ok",
            gates=gate_state,
        )
        return (
            verdict,
            {
                "contract": contract.model_dump(mode="json"),
                "primary": {"p": 0.01},
                "support_decision": classify_support(
                    gate_state,
                    minimum_evidence_tier,
                ),
            },
            [tmp_path / "ADNI_DISC.parquet"],
        )

    monkeypatch.setattr(gates, "evaluate_contract", fake_execute)

    out = tmp_path / "out"
    args = argparse.Namespace(
        contracts=str(source),
        out_dir=str(out),
        data_root=[str(tmp_path)],
        max_workers=1,
        parallel_backend="thread",
        minimum_evidence_tier="confirmed",
        no_progress=True,
        limit=None,
    )
    payload = gates.run(args)

    claim = payload["claims"][0]
    assert claim["final_label"] == "confirmed"
    assert claim["contract"]["claim_id"] == "source_id"
    assert claim["drafted_contract"]["claim_id"] == "source_id"
    assert claim["gate_results"]["primary"]["p"] == 0.01
    assert (out / "combined_benchmark_results.json").exists()
    assert (out / "claim_gate_audit.csv").exists()
