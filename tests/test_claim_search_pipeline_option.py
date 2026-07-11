from __future__ import annotations

import json

from confirm.agent import _parse_contract_text
from confirm.agent import run_claim
from confirm.cli import build_parser
from confirm.contract import ClaimContract
from confirm.verdict import Verdict


class _FakeCandidateLLM:
    model = "fake-claim-generator"

    def complete(self, system, user):
        payload = json.loads(user)
        evidence = payload["failure_localization"]["evidence"][:1]
        proposed_contract = payload["original_contract"]
        return json.dumps(
            {
                "candidates": [
                    {
                        "proposal_type": "exploratory_followup_claim",
                        "transform_type": "narrower_outcome_family",
                        "domain_core": {
                            "population_or_disease": "age",
                            "cohort_family": "ADNI;OASIS3",
                            "predictor_or_contrast": "age",
                            "outcome_modality": "smri",
                            "outcome_family": "smri_hippocampus",
                            "direction_family": "negative",
                            "scientific_motivation": "Feedback CLI test.",
                        },
                        "preservation_check": {
                            "preserves_population": True,
                            "preserves_cohort_family": True,
                            "preserves_predictor_or_contrast": True,
                            "preserves_outcome_modality": True,
                            "preserves_direction_family": True,
                            "preserves_scientific_motivation": True,
                            "changed_fields": ["outcome_family"],
                            "allowed_change_rationale": "Preserves the age predictor and smri modality.",
                        },
                        "proposed_question": "Under adaptive same-data evaluation, test a narrower smri age association.",
                        "proposed_contract": proposed_contract,
                        "rationale": "This connected same-modality follow-up can be adaptively evaluated on current data.",
                        "connection_rationale": "Preserves the age predictor and smri modality.",
                        "evidence_policy": {
                            "provenance": "post_hoc_followup",
                            "requires_new_evidence": False,
                            "can_confirm_on_current_data": True,
                            "validation_split": "current_data_adaptive",
                        },
                        "supported_by_evidence": evidence,
                        "disposition_label": None,
                    }
                ]
            }
        )


def _contract_path(tmp_path):
    path = tmp_path / "contract.yaml"
    path.write_text(
        """
claim_id: feedback_cli_claim
question: Feedback CLI test.
estimand:
  type: association
  outcome: smri_hippocampus
  predictor: age
  group: null
  direction: negative
  unit: scalar
  region_set: null
covariates: [sex]
inclusion: null
discovery_cohort: ADNI
replication_cohorts: [OASIS3]
search_provenance:
  declared: true
  family_size: 1
  selection: preregistered
gates:
  multiplicity: {method: fdr_bh, alpha: 0.05, family_size: 1}
  confound: {require_covariates: [sex], motion_check: false}
  power: {min_power: 0.8, ref_effect: null}
  multiverse: {min_fraction_consistent: 0.6}
  replication:
    alpha: 0.05
    require_same_sign: true
    require_ci_overlap: false
    harmonize: combat
    pattern_corr_min: 0.5
    region_replication_frac_min: 0.5
    dice_min: 0.0
reporting_language_allowed: [confirmed, non_replicated, under_powered, fragile]
""",
        encoding="utf-8",
    )
    return path


def test_cli_claim_search_option_parses_for_run_and_ask():
    parser = build_parser()
    run = parser.parse_args(
        [
            "run",
            "--contract",
            "c.yaml",
            "--data-dir",
            "data",
            "--out",
            "out",
            "--claim-search",
            "on",
            "--claim-search-max-rounds",
            "2",
            "--claim-search-max-candidates",
            "3",
            "--claim-search-schema-retries",
            "1",
            "--claim-search-external-data-dir",
            "external",
        ]
    )
    ask = parser.parse_args(
        [
            "ask",
            "question",
            "--out",
            "out",
            "--claim-search",
            "on",
            "--claim-search-max-rounds",
            "2",
            "--claim-search-max-candidates",
            "3",
            "--claim-search-schema-retries",
            "1",
            "--claim-search-external-data-dir",
            "external",
            "--auto",
        ]
    )

    assert run.claim_search == "on"
    assert run.claim_search_max_rounds == 2
    assert run.claim_search_max_candidates == 3
    assert run.claim_search_schema_retries == 1
    assert run.claim_search_external_data_dir == "external"
    assert ask.claim_search == "on"
    assert ask.claim_search_schema_retries == 1
    assert ask.claim_search_external_data_dir == "external"


def test_contract_parser_extracts_embedded_code_fence():
    assert _parse_contract_text("Here is the revision:\n```yaml\nanswer: 1\n```\n") == {"answer": 1}


def test_run_claim_does_not_write_claim_search_artifacts_by_default(tmp_path, monkeypatch):
    contract_path = _contract_path(tmp_path)
    contract = ClaimContract.model_validate(
        {
            "claim_id": "feedback_cli_claim",
            "question": "Feedback CLI test.",
            "estimand": {
                "type": "association",
                "outcome": "smri_hippocampus",
                "predictor": "age",
                "group": None,
                "direction": "negative",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["sex"],
            "inclusion": None,
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["sex"], "motion_check": False},
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
    )
    verdict = Verdict(
        label="fragile",
        abstained=True,
        rationale="Failed gates: multiverse",
        gates={"multiplicity": True, "confound": True, "power": True, "multiverse": False, "replication": False},
    )

    monkeypatch.setattr("confirm.agent.load_contract", lambda _: contract)
    monkeypatch.setattr("confirm.agent.lookup_ref_effect", lambda *_: None)
    monkeypatch.setattr(
        "confirm.agent._execute_contract",
        lambda *_args, **_kwargs: (verdict, {"contract": contract.model_dump(mode="json")}, []),
    )
    monkeypatch.setattr("confirm.provenance.file_sha256", lambda _: "sha")
    monkeypatch.setattr("confirm.provenance.git_sha", lambda: None)

    out_dir = tmp_path / "out"
    run_claim(contract_path, tmp_path, out_dir)

    assert not (out_dir / "claim_search_trace.json").exists()


def test_run_claim_writes_claim_search_artifacts_when_enabled(tmp_path, monkeypatch):
    contract_path = _contract_path(tmp_path)
    contract = ClaimContract.model_validate(
        {
            "claim_id": "feedback_cli_claim",
            "question": "Feedback CLI test.",
            "estimand": {
                "type": "association",
                "outcome": "smri_hippocampus",
                "predictor": "age",
                "group": None,
                "direction": "negative",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["sex"],
            "inclusion": None,
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["sex"], "motion_check": False},
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
    )
    verdict = Verdict(
        label="fragile",
        abstained=True,
        rationale="Failed gates: multiverse",
        gates={"multiplicity": True, "confound": True, "power": True, "multiverse": False, "replication": True},
    )

    monkeypatch.setattr("confirm.agent.load_contract", lambda _: contract)
    monkeypatch.setattr("confirm.agent.lookup_ref_effect", lambda *_: None)
    monkeypatch.setattr(
        "confirm.agent._execute_contract",
        lambda *_args, **_kwargs: (verdict, {"contract": contract.model_dump(mode="json")}, []),
    )
    monkeypatch.setattr("confirm.provenance.file_sha256", lambda _: "sha")
    monkeypatch.setattr("confirm.provenance.git_sha", lambda: None)
    monkeypatch.setattr("confirm.agent.get_llm", lambda: _FakeCandidateLLM())

    out_dir = tmp_path / "out"
    run_claim(contract_path, tmp_path, out_dir, claim_search=True)

    assert (out_dir / "claim_search_config.json").exists()
    assert (out_dir / "failure_localization.json").exists()
    assert (out_dir / "claim_search_trace.json").exists()
    assert (out_dir / "candidate_claims.json").exists()
    assert (out_dir / "duplicate_candidates.json").exists()
    assert (out_dir / "proposal_validation.json").exists()
    assert (out_dir / "candidate_evaluations.json").exists()
    assert (out_dir / "claim_lineage.json").exists()
    assert (out_dir / "llm_candidate_prompts.jsonl").exists()
    assert (out_dir / "llm_candidate_responses.jsonl").exists()
    trace = json.loads((out_dir / "claim_search_trace.json").read_text(encoding="utf-8"))
    receipt = json.loads((out_dir / "receipt.json").read_text(encoding="utf-8"))
    assert trace["stopped_reason"] == "no_candidates"
    assert len(trace["duplicate_candidates"]) == 1
    assert "candidate_claims" in receipt["results"]
    assert "duplicate_candidates" in receipt["results"]
