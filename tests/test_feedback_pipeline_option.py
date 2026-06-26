from __future__ import annotations

import json

from confirm.agent import _parse_contract_text
from confirm.agent import run_claim
from confirm.cli import build_parser
from confirm.contract import ClaimContract
from confirm.verdict import Verdict


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


def test_cli_feedback_option_parses_for_run_and_ask():
    parser = build_parser()
    run = parser.parse_args(
        ["run", "--contract", "c.yaml", "--data-dir", "data", "--out", "out", "--feedback", "on"]
    )
    ask = parser.parse_args(["ask", "question", "--out", "out", "--feedback", "on", "--auto"])

    assert run.feedback == "on"
    assert ask.feedback == "on"


def test_contract_parser_extracts_embedded_code_fence():
    assert _parse_contract_text("Here is the revision:\n```yaml\nanswer: 1\n```\n") == {"answer": 1}


def test_run_claim_writes_feedback_when_enabled(tmp_path, monkeypatch):
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
    run_claim(contract_path, tmp_path, out_dir, feedback=True)

    payload = json.loads((out_dir / "feedback.json").read_text(encoding="utf-8"))
    assert payload["claim_id"] == "feedback_cli_claim"
    assert payload["primary_failure"] == "multiverse"
