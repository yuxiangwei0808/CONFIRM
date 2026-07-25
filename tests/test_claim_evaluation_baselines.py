from __future__ import annotations

import json

from bench.claim_evaluation_baselines import (
    DirectJudgeOutput,
    NeuroClawPersonaOutput,
    _aggregate_neuroclaw,
    conventional_significance_decision,
    direct_judge_prompt,
    direct_llm_decision,
    label_blind_evidence,
    neuroclaw_adapted_decision,
    neuroclaw_persona_prompt,
    neuroclaw_persona_system,
    veritas_adapted_decision,
    veritas_evidence_label,
)
from confirm.contract import ClaimContract


def _contract(*, direction: str = "negative", unit: str = "scalar") -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "baseline_fixture",
            "question": "Do cases have lower hippocampal volume than controls?",
            "estimand": {
                "type": "group_diff",
                "outcome": "smri_hippocampus" if unit == "scalar" else "smri_",
                "predictor": "dx",
                "group": {"var": "dx", "case": "case", "control": "control"},
                "direction": direction,
                "unit": unit,
                "region_set": None if unit == "scalar" else "smri",
            },
            "covariates": ["age", "sex"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
            "search_provenance": {
                "declared": True,
                "family_size": 4,
                "selection": "preregistered",
            },
            "gates": {
                "multiplicity": {
                    "method": "fdr_bh",
                    "alpha": 0.05,
                    "family_size": 4,
                },
                "confound": {
                    "require_covariates": ["age", "sex"],
                    "motion_check": False,
                },
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
            "reporting_language_allowed": ["confirmed", "fragile"],
        }
    )


def _checkpoint(
    *,
    direction: str = "negative",
    discovery_beta: float = -0.4,
    discovery_p: float = 0.01,
    replication_beta: float = -0.3,
    replication_p: float = 0.02,
    unit: str = "scalar",
) -> dict:
    contract = _contract(direction=direction, unit=unit)
    return {
        "task_id": "task_fixture",
        "benchmark_item_id": "case_fixture",
        "confirm_outcome": "confirmed",
        "gate_verdict": {"label": "confirmed"},
        "gate_results": {
            "contract": contract.model_dump(mode="json"),
            "primary": {"beta": discovery_beta, "p": discovery_p},
            "power": {
                "achieved_power": 0.91,
                "under_powered": False,
                "rationale": "passed",
            },
            "multiverse": {
                "fraction_consistent": 0.75,
                "passed": True,
                "specs": [
                    {
                        "beta": discovery_beta,
                        "p": discovery_p,
                        "same_sign": True,
                        "significant": True,
                    }
                ],
            },
            "replication": {
                "passed": True,
                "reason": "replicated",
                "cohort_results": [
                    {
                        "cohort": "REP",
                        "passed": True,
                        "effect": {
                            "beta": replication_beta,
                            "p": replication_p,
                        },
                    }
                ],
            },
            "verdict": {"label": "confirmed"},
        },
    }


def test_conventional_significance_requires_both_cohorts() -> None:
    decision = conventional_significance_decision(_checkpoint(), "protocol")
    assert decision.available
    assert decision.supported
    assert decision.details["multiplicity_adjustment"] == "none"

    failed = conventional_significance_decision(
        _checkpoint(replication_p=0.08),
        "protocol",
    )
    assert failed.available
    assert not failed.supported
    assert failed.reason == "not_significant"


def test_conventional_significance_requires_direction_agreement() -> None:
    decision = conventional_significance_decision(
        _checkpoint(replication_beta=0.3),
        "protocol",
    )
    assert not decision.supported
    assert decision.reason == "direction_mismatch"


def test_conventional_significance_is_scalar_only() -> None:
    decision = conventional_significance_decision(
        _checkpoint(unit="brainwide"),
        "protocol",
    )
    assert not decision.available
    assert decision.reason == "scalar_contract_required"


def test_direct_judge_packet_removes_gate_decisions_and_references() -> None:
    checkpoint = _checkpoint()
    checkpoint["reference_disposition"] = "confirm"
    checkpoint["gate_results"]["replication"]["cohort_results"][0][
        "cohort"
    ] = "neg_p_fishing_fixture_REP"
    packet = label_blind_evidence(checkpoint)
    serialized = json.dumps(packet, sort_keys=True)
    assert "gate_verdict" not in serialized
    assert "reference_disposition" not in serialized
    assert '"passed"' not in serialized
    assert '"under_powered"' not in serialized
    assert '"significant"' not in serialized
    assert "reporting_language_allowed" not in serialized
    assert "baseline_fixture" not in serialized
    assert "hippocampal" not in serialized
    assert '"discovery_cohort"' not in serialized
    assert '"gates"' not in serialized
    assert "neg_p_fishing" not in serialized
    assert "replication_1" in serialized
    assert "fraction_consistent" in serialized
    assert "achieved_power" in serialized


def test_direct_judge_prompt_contains_claim_and_numerical_evidence() -> None:
    prompt = direct_judge_prompt(_checkpoint())
    assert '"direction": "negative"' in prompt
    assert '"outcome": "outcome"' in prompt
    assert '"beta": -0.4' in prompt
    assert '"p": 0.02' in prompt


def test_direct_llm_output_maps_to_binary_decision() -> None:
    output = DirectJudgeOutput(
        decision="abstain",
        confidence="high",
        rationale="The replication estimate is not convincing.",
        key_evidence=["replication"],
    )
    decision = direct_llm_decision(_checkpoint(), "protocol", output)
    assert decision.available
    assert not decision.supported
    assert decision.details["confidence"] == "high"


def _scalar_primary(ckpt: dict, **primary) -> dict:
    ckpt["gate_results"]["primary"] = primary
    return ckpt


def test_veritas_supported_on_significant_matching_direction() -> None:
    ckpt = _scalar_primary(
        _checkpoint(direction="negative"),
        beta=-0.4,
        p=0.001,
        n=200,
        standardized_effect=-0.4,
    )
    result = veritas_evidence_label(ckpt)
    assert result["evidence_label"] == "SUPPORTED"
    decision = veritas_adapted_decision(ckpt, "protocol")
    assert decision.available
    assert decision.supported
    assert decision.reason == "veritas_supported"


def test_veritas_refuted_on_opposite_direction() -> None:
    ckpt = _scalar_primary(
        _checkpoint(direction="negative"),
        beta=0.4,
        p=0.001,
        n=200,
        standardized_effect=0.4,
    )
    assert veritas_evidence_label(ckpt)["evidence_label"] == "REFUTED"
    assert not veritas_adapted_decision(ckpt, "protocol").supported


def test_veritas_underpowered_when_null_and_small_n() -> None:
    ckpt = _scalar_primary(
        _checkpoint(direction="negative"),
        beta=-0.05,
        p=0.6,
        n=40,
        standardized_effect=-0.05,
    )
    assert veritas_evidence_label(ckpt)["evidence_label"] == "UNDERPOWERED"


def test_veritas_refuted_when_null_and_well_powered() -> None:
    ckpt = _scalar_primary(
        _checkpoint(direction="negative"),
        beta=-0.02,
        p=0.6,
        n=2000,
        standardized_effect=-0.02,
    )
    assert veritas_evidence_label(ckpt)["evidence_label"] == "REFUTED"
    assert not veritas_adapted_decision(ckpt, "protocol").supported


def test_veritas_is_scalar_only() -> None:
    decision = veritas_adapted_decision(_checkpoint(unit="brainwide"), "protocol")
    assert not decision.available
    assert decision.reason == "scalar_contract_required"


def _votes(bio: bool, clin: bool, method: bool) -> dict[str, NeuroClawPersonaOutput]:
    return {
        "biostatistician": NeuroClawPersonaOutput(
            supports_claim=bio, confidence="high", concern="c"
        ),
        "clinical_neuroscientist": NeuroClawPersonaOutput(
            supports_claim=clin, confidence="medium", concern="c"
        ),
        "methodology_expert": NeuroClawPersonaOutput(
            supports_claim=method, confidence="high", concern="c"
        ),
    }


def test_neuroclaw_confirms_only_with_panel_support() -> None:
    assert _aggregate_neuroclaw(_votes(True, True, True)) == (True, "pass")
    # lone methodology veto fails the panel
    assert _aggregate_neuroclaw(_votes(True, True, False)) == (False, "fail")
    # a single non-methodology dissent is a non-passing 'revise'
    assert _aggregate_neuroclaw(_votes(False, True, True)) == (False, "revise")
    # two dissents fail
    assert _aggregate_neuroclaw(_votes(False, False, True)) == (False, "fail")


def test_neuroclaw_decision_records_persona_votes() -> None:
    decision = neuroclaw_adapted_decision(
        _checkpoint(), "protocol", _votes(True, True, True)
    )
    assert decision.available
    assert decision.supported
    assert decision.reason == "neuroclaw_pass"
    assert set(decision.details["persona_votes"]) == {
        "biostatistician",
        "clinical_neuroscientist",
        "methodology_expert",
    }


def test_neuroclaw_persona_prompt_is_label_blind() -> None:
    checkpoint = _checkpoint()
    checkpoint["reference_disposition"] = "confirm"
    prompt = neuroclaw_persona_prompt(checkpoint)
    assert "gate_verdict" not in prompt
    assert "reference_disposition" not in prompt
    assert '"passed"' not in prompt
    assert "baseline_fixture" not in prompt
    assert "hippocampal" not in prompt
    # the verbatim NeuroClaw persona is used as the system prompt
    assert "biostatistician" in neuroclaw_persona_system("biostatistician")
