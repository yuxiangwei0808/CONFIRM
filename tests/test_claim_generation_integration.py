from __future__ import annotations

from bench.claim_generation_integration import (
    DOMAIN_PRIOR_SYSTEM_PROMPT,
    NEUROCLAW_DRAFTER_SYSTEM,
    build_negative_controls,
    classify_draft_failure,
    drafter_system_prompt,
)
from bench.run_initial_claim_drafting import ClaimQuestion


def _question(claim_id: str, family: str) -> ClaimQuestion:
    return ClaimQuestion.model_validate(
        {
            "claim_id": claim_id,
            "target_family": family,
            "source_mode": "llm_proposed",
            "question": "Do cases differ from controls on the imaging outcome?",
            "discovery_cohort": "ADNI",
            "replication_cohorts": "OASIS3",
            "group_var": "dx",
            "case_label": "Dementia",
            "control_label": "CN",
            "allowed_covariates": "age;sex",
            "shared_outcome_prefixes": "smri_",
        }
    )


def test_drafter_system_prompt_selects_persona() -> None:
    assert drafter_system_prompt("direct_gpt_drafter") is None
    system = drafter_system_prompt("neuroclaw_adapted_drafter")
    assert system == NEUROCLAW_DRAFTER_SYSTEM
    # NeuroClaw persona is prepended to the unchanged domain-prior instructions
    assert "biostatistician" in system
    assert "methodology expert" in system
    assert DOMAIN_PRIOR_SYSTEM_PROMPT in system


def test_classify_draft_failure_distinguishes_schema_and_preflight() -> None:
    assert classify_draft_failure([{"preflight_error": "x"}]) == (
        True,
        "unsupported_variable_or_preflight",
    )
    assert classify_draft_failure([{"source_preservation_error": "x"}]) == (
        True,
        "unsupported_cohort_or_predictor",
    )
    assert classify_draft_failure([{"schema_error": "bad json"}]) == (
        False,
        "schema_invalid",
    )
    assert classify_draft_failure([]) == (False, "no_response")


def test_build_negative_controls_are_executable_gate_targeted() -> None:
    positives = [
        _question(f"q_{family}_{i}", family)
        for family in ("adhd", "asd")
        for i in range(3)
    ]
    negatives = build_negative_controls(positives, per_family=2)
    assert len(negatives) == 4  # 2 families x 2
    for negative in negatives:
        assert negative.label_class == "negative_control"
        assert negative.source_mode == "synthetic_stress"
        assert not negative.include_in_main
        assert "post hoc" in negative.question.lower()
        # predictor, group contrast, and cohorts are preserved so the control
        # stays executable and reaches the CONFIRM gates
        assert negative.group_var == "dx"
        assert negative.case_label == "Dementia"
        assert negative.discovery_cohort == "ADNI"
