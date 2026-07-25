"""Experiment 2: question->claim generation integration.

Two drafters translate the same fixed natural-language questions into a frozen
``ClaimContract`` using the identical schema, catalog, preflight, and
source-preservation checks:

* ``direct_gpt_drafter``       -- the native CONFIRM Stage-1 drafter.
* ``neuroclaw_adapted_drafter`` -- the same drafter driven by NeuroClaw's
  methodology + biostatistician persona (adapted; backbone held fixed).

Every successfully drafted, executable contract is then evaluated by the
unchanged CONFIRM gates. This is an *integration* study (can CONFIRM govern
claims authored by another agent?), not a benchmark-accuracy measurement: a
newly generated claim may differ from any original labeled claim.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.claim_evaluation_baselines import NEUROCLAW_PERSONAS
from bench.run_initial_claim_drafting import (
    DEFAULT_DATA_ROOTS,
    ClaimQuestion,
    DraftContractError,
    _source_preservation_violations,
    draft_contract_with_trace,
)
from confirm.agent import DOMAIN_PRIOR_SYSTEM_PROMPT
from confirm.contract import ClaimContract
from confirm.execution import evaluate_contract, resolve_execution_root

DrafterMethod = Literal["direct_gpt_drafter", "neuroclaw_adapted_drafter"]
QuestionClass = Literal["positive", "negative_control"]

# NeuroClaw drives drafting with its verbatim methodology + biostatistician
# personas (github.com/CUHK-AIM-Group/NeuroClaw @ b9e3833), prepended to the
# unchanged CONFIRM domain-prior drafting instructions so both arms share the
# same contract schema and validation.
NEUROCLAW_DRAFTER_PERSONA = (
    NEUROCLAW_PERSONAS["methodology_expert"]
    + " "
    + NEUROCLAW_PERSONAS["biostatistician"]
    + " As the NeuroClaw research-design panel, translate the neuroimaging "
    "research question into a rigorous, executable analysis specification."
)
NEUROCLAW_DRAFTER_SYSTEM = (
    NEUROCLAW_DRAFTER_PERSONA + "\n\n" + DOMAIN_PRIOR_SYSTEM_PROMPT
)


def drafter_system_prompt(method: DrafterMethod) -> Optional[str]:
    """System prompt override for a drafter (None = native domain prior)."""

    if method == "neuroclaw_adapted_drafter":
        return NEUROCLAW_DRAFTER_SYSTEM
    return None


class GenerationOutcome(BaseModel):
    """One drafter's result on one question, plus the CONFIRM gate outcome."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    target_family: str
    question_class: QuestionClass
    drafter: DrafterMethod
    schema_valid: bool
    executable: bool
    aligned: bool
    alignment_violations: list[str] = Field(default_factory=list)
    error_disposition: Optional[str] = None
    gate_available: bool = False
    gate_label: Optional[str] = None
    confirm_support: bool = False
    unsafe_support: bool = False
    drafted_contract: Optional[dict[str, Any]] = None
    details: dict[str, Any] = Field(default_factory=dict)


def classify_draft_failure(responses: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return (schema_valid, disposition) from a failed drafting trace.

    A response that reached preflight or source-preservation checks was
    schema-valid (it parsed into a ClaimContract) but not executable/aligned.
    """

    if not responses:
        return False, "no_response"
    last = responses[-1]
    if "preflight_error" in last:
        return True, "unsupported_variable_or_preflight"
    if "source_preservation_error" in last:
        return True, "unsupported_cohort_or_predictor"
    if "evidence_policy_error" in last:
        return True, "excluded_evidence"
    if "schema_error" in last or "validation_error" in last:
        return False, "schema_invalid"
    return False, "unknown"


def _gate_contract(
    contract: ClaimContract,
    data_roots: list[str],
) -> tuple[bool, Optional[str], bool, dict[str, Any]]:
    """Evaluate a drafted contract with the unchanged CONFIRM gates."""

    try:
        roots = [Path(item) for item in data_roots]
        root = resolve_execution_root(contract, roots)
        verdict, _results, _paths = evaluate_contract(
            contract,
            root,
            ref_effect=contract.gates.power.ref_effect,
        )
        confirmed = not verdict.abstained and verdict.label == "confirmed"
        return True, verdict.label, confirmed, {"root": str(root)}
    except Exception as exc:  # noqa: BLE001 - recorded per item
        return False, None, False, {"gate_error": str(exc)}


def draft_and_gate(
    question: ClaimQuestion,
    question_class: QuestionClass,
    catalog: dict[str, Any],
    llm: Any,
    method: DrafterMethod,
    *,
    schema_retries: int,
    preflight_context: Any = None,
    data_roots: Optional[list[str]] = None,
) -> tuple[GenerationOutcome, list[dict[str, Any]], list[dict[str, Any]]]:
    """Draft a contract with one drafter and evaluate it through CONFIRM."""

    roots = data_roots or [str(path) for path in DEFAULT_DATA_ROOTS]
    base = dict(
        claim_id=question.claim_id,
        target_family=question.target_family,
        question_class=question_class,
        drafter=method,
    )
    system = drafter_system_prompt(method)
    try:
        contract, prompts, responses = draft_contract_with_trace(
            question,
            catalog,
            llm,
            schema_retries=schema_retries,
            preflight_context=preflight_context,
            system_prompt=system,
        )
    except DraftContractError as exc:
        schema_valid, disposition = classify_draft_failure(exc.responses)
        outcome = GenerationOutcome(
            **base,
            schema_valid=schema_valid,
            executable=False,
            aligned=False,
            error_disposition=disposition,
        )
        return outcome, exc.prompts, exc.responses

    violations = _source_preservation_violations(question, contract)
    gate_available, gate_label, confirmed, gate_details = _gate_contract(
        contract, roots
    )
    unsafe = bool(confirmed and question_class == "negative_control")
    outcome = GenerationOutcome(
        **base,
        schema_valid=True,
        executable=True,
        aligned=not violations,
        alignment_violations=violations,
        error_disposition=None,
        gate_available=gate_available,
        gate_label=gate_label,
        confirm_support=confirmed,
        unsafe_support=unsafe,
        drafted_contract=contract.model_dump(mode="json"),
        details=gate_details,
    )
    return outcome, prompts, responses


def build_negative_controls(
    positives: list[ClaimQuestion],
    *,
    per_family: int = 2,
) -> list[ClaimQuestion]:
    """Derive executable, gate-targeted negative-control questions.

    Each control keeps a real question's cohorts, predictor, and group contrast
    so the drafted contract stays executable and reaches the CONFIRM gates, but
    reframes the design as an unregistered post-hoc search over many outcomes in
    a small opportunistic subsample. A faithful contract should therefore abstain
    via the search-multiplicity and/or power gates; an ``unsafe_support`` outcome
    means the drafter+CONFIRM confirmed a design that should have abstained.
    """

    by_family: dict[str, list[ClaimQuestion]] = {}
    for question in positives:
        by_family.setdefault(question.target_family, []).append(question)

    controls: list[ClaimQuestion] = []
    for family, questions in sorted(by_family.items()):
        for question in questions[:per_family]:
            payload = question.model_dump(mode="json")
            payload["claim_id"] = f"negctrl_search_{question.claim_id}"
            payload["source_mode"] = "synthetic_stress"
            payload["label_class"] = "negative_control"
            payload["label_basis"] = "constructed_control"
            payload["include_in_main"] = False
            payload["notes"] = (
                "Constructed gate-targeted negative control: an unregistered "
                "post-hoc search over many candidate outcomes reported from a "
                "small opportunistic subsample. CONFIRM should abstain via the "
                "search-multiplicity and/or power gates, not confirm."
            )
            outcome_hint = question.shared_outcome_prefixes or "imaging"
            predictor = question.group_var or "the predictor of interest"
            payload["question"] = (
                "As a deliberate negative control, report the single strongest "
                f"association between {predictor} and one of many {outcome_hint} "
                "outcomes that were scanned post hoc with no prespecified "
                f"hypothesis, in a small opportunistic subsample of "
                f"{question.discovery_cohort or family} with replication attempted "
                f"in {question.replication_cohorts or 'a held-in cohort'}. This "
                "unregistered, multiplicity-laden, underpowered design should not "
                "be reported as a confirmed scientific claim."
            )
            controls.append(ClaimQuestion.model_validate(payload))
    return controls
