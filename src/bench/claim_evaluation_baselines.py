"""Simple claim-evaluation baselines for frozen CONFIRM evidence."""

from __future__ import annotations

import copy
import json
import math
from hashlib import sha256
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from confirm.contract import ClaimContract

BaselineMethod = Literal[
    "conventional_significance",
    "direct_llm_judge",
    "veritas_adapted",
    "neuroclaw_adapted_judge",
]


class DirectJudgeOutput(BaseModel):
    """Structured decision returned by the direct LLM judge."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["confirm", "abstain"]
    confidence: Literal["low", "medium", "high"]
    rationale: str = Field(min_length=1)
    key_evidence: list[str] = Field(default_factory=list, max_length=5)


class ClaimEvaluationDecision(BaseModel):
    """One label-blind baseline decision for one frozen evaluation task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    benchmark_case_id: str
    method: BaselineMethod
    available: bool
    supported: bool
    reason: str
    protocol_sha256: str
    direction: str
    unit: str
    evidence_count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


DIRECT_JUDGE_SYSTEM_PROMPT = """\
You are a scientific reviewer deciding whether a frozen neuroimaging claim is
supported by the supplied statistical evidence.

Return CONFIRM when the exact frozen claim is sufficiently supported by the
evidence; otherwise return ABSTAIN. Use your own scientific judgment. Do not
invent statistics or alter the claim.
"""


_DECISION_KEYS = {
    "abstained",
    "confirmation_i2",
    "confirmation_subtype",
    "failures",
    "gates",
    "high_i2",
    "label",
    "passed",
    "rationale",
    "reason",
    "replicated_but_heterogeneous",
    "same_sign",
    "significant",
    "under_powered",
}


def _scrub_decisions(value: Any) -> Any:
    """Remove deterministic verdict fields while preserving numerical evidence."""

    if isinstance(value, dict):
        return {
            key: _scrub_decisions(item)
            for key, item in value.items()
            if key not in _DECISION_KEYS
        }
    if isinstance(value, list):
        return [_scrub_decisions(item) for item in value]
    return value


def label_blind_evidence(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Build the evidence packet shown to the direct LLM judge."""

    gate_results = checkpoint.get("gate_results") or {}
    contract = dict(gate_results["contract"])
    for key in (
        "claim_id",
        "question",
        "discovery_cohort",
        "replication_cohorts",
        "reporting_language_allowed",
        "gates",
    ):
        contract.pop(key, None)
    estimand = dict(contract["estimand"])
    estimand["predictor"] = "predictor"
    estimand["outcome"] = (
        "outcome" if estimand.get("unit") == "scalar" else "outcome_family"
    )
    if estimand.get("group") is not None:
        estimand["group"] = {
            "var": "predictor",
            "case": "group_a",
            "control": "group_b",
        }
    contract["estimand"] = estimand
    contract["replication_cohort_count"] = len(
        gate_results["contract"].get("replication_cohorts") or []
    )
    evidence = {
        key: copy.deepcopy(gate_results.get(key))
        for key in (
            "primary",
            "regions",
            "replication",
            "multiverse",
            "power",
            "confound_completeness",
        )
        if gate_results.get(key) is not None
    }
    replication = evidence.get("replication")
    if isinstance(replication, dict):
        for index, row in enumerate(
            replication.get("cohort_results") or [],
            start=1,
        ):
            if isinstance(row, dict) and "cohort" in row:
                row["cohort"] = f"replication_{index}"
        heterogeneity = replication.get("heterogeneity")
        if isinstance(heterogeneity, dict):
            for index, row in enumerate(
                heterogeneity.get("cohort_effects") or [],
                start=1,
            ):
                if isinstance(row, dict) and "cohort" in row:
                    row["cohort"] = (
                        "discovery" if index == 1 else f"replication_{index - 1}"
                    )
    confound = evidence.get("confound_completeness")
    if isinstance(confound, dict) and "predictor" in confound:
        confound["predictor"] = "predictor"
    return {
        "frozen_claim": contract,
        "statistical_evidence": _scrub_decisions(evidence),
    }


def direct_judge_prompt(checkpoint: dict[str, Any]) -> str:
    """Serialize the label-blind direct-judge input."""

    payload = label_blind_evidence(checkpoint)
    payload["instruction"] = (
        "Decide whether the exact frozen claim should be reported as confirmed "
        "or whether the system should abstain."
    )
    return json.dumps(payload, indent=2, sort_keys=True)


def prompt_sha256(checkpoint: dict[str, Any]) -> str:
    """Hash the complete direct-judge prompt."""

    prompt = DIRECT_JUDGE_SYSTEM_PROMPT + "\n" + direct_judge_prompt(checkpoint)
    return sha256(prompt.encode("utf-8")).hexdigest()


def _base(
    checkpoint: dict[str, Any],
    method: BaselineMethod,
    protocol_sha256: str,
) -> dict[str, Any]:
    gate_results = checkpoint.get("gate_results") or {}
    contract = ClaimContract.model_validate(gate_results["contract"])
    return {
        "task_id": str(checkpoint["task_id"]),
        "benchmark_case_id": str(
            checkpoint.get("benchmark_item_id")
            or checkpoint.get("benchmark_case_id")
        ),
        "method": method,
        "protocol_sha256": protocol_sha256,
        "direction": contract.estimand.direction,
        "unit": contract.estimand.unit,
    }


def _direction_matches(beta: float, direction: str) -> bool:
    if direction == "positive":
        return beta > 0
    if direction == "negative":
        return beta < 0
    return beta != 0


def conventional_significance_decision(
    checkpoint: dict[str, Any],
    protocol_sha256: str,
    *,
    alpha: float = 0.05,
) -> ClaimEvaluationDecision:
    """Require unadjusted significance and matching direction in every cohort."""

    method: BaselineMethod = "conventional_significance"
    gate_results = checkpoint.get("gate_results") or {}
    contract = ClaimContract.model_validate(gate_results["contract"])
    base = _base(checkpoint, method, protocol_sha256)
    if contract.estimand.unit != "scalar":
        return ClaimEvaluationDecision(
            **base,
            available=False,
            supported=False,
            reason="scalar_contract_required",
        )

    primary = gate_results.get("primary")
    if not isinstance(primary, dict):
        return ClaimEvaluationDecision(
            **base,
            available=False,
            supported=False,
            reason="discovery_effect_unavailable",
        )
    cohort_results = [
        row
        for row in (gate_results.get("replication") or {}).get("cohort_results") or []
        if isinstance(row, dict) and isinstance(row.get("effect"), dict)
    ]
    if not cohort_results:
        return ClaimEvaluationDecision(
            **base,
            available=False,
            supported=False,
            reason="replication_effect_unavailable",
            evidence_count=1,
        )

    effects = [primary, *[row["effect"] for row in cohort_results]]
    significant = all(float(effect["p"]) < alpha for effect in effects)
    directions = [
        _direction_matches(float(effect["beta"]), contract.estimand.direction)
        for effect in effects
    ]
    if contract.estimand.direction == "two_sided":
        discovery_sign = 1 if float(primary["beta"]) > 0 else -1
        directions = [
            (1 if float(effect["beta"]) > 0 else -1) == discovery_sign
            for effect in effects
        ]
    direction_consistent = all(directions)
    supported = significant and direction_consistent
    reason = (
        "supported"
        if supported
        else "not_significant"
        if not significant
        else "direction_mismatch"
    )
    return ClaimEvaluationDecision(
        **base,
        available=True,
        supported=supported,
        reason=reason,
        evidence_count=len(effects),
        details={
            "alpha": alpha,
            "discovery": {
                "beta": float(primary["beta"]),
                "p": float(primary["p"]),
            },
            "replication": [
                {
                    "cohort": str(row.get("cohort") or ""),
                    "beta": float(row["effect"]["beta"]),
                    "p": float(row["effect"]["p"]),
                }
                for row in cohort_results
            ],
            "multiplicity_adjustment": "none",
        },
    )


def direct_llm_decision(
    checkpoint: dict[str, Any],
    protocol_sha256: str,
    output: DirectJudgeOutput,
) -> ClaimEvaluationDecision:
    """Convert a validated direct-judge response into a frozen decision."""

    return ClaimEvaluationDecision(
        **_base(checkpoint, "direct_llm_judge", protocol_sha256),
        available=True,
        supported=output.decision == "confirm",
        reason="llm_judgment",
        evidence_count=1,
        details=output.model_dump(mode="json"),
    )


# --------------------------------------------------------------------------- #
# VERITAS-adapted deterministic epistemic label
# --------------------------------------------------------------------------- #
# Faithful port of VERITAS ``experiments/evaluation.py::compute_evidence_label``
# (github.com/LucZot/veritas @ 17dbdc9). The published epistemic-label function
# maps frozen statistics to SUPPORTED / REFUTED / UNDERPOWERED / INVALID; we map
# SUPPORTED -> confirm and everything else -> abstain. SESOI uses the VERITAS
# "standard" profile and power uses the VERITAS normal-approximation formula.
# VERITAS operates on a single hypothesis test (discovery), so this baseline is
# scalar-only and does not use the replication cohort.
VERITAS_ALPHA = 0.05
VERITAS_SESOI_STANDARD = {
    "group_difference": 0.5,
    "correlation": 0.3,
    "regression": 0.3,
}


def _veritas_test_family(estimand: Any) -> str:
    return "group_difference" if estimand.type == "group_diff" else "regression"


def _veritas_power_group_difference(d: float, n1: int, n2: int) -> Optional[float]:
    if n1 <= 1 or n2 <= 1:
        return None
    n_eff = (n1 * n2) / (n1 + n2)
    z_alpha = float(stats.norm.ppf(1 - VERITAS_ALPHA / 2))
    ncp = d * math.sqrt(n_eff)
    return float(stats.norm.cdf(-z_alpha - ncp) + (1 - stats.norm.cdf(z_alpha - ncp)))


def _veritas_power_correlation(rho: float, n: int) -> Optional[float]:
    if n <= 3:
        return None
    rho = max(min(rho, 0.999999), -0.999999)
    z = math.atanh(rho)
    se = 1.0 / math.sqrt(n - 3)
    z_alpha = float(stats.norm.ppf(1 - VERITAS_ALPHA / 2))
    ncp = z / se
    return float(stats.norm.cdf(-z_alpha - ncp) + (1 - stats.norm.cdf(z_alpha - ncp)))


def _veritas_power_at_sesoi(
    test_family: str, n_total: Optional[int]
) -> tuple[Optional[float], Optional[float], str]:
    """Return (power, sesoi, allocation_note) at the VERITAS standard SESOI.

    Group sizes are a balanced split of the frozen total sample: the VERITAS
    two-sample power formula is symmetric in allocation and near-invariant for
    these sample sizes, and an unbalanced split only lowers effective n.
    """

    if n_total is None or n_total <= 1:
        return None, None, "sample_size_unavailable"
    sesoi = VERITAS_SESOI_STANDARD.get(test_family)
    if sesoi is None:
        return None, None, "sesoi_unavailable"
    if test_family == "group_difference":
        n1 = n_total // 2
        n2 = n_total - n1
        return (
            _veritas_power_group_difference(abs(sesoi), n1, n2),
            sesoi,
            "balanced_from_total_n",
        )
    return _veritas_power_correlation(abs(sesoi), n_total), sesoi, "total_n"


def veritas_evidence_label(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Compute the VERITAS deterministic epistemic label for a scalar claim."""

    gate_results = checkpoint.get("gate_results") or {}
    contract = ClaimContract.model_validate(gate_results["contract"])
    estimand = contract.estimand
    primary = gate_results.get("primary")
    test_family = _veritas_test_family(estimand)
    direction_intent = (
        estimand.direction if estimand.direction in {"positive", "negative"} else None
    )
    if not isinstance(primary, dict):
        return {
            "evidence_label": "INVALID",
            "reason": "primary_effect_unavailable",
            "test_family": test_family,
        }
    p_value = primary.get("p")
    effect = primary.get("standardized_effect")
    if effect is None:
        effect = primary.get("beta")
    if p_value is None or not (0.0 <= float(p_value) <= 1.0):
        return {
            "evidence_label": "INVALID",
            "reason": "p_value_invalid",
            "test_family": test_family,
        }
    if effect is None or not math.isfinite(float(effect)):
        return {
            "evidence_label": "INVALID",
            "reason": "effect_size_invalid",
            "test_family": test_family,
        }
    n_total = primary.get("n")
    n_total = int(n_total) if isinstance(n_total, (int, float)) else None
    power_at_sesoi, sesoi, allocation = _veritas_power_at_sesoi(test_family, n_total)
    p_value = float(p_value)
    effect = float(effect)
    if p_value < VERITAS_ALPHA:
        if direction_intent == "positive" and effect < 0:
            label = "REFUTED"
        elif direction_intent == "negative" and effect > 0:
            label = "REFUTED"
        else:
            label = "SUPPORTED"
    else:
        if power_at_sesoi is not None and power_at_sesoi >= 0.8:
            label = "REFUTED"
        else:
            label = "UNDERPOWERED"
    return {
        "evidence_label": label,
        "reason": "evaluated",
        "test_family": test_family,
        "direction_intent": direction_intent,
        "observed_effect": effect,
        "p_value": p_value,
        "sesoi": sesoi,
        "power_at_sesoi": power_at_sesoi,
        "power_allocation": allocation,
        "alpha": VERITAS_ALPHA,
    }


def veritas_adapted_decision(
    checkpoint: dict[str, Any],
    protocol_sha256: str,
) -> ClaimEvaluationDecision:
    """VERITAS-adapted confirm/abstain (SUPPORTED -> confirm, else abstain)."""

    method: BaselineMethod = "veritas_adapted"
    gate_results = checkpoint.get("gate_results") or {}
    contract = ClaimContract.model_validate(gate_results["contract"])
    base = _base(checkpoint, method, protocol_sha256)
    if contract.estimand.unit != "scalar":
        return ClaimEvaluationDecision(
            **base,
            available=False,
            supported=False,
            reason="scalar_contract_required",
        )
    result = veritas_evidence_label(checkpoint)
    label = result["evidence_label"]
    return ClaimEvaluationDecision(
        **base,
        available=True,
        supported=label == "SUPPORTED",
        reason=f"veritas_{label.lower()}",
        evidence_count=1,
        details=result,
    )


# --------------------------------------------------------------------------- #
# NeuroClaw-adapted three-perspective critic judge
# --------------------------------------------------------------------------- #
# Adapts NeuroClaw's own statistical-critic subagents
# (core/subagent/manager.py::_build_persona_prefix and
# neurooracle/src/critic_agent.py, github.com/CUHK-AIM-Group/NeuroClaw @ b9e3833)
# to the frozen-claim confirm/abstain task. Each verbatim persona votes on the
# same label-blind evidence packet; votes are aggregated with NeuroClaw's
# majority + methodology-weighted rule (a lone methodology veto fails; a claim
# only "passes" -> confirm when supported by the panel).
NEUROCLAW_PERSONAS = {
    "biostatistician": (
        "You are a biostatistician specializing in neuroscience research. "
        "Focus on statistical evidence: p-values, sample sizes, effect sizes, "
        "confidence intervals, multiple comparison corrections. "
        "Flag overclaimed significance."
    ),
    "clinical_neuroscientist": (
        "You are a clinical neuroscientist. "
        "Focus on biological plausibility: molecular mechanisms, disease pathways, "
        "clinical translation feasibility. Flag biologically implausible connections."
    ),
    "methodology_expert": (
        "You are a research methodology expert. "
        "Focus on study design: causal inference validity, confounding control, "
        "selection bias, measurement validity. "
        "Flag correlational claims presented as causal."
    ),
}
NEUROCLAW_PERSONA_ORDER = (
    "biostatistician",
    "clinical_neuroscientist",
    "methodology_expert",
)
NEUROCLAW_TASK_INSTRUCTION = (
    "You are one perspective on a NeuroClaw review panel deciding whether a "
    "frozen neuroimaging claim should be reported as CONFIRMED, or whether the "
    "panel should ABSTAIN. From your assigned perspective, decide whether the "
    "supplied statistical evidence supports reporting the exact frozen claim as "
    "confirmed. Do not invent statistics or alter the claim."
)


class NeuroClawPersonaOutput(BaseModel):
    """One NeuroClaw persona's vote on a frozen claim."""

    model_config = ConfigDict(extra="forbid")

    supports_claim: bool
    confidence: Literal["low", "medium", "high"]
    concern: str = Field(min_length=1)


def neuroclaw_persona_system(persona_key: str) -> str:
    """Verbatim NeuroClaw persona prefix plus the panel task instruction."""

    return NEUROCLAW_PERSONAS[persona_key] + "\n\n" + NEUROCLAW_TASK_INSTRUCTION


def neuroclaw_persona_prompt(checkpoint: dict[str, Any]) -> str:
    """Serialize the shared label-blind packet shown to every persona."""

    payload = label_blind_evidence(checkpoint)
    payload["instruction"] = (
        "From your assigned perspective, decide whether the exact frozen claim "
        "should be reported as confirmed or whether the panel should abstain."
    )
    return json.dumps(payload, indent=2, sort_keys=True)


def _aggregate_neuroclaw(
    votes: dict[str, NeuroClawPersonaOutput],
) -> tuple[bool, str]:
    """Apply NeuroClaw's majority + methodology-weighted aggregation.

    Mirrors ``CriticAgent.review``: two or more opposing perspectives, or a lone
    methodology-expert veto, force a fail; the panel passes (confirm) only with
    supportive agreement; the remaining 'revise' state is not a pass.
    """

    pass_count = sum(1 for vote in votes.values() if vote.supports_claim)
    fail_count = len(votes) - pass_count
    methodology_failed = not votes["methodology_expert"].supports_claim
    if pass_count >= 3:
        verdict = "pass"
    elif pass_count >= 2 and fail_count == 0:
        verdict = "pass"
    elif fail_count >= 2:
        verdict = "fail"
    elif fail_count == 1 and methodology_failed:
        verdict = "fail"
    else:
        verdict = "revise"
    return verdict == "pass", verdict


def neuroclaw_adapted_decision(
    checkpoint: dict[str, Any],
    protocol_sha256: str,
    votes: dict[str, NeuroClawPersonaOutput],
) -> ClaimEvaluationDecision:
    """Aggregate persona votes into a frozen NeuroClaw-adapted decision."""

    supported, verdict = _aggregate_neuroclaw(votes)
    return ClaimEvaluationDecision(
        **_base(checkpoint, "neuroclaw_adapted_judge", protocol_sha256),
        available=True,
        supported=supported,
        reason=f"neuroclaw_{verdict}",
        evidence_count=len(votes),
        details={
            "neuroclaw_verdict": verdict,
            "persona_votes": {
                key: vote.model_dump(mode="json") for key, vote in votes.items()
            },
        },
    )
