"""LLM client boundary for the CONFIRM agent layer."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Protocol

from confirm.env import load_env


class LLMClient(Protocol):
    """Minimal text-completion protocol used by the agent."""

    model: str

    def complete(self, system: str, user: str) -> str:
        """Return a completion for the supplied system and user prompts."""


class OpenAIClient:
    """OpenAI-backed LLM client."""

    provider = "openai"

    def __init__(self, model: str | None = None, *, max_tokens: int | None = 2048) -> None:
        load_env()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.timeout = _client_timeout()

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI(timeout=self.timeout)
        response = _create_chat_completion_with_param_fallback(
            client.chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def complete_structured(self, system: str, user: str, response_model: type[Any]) -> str:
        """Return JSON constrained by a Pydantic response model when supported."""

        from openai import OpenAI

        client = OpenAI(timeout=self.timeout)
        response = _create_chat_completion_with_param_fallback(
            client.chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _openai_strict_json_schema(response_model),
                },
            },
        )
        return response.choices[0].message.content or ""


class OpenRouterClient:
    """OpenRouter-backed OpenAI-compatible LLM client."""

    provider = "openrouter"

    def __init__(self, model: str | None = None, *, max_tokens: int | None = 2048) -> None:
        load_env()
        self.model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        self.max_tokens = max_tokens
        self.timeout = _client_timeout()

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=self.timeout,
        )
        response = _create_chat_completion_with_param_fallback(
            client.chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    """Anthropic-backed LLM client."""

    provider = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        load_env()
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def complete(self, system: str, user: str) -> str:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=2048,
            temperature=0,
        )
        return "".join(getattr(block, "text", "") for block in response.content)


class StandInClient:
    """Deterministic offline client for tests and local development."""

    provider = "standin"
    model = "stand-in-offline"

    def complete(self, system: str, user: str) -> str:
        if "generate scientifically connected follow-up claim candidates" in system.lower():
            return _standin_candidate_response(user)
        if "interpret" in system.lower() or "narrative" in system.lower():
            return "CONFIRM verdict: see the engine-computed result bundle for effect estimates and gate decisions."
        if "site-confound null/control" in user or "site NYU" in user:
            return """
claim_id: agent_site_confound_null
question: "Offline stand-in ABIDE1 site-confound null."
estimand:
  type: group_diff
  outcome: fc_mean_abs
  predictor: site
  group:
    var: site
    case: NYU
    control: UCLA_1
  direction: two_sided
  unit: scalar
  region_set: null
covariates: [age, sex]
inclusion: null
discovery_cohort: ABIDE1
replication_cohorts: [ABIDE1]
gates:
  multiplicity:
    method: fdr_bh
    alpha: 0.05
    family_size: 1
  confound:
    require_covariates: [age, sex]
    motion_check: false
  power:
    min_power: 0.8
    ref_effect: null
  multiverse:
    min_fraction_consistent: 0.6
  replication:
    alpha: 0.05
    require_same_sign: true
    require_ci_overlap: false
    harmonize: combat
    pattern_corr_min: 0.5
    region_replication_frac_min: 0.5
    dice_min: 0.0
reporting_language_allowed: [confirmed, non_replicated, under_powered, fragile]
"""
        if "female participants" in user and "male participants" in user:
            return """
claim_id: agent_sex_hippocampus_cn
question: "Offline stand-in ADNI/OASIS3 CN sex contrast."
estimand:
  type: group_diff
  outcome: smri_hippocampus
  predictor: sex
  group:
    var: sex
    case: F
    control: M
  direction: negative
  unit: scalar
  region_set: null
covariates: [age, eTIV]
inclusion: 'dx == "CN"'
discovery_cohort: ADNI
replication_cohorts: [OASIS3]
gates:
  multiplicity:
    method: fdr_bh
    alpha: 0.05
    family_size: 1
  confound:
    require_covariates: [age, eTIV]
    motion_check: false
  power:
    min_power: 0.8
    ref_effect: null
  multiverse:
    min_fraction_consistent: 0.6
  replication:
    alpha: 0.05
    require_same_sign: true
    require_ci_overlap: false
    harmonize: combat
    pattern_corr_min: 0.5
    region_replication_frac_min: 0.5
    dice_min: 0.0
reporting_language_allowed: [confirmed, non_replicated, under_powered, fragile]
"""
        discovery = "OASIS1"
        replication = "ABIDE"
        if "ADNI" in user and "OASIS3" in user:
            discovery = "ADNI"
            replication = "OASIS3"
        elif "OASIS1" in user:
            discovery = "OASIS1"
            replication = "OASIS1"
        return f"""
claim_id: agent_brainwide_ad_cn
question: "Agent-drafted brain-wide regional claim."
estimand:
  type: group_diff
  outcome: "smri_*"
  predictor: dx
  group:
    var: dx
    case: Dementia
    control: CN
  direction: negative
  unit: brainwide
  region_set: shared_ad_signature
covariates: [age, sex, eTIV]
inclusion: null
discovery_cohort: {discovery}
replication_cohorts: [{replication}]
gates:
  multiplicity:
    method: fdr_bh
    alpha: 0.05
    family_size: 1
  confound:
    require_covariates: [age, sex, eTIV]
    motion_check: false
  power:
    min_power: 0.8
    ref_effect: null
  multiverse:
    min_fraction_consistent: 0.6
  replication:
    alpha: 0.05
    require_same_sign: true
    require_ci_overlap: false
    harmonize: combat
    pattern_corr_min: 0.5
    region_replication_frac_min: 0.5
    dice_min: 0.0
reporting_language_allowed: [confirmed, non_replicated, under_powered, fragile]
"""


def _standin_candidate_response(user: str) -> str:
    try:
        payload = json.loads(user)
        contract = payload["original_contract"]
        localization = payload["failure_localization"]
        max_candidates = int(payload.get("max_candidates") or 2)
    except Exception:
        contract = {}
        localization = {}
        max_candidates = 2
    estimand = contract.get("estimand") if isinstance(contract, dict) else {}
    outcome = estimand.get("outcome", "original_outcome") if isinstance(estimand, dict) else "original_outcome"
    if isinstance(outcome, list):
        outcome_text = str(outcome[0]) if outcome else "original_outcome"
    else:
        outcome_text = str(outcome)
    modality = outcome_text.split("_", 1)[0] if "_" in outcome_text else outcome_text
    group = estimand.get("group") if isinstance(estimand, dict) else None
    if isinstance(group, dict):
        contrast = f"{group.get('case')} vs {group.get('control')}"
    else:
        contrast = str(estimand.get("predictor", "original predictor")) if isinstance(estimand, dict) else "original predictor"
    direction = str(estimand.get("direction", "same_direction")) if isinstance(estimand, dict) else "same_direction"
    discovery = str(contract.get("discovery_cohort", "original_discovery")) if isinstance(contract, dict) else "original_discovery"
    replication = contract.get("replication_cohorts", []) if isinstance(contract, dict) else []
    if not isinstance(replication, list):
        replication = []
    domain_core = {
        "population_or_disease": contrast,
        "cohort_family": ";".join([discovery, *[str(item) for item in replication]]),
        "predictor_or_contrast": contrast,
        "outcome_modality": modality,
        "outcome_family": outcome_text,
        "direction_family": direction,
        "scientific_motivation": str(contract.get("question", "")) if isinstance(contract, dict) else "",
    }
    preservation_check = {
        "preserves_population": True,
        "preserves_cohort_family": True,
        "preserves_predictor_or_contrast": True,
        "preserves_outcome_modality": True,
        "preserves_direction_family": True,
        "preserves_scientific_motivation": True,
        "changed_fields": ["outcome_family"],
        "allowed_change_rationale": f"Preserves the {contrast} contrast and {modality} outcome modality while narrowing the outcome family.",
    }
    evidence = localization.get("evidence") if isinstance(localization, dict) else []
    if not isinstance(evidence, list):
        evidence = []
    proposed_contract_1 = json.loads(json.dumps(contract)) if isinstance(contract, dict) else {}
    proposed_contract_2 = json.loads(json.dumps(contract)) if isinstance(contract, dict) else {}
    if proposed_contract_1:
        proposed_contract_1["claim_id"] = f"{proposed_contract_1.get('claim_id', 'claim')}_narrower_followup"
        proposed_contract_1["question"] = (
            f"Under adaptive same-data evaluation, does the {contrast} effect appear in a narrower "
            f"{modality} outcome family related to {outcome_text}?"
        )
        proposed_contract_1.setdefault("search_provenance", {})["selection"] = "discovery_only"
    if proposed_contract_2:
        proposed_contract_2["claim_id"] = f"{proposed_contract_2.get('claim_id', 'claim')}_replication_ready_followup"
        proposed_contract_2["question"] = (
            f"Does a replication-ready version of the original {contrast} claim for {outcome_text} "
            "show adaptive support before optional external validation?"
        )
        proposed_contract_2.setdefault("search_provenance", {})["selection"] = "discovery_only"
    candidates = [
        {
            "proposal_type": "exploratory_followup_claim",
            "transform_type": "narrower_outcome_family",
            "domain_core": domain_core,
            "preservation_check": preservation_check,
            "proposed_question": f"Under adaptive same-data evaluation, does the {contrast} effect appear in a narrower predeclared {modality} outcome family related to {outcome_text}?",
            "proposed_contract": proposed_contract_1,
            "rationale": "A narrower same-modality follow-up may be scientifically useful and should be labeled exploratory_confirmed unless optional external validation also passes.",
            "connection_rationale": f"Preserves the {contrast} contrast and {modality} outcome modality while narrowing the outcome family.",
            "evidence_policy": {
                "provenance": "post_hoc_followup",
                "requires_new_evidence": False,
                "can_confirm_on_current_data": True,
                "validation_split": "current_data_adaptive",
            },
            "supported_by_evidence": evidence[:2],
            "disposition_label": None,
        },
        {
            "proposal_type": "independent_replication_claim",
            "transform_type": "stronger_design",
            "domain_core": domain_core,
            "preservation_check": {
                **preservation_check,
                "changed_fields": ["validation_evidence"],
                "allowed_change_rationale": "Preserves the scientific claim while moving confirmation to independent evidence.",
            },
            "proposed_question": f"Replicate the original {contrast} claim for {outcome_text} in an independent cohort or reserved holdout with unchanged CONFIRM gates.",
            "proposed_contract": proposed_contract_2,
            "rationale": "The follow-up keeps the independent-validation target while allowing same-data adaptive screening with an exploratory_confirmed label.",
            "connection_rationale": f"Preserves the original {contrast} contrast, {modality} modality, and gate stack.",
            "evidence_policy": {
                "provenance": "independent_replication",
                "requires_new_evidence": False,
                "can_confirm_on_current_data": True,
                "validation_split": "current_data_adaptive",
            },
            "supported_by_evidence": evidence[:2],
            "disposition_label": None,
        },
    ][:max_candidates]
    return json.dumps({"candidates": candidates}, indent=2, sort_keys=True)


def _looks_like_param_error(text: str) -> bool:
    markers = (
        "unsupported",
        "not support",
        "not supported",
        "unknown parameter",
        "unrecognized",
        "invalid parameter",
        "extra inputs",
    )
    return any(marker in text for marker in markers)


def _client_timeout() -> float:
    try:
        return float(os.getenv("CONFIRM_LLM_TIMEOUT", "60"))
    except ValueError:
        return 60.0


def _openai_strict_json_schema(response_model: type[Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema to OpenAI strict structured-output shape."""

    schema = response_model.model_json_schema()
    _make_openai_schema_strict(schema)
    return schema


def _make_openai_schema_strict(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            node["additionalProperties"] = False
        for value in node.values():
            _make_openai_schema_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_openai_schema_strict(item)


def _create_chat_completion_with_param_fallback(
    create: Callable[..., Any],
    **kwargs: Any,
) -> Any:
    """Call an OpenAI-compatible chat endpoint, retrying without brittle params.

    Some OpenAI-compatible models reject explicit ``temperature`` values, while
    others reject ``max_tokens`` in favor of provider-specific token controls.
    This helper only retries parameter-shape failures; authentication, network,
    quota, and model errors still propagate to the caller.
    """

    active = dict(kwargs)
    last_error: Exception | None = None
    for _ in range(3):
        try:
            return create(**active)
        except Exception as exc:
            text = str(exc).lower()
            next_active = dict(active)
            changed = False
            if "temperature" in text and "temperature" in next_active:
                next_active.pop("temperature", None)
                changed = True
            if ("max_tokens" in text or "max completion" in text or "max_completion_tokens" in text) and "max_tokens" in next_active:
                next_active.pop("max_tokens", None)
                changed = True
            if not changed and _looks_like_param_error(text):
                if "temperature" in next_active:
                    next_active.pop("temperature", None)
                    changed = True
                elif "max_tokens" in next_active:
                    next_active.pop("max_tokens", None)
                    changed = True
            if not changed:
                raise
            active = next_active
            last_error = exc
    if last_error is not None:
        raise last_error
    return create(**active)


def make_llm(spec: str) -> LLMClient:
    """Create an LLM client from ``provider:model`` or a stand-in alias."""

    load_env()
    text = spec.strip()
    if not text:
        raise ValueError("LLM spec cannot be empty")

    if ":" in text:
        provider, model = text.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if not model and provider not in {"standin", "stand-in", "offline", "manual"}:
            raise ValueError(f"LLM spec missing model: {spec!r}")
    else:
        provider = text.lower()
        model = ""

    if provider == "openai":
        return OpenAIClient(model or None)
    if provider == "anthropic":
        return AnthropicClient(model or None)
    if provider == "openrouter":
        return OpenRouterClient(model or None)
    if provider in {"standin", "stand-in", "offline", "manual"}:
        return StandInClient()
    raise ValueError(f"Unknown LLM provider in spec {spec!r}")


def get_llm() -> LLMClient:
    """Select an LLM client from ``CONFIRM_LLM``."""

    load_env()
    backend = os.getenv("CONFIRM_LLM", "openai").strip()
    return make_llm(backend)
