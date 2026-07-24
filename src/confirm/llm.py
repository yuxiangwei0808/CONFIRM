"""LLM client boundary for the CONFIRM agent layer."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from collections.abc import Callable
from typing import Any, Protocol

from confirm.env import load_env


class LLMClient(Protocol):
    """Minimal text-completion protocol used by the agent."""

    model: str

    def complete(self, system: str, user: str) -> str:
        """Return a completion for the supplied system and user prompts."""


def complete_structured(
    llm: LLMClient,
    system: str,
    user: str,
    response_model: type[Any],
) -> str:
    """Request structured output when the provider supports it."""

    method = getattr(llm, "complete_structured", None)
    if callable(method):
        return str(method(system, user, response_model))
    return llm.complete(system, user)


def parse_structured(text: str, response_model: type[Any]) -> Any:
    """Parse a structured response through the supplied Pydantic model."""

    try:
        return response_model.model_validate_json(text)
    except Exception:
        return response_model.model_validate(json.loads(text))


def complete_structured_with_retries(
    llm: LLMClient,
    *,
    system: str,
    prompt: str,
    response_model: type[Any],
    retries: int,
    validator: Callable[[Any], None] | None = None,
) -> tuple[Any, str, int, list[dict[str, Any]]]:
    """Run the frozen structured-output retry policy with full traces."""

    attempts: list[dict[str, Any]] = []
    active_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        raw = ""
        try:
            raw = complete_structured(
                llm,
                system,
                active_prompt,
                response_model,
            )
            parsed = response_model.model_validate_json(raw)
            if validator is not None:
                validator(parsed)
            attempts.append(
                {
                    "attempt": attempt,
                    "prompt": active_prompt,
                    "prompt_sha256": sha256(
                        active_prompt.encode("utf-8")
                    ).hexdigest(),
                    "raw_response": raw,
                    "response_sha256": sha256(raw.encode("utf-8")).hexdigest(),
                    "schema_valid": True,
                    "call_metadata": dict(
                        getattr(llm, "last_call_metadata", {}) or {}
                    ),
                }
            )
            return parsed, raw, attempt, attempts
        except Exception as exc:
            last_error = exc
            attempts.append(
                {
                    "attempt": attempt,
                    "prompt": active_prompt,
                    "prompt_sha256": sha256(
                        active_prompt.encode("utf-8")
                    ).hexdigest(),
                    "raw_response": raw,
                    "response_sha256": (
                        sha256(raw.encode("utf-8")).hexdigest()
                        if raw
                        else None
                    ),
                    "schema_valid": False,
                    "error": str(exc),
                    "call_metadata": dict(
                        getattr(llm, "last_call_metadata", {}) or {}
                    ),
                }
            )
            if is_non_retryable_provider_error(exc):
                raise RuntimeError(
                    "Non-retryable LLM provider error on "
                    f"attempt {attempt}: {exc}"
                ) from exc
            active_prompt = (
                f"{prompt}\n\nPrevious structured-output error: {exc}. "
                "Return a corrected response matching the schema exactly."
            )
    raise RuntimeError(
        f"Structured output failed after {retries + 1} attempts: {last_error}"
    )


def is_non_retryable_provider_error(exc: Exception) -> bool:
    """Return whether retrying cannot repair provider authorization/billing."""

    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "insufficient credits",
            "error code: 401",
            "error code: 402",
            "error code: 403",
            "'code': 401",
            "'code': 402",
            "'code': 403",
            '"code": 401',
            '"code": 402',
            '"code": 403',
        )
    )


class OpenAIClient:
    """OpenAI-backed LLM client."""

    provider = "openai"

    def __init__(self, model: str | None = None, *, max_tokens: int | None = 2048) -> None:
        load_env()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_tokens = max_tokens
        self.timeout = _client_timeout()
        self.last_call_metadata: dict[str, Any] = {}

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
        self.last_call_metadata = _openai_response_metadata(response, provider=self.provider, model=self.model)
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
        self.last_call_metadata = _openai_response_metadata(response, provider=self.provider, model=self.model)
        return response.choices[0].message.content or ""


class GoogleClient:
    """Google Gen AI client with Pydantic structured-output support."""

    provider = "google"

    def __init__(self, model: str | None = None, *, max_tokens: int | None = 2048) -> None:
        load_env()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.max_tokens = max_tokens
        self.last_call_metadata: dict[str, Any] = {}

    def _client(self) -> Any:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key) if api_key else genai.Client()

    def complete(self, system: str, user: str) -> str:
        client = self._client()
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": 0,
                "max_output_tokens": self.max_tokens,
            },
        )
        self.last_call_metadata = _google_response_metadata(response, model=self.model)
        return str(getattr(response, "text", "") or "")

    def complete_structured(self, system: str, user: str, response_model: type[Any]) -> str:
        client = self._client()
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config={
                "system_instruction": system,
                "temperature": 0,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": _google_response_json_schema(response_model),
            },
        )
        self.last_call_metadata = _google_response_metadata(response, model=self.model)
        return str(getattr(response, "text", "") or "")


class OpenRouterClient:
    """OpenRouter-backed OpenAI-compatible LLM client."""

    provider = "openrouter"

    def __init__(self, model: str | None = None, *, max_tokens: int | None = 2048) -> None:
        load_env()
        self.model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        self.max_tokens = max_tokens
        self.timeout = _client_timeout()
        self.last_call_metadata: dict[str, Any] = {}

    def _client(self) -> Any:
        from openai import OpenAI

        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=self.timeout,
        )

    def complete(self, system: str, user: str) -> str:
        response = _create_chat_completion_with_param_fallback(
            self._client().chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        self.last_call_metadata = _openai_response_metadata(response, provider=self.provider, model=self.model)
        return response.choices[0].message.content or ""

    def complete_structured(self, system: str, user: str, response_model: type[Any]) -> str:
        """Return strict JSON and require a route that supports the schema."""

        response = _create_chat_completion_with_param_fallback(
            self._client().chat.completions.create,
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": _openrouter_strict_json_schema(response_model),
                },
            },
            extra_body={"provider": {"require_parameters": True}},
        )
        self.last_call_metadata = _openai_response_metadata(response, provider=self.provider, model=self.model)
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
        if "pubmed abstracts into literature-grounded confirm claim seeds" in system.lower():
            return _standin_literature_extraction_response(user)
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


def _standin_literature_extraction_response(user: str) -> str:
    try:
        payload = json.loads(user)
        record = payload.get("record", {})
    except Exception:
        record = {}
    target_family = str(record.get("target_family") or "ad_aging")
    pmid = str(record.get("pmid") or "0")
    title = str(record.get("title") or "stand-in PubMed record")
    doi = str(record.get("doi") or "")
    year = str(record.get("year") or "")
    if target_family == "ad_aging":
        outcome_modality = "sMRI"
        predictor_or_group = "Alzheimer disease diagnosis versus cognitively normal controls"
        outcome_family = "hippocampal volume"
        expected_direction = "negative"
        question = "AD diagnosis is associated with lower hippocampal volume."
    else:
        outcome_modality = "fMRI-FC"
        predictor_or_group = "case-control diagnosis"
        outcome_family = "functional connectivity"
        expected_direction = "two_sided"
        question = "Diagnosis is associated with functional connectivity differences."
    return json.dumps(
        {
            "claims": [
                {
                    "seed_id": f"pmid_{pmid}_seed_1",
                    "source_pmid": pmid,
                    "source_doi": doi,
                    "source_title": title,
                    "source_year": year,
                    "target_family": target_family,
                    "outcome_modality": outcome_modality,
                    "predictor_or_group": predictor_or_group,
                    "outcome_family": outcome_family,
                    "expected_direction": expected_direction,
                    "covariates": ["age", "sex"],
                    "candidate_question": question,
                    "evidence_snippet": str(record.get("abstract") or question)[:300],
                    "support_level": "direct",
                    "rationale": "Offline stand-in extraction for local smoke tests.",
                }
            ]
        },
        sort_keys=True,
    )


def _standin_candidate_response(user: str) -> str:
    try:
        payload = json.loads(user)
        contract = payload["original_contract"]
        localization = payload["failure_localization"]
        max_candidates = int(payload.get("max_candidates") or 2)
    except Exception:
        payload = {}
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
    catalog = payload.get("executable_data_catalog") if isinstance(payload, dict) else {}
    if not isinstance(catalog, dict):
        catalog = {}
    alternatives = [
        str(item)
        for item in catalog.get("common_outcome_columns_sample", [])
        if str(item) != outcome_text and str(item).split("_", 1)[0] == modality
    ]
    inclusions = [
        item
        for item in catalog.get("allowed_inclusion_examples", [])
        if item is not None and item != contract.get("inclusion")
    ]
    failure_context = payload.get("round_failure_context") if isinstance(payload, dict) else None
    responds_to = []
    if isinstance(failure_context, dict):
        failures = failure_context.get("failed_candidates")
        if isinstance(failures, list):
            responds_to = [str(item.get("candidate_id")) for item in failures if isinstance(item, dict) and item.get("candidate_id")]
        elif isinstance(failure_context.get("failed_candidate_ids"), list):
            responds_to = [str(item) for item in failure_context["failed_candidate_ids"]]
    candidates = []
    if alternatives:
        proposed_contract = json.loads(json.dumps(contract))
        proposed_contract["claim_id"] = f"{proposed_contract.get('claim_id', 'claim')}_narrower_followup"
        proposed_contract["question"] = f"Does the original contrast extend to {alternatives[0]}?"
        proposed_contract.setdefault("estimand", {})["outcome"] = alternatives[0]
        candidates.append({
            "proposal_type": "exploratory_followup_claim",
            "transform_type": "narrower_outcome_family",
            "domain_core": domain_core,
            "preservation_check": preservation_check,
            "proposed_question": f"Under adaptive same-data evaluation, does the {contrast} effect appear in a narrower predeclared {modality} outcome family related to {outcome_text}?",
            "proposed_contract": proposed_contract,
            "rationale": "A distinct source-measured outcome in the same modality is an executable connected follow-up.",
            "connection_rationale": f"Preserves the {contrast} contrast and {modality} outcome modality while narrowing the outcome family.",
            "evidence_policy": {
                "provenance": "post_hoc_followup",
                "requires_new_evidence": False,
                "can_confirm_on_current_data": True,
                "validation_split": "current_data_adaptive",
            },
            "supported_by_evidence": evidence[:2],
            "disposition_label": None,
            "responds_to_candidate_ids": responds_to,
        })
    if inclusions:
        proposed_contract = json.loads(json.dumps(contract))
        proposed_contract["claim_id"] = f"{proposed_contract.get('claim_id', 'claim')}_subgroup_followup"
        proposed_contract["question"] = f"Does the original claim hold under {inclusions[0]}?"
        proposed_contract["inclusion"] = inclusions[0]
        candidates.append({
            "proposal_type": "exploratory_followup_claim",
            "transform_type": "moderator_or_subgroup",
            "domain_core": domain_core,
            "preservation_check": {
                **preservation_check,
                "changed_fields": ["inclusion"],
                "allowed_change_rationale": "Uses a source-data-feasible subgroup while preserving the scientific core.",
            },
            "proposed_question": proposed_contract["question"],
            "proposed_contract": proposed_contract,
            "rationale": "The subgroup predicate was derived from parent source covariates before excluded evaluation.",
            "connection_rationale": f"Preserves the original {contrast} contrast, {modality} modality, and gate stack while refining inclusion.",
            "evidence_policy": {
                "provenance": "post_hoc_followup",
                "requires_new_evidence": False,
                "can_confirm_on_current_data": True,
                "validation_split": "current_data_adaptive",
            },
            "supported_by_evidence": evidence[:2],
            "disposition_label": None,
            "responds_to_candidate_ids": responds_to,
        })
    candidates = candidates[:max_candidates]
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


def _openai_response_metadata(response: Any, *, provider: str, model: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    usage_payload: dict[str, Any] = {}
    if usage is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            value = getattr(usage, key, None)
            if value is not None:
                usage_payload[key] = float(value) if key == "cost" else int(value)
    response_id = str(getattr(response, "id", "") or "")
    return {
        "provider": provider,
        "model": str(getattr(response, "model", "") or model),
        "routed_provider": str(getattr(response, "provider", "") or "") or None,
        "response_id": response_id,
        "response_id_sha256": sha256(response_id.encode("utf-8")).hexdigest() if response_id else None,
        "usage": usage_payload,
    }


def _google_response_metadata(response: Any, *, model: str) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    usage_payload: dict[str, Any] = {}
    if usage is not None:
        mappings = {
            "prompt_token_count": "prompt_tokens",
            "candidates_token_count": "completion_tokens",
            "total_token_count": "total_tokens",
        }
        for source, target in mappings.items():
            value = getattr(usage, source, None)
            if value is not None:
                usage_payload[target] = int(value)
    return {"provider": "google", "model": model, "response_id": None, "response_id_sha256": None, "usage": usage_payload}


def _openai_strict_json_schema(response_model: type[Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema to OpenAI strict structured-output shape."""

    schema = response_model.model_json_schema()
    _make_openai_schema_strict(schema)
    return schema


def _google_response_json_schema(response_model: type[Any]) -> dict[str, Any]:
    """Return JSON schema without fields rejected by Gemini's schema endpoint."""

    schema = response_model.model_json_schema()
    _remove_google_unsupported_schema_fields(schema)
    return schema


def _openrouter_strict_json_schema(response_model: type[Any]) -> dict[str, Any]:
    """Return strict JSON schema without array bounds rejected by Claude routes."""

    schema = _openai_strict_json_schema(response_model)
    _remove_openrouter_unsupported_schema_fields(schema)
    return schema


def _remove_openrouter_unsupported_schema_fields(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("minItems", None)
        node.pop("maxItems", None)
        for value in node.values():
            _remove_openrouter_unsupported_schema_fields(value)
    elif isinstance(node, list):
        for item in node:
            _remove_openrouter_unsupported_schema_fields(item)


def _remove_google_unsupported_schema_fields(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("additionalProperties", None)
        node.pop("minItems", None)
        node.pop("maxItems", None)
        for value in node.values():
            _remove_google_unsupported_schema_fields(value)
    elif isinstance(node, list):
        for item in node:
            _remove_google_unsupported_schema_fields(item)


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
    if provider in {"google", "gemini"}:
        return GoogleClient(model or None)
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
