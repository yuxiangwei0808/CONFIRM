"""Deterministic failure evidence extraction for post-verdict layers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DispositionLabel = Literal["needs_more_data", "non_replicated", "under_powered", "fragile", "abandon_claim"]


class ClaimFeedback(BaseModel):
    """Minimal verdict-derived failure evidence used by proposal/search layers."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    source_verdict: str
    failed_gates: list[str] = Field(default_factory=list)
    primary_failure: str
    diagnosis: str
    evidence: list[str] = Field(default_factory=list)
    next_agent_instruction: str


def feedback_from_verdict(
    claim_id: str,
    verdict: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
) -> ClaimFeedback:
    """Create deterministic failure evidence from a serialized Verdict-like mapping."""

    verdict_payload = _plain(verdict)
    results_payload = _plain(results or {})
    source_verdict = str(verdict_payload.get("label", verdict_payload.get("final_label", "unknown")))
    gates = verdict_payload.get("gates", {})
    failed: list[str] = []
    if isinstance(gates, Mapping):
        failed = [str(name) for name, passed in gates.items() if isinstance(passed, bool) and not passed]
    rationale = str(verdict_payload.get("rationale", "") or "")
    primary = _primary_failure(source_verdict, failed, rationale, results_payload)
    evidence = _evidence_from_results(verdict_payload, results_payload, failed, primary)
    return ClaimFeedback(
        claim_id=claim_id,
        source_verdict=source_verdict,
        failed_gates=failed,
        primary_failure=primary,
        diagnosis=_diagnosis(primary, rationale),
        evidence=evidence,
        next_agent_instruction=_instruction(primary, rationale),
    )


def _primary_failure(
    source_verdict: str,
    failed: list[str],
    rationale: str,
    results: Mapping[str, Any],
) -> str:
    if source_verdict == "confirmed":
        return "none"
    text = rationale.lower()
    contract = _mapping(results.get("contract"))
    search = _mapping(contract.get("search_provenance"))
    selection = str(search.get("selection") or "").lower()
    if "execution_error" in text:
        return "execution_error"
    if "unverifiable_search" in text or "search_provenance" in failed:
        return "search_provenance"
    if selection in {"discovery_only", "full_data", "unknown"} and "multiplicity" in failed:
        return "search_provenance"
    if "confound_completeness" in failed or "confound_incomplete" in text:
        return "confound"
    if "confound" in failed or "predictor is nested" in text:
        return "confound"
    if "multiplicity" in failed:
        return "multiplicity"
    if "power" in failed or source_verdict == "under_powered":
        return "power"
    if "multiverse" in failed or "pattern_corr" in failed or "region_replication_fraction" in failed:
        return "multiverse"
    if "replication" in failed or source_verdict == "non_replicated":
        return "replication"
    return "abstention"


def _diagnosis(primary: str, rationale: str) -> str:
    if primary == "none":
        return "The claim is already confirmed by the current gate stack."
    if primary == "execution_error":
        return "The claim failed because the contract did not execute cleanly."
    if primary == "search_provenance":
        return "The claim failed because the hypothesis lineage/search family is not valid for confirmatory reporting."
    if primary == "confound":
        if "nested" in rationale.lower():
            return "The claim failed because the predictor is structurally nested in a declared confound."
        return "The claim failed because measured confounding is missing or incomplete."
    if primary == "power":
        return "The claim failed because the design is under-powered for the declared effect target."
    if primary == "multiverse":
        return "The claim failed because the result is not stable across reasonable analysis specifications."
    if primary == "replication":
        return "The claim failed because it did not replicate under the declared replication gate."
    if primary == "multiplicity":
        return "The primary effect does not survive the declared multiplicity correction."
    return "The current evidence does not support confirmation."


def _instruction(primary: str, rationale: str) -> str:
    if primary == "confound" and "nested" in rationale.lower():
        return "Require a balanced independent design; do not treat a nested predictor as confirmatory evidence."
    if primary == "confound":
        return "Repair measured confound specification or require new evidence before confirmation."
    if primary == "search_provenance":
        return "Keep the true search lineage and require future/split-confirmatory evidence for post-hoc claims."
    if primary == "power":
        return "Require additional sample size or report an under-powered disposition."
    if primary == "multiverse":
        return "Treat current evidence as fragile and require a predeclared future specification."
    if primary == "replication":
        return "Require independent replication evidence or report non-replication."
    if primary == "multiplicity":
        return "Do not shrink the searched family; require a connected follow-up with excluded validation evidence."
    if primary == "execution_error":
        return "Only true executable contract corrections may be rerun on current data."
    return "Do not force confirmation; produce a disposition or a provenance-safe follow-up."


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _num(value)
    if number is None:
        return str(value)
    if number == 0:
        return "0"
    if abs(number) < 0.001 or abs(number) >= 10000:
        return f"{number:.3g}"
    return f"{number:.4g}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _evidence_from_results(
    verdict: Mapping[str, Any],
    results: Mapping[str, Any],
    failed: list[str],
    primary: str,
) -> list[str]:
    evidence: list[str] = []
    gates = _mapping(verdict.get("gates"))
    contract = _mapping(results.get("contract"))
    gate_config = _mapping(contract.get("gates"))

    primary_result = _mapping(results.get("primary"))
    if "multiplicity" in failed:
        parts = ["Multiplicity gate failed"]
        p_value = _num(primary_result.get("p"))
        if p_value is not None:
            parts.append(f"primary p={_fmt(p_value)}")
        family_size = gates.get("multiplicity_effective_family_size")
        if family_size is None:
            family_size = _mapping(gate_config.get("multiplicity")).get("family_size")
        method = _mapping(gate_config.get("multiplicity")).get("method")
        alpha = _mapping(gate_config.get("multiplicity")).get("alpha")
        if method:
            parts.append(f"method={method}")
        if alpha is not None:
            parts.append(f"alpha={_fmt(alpha)}")
        if family_size is not None:
            parts.append(f"effective family_size={_fmt(family_size)}")
        evidence.append("; ".join(parts) + ".")

    multiverse = _mapping(results.get("multiverse"))
    if "multiverse" in failed or "pattern_corr" in failed or "region_replication_fraction" in failed:
        fraction = _num(multiverse.get("fraction_consistent"))
        threshold = _num(_mapping(gate_config.get("multiverse")).get("min_fraction_consistent"))
        specs = multiverse.get("specs")
        parts = ["Multiverse gate failed"]
        if fraction is not None:
            parts.append(f"fraction_consistent={_fmt(fraction)}")
        if threshold is not None:
            parts.append(f"required>={_fmt(threshold)}")
        if isinstance(specs, list):
            valid_specs = [_mapping(spec) for spec in specs if isinstance(spec, Mapping)]
            consistent = [spec for spec in valid_specs if bool(spec.get("same_sign")) and bool(spec.get("significant"))]
            parts.append(f"consistent_specs={len(consistent)}/{len(valid_specs)}")
        evidence.append("; ".join(parts) + ".")

    replication = _mapping(results.get("replication"))
    if "replication" in failed:
        reason = str(replication.get("reason") or "").strip()
        parts = ["Replication gate failed"]
        if reason:
            parts.append(f"reason={reason}")
        cohort_results = replication.get("cohort_results")
        if isinstance(cohort_results, list):
            cohort_parts = []
            for item in cohort_results[:3]:
                cohort = _mapping(item)
                effect = _mapping(cohort.get("effect"))
                label = str(cohort.get("cohort") or "replication")
                details = [label]
                if cohort.get("reason"):
                    details.append(f"reason={cohort.get('reason')}")
                if effect.get("beta") is not None:
                    details.append(f"beta={_fmt(effect.get('beta'))}")
                if effect.get("p") is not None:
                    details.append(f"p={_fmt(effect.get('p'))}")
                if effect.get("n") is not None:
                    details.append(f"n={_fmt(effect.get('n'))}")
                cohort_parts.append(" ".join(details))
            if cohort_parts:
                parts.append("cohorts: " + " | ".join(cohort_parts))
        evidence.append("; ".join(parts) + ".")

    power = _mapping(results.get("power"))
    if "power" in failed or primary == "power":
        parts = ["Power gate failed"]
        achieved = power.get("achieved_power")
        threshold = _mapping(gate_config.get("power")).get("min_power")
        needed = power.get("n_needed_80")
        if achieved is not None:
            parts.append(f"achieved_power={_fmt(achieved)}")
        if threshold is not None:
            parts.append(f"required>={_fmt(threshold)}")
        if needed is not None:
            parts.append(f"n_needed_80={_fmt(needed)}")
        evidence.append("; ".join(parts) + ".")

    confound = _mapping(results.get("confound_completeness")) or _mapping(gates.get("confound_completeness_audit"))
    if "confound" in failed or "confound_completeness" in failed:
        parts = ["Confound gate failed"]
        if confound.get("reason"):
            parts.append(f"reason={confound.get('reason')}")
        failures = confound.get("failures")
        if isinstance(failures, list) and failures:
            parts.append("failures=" + ", ".join(str(item) for item in failures[:5]))
        details = confound.get("details")
        if isinstance(details, list) and details:
            detail_parts = []
            for item in details[:3]:
                detail = _mapping(item)
                confound_name = detail.get("confound")
                p_value = detail.get("p")
                associated = detail.get("associated")
                if confound_name:
                    text = str(confound_name)
                    if p_value is not None:
                        text += f" p={_fmt(p_value)}"
                    if associated is not None:
                        text += f" associated={associated}"
                    detail_parts.append(text)
            if detail_parts:
                parts.append("tested=" + " | ".join(detail_parts))
        evidence.append("; ".join(parts) + ".")

    if primary == "search_provenance":
        search = _mapping(contract.get("search_provenance"))
        parts = ["Search provenance is not confirmatory/auditable"]
        if search.get("selection"):
            parts.append(f"selection={search.get('selection')}")
        if search.get("family_size") is not None:
            parts.append(f"declared family_size={_fmt(search.get('family_size'))}")
        evidence.append("; ".join(parts) + ".")

    rationale = str(verdict.get("rationale") or "").strip()
    if not evidence and rationale:
        evidence.append(f"Verdict rationale: {rationale}")
    if not evidence and failed:
        evidence.append(f"Failed gates: {', '.join(failed)}.")
    return evidence
