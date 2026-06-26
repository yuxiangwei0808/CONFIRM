"""Deterministic post-verdict feedback for claim repair and triage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confirm.contract import ClaimContract

Repairability = Literal["contract_repairable", "needs_new_data", "downgrade_only", "not_repairable"]
DispositionLabel = Literal["needs_more_data", "non_replicated", "under_powered", "fragile", "abandon_claim"]
ResponseType = Literal["revised_contract", "claim_disposition"]
ContractChange = Literal["covariates", "search_family_size", "replication_cohorts", "cohorts", "estimand", "disposition"]


class ClaimFeedback(BaseModel):
    """Structured repair guidance emitted after a CONFIRM verdict."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    source_verdict: str
    failed_gates: list[str] = Field(default_factory=list)
    primary_failure: str
    repairability: Repairability
    diagnosis: str
    evidence: list[str] = Field(default_factory=list)
    allowed_revisions: list[str]
    forbidden_revisions: list[str]
    refinement_actions: list[str] = Field(default_factory=list)
    next_agent_instruction: str
    must_preserve: list[str]
    allowed_contract_changes: list[ContractChange] = Field(default_factory=list)


class RevisionResponse(BaseModel):
    """Agent response after receiving feedback."""

    model_config = ConfigDict(extra="forbid")

    response_type: ResponseType
    revised_contract: Optional[ClaimContract] = None
    disposition_label: Optional[DispositionLabel] = None
    rationale: Optional[str] = None

    @model_validator(mode="after")
    def validate_response_shape(self) -> "RevisionResponse":
        if self.response_type == "revised_contract" and self.revised_contract is None:
            raise ValueError("revised_contract responses require revised_contract")
        if self.response_type == "claim_disposition" and self.disposition_label is None:
            raise ValueError("claim_disposition responses require disposition_label")
        return self


class PolicyValidation(BaseModel):
    """Result of validating a revised response against feedback policy."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    accepted_disposition: bool = False
    checked_contract: bool = False


def feedback_from_row(row: Mapping[str, Any]) -> ClaimFeedback:
    """Create feedback from an audit/result row."""

    claim_id = str(row.get("claim_id", "") or "unknown_claim")
    source_verdict = _source_verdict(row)
    failed = _failed_gates(row)
    primary = _primary_failure(row, failed)
    evidence = _evidence_from_row(row, failed, primary)
    return _feedback_for_failure(
        claim_id,
        source_verdict,
        failed,
        primary,
        str(row.get("rationale", "") or ""),
        evidence=evidence,
    )


def feedback_from_verdict(
    claim_id: str,
    verdict: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
) -> ClaimFeedback:
    """Create feedback from a serialized Verdict-like mapping."""

    verdict_payload = _plain(verdict)
    results_payload = _plain(results or {})
    source_verdict = str(verdict_payload.get("label", verdict_payload.get("final_label", "unknown")))
    gates = verdict_payload.get("gates", {})
    failed: list[str] = []
    if isinstance(gates, Mapping):
        failed = [str(name) for name, passed in gates.items() if isinstance(passed, bool) and not passed]
    rationale = str(verdict_payload.get("rationale", "") or "")
    primary = _primary_failure({"final_label": source_verdict, "rationale": rationale}, failed)
    evidence = _evidence_from_results(verdict_payload, results_payload, failed, primary)
    return _feedback_for_failure(claim_id, source_verdict, failed, primary, rationale, evidence=evidence)


def feedback_for_estimand_mismatch(
    claim_id: str,
    *,
    source_verdict: str = "draft_mismatch",
    mismatches: Mapping[str, Any] | None = None,
) -> ClaimFeedback:
    """Feedback for an agent draft whose estimand differs from the requested task."""

    mismatch_names = sorted(str(key) for key in (mismatches or {}).keys())
    detail = ", ".join(mismatch_names) if mismatch_names else "estimand fields"
    return ClaimFeedback(
        claim_id=claim_id,
        source_verdict=source_verdict,
        failed_gates=["estimand_match"],
        primary_failure="estimand_mismatch",
        repairability="contract_repairable",
        diagnosis=f"The drafted contract does not match the intended scientific estimand: {detail}.",
        evidence=[
            f"Mismatched contract fields: {detail}.",
        ],
        allowed_revisions=[
            "Correct the mismatched estimand fields to match the original user question.",
            "Keep the same gate thresholds and confound requirements.",
            "Return a downgrade disposition if the requested estimand cannot be represented.",
        ],
        forbidden_revisions=[
            "Do not change the scientific question to an easier claim.",
            "Do not lower statistical gates or remove required covariates.",
            "Do not switch to an outcome selected from observed significance.",
        ],
        refinement_actions=[
            "Revise only the named mismatched fields.",
            "Preserve the original scientific question and all gate thresholds.",
            "Use a claim disposition if the intended estimand cannot be encoded without changing the question.",
        ],
        next_agent_instruction="Revise only the mismatched contract fields, or return a claim disposition if the requested estimand is not executable.",
        must_preserve=["scientific_question", "gate_thresholds"],
        allowed_contract_changes=["estimand", "cohorts", "covariates", "disposition"],
    )


def validate_revision_response(
    original: ClaimContract | None,
    response: RevisionResponse,
    feedback: ClaimFeedback,
) -> PolicyValidation:
    """Validate an agent revision response against deterministic repair policy."""

    if response.response_type == "claim_disposition":
        return PolicyValidation(ok=True, accepted_disposition=True)
    if feedback.repairability == "downgrade_only":
        return PolicyValidation(
            ok=False,
            violations=["This feedback is downgrade-only; return a claim disposition instead of a revised contract."],
            checked_contract=True,
        )
    if original is None:
        return PolicyValidation(
            ok=False,
            violations=["Cannot policy-check a revised contract without an original contract."],
            checked_contract=True,
        )
    if response.revised_contract is None:
        return PolicyValidation(
            ok=False,
            violations=["Missing revised_contract for revised_contract response."],
            checked_contract=True,
        )
    if feedback.repairability == "needs_new_data":
        original_replication = sorted(original.replication_cohorts)
        revised_replication = sorted(response.revised_contract.replication_cohorts)
        if "replication_cohorts" not in feedback.allowed_contract_changes or original_replication == revised_replication:
            return PolicyValidation(
                ok=False,
                violations=[
                    "This feedback requires new evidence; return a disposition unless adding a valid independent replication cohort is explicitly allowed."
                ],
                checked_contract=True,
            )
    return validate_revised_contract(original, response.revised_contract, feedback)


def validate_revised_contract(
    original: ClaimContract,
    revised: ClaimContract,
    feedback: ClaimFeedback,
) -> PolicyValidation:
    """Reject gate gaming while allowing narrow feedback-directed repairs."""

    violations: list[str] = []
    warnings: list[str] = []
    allowed = set(feedback.allowed_contract_changes)
    estimand_repair_allowed = feedback.primary_failure == "estimand_mismatch" and "estimand" in allowed
    cohort_repair_allowed = "cohorts" in allowed or estimand_repair_allowed
    covariate_repair_allowed = "covariates" in allowed or estimand_repair_allowed

    if _outcomes(original) != _outcomes(revised) and "estimand" not in allowed:
        violations.append("Outcome changed outside an estimand-repair allowance.")
    if original.estimand.direction != revised.estimand.direction and "estimand" not in allowed:
        violations.append("Direction changed outside an estimand-repair allowance.")
    if original.estimand.type != revised.estimand.type and "estimand" not in allowed:
        violations.append("Estimand type changed outside an estimand-repair allowance.")
    if _group_dict(original) != _group_dict(revised) and "estimand" not in allowed:
        violations.append("Group contrast changed outside an estimand-repair allowance.")
    if original.estimand.predictor != revised.estimand.predictor and "estimand" not in allowed:
        violations.append("Predictor changed outside an estimand-repair allowance.")

    if original.discovery_cohort != revised.discovery_cohort and not cohort_repair_allowed:
        violations.append("Discovery cohort changed; feedback never permits changing the tested discovery evidence.")
    if (
        sorted(original.replication_cohorts) != sorted(revised.replication_cohorts)
        and "replication_cohorts" not in allowed
        and not cohort_repair_allowed
    ):
        violations.append("Replication cohort changed without a replication-repair allowance.")
    if revised.discovery_cohort in set(revised.replication_cohorts):
        violations.append("Replication cohorts must be independent from the discovery cohort.")

    original_required = set(original.gates.confound.require_covariates)
    revised_required = set(revised.gates.confound.require_covariates)
    if not original_required.issubset(revised_required) and not estimand_repair_allowed:
        violations.append("Required confound covariates were removed.")
    if not set(original.covariates).issubset(set(revised.covariates)) and not covariate_repair_allowed:
        violations.append("Original covariates were removed without a covariate-repair allowance.")

    violations.extend(_gate_weakening_violations(original, revised))
    violations.extend(_search_provenance_violations(original, revised))

    if "covariates" in allowed and revised_required == original_required:
        warnings.append("Feedback allowed covariate repair, but no new required confound covariate was added.")
    if "search_family_size" in allowed and revised.search_provenance.family_size == original.search_provenance.family_size:
        warnings.append("Feedback allowed search-family repair, but family_size did not change.")

    return PolicyValidation(ok=not violations, violations=violations, warnings=warnings, checked_contract=True)


def summarize_feedback(feedbacks: list[ClaimFeedback]) -> dict[str, Any]:
    """Summarize feedback objects for replay experiments."""

    abstentions = [item for item in feedbacks if item.source_verdict != "confirmed"]
    actionable = [
        item
        for item in abstentions
        if item.next_agent_instruction.strip()
        and item.primary_failure != "none"
        and item.repairability != "not_repairable"
    ]
    forbidden_total = sum(len(item.forbidden_revisions) for item in abstentions)
    return {
        "n_feedback": len(feedbacks),
        "n_abstentions": len(abstentions),
        "feedback_coverage": (len(actionable) / len(abstentions)) if abstentions else 1.0,
        "actionable_count": len(actionable),
        "primary_failure_counts": dict(Counter(item.primary_failure for item in abstentions)),
        "repairability_counts": dict(Counter(item.repairability for item in abstentions)),
        "forbidden_action_template_count": forbidden_total,
        "forbidden_action_template_rate": (forbidden_total / len(abstentions)) if abstentions else 0.0,
    }


def _source_verdict(row: Mapping[str, Any]) -> str:
    value = row.get("final_label", row.get("label", row.get("source_verdict", "")))
    text = str(value or "").strip()
    return text or "unknown"


def _failed_gates(row: Mapping[str, Any]) -> list[str]:
    failed: list[str] = []
    for name in [
        "search_provenance",
        "confound",
        "confound_completeness",
        "multiplicity",
        "power",
        "multiverse",
        "replication",
        "pattern_corr",
        "region_replication_fraction",
        "dice",
    ]:
        value = row.get(name)
        if _is_false(value):
            failed.append(name)

    failed.extend(_failed_from_cumulative_rungs(row))

    rationale = str(row.get("rationale", "") or "").lower()
    for gate in [
        "search_provenance",
        "confound_completeness",
        "confound",
        "multiplicity",
        "power",
        "multiverse",
        "replication",
        "pattern_corr",
        "region_replication_fraction",
    ]:
        if gate in rationale:
            failed.append(gate)
    if "unverifiable_search" in rationale:
        failed.append("search_provenance")
    if "non_replicated" in rationale:
        failed.append("replication")
    return list(dict.fromkeys(failed))


def _failed_from_cumulative_rungs(row: Mapping[str, Any]) -> list[str]:
    """Infer the first failed gate from cumulative audit columns."""

    if "exec_only" not in row:
        return []
    if _is_false(row.get("exec_only")):
        return ["multiplicity"]
    if "+confound" in row and _is_false(row.get("+confound")):
        return ["confound"]
    if "+power" in row and _is_false(row.get("+power")):
        return ["power"]
    if "+multiverse" in row and _is_false(row.get("+multiverse")):
        return ["multiverse"]
    if "+replication" in row and _is_false(row.get("+replication")):
        return ["replication"]
    return []


def _primary_failure(row: Mapping[str, Any], failed: list[str]) -> str:
    if _source_verdict(row) == "confirmed":
        return "none"
    rationale = str(row.get("rationale", "") or "").lower()
    search_selection = str(row.get("search_selection", "") or "").lower()
    if "execution_error" in rationale:
        return "execution_error"
    if "unverifiable_search" in rationale or "search_provenance" in failed:
        return "search_provenance"
    if search_selection in {"discovery_only", "full_data", "unknown"} and "multiplicity" in failed:
        return "search_provenance"
    if "confound_completeness" in failed or "confound_incomplete" in rationale:
        return "confound"
    if "confound" in failed or "predictor is nested" in rationale:
        return "confound"
    if "multiplicity" in failed:
        return "multiplicity"
    if "power" in failed or _source_verdict(row) == "under_powered":
        return "power"
    if "multiverse" in failed or "pattern_corr" in failed or "region_replication_fraction" in failed:
        return "multiverse"
    if "replication" in failed or _source_verdict(row) == "non_replicated":
        return "replication"
    return "abstention"


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


def _evidence_from_row(row: Mapping[str, Any], failed: list[str], primary: str) -> list[str]:
    evidence: list[str] = []
    rationale = str(row.get("rationale", "") or "").strip()
    if rationale:
        evidence.append(f"Verdict rationale: {rationale}")

    search_selection = str(row.get("search_selection", "") or "").strip()
    if primary == "search_provenance" and search_selection:
        evidence.append(f"Search provenance selection was {search_selection!r}.")

    for key, label in [
        ("p", "p"),
        ("q", "q"),
        ("beta", "effect beta"),
        ("n", "sample size"),
        ("power", "power"),
        ("achieved_power", "achieved power"),
        ("family_size", "family size"),
    ]:
        if key in row and str(row.get(key, "")).strip():
            evidence.append(f"Observed {label}: {_fmt(row[key])}.")

    if not evidence and failed:
        evidence.append(f"Failed gates: {', '.join(failed)}.")
    return evidence


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
            family_size = _mapping(_mapping(gate_config.get("multiplicity")).get("family_size")).get("value")
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
            consistent = [
                spec
                for spec in valid_specs
                if bool(spec.get("same_sign")) and bool(spec.get("significant"))
            ]
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
        rationale = str(power.get("rationale") or "").strip()
        if rationale:
            parts.append(f"power_rationale={rationale}")
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


def _refinement_actions(primary: str) -> list[str]:
    if primary == "confound":
        return [
            "If the issue is missing measured covariates, add them to both `covariates` and `gates.confound.require_covariates`.",
            "If the predictor is nested in site/scanner/cohort, return `needs_more_data` unless a balanced independent design is available.",
            "Do not change the outcome or biological direction to avoid the confound.",
        ]
    if primary == "search_provenance":
        return [
            "Set `search_provenance.family_size` to the full searched family and keep the true selection mode.",
            "Use `claim_disposition: fragile` if the search family cannot be reconstructed.",
            "For confirmation, rerun as a new split-confirmatory or preregistered design.",
        ]
    if primary == "power":
        return [
            "Return `claim_disposition: under_powered` or `needs_more_data` for the current evidence.",
            "Request additional sample size or a better-powered independent replication cohort.",
            "Do not set the reference effect from the observed effect in the failed run.",
        ]
    if primary == "multiverse":
        return [
            "Return `claim_disposition: fragile` for the current evidence.",
            "For a future run, predeclare one scientifically justified analysis specification before looking at results.",
            "Do not select only the favorable multiverse branch as a revised contract.",
        ]
    if primary == "replication":
        return [
            "Return `claim_disposition: non_replicated` or `needs_more_data` unless a valid independent replication cohort already exists.",
            "A revised contract may add an independent replication cohort only if it preserves the original question, outcome, direction, and cohort scope.",
            "Do not swap to a replication cohort because it has a favorable observed p-value.",
        ]
    if primary == "multiplicity":
        return [
            "Return `claim_disposition: fragile` for the current evidence.",
            "In the rationale, cite only evidence values supplied by CONFIRM feedback or the initial verdict.",
            "For a future confirmatory run, predeclare the same outcome family and keep the full family size instead of lowering alpha or family_size.",
        ]
    if primary == "none":
        return ["No repair action is needed."]
    if primary == "estimand_mismatch":
        return [
            "Revise only the mismatched estimand fields.",
            "Preserve the original scientific question and all gate thresholds.",
        ]
    return [
        "Repair only schema or execution errors that preserve the same scientific question.",
        "Return an abstention disposition if the evidence cannot support confirmation.",
    ]


def _feedback_for_failure(
    claim_id: str,
    source_verdict: str,
    failed: list[str],
    primary: str,
    rationale: str,
    *,
    evidence: list[str] | None = None,
) -> ClaimFeedback:
    preserve = ["scientific_question", "outcome_family", "direction", "discovery_cohort", "gate_thresholds"]
    if primary != "replication":
        preserve.append("replication_cohorts")
    evidence_items = list(evidence or [])
    action_items = _refinement_actions(primary)

    if source_verdict == "confirmed" or primary == "none":
        return ClaimFeedback(
            claim_id=claim_id,
            source_verdict="confirmed",
            failed_gates=[],
            primary_failure="none",
            repairability="not_repairable",
            diagnosis="The claim is already confirmed by the current gate stack.",
            evidence=evidence_items,
            allowed_revisions=["No revision is needed."],
            forbidden_revisions=["Do not revise a confirmed claim to chase larger effects."],
            refinement_actions=action_items,
            next_agent_instruction="Report the confirmed claim using only engine-computed numbers.",
            must_preserve=preserve,
            allowed_contract_changes=[],
        )

    if primary == "confound":
        nested = "nested" in rationale.lower()
        return ClaimFeedback(
            claim_id=claim_id,
            source_verdict=source_verdict,
            failed_gates=failed,
            primary_failure="confound",
            repairability="needs_new_data" if nested else "contract_repairable",
            diagnosis=(
                "The apparent effect is structurally confounded: the predictor is nested in a declared confound."
                if nested
                else "A required measured confound is missing or incomplete in the claim contract."
            ),
            evidence=evidence_items,
            allowed_revisions=[
                "Add missing measured confounds to the covariate and confound-gate requirements.",
                "Use a balanced or independent cohort/split where the predictor is not nested in site/scanner.",
                "Downgrade the claim to a confounded association if the design cannot be repaired.",
            ],
            forbidden_revisions=[
                "Do not remove the confound gate or required covariates.",
                "Do not report the result as biological confirmation while the design is confounded.",
                "Do not change the outcome to a more significant feature.",
            ],
            refinement_actions=action_items,
            next_agent_instruction="Repair the confound specification/design, or return a fragile/needs_more_data disposition.",
            must_preserve=preserve,
            allowed_contract_changes=["covariates", "disposition"],
        )

    if primary == "search_provenance":
        return ClaimFeedback(
            claim_id=claim_id,
            source_verdict=source_verdict,
            failed_gates=failed,
            primary_failure="search_provenance",
            repairability="contract_repairable",
            diagnosis="The reported effect depends on a search family that is broader than the declared confirmatory test.",
            evidence=evidence_items,
            allowed_revisions=[
                "Declare the full searched family size and keep discovery-only provenance visible.",
                "Rerun as a new split-confirmatory design before reporting confirmation.",
                "Return a fragile disposition if the search cannot be audited.",
            ],
            forbidden_revisions=[
                "Do not relabel a discovery-only search as preregistered after seeing results.",
                "Do not shrink the multiplicity family size.",
                "Do not switch to another selected feature after seeing p-values.",
            ],
            refinement_actions=action_items,
            next_agent_instruction="Revise search provenance/family size honestly, or return a fragile disposition.",
            must_preserve=preserve,
            allowed_contract_changes=["search_family_size", "disposition"],
        )

    if primary == "power":
        return ClaimFeedback(
            claim_id=claim_id,
            source_verdict=source_verdict,
            failed_gates=failed,
            primary_failure="power",
            repairability="needs_new_data",
            diagnosis="The design is under-powered for the predeclared minimal effect of interest.",
            evidence=evidence_items,
            allowed_revisions=[
                "Request more subjects or a better-powered replication cohort.",
                "Use an externally justified reference effect before execution.",
                "Return an under_powered or needs_more_data disposition.",
            ],
            forbidden_revisions=[
                "Do not lower the minimum power threshold.",
                "Do not use the observed effect as the reference effect.",
                "Do not remove the power gate.",
            ],
            refinement_actions=action_items,
            next_agent_instruction="Do not force confirmation; request more data or return an under_powered/needs_more_data disposition.",
            must_preserve=preserve,
            allowed_contract_changes=["disposition"],
        )

    if primary == "multiverse":
        return ClaimFeedback(
            claim_id=claim_id,
            source_verdict=source_verdict,
            failed_gates=failed,
            primary_failure="multiverse",
            repairability="downgrade_only",
            diagnosis="The result is not stable across reasonable analysis specifications.",
            evidence=evidence_items,
            allowed_revisions=[
                "State that the claim is fragile under multiverse analysis.",
                "Predeclare a narrower estimand for a future run if scientifically justified.",
                "Return a fragile disposition.",
            ],
            forbidden_revisions=[
                "Do not select only the favorable analysis branch after seeing the multiverse.",
                "Do not lower the multiverse consistency threshold.",
                "Do not present a fragile result as confirmed.",
            ],
            refinement_actions=action_items,
            next_agent_instruction="Return a fragile disposition unless a new predeclared study design is available.",
            must_preserve=preserve,
            allowed_contract_changes=["disposition"],
        )

    if primary == "replication":
        return ClaimFeedback(
            claim_id=claim_id,
            source_verdict=source_verdict,
            failed_gates=failed,
            primary_failure="replication",
            repairability="needs_new_data",
            diagnosis="The discovery effect did not reproduce under the required replication check.",
            evidence=evidence_items,
            allowed_revisions=[
                "Use a valid independent replication cohort that matches the original claim scope.",
                "Narrow the claim scope only as a future predeclared claim.",
                "Return a non_replicated or needs_more_data disposition.",
            ],
            forbidden_revisions=[
                "Do not swap replication cohorts based on observed significance.",
                "Do not lower replication thresholds.",
                "Do not report a non-replicated effect as confirmed.",
            ],
            refinement_actions=action_items,
            next_agent_instruction="Request independent replication evidence or return a non_replicated/needs_more_data disposition.",
            must_preserve=preserve,
            allowed_contract_changes=["replication_cohorts", "disposition"],
        )

    if primary == "multiplicity":
        return ClaimFeedback(
            claim_id=claim_id,
            source_verdict=source_verdict,
            failed_gates=failed,
            primary_failure="multiplicity",
            repairability="downgrade_only",
            diagnosis="The primary effect does not survive the declared multiplicity correction.",
            evidence=evidence_items,
            allowed_revisions=[
                "Return a fragile disposition.",
                "Treat the finding as exploratory and design a new confirmatory study.",
            ],
            forbidden_revisions=[
                "Do not lower alpha or family size.",
                "Do not switch to a more significant outcome.",
                "Do not report nominal significance as confirmation.",
            ],
            refinement_actions=action_items,
            next_agent_instruction="Return a fragile disposition or propose a new confirmatory design.",
            must_preserve=preserve,
            allowed_contract_changes=["disposition"],
        )

    return ClaimFeedback(
        claim_id=claim_id,
        source_verdict=source_verdict,
        failed_gates=failed,
        primary_failure=primary,
        repairability="contract_repairable" if primary == "execution_error" else "downgrade_only",
        diagnosis="The claim did not reach a confirmed verdict; use the failed gates to triage the next step.",
        evidence=evidence_items,
        allowed_revisions=[
            "Fix schema/execution issues without changing the scientific question.",
            "Return an appropriate abstention disposition if evidence is insufficient.",
        ],
        forbidden_revisions=[
            "Do not weaken gates.",
            "Do not change outcomes or directions after seeing results.",
            "Do not report an abstained claim as confirmed.",
        ],
        refinement_actions=action_items,
        next_agent_instruction="Repair only auditable contract errors, or return an abstention disposition.",
        must_preserve=preserve,
        allowed_contract_changes=["disposition"],
    )


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    text = str(value).strip().lower()
    return text in {"false", "0", "no", "failed"}


def _outcomes(contract: ClaimContract) -> tuple[str, ...]:
    outcome = contract.estimand.outcome
    if isinstance(outcome, list):
        return tuple(str(item) for item in outcome)
    return (str(outcome),)


def _group_dict(contract: ClaimContract) -> dict[str, str] | None:
    return contract.estimand.group.model_dump() if contract.estimand.group is not None else None


def _gate_weakening_violations(original: ClaimContract, revised: ClaimContract) -> list[str]:
    violations: list[str] = []
    if revised.gates.multiplicity.alpha > original.gates.multiplicity.alpha:
        violations.append("Multiplicity alpha was weakened.")
    if revised.gates.multiplicity.family_size < original.gates.multiplicity.family_size:
        violations.append("Multiplicity family_size was reduced.")
    if revised.gates.power.min_power < original.gates.power.min_power:
        violations.append("Power threshold was weakened.")
    if revised.gates.multiverse.min_fraction_consistent < original.gates.multiverse.min_fraction_consistent:
        violations.append("Multiverse threshold was weakened.")
    if revised.gates.replication.alpha > original.gates.replication.alpha:
        violations.append("Replication alpha was weakened.")
    if original.gates.replication.require_same_sign and not revised.gates.replication.require_same_sign:
        violations.append("Replication same-sign requirement was removed.")
    if original.gates.replication.require_ci_overlap and not revised.gates.replication.require_ci_overlap:
        violations.append("Replication CI-overlap requirement was removed.")
    if revised.gates.replication.pattern_corr_min < original.gates.replication.pattern_corr_min:
        violations.append("Replication pattern-correlation threshold was weakened.")
    if revised.gates.replication.region_replication_frac_min < original.gates.replication.region_replication_frac_min:
        violations.append("Replication region-fraction threshold was weakened.")
    if revised.gates.replication.dice_min < original.gates.replication.dice_min:
        violations.append("Replication Dice threshold was weakened.")
    return violations


def _search_provenance_violations(original: ClaimContract, revised: ClaimContract) -> list[str]:
    violations: list[str] = []
    if revised.search_provenance.family_size < original.search_provenance.family_size:
        violations.append("Search-provenance family_size was reduced.")
    if (
        original.search_provenance.selection != "preregistered"
        and revised.search_provenance.selection == "preregistered"
    ):
        violations.append("Discovery-only or unknown search was relabeled as preregistered.")
    if original.search_provenance.declared and not revised.search_provenance.declared:
        violations.append("Declared search provenance was removed.")
    return violations
