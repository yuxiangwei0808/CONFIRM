"""Failure diagnosis and typed new-claim proposals for CONFIRM."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confirm.contract import ClaimContract
from confirm.feedback import DispositionLabel, feedback_from_verdict

FailureKind = Literal[
    "contract_error",
    "evidence_failure",
    "design_limitation",
    "search_lineage_failure",
    "confound_failure",
    "none",
]
ProposalType = Literal[
    "corrected_contract",
    "downgraded_claim",
    "exploratory_followup_claim",
    "independent_replication_claim",
    "abandon_claim",
]
ProposalProvenance = Literal[
    "contract_correction",
    "post_hoc",
    "post_hoc_followup",
    "future_design",
    "independent_replication",
    "abstention",
    "none",
]


class FailureLocalization(BaseModel):
    """Deterministic localization of why a claim did not confirm."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    source_verdict: str
    failed_gates: list[str] = Field(default_factory=list)
    primary_failure: str
    failure_kind: FailureKind
    diagnosis: str
    evidence: list[str] = Field(default_factory=list)
    allowed_proposal_types: list[ProposalType] = Field(default_factory=list)
    current_data_repair_allowed: bool = False
    requires_new_evidence_for_confirmation: bool = True
    must_preserve: list[str] = Field(default_factory=list)


class NewClaimProposal(BaseModel):
    """A typed next claim/follow-up proposal after a failed CONFIRM verdict."""

    model_config = ConfigDict(extra="forbid")

    source_claim_id: str
    proposal_type: ProposalType
    rationale: str
    proposed_question: Optional[str] = None
    proposed_contract: Optional[ClaimContract] = None
    disposition_label: Optional[DispositionLabel] = None
    provenance: ProposalProvenance
    requires_new_evidence: bool
    can_confirm_on_current_data: bool
    supported_by_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_shape(self) -> "NewClaimProposal":
        # Shape policy is enforced by ProposalValidation so malformed LLM
        # proposals can be serialized and counted instead of aborting a run.
        return self


class ProposalValidation(BaseModel):
    """Deterministic validation of a new-claim proposal."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance_compliant: bool = False
    current_data_confirmability_ok: bool = False
    checked_contract: bool = False
    useful: bool = False
    accepted_proposal_type: bool = False
    design_diagnostics: dict[str, Any] = Field(default_factory=dict)


def localize_failure(
    contract: ClaimContract,
    verdict: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
) -> FailureLocalization:
    """Localize the failure mode behind a CONFIRM verdict."""

    feedback = feedback_from_verdict(contract.claim_id, verdict, results)
    failure_kind = _failure_kind(contract, feedback.primary_failure, feedback.failed_gates, feedback.evidence)
    structural_confound = _is_structural_confound(feedback)
    allowed = _allowed_proposal_types(failure_kind, feedback.primary_failure, structural_confound)
    current_repair = _current_data_repair_allowed(failure_kind, feedback.primary_failure, structural_confound)
    diagnosis = _diagnosis_for_kind(failure_kind, feedback.primary_failure, feedback.diagnosis)
    return FailureLocalization(
        claim_id=contract.claim_id,
        source_verdict=feedback.source_verdict,
        failed_gates=feedback.failed_gates,
        primary_failure=feedback.primary_failure,
        failure_kind=failure_kind,
        diagnosis=diagnosis,
        evidence=feedback.evidence,
        allowed_proposal_types=allowed,
        current_data_repair_allowed=current_repair,
        requires_new_evidence_for_confirmation=not current_repair,
        must_preserve=_must_preserve(failure_kind, feedback.primary_failure),
    )


def localization_for_estimand_mismatch(
    claim_id: str,
    *,
    mismatches: Mapping[str, Any] | None = None,
) -> FailureLocalization:
    """Localize an agent draft mismatch before gate execution."""

    mismatch_names = sorted(str(key) for key in (mismatches or {}).keys())
    detail = ", ".join(mismatch_names) if mismatch_names else "estimand fields"
    return FailureLocalization(
        claim_id=claim_id,
        source_verdict="draft_mismatch",
        failed_gates=["estimand_match"],
        primary_failure="estimand_mismatch",
        failure_kind="contract_error",
        diagnosis=f"The draft does not encode the intended scientific estimand: {detail}.",
        evidence=[f"Mismatched contract fields: {detail}."],
        allowed_proposal_types=["corrected_contract", "downgraded_claim", "abandon_claim"],
        current_data_repair_allowed=True,
        requires_new_evidence_for_confirmation=False,
        must_preserve=["scientific_question", "gate_thresholds"],
    )


def default_new_claim_proposal(
    localization: FailureLocalization,
    contract: ClaimContract | None = None,
) -> NewClaimProposal:
    """Emit a deterministic safe proposal for replay and pipeline artifacts."""

    if localization.primary_failure == "replication":
        disposition: DispositionLabel = "non_replicated"
    elif localization.primary_failure == "power":
        disposition = "under_powered"
    elif localization.failure_kind == "design_limitation":
        disposition = "needs_more_data"
    elif localization.failure_kind == "none":
        return NewClaimProposal(
            source_claim_id=localization.claim_id,
            proposal_type="abandon_claim",
            rationale="No new claim is needed because the source claim is already confirmed.",
            proposed_question=None,
            proposed_contract=None,
            disposition_label=None,
            provenance="none",
            requires_new_evidence=False,
            can_confirm_on_current_data=False,
            supported_by_evidence=localization.evidence,
        )
    else:
        disposition = "fragile"
    return NewClaimProposal(
        source_claim_id=localization.claim_id,
        proposal_type="downgraded_claim",
        rationale=f"Current evidence supports a {disposition} disposition rather than a new same-data confirmation.",
        proposed_question=contract.question if contract is not None else None,
        proposed_contract=None,
        disposition_label=disposition,
        provenance="abstention",
        requires_new_evidence=False,
        can_confirm_on_current_data=False,
        supported_by_evidence=localization.evidence,
    )


def validate_new_claim_proposal(
    original: ClaimContract | None,
    proposal: NewClaimProposal,
    localization: FailureLocalization,
) -> ProposalValidation:
    """Validate a typed proposal against provenance-gated scientific policy."""

    violations: list[str] = []
    warnings: list[str] = []
    checked_contract = False

    if proposal.source_claim_id != localization.claim_id:
        violations.append("Proposal source_claim_id does not match the localized claim.")
    if proposal.proposal_type not in localization.allowed_proposal_types:
        violations.append("Proposal type is not allowed for this failure localization.")

    if proposal.can_confirm_on_current_data:
        if proposal.proposal_type == "corrected_contract" and not localization.current_data_repair_allowed:
            violations.append("Only validator-approved contract corrections may be confirmable on current data.")
        elif proposal.proposal_type not in {"corrected_contract", "exploratory_followup_claim", "independent_replication_claim"}:
            violations.append("Only corrected contracts or connected follow-up proposals may be evaluated on current data.")
    if proposal.proposal_type == "downgraded_claim" and proposal.can_confirm_on_current_data:
        violations.append("Downgraded claims cannot be marked confirmable on current data.")

    unsupported = _unsupported_numbers(proposal, localization)
    if unsupported:
        violations.append(f"Proposal rationale contains unsupported numeric values: {unsupported}.")

    if proposal.proposed_contract is not None:
        checked_contract = True
        if original is None:
            violations.append("Cannot validate a proposed contract without the original contract.")
        else:
            violations.extend(_contract_policy_violations(original, proposal.proposed_contract, proposal, localization))
    elif proposal.proposal_type == "corrected_contract":
        violations.append("corrected_contract proposal is missing proposed_contract.")

    if proposal.proposal_type == "downgraded_claim" and proposal.proposed_contract is not None:
        warnings.append("Downgraded claims do not need a proposed contract.")

    provenance_compliant = not any(
        "Post-hoc" in item
        or "confirmable on current data" in item
        or "contract corrections" in item
        or "provenance" in item
        for item in violations
    )
    current_data_ok = not any("current data" in item for item in violations)
    accepted_type = proposal.proposal_type in localization.allowed_proposal_types
    ok = not violations
    return ProposalValidation(
        ok=ok,
        violations=violations,
        warnings=warnings,
        provenance_compliant=ok and provenance_compliant,
        current_data_confirmability_ok=ok and current_data_ok,
        checked_contract=checked_contract,
        useful=ok and accepted_type,
        accepted_proposal_type=accepted_type,
    )


def summarize_proposals(
    localizations: list[FailureLocalization],
    validations: list[ProposalValidation],
) -> dict[str, Any]:
    """Summarize localization/proposal validation outputs."""

    failed = [item for item in localizations if item.source_verdict != "confirmed" or item.failure_kind != "none"]
    covered = [item for item in failed if item.failure_kind != "none"]
    current_data = [item for item in localizations if item.current_data_repair_allowed]
    blocked = [item for item in validations if not item.ok]
    allowed_type_counts: Counter[str] = Counter()
    for item in localizations:
        allowed_type_counts.update(item.allowed_proposal_types)
    return {
        "n_localizations": len(localizations),
        "n_failed_claims": len(failed),
        "localized_failed_claim_count": len(covered),
        "localization_coverage": len(covered) / len(failed) if failed else 1.0,
        "failure_kind_counts": dict(Counter(item.failure_kind for item in localizations)),
        "primary_failure_counts": dict(Counter(item.primary_failure for item in localizations)),
        "allowed_proposal_type_counts": dict(allowed_type_counts),
        "current_data_repairable_count": len(current_data),
        "current_data_repairable_rate": len(current_data) / len(localizations) if localizations else 0.0,
        "unsafe_proposal_block_count": len(blocked),
        "unsafe_proposal_block_rate": len(blocked) / len(validations) if validations else 0.0,
    }


def _failure_kind(
    contract: ClaimContract,
    primary_failure: str,
    failed_gates: list[str],
    evidence: list[str],
) -> FailureKind:
    if primary_failure == "none":
        return "none"
    if primary_failure in {"execution_error", "estimand_mismatch"}:
        return "contract_error"
    if primary_failure == "search_provenance":
        return "search_lineage_failure"
    if primary_failure == "confound":
        return "confound_failure"
    if primary_failure == "power":
        return "design_limitation"
    if primary_failure == "replication" and contract.discovery_cohort in set(contract.replication_cohorts):
        return "design_limitation"
    if primary_failure in {"multiplicity", "multiverse", "replication"}:
        return "evidence_failure"
    if "replication" in failed_gates or any("Replication gate failed" in item for item in evidence):
        return "evidence_failure"
    return "evidence_failure"


def _allowed_proposal_types(
    failure_kind: FailureKind,
    primary_failure: str,
    structural_confound: bool,
) -> list[ProposalType]:
    if failure_kind == "none":
        return ["abandon_claim"]
    if failure_kind == "contract_error":
        return ["corrected_contract", "downgraded_claim", "abandon_claim"]
    if failure_kind == "confound_failure":
        if structural_confound:
            return ["downgraded_claim", "independent_replication_claim", "abandon_claim"]
        return ["corrected_contract", "downgraded_claim", "independent_replication_claim", "abandon_claim"]
    if failure_kind == "search_lineage_failure":
        return ["corrected_contract", "downgraded_claim", "exploratory_followup_claim", "abandon_claim"]
    if failure_kind == "design_limitation":
        return ["downgraded_claim", "independent_replication_claim", "abandon_claim"]
    if primary_failure == "replication":
        return ["downgraded_claim", "independent_replication_claim", "abandon_claim"]
    return ["downgraded_claim", "exploratory_followup_claim", "independent_replication_claim", "abandon_claim"]


def _current_data_repair_allowed(
    failure_kind: FailureKind,
    primary_failure: str,
    structural_confound: bool,
) -> bool:
    if failure_kind == "contract_error":
        return True
    if failure_kind == "confound_failure" and not structural_confound:
        return True
    if failure_kind == "search_lineage_failure":
        return True
    return False


def _diagnosis_for_kind(failure_kind: FailureKind, primary_failure: str, base: str) -> str:
    if failure_kind == "contract_error":
        return "The claim failed because the contract does not faithfully encode an executable scientific question."
    if failure_kind == "search_lineage_failure":
        return "The claim failed because the hypothesis lineage/search family is not valid for confirmatory reporting."
    if failure_kind == "confound_failure":
        return "The claim failed because the current design does not sufficiently address confounding."
    if failure_kind == "design_limitation":
        return "The claim failed because the study design lacks the power or independent evidence needed for confirmation."
    if failure_kind == "evidence_failure":
        return f"The claim failed because the observed evidence does not support confirmation: {base}"
    if failure_kind == "none":
        return "The claim is already confirmed; no new proposal is needed."
    return base


def _must_preserve(failure_kind: FailureKind, primary_failure: str) -> list[str]:
    preserve = ["scientific_question", "gate_thresholds"]
    if failure_kind != "contract_error":
        preserve.extend(["outcome_family", "direction", "discovery_cohort"])
    if primary_failure != "replication":
        preserve.append("replication_cohorts")
    return preserve


def _is_structural_confound(feedback: Any) -> bool:
    text = " ".join(
        [
            str(feedback.diagnosis),
            str(feedback.next_agent_instruction),
            " ".join(str(item) for item in feedback.evidence),
        ]
    ).lower()
    return "nested" in text or "structurally confounded" in text


def _contract_policy_violations(
    original: ClaimContract,
    revised: ClaimContract,
    proposal: NewClaimProposal,
    localization: FailureLocalization,
) -> list[str]:
    violations: list[str] = []
    estimand_repair = localization.primary_failure == "estimand_mismatch"
    covariate_repair = localization.failure_kind == "confound_failure" and localization.current_data_repair_allowed
    if proposal.proposal_type == "corrected_contract":
        if not proposal.can_confirm_on_current_data:
            violations.append("corrected_contract proposals intended for rerun must set can_confirm_on_current_data=true.")
        if not localization.current_data_repair_allowed:
            violations.append("This failure localization is not current-data repairable.")
        if _material_contract_payload(original) == _material_contract_payload(revised):
            violations.append("Corrected contract does not change executable or governance contract fields.")

    if not estimand_repair:
        if _outcomes(original) != _outcomes(revised):
            violations.append("Outcome changed outside an estimand-repair allowance.")
        if original.estimand.direction != revised.estimand.direction:
            violations.append("Direction changed outside an estimand-repair allowance.")
        if original.estimand.type != revised.estimand.type:
            violations.append("Estimand type changed outside an estimand-repair allowance.")
        if _group_dict(original) != _group_dict(revised):
            violations.append("Group contrast changed outside an estimand-repair allowance.")
        if original.estimand.predictor != revised.estimand.predictor:
            violations.append("Predictor changed outside an estimand-repair allowance.")

    if proposal.proposal_type in {"corrected_contract", "independent_replication_claim"}:
        if revised.discovery_cohort in set(revised.replication_cohorts):
            violations.append("Replication cohorts must be independent from the discovery cohort.")

    if proposal.proposal_type == "corrected_contract" and original.discovery_cohort != revised.discovery_cohort and not estimand_repair:
        violations.append("Corrected contracts cannot change the tested discovery cohort.")
    if proposal.proposal_type == "corrected_contract" and sorted(original.replication_cohorts) != sorted(revised.replication_cohorts):
        if not estimand_repair and localization.primary_failure != "replication":
            violations.append("Corrected contracts cannot swap replication cohorts for this failure mode.")

    original_required = set(original.gates.confound.require_covariates)
    revised_required = set(revised.gates.confound.require_covariates)
    if not original_required.issubset(revised_required) and not estimand_repair:
        violations.append("Required confound covariates were removed.")
    if not set(original.covariates).issubset(set(revised.covariates)) and not (estimand_repair or covariate_repair):
        violations.append("Original covariates were removed without a covariate-repair allowance.")

    violations.extend(_gate_weakening_violations(original, revised))
    violations.extend(_search_provenance_violations(original, revised))
    return violations


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


def _search_provenance_violations(
    original: ClaimContract,
    revised: ClaimContract,
) -> list[str]:
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


def _outcomes(contract: ClaimContract) -> tuple[str, ...]:
    outcome = contract.estimand.outcome
    if isinstance(outcome, list):
        return tuple(str(item) for item in outcome)
    return (str(outcome),)


def _material_contract_payload(contract: ClaimContract) -> dict[str, Any]:
    data = contract.model_dump(mode="json")
    return {
        key: data[key]
        for key in [
            "estimand",
            "covariates",
            "inclusion",
            "discovery_cohort",
            "replication_cohorts",
            "search_provenance",
            "gates",
            "reporting_language_allowed",
        ]
    }


def _group_dict(contract: ClaimContract) -> dict[str, str] | None:
    return contract.estimand.group.model_dump() if contract.estimand.group is not None else None


_NUMBER_RE = re.compile(r"(?<![\w-])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w-])")


def _unsupported_numbers(proposal: NewClaimProposal, localization: FailureLocalization) -> list[str]:
    text = proposal.rationale or ""
    numbers = _NUMBER_RE.findall(text)
    if not numbers or not localization.evidence:
        return []
    allowed = [float(item) for item in _NUMBER_RE.findall(" ".join(localization.evidence))]
    if not allowed:
        return []
    unsupported: list[str] = []
    for item in numbers:
        value = float(item)
        if value in {0.0, 1.0}:
            continue
        if not any(abs(value - allowed_value) <= max(1e-6, abs(allowed_value) * 1e-3) for allowed_value in allowed):
            unsupported.append(item)
    return unsupported
