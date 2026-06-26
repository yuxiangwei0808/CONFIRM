"""Verdict gate ordering and final label selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from confirm.analysis import directionally_consistent, effective_multiplicity_family_size, fdr_bh_q_values, multiplicity_threshold
from confirm.contract import ClaimContract
from confirm.results import BrainwideReplicationResult, EffectResult, MultiverseResult, PowerResult, RegionTable, ReplicationResult

UNVERIFIABLE_SEARCH_PROVENANCE = "unverifiable_search_provenance"
CONFOUND_INCOMPLETE = "confound_incomplete"


@dataclass(frozen=True)
class Verdict:
    """Final CONFIRM verdict."""

    label: str
    abstained: bool
    rationale: str
    gates: dict[str, Any]
    confirmation_subtype: str | None = None
    heterogeneity_i2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _multiplicity_pass(effect: EffectResult, contract: ClaimContract) -> bool:
    threshold = multiplicity_threshold(contract)
    return effect.p <= threshold and directionally_consistent(effect.beta, contract)


def _confound_pass(contract: ClaimContract) -> bool:
    return set(contract.gates.confound.require_covariates).issubset(set(contract.covariates))


def _search_provenance_pass(contract: ClaimContract) -> bool:
    provenance = contract.search_provenance
    return bool(provenance.declared and provenance.selection not in {"unknown", "full_data"})


def _confound_completeness_pass(confound_audit: dict[str, Any] | None) -> bool:
    return bool(confound_audit is None or confound_audit.get("passed", True))


def _brainwide_multiplicity_pass(regions: RegionTable, contract: ClaimContract) -> bool:
    family_size = effective_multiplicity_family_size(contract, observed_family_size=len(regions.regions))
    q_values = fdr_bh_q_values((region.effect.p for region in regions.regions), family_size=family_size)
    return any(
        q <= contract.gates.multiplicity.alpha and directionally_consistent(region.effect.beta, contract)
        for region, q in zip(regions.regions, q_values)
    )


def decide(
    primary: EffectResult,
    multiverse: MultiverseResult,
    power: PowerResult,
    replication: ReplicationResult,
    contract: ClaimContract,
    *,
    confound_audit: dict[str, Any] | None = None,
) -> Verdict:
    """Apply CONFIRM gates in the order required by the architecture."""

    gate_state: dict[str, Any] = {
        "search_provenance": _search_provenance_pass(contract),
        "confound": _confound_pass(contract),
        "confound_completeness": _confound_completeness_pass(confound_audit),
        "multiplicity": _multiplicity_pass(primary, contract),
        "power": not power.under_powered,
        "multiverse": multiverse.passed,
        "replication": replication.passed,
    }
    if confound_audit is not None:
        gate_state["confound_completeness_audit"] = confound_audit
        if not gate_state["confound_completeness"]:
            gate_state["reason"] = CONFOUND_INCOMPLETE
    failures = [name for name, passed in gate_state.items() if isinstance(passed, bool) and not passed]
    gate_state["multiplicity_effective_family_size"] = effective_multiplicity_family_size(contract)
    if (
        not gate_state["search_provenance"]
        or not gate_state["confound"]
        or not gate_state["confound_completeness"]
        or not gate_state["multiplicity"]
    ):
        label = "fragile"
    elif not gate_state["power"]:
        label = "under_powered"
    elif not gate_state["multiverse"]:
        label = "fragile"
    elif not gate_state["replication"]:
        label = "non_replicated"
    else:
        label = "confirmed"

    abstained = label != "confirmed"
    rationale = "All gates passed." if not failures else "Failed gates: " + ", ".join(failures)
    if not gate_state["search_provenance"]:
        rationale += f"; {UNVERIFIABLE_SEARCH_PROVENANCE}"
    if not gate_state["confound_completeness"]:
        rationale += f"; {CONFOUND_INCOMPLETE}"
    if replication.reason != "passed":
        rationale += f"; replication={replication.reason}"
    if power.under_powered:
        rationale += f"; {power.rationale}"
    confirmation_subtype = replication.confirmation_subtype if label == "confirmed" else None
    heterogeneity_i2 = replication.confirmation_i2 if label == "confirmed" else None
    if confirmation_subtype:
        gate_state["confirmation_subtype"] = confirmation_subtype
        gate_state["heterogeneity_i2"] = heterogeneity_i2
        rationale += f"; confirmation_subtype={confirmation_subtype}"
        if heterogeneity_i2 is not None:
            rationale += f"; heterogeneity_i2={heterogeneity_i2:.1f}"
    return Verdict(
        label=label,
        abstained=abstained,
        rationale=rationale,
        gates=gate_state,
        confirmation_subtype=confirmation_subtype,
        heterogeneity_i2=heterogeneity_i2,
    )


def decide_brainwide(
    regions: RegionTable,
    multiverse: MultiverseResult,
    power: PowerResult,
    replication: BrainwideReplicationResult,
    contract: ClaimContract,
    *,
    confound_audit: dict[str, Any] | None = None,
) -> Verdict:
    """Apply CONFIRM gates for a brain-wide regional claim."""

    gate = contract.gates.replication
    gate_state: dict[str, Any] = {
        "search_provenance": _search_provenance_pass(contract),
        "confound": _confound_pass(contract),
        "confound_completeness": _confound_completeness_pass(confound_audit),
        "multiplicity": _brainwide_multiplicity_pass(regions, contract),
        "power": not power.under_powered,
        "multiverse": multiverse.passed,
        "replication": replication.passed,
        "pattern_corr": replication.pattern_corr >= gate.pattern_corr_min,
        "region_replication_fraction": replication.region_replication_fraction >= gate.region_replication_frac_min,
        "dice": replication.dice >= gate.dice_min,
    }
    if confound_audit is not None:
        gate_state["confound_completeness_audit"] = confound_audit
        if not gate_state["confound_completeness"]:
            gate_state["reason"] = CONFOUND_INCOMPLETE
    failures = [name for name, passed in gate_state.items() if isinstance(passed, bool) and not passed]
    gate_state["multiplicity_effective_family_size"] = effective_multiplicity_family_size(
        contract, observed_family_size=len(regions.regions)
    )
    if (
        not gate_state["search_provenance"]
        or not gate_state["confound"]
        or not gate_state["confound_completeness"]
        or not gate_state["multiplicity"]
    ):
        label = "fragile"
    elif not gate_state["power"]:
        label = "under_powered"
    elif not gate_state["multiverse"]:
        label = "fragile"
    elif not gate_state["replication"]:
        label = "non_replicated"
    else:
        label = "confirmed"

    abstained = label != "confirmed"
    rationale = "All brain-wide gates passed." if not failures else "Failed gates: " + ", ".join(failures)
    if not gate_state["search_provenance"]:
        rationale += f"; {UNVERIFIABLE_SEARCH_PROVENANCE}"
    if not gate_state["confound_completeness"]:
        rationale += f"; {CONFOUND_INCOMPLETE}"
    if replication.reason != "passed":
        rationale += f"; replication={replication.reason}"
    if power.under_powered:
        rationale += f"; {power.rationale}"
    rationale += (
        f"; pattern_corr={replication.pattern_corr:.3f}; "
        f"dice={replication.dice:.3f}; "
        f"region_replication_fraction={replication.region_replication_fraction:.3f}"
    )
    return Verdict(label=label, abstained=abstained, rationale=rationale, gates=gate_state)
