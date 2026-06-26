"""Serializable result containers shared across CONFIRM gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TRANSPORTABLE_CONFIRMED = "transportable_confirmed"
MAGNITUDE_CONFIRMED = "magnitude_confirmed"
DIRECTION_CONFIRMED = "direction_confirmed"


@dataclass(frozen=True)
class EffectResult:
    """A fitted effect estimate for one contract outcome/predictor."""

    beta: float
    se: float
    ci_low: float
    ci_high: float
    p: float
    n: int
    dof: float
    standardized_effect: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiverseSpecResult:
    """One fitted fork in the bounded multiverse."""

    spec_id: str
    same_sign: bool
    significant: bool
    beta: float
    p: float
    n: int
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiverseResult:
    """Aggregate multiverse stability result."""

    fraction_consistent: float
    passed: bool
    specs: list[MultiverseSpecResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction_consistent": self.fraction_consistent,
            "passed": self.passed,
            "specs": [spec.to_dict() for spec in self.specs],
        }


@dataclass(frozen=True)
class PowerResult:
    """Power and winner's-curse gate result."""

    achieved_power: float
    n_needed_80: float
    shrinkage_factor: float
    shrunken_effect: float
    under_powered: bool
    ref_effect: float | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CohortReplicationResult:
    """Replication result for one independent cohort."""

    cohort: str
    passed: bool
    reason: str
    effect: EffectResult | None = None
    feature_coverage: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["effect"] = self.effect.to_dict() if self.effect else None
        return data


@dataclass(frozen=True)
class CohortEffectSummary:
    """Compact per-cohort scalar effect summary for heterogeneity reporting."""

    cohort: str
    standardized_effect: float
    se_standardized_effect: float
    sign: str
    p: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeterogeneityResult:
    """DerSimonian-Laird random-effects heterogeneity summary."""

    cohort_effects: list[CohortEffectSummary]
    random_effect: float
    random_effect_se: float
    ci_low: float
    ci_high: float
    q: float
    tau2: float
    i2: float
    high_i2: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cohort_effects"] = [effect.to_dict() for effect in self.cohort_effects]
        return data


def _standardized_ci(effect: CohortEffectSummary) -> tuple[float, float]:
    half_width = 1.96 * effect.se_standardized_effect
    return effect.standardized_effect - half_width, effect.standardized_effect + half_width


def _cohort_effect_cis_overlap(heterogeneity: HeterogeneityResult) -> bool:
    intervals = [_standardized_ci(effect) for effect in heterogeneity.cohort_effects]
    if len(intervals) < 2:
        return True
    return max(low for low, _ in intervals) <= min(high for _, high in intervals)


def scalar_confirmation_subtype(heterogeneity: HeterogeneityResult | None) -> str | None:
    """Classify a passed scalar replication by transportability strength."""

    if heterogeneity is None:
        return None
    if heterogeneity.i2 >= 75.0:
        return DIRECTION_CONFIRMED
    if heterogeneity.i2 < 40.0 and _cohort_effect_cis_overlap(heterogeneity):
        return TRANSPORTABLE_CONFIRMED
    return MAGNITUDE_CONFIRMED


@dataclass(frozen=True)
class ReplicationResult:
    """Cross-cohort replication gate result."""

    passed: bool
    reason: str
    cohort_results: list[CohortReplicationResult]
    harmonized: bool
    heterogeneity: HeterogeneityResult | None = None
    replicated_but_heterogeneous: bool = False
    confirmation_subtype: str | None = None
    confirmation_i2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "cohort_results": [result.to_dict() for result in self.cohort_results],
            "harmonized": self.harmonized,
            "heterogeneity": self.heterogeneity.to_dict() if self.heterogeneity else None,
            "replicated_but_heterogeneous": self.replicated_but_heterogeneous,
            "confirmation_subtype": self.confirmation_subtype,
            "confirmation_i2": self.confirmation_i2,
        }


@dataclass(frozen=True)
class RegionEffectResult:
    """A fitted effect and FDR decision for one brain region."""

    region: str
    effect: EffectResult
    q: float
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "effect": self.effect.to_dict(),
            "q": self.q,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class RegionTable:
    """Brain-wide region-wise fitted effects."""

    regions: list[RegionEffectResult]
    alpha: float
    method: str = "fdr_bh"
    region_set: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": [region.to_dict() for region in self.regions],
            "alpha": self.alpha,
            "method": self.method,
            "region_set": self.region_set,
        }


@dataclass(frozen=True)
class CohortBrainwideReplicationResult:
    """Brain-wide replication metrics for one cohort."""

    cohort: str
    passed: bool
    reason: str
    pattern_corr: float
    dice: float
    region_replication_fraction: float
    region_table: RegionTable | None = None
    feature_coverage: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["region_table"] = self.region_table.to_dict() if self.region_table else None
        return data


@dataclass(frozen=True)
class BrainwideReplicationResult:
    """Cross-cohort brain-wide pattern replication gate result."""

    passed: bool
    reason: str
    cohort_results: list[CohortBrainwideReplicationResult]
    harmonized: bool
    pattern_corr: float
    dice: float
    region_replication_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "cohort_results": [result.to_dict() for result in self.cohort_results],
            "harmonized": self.harmonized,
            "pattern_corr": self.pattern_corr,
            "dice": self.dice,
            "region_replication_fraction": self.region_replication_fraction,
        }
