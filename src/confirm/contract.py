"""Claim-contract schema and loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ALLOWED_LABELS = {"confirmed", "non_replicated", "under_powered", "fragile"}


class GroupSpec(BaseModel):
    """Case/control definition for a group-difference estimand."""

    model_config = ConfigDict(extra="forbid")

    var: str
    case: str
    control: str


class Estimand(BaseModel):
    """Primary statistical estimand."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["association", "group_diff"]
    outcome: Union[str, list[str]]
    predictor: str
    group: Optional[GroupSpec] = None
    direction: Literal["negative", "positive", "two_sided"]
    unit: Literal["scalar", "brainwide"] = "scalar"
    region_set: Optional[str] = None

    @model_validator(mode="after")
    def validate_estimand(self) -> "Estimand":
        if self.type == "group_diff" and self.group is None:
            raise ValueError("group_diff estimands require a group specification")
        if self.unit == "scalar" and not isinstance(self.outcome, str):
            raise ValueError("scalar estimands require a single string outcome")
        return self


class MultiplicityGate(BaseModel):
    """Multiplicity-control gate."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["fdr_bh"]
    alpha: float = Field(gt=0.0, lt=1.0)
    family_size: int = Field(ge=1)


class SearchProvenance(BaseModel):
    """Lineage for hypotheses searched before the reported claim was selected."""

    model_config = ConfigDict(extra="forbid")

    declared: bool = True
    family_size: int = Field(default=1, ge=1)
    selection: Literal["preregistered", "discovery_only", "full_data", "unknown"] = "preregistered"


class ConfoundGate(BaseModel):
    """Declared confound-control requirements."""

    model_config = ConfigDict(extra="forbid")

    require_covariates: list[str]
    motion_check: bool = False


class PowerGate(BaseModel):
    """Power-gate configuration."""

    model_config = ConfigDict(extra="forbid")

    min_power: float = Field(gt=0.0, le=1.0)
    ref_effect: Optional[float] = None


class MultiverseGate(BaseModel):
    """Bounded multiverse stability gate."""

    model_config = ConfigDict(extra="forbid")

    min_fraction_consistent: float = Field(ge=0.0, le=1.0)


class ReplicationGate(BaseModel):
    """Cross-cohort replication gate."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(gt=0.0, lt=1.0)
    require_same_sign: bool = True
    require_ci_overlap: bool = False
    harmonize: Literal["combat", "none"] = "combat"
    pattern_corr_min: float = Field(default=0.5, ge=-1.0, le=1.0)
    region_replication_frac_min: float = Field(default=0.5, ge=0.0, le=1.0)
    dice_min: float = Field(default=0.0, ge=0.0, le=1.0)


class Gates(BaseModel):
    """All gate settings in a claim contract."""

    model_config = ConfigDict(extra="forbid")

    multiplicity: MultiplicityGate
    confound: ConfoundGate
    power: PowerGate
    multiverse: MultiverseGate
    replication: ReplicationGate


class ClaimContract(BaseModel):
    """Strict pydantic v2 model for CONFIRM claim contracts."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    question: str
    estimand: Estimand
    covariates: list[str]
    inclusion: Optional[str] = None
    discovery_cohort: str
    replication_cohorts: list[str]
    search_provenance: SearchProvenance = Field(default_factory=SearchProvenance)
    gates: Gates
    reporting_language_allowed: list[Literal["confirmed", "non_replicated", "under_powered", "fragile"]]

    @model_validator(mode="after")
    def validate_contract(self) -> "ClaimContract":
        missing = set(self.gates.confound.require_covariates) - set(self.covariates)
        if missing:
            raise ValueError(f"Required confound covariates are absent from covariates: {sorted(missing)}")
        estimand_covariate_collisions = []
        covariates = set(self.covariates)
        if self.estimand.predictor in covariates:
            estimand_covariate_collisions.append(f"predictor {self.estimand.predictor!r}")
        if self.estimand.group is not None and self.estimand.group.var in covariates:
            estimand_covariate_collisions.append(f"group.var {self.estimand.group.var!r}")
        if estimand_covariate_collisions:
            joined = ", ".join(estimand_covariate_collisions)
            raise ValueError(f"Estimand variables cannot also be covariates: {joined}")
        if not set(self.reporting_language_allowed).issubset(ALLOWED_LABELS):
            raise ValueError("Unknown reporting label")
        outcomes = self.estimand.outcome if isinstance(self.estimand.outcome, list) else [self.estimand.outcome]
        if any(outcome in self.covariates for outcome in outcomes):
            raise ValueError("Outcome cannot also be a covariate")
        return self


def load_contract(path: Union[str, Path]) -> ClaimContract:
    """Load and validate a YAML or JSON claim contract."""

    src = Path(path)
    with src.open("r", encoding="utf-8") as handle:
        if src.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)
    return ClaimContract.model_validate(data)
