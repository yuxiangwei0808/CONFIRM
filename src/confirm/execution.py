"""Public execution boundary for frozen CONFIRM claim contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from confirm.analysis import audit_confound_completeness, run_primary
from confirm.brainwide import run_brainwide
from confirm.contract import ClaimContract
from confirm.derived_columns import add_virtual_columns
from confirm.multiverse import run_brainwide_multiverse, run_multiverse
from confirm.power import power_check
from confirm.replication import replicate, replicate_brainwide
from confirm.results import EffectResult, RegionTable
from confirm.schema import validate_canonical
from confirm.verdict import Verdict, decide, decide_brainwide


def cohort_path(data_dir: Path, cohort: str) -> Path:
    """Resolve an exact canonical cohort parquet."""

    path = data_dir / f"{cohort}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Canonical cohort parquet not found: {path}")
    return path


def load_canonical(path: Path) -> pd.DataFrame:
    """Load canonical data and attach deterministic virtual columns."""

    df = validate_canonical(pd.read_parquet(path), drop_invalid_demographics=True)
    return add_virtual_columns(df, path.stem)


def jsonable(value: Any) -> Any:
    """Convert CONFIRM result models and tables to JSON-compatible values."""

    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def run_scalar_contract(
    contract: ClaimContract,
    discovery_df: pd.DataFrame,
    replication_dfs: list[pd.DataFrame],
    ref_effect: float | None,
) -> tuple[Verdict, dict[str, Any]]:
    """Execute unchanged scalar CONFIRM gates."""

    confound_audit = audit_confound_completeness(discovery_df, contract)
    primary = run_primary(discovery_df, contract)
    multiverse = run_multiverse(discovery_df, contract, forks=None)
    power = power_check(primary, contract, ref_effect=ref_effect)
    replication = replicate(primary, discovery_df, replication_dfs, contract)
    verdict = decide(
        primary,
        multiverse,
        power,
        replication,
        contract,
        confound_audit=confound_audit,
    )
    return verdict, {
        "primary": primary,
        "confound_completeness": confound_audit,
        "multiverse": multiverse,
        "power": power,
        "replication": replication,
        "verdict": verdict,
    }


def _best_region_effect(regions: RegionTable) -> EffectResult:
    ordered = sorted(regions.regions, key=lambda region: (not region.significant, region.effect.p))
    return ordered[0].effect


def run_brainwide_contract(
    contract: ClaimContract,
    discovery_df: pd.DataFrame,
    replication_dfs: list[pd.DataFrame],
) -> tuple[Verdict, dict[str, Any]]:
    """Execute unchanged brainwide CONFIRM gates."""

    confound_audit = audit_confound_completeness(discovery_df, contract)
    regions = run_brainwide(discovery_df, contract)
    multiverse = run_brainwide_multiverse(discovery_df, regions, contract)
    power = power_check(
        _best_region_effect(regions),
        contract,
        ref_effect=contract.gates.power.ref_effect,
    )
    replication = replicate_brainwide(regions, discovery_df, replication_dfs, contract)
    verdict = decide_brainwide(
        regions,
        multiverse,
        power,
        replication,
        contract,
        confound_audit=confound_audit,
    )
    return verdict, {
        "regions": regions,
        "confound_completeness": confound_audit,
        "multiverse": multiverse,
        "power": power,
        "replication": replication,
        "verdict": verdict,
    }


def evaluate_contract(
    contract: ClaimContract,
    data_root: Path,
    ref_effect: float | None = None,
) -> tuple[Verdict, dict[str, Any], list[Path]]:
    """Evaluate one frozen contract against exact discovery/replication cohorts."""

    discovery_path = cohort_path(data_root, contract.discovery_cohort)
    replication_paths = [
        cohort_path(data_root, cohort) for cohort in contract.replication_cohorts
    ]
    discovery_df = load_canonical(discovery_path)
    replication_dfs = [load_canonical(path) for path in replication_paths]
    if contract.estimand.unit == "brainwide":
        verdict, results = run_brainwide_contract(
            contract,
            discovery_df,
            replication_dfs,
        )
    else:
        verdict, results = run_scalar_contract(
            contract,
            discovery_df,
            replication_dfs,
            ref_effect,
        )
    return (
        verdict,
        {"contract": contract.model_dump(), **results},
        [discovery_path, *replication_paths],
    )
