"""Shared execution and mapping helpers for excluded-evidence analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

from confirm.contract import ClaimContract
from confirm.execution import (
    jsonable,
    load_canonical,
    run_brainwide_contract,
    run_scalar_contract,
)
from confirm.evidence_partitions import (
    EvidencePartitionManifest,
    EvidencePartitionRecord,
    canonical_base_cohort,
    infer_target_family,
)

EvidenceScope = Literal["current", "holdout", "external", "external_robustness"]


class ExcludedEvidenceUnavailableError(FileNotFoundError):
    """Excluded evidence could not be mapped before any gate was evaluated."""

    code = "excluded_evidence_unavailable"


def cohort_aliases(cohort: str) -> list[str]:
    """Return permitted aliases for current-data cohort resolution."""

    aliases = [cohort]
    suffixes = (
        "_DISC_SITES",
        "_REP_SITES",
        "_HOLDOUT_SITES",
        "_HOLDOUT_DISC",
        "_HOLDOUT_REP",
        "_EXTERNAL_DISC",
        "_EXTERNAL_REP",
        "_HOLDOUT",
        "_DISC",
        "_REP",
        "_CN",
    )
    for suffix in suffixes:
        if cohort.endswith(suffix):
            aliases.append(cohort[: -len(suffix)])
    for split in ("_DISC_s", "_REP_s", "_HOLDOUT_s"):
        if split in cohort:
            aliases.append(cohort.split(split, 1)[0])
    return list(dict.fromkeys(item for item in aliases if item))


def cohort_path(
    data_roots: Iterable[str | Path],
    cohort: str,
    *,
    allow_aliases: bool = True,
) -> Path:
    """Resolve a cohort parquet, optionally permitting current-data aliases."""

    roots = [Path(root) for root in data_roots]
    candidates = cohort_aliases(cohort) if allow_aliases else [cohort]
    for candidate in candidates:
        for root in roots:
            path = root / f"{candidate}.parquet"
            if path.exists():
                return path
    rendered_roots = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"Cohort {cohort!r} was not found in data roots: {rendered_roots}")


def exact_contract_paths(contract: ClaimContract, data_roots: Iterable[str | Path]) -> tuple[Path, list[Path]]:
    """Resolve all contract paths without falling back to base-dataset aliases."""

    discovery = cohort_path(data_roots, contract.discovery_cohort, allow_aliases=False)
    replications = [
        cohort_path(data_roots, cohort, allow_aliases=False)
        for cohort in contract.replication_cohorts
    ]
    return discovery, replications


def execute_contract(
    contract: ClaimContract,
    data_roots: Iterable[str | Path],
    *,
    evidence_scope: EvidenceScope = "current",
    target_family: str | None = None,
    source_contract: ClaimContract | None = None,
    evidence_set_id: str | None = None,
) -> dict[str, Any]:
    """Run unchanged CONFIRM gates for one frozen contract."""

    roots = [Path(root) for root in data_roots]
    allow_aliases = evidence_scope == "current"
    discovery_path = cohort_path(roots, contract.discovery_cohort, allow_aliases=allow_aliases)
    replication_paths = [
        cohort_path(roots, cohort, allow_aliases=allow_aliases)
        for cohort in contract.replication_cohorts
    ]
    discovery_df = load_canonical(discovery_path)
    replication_dfs = [load_canonical(path) for path in replication_paths]
    if contract.estimand.unit == "brainwide":
        verdict, results = run_brainwide_contract(contract, discovery_df, replication_dfs)
    else:
        verdict, results = run_scalar_contract(
            contract,
            discovery_df,
            replication_dfs,
            ref_effect=contract.gates.power.ref_effect,
        )
    return {
        "final_label": verdict.label,
        "gate_results": jsonable(
            {
                "contract": contract.model_dump(mode="json"),
                "data_paths": {
                    "discovery": str(discovery_path),
                    "replication": [str(path) for path in replication_paths],
                },
                "evidence_scope": {
                    "scope": evidence_scope,
                    "target_family": target_family or infer_target_family(source_contract or contract),
                    "evidence_set_id": evidence_set_id,
                    "source_contract": (
                        source_contract.model_dump(mode="json")
                        if source_contract is not None
                        else None
                    ),
                },
                **results,
            }
        ),
    }


def mapped_contract_for_evidence(
    contract: ClaimContract,
    manifest: EvidencePartitionManifest,
    evidence_kind: Literal["holdout", "external"],
    *,
    evidence_set_id: str | None = None,
) -> tuple[ClaimContract, EvidencePartitionRecord, list[EvidencePartitionRecord], str | None]:
    """Map a frozen source contract to one predeclared excluded-evidence pair."""

    payload = contract.model_dump(mode="json")
    target_family = infer_target_family(contract)
    if evidence_kind == "holdout":
        pair = manifest.holdout_evaluation_pair_for_contract(contract)
        if pair is None:
            bases = [
                canonical_base_cohort(contract.discovery_cohort),
                *[canonical_base_cohort(cohort) for cohort in contract.replication_cohorts],
            ]
            raise ExcludedEvidenceUnavailableError(
                f"No holdout evaluation pair for bases={bases!r} target_family={target_family!r}"
            )
        discovery, replications = pair
        resolved_set_id = None
    else:
        pair = manifest.external_pair_for_contract(contract, evidence_set_id=evidence_set_id)
        if pair is None:
            detail = f" evidence_set_id={evidence_set_id!r}" if evidence_set_id else ""
            raise ExcludedEvidenceUnavailableError(
                f"No contract-compatible external pair for target_family={target_family!r}{detail}"
            )
        discovery, replications, evidence_set = pair
        resolved_set_id = evidence_set.evidence_set_id

    payload["discovery_cohort"] = discovery.partition_id
    payload["replication_cohorts"] = [record.partition_id for record in replications]
    return ClaimContract.model_validate(payload), discovery, replications, resolved_set_id


def external_evidence_set_ids(
    contract: ClaimContract,
    manifest: EvidencePartitionManifest,
) -> list[str]:
    """Return every schema-compatible external evidence set in frozen priority order."""

    return [item.evidence_set_id for item in manifest.external_sets_for_contract(contract)]
