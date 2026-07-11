"""Evidence-partition manifests for excluded claim-search evaluation."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from confirm.contract import ClaimContract
from confirm.derived_columns import add_virtual_columns, columns_with_virtuals
from confirm.schema import columns_with_canonical_aliases, normalize_sex

PartitionRole = Literal["discovery", "replication", "holdout", "external_eval"]
EvaluationRole = Literal["discovery", "replication", "holdout", "external"]
EvidenceKind = Literal["holdout", "external"]

ROLE_SUFFIX = {
    "discovery": "DISC",
    "replication": "REP",
    "holdout": "HOLDOUT",
}


class EvidencePartitionRecord(BaseModel):
    """One materialized cohort partition."""

    model_config = ConfigDict(extra="forbid")

    partition_id: str
    base_dataset: str
    target_family: str
    role: PartitionRole
    evaluation_role: EvaluationRole
    path: str
    source_path: str
    split_method: str
    seed: int
    n_rows: int
    site_count: int
    exclusion_role: str
    subject_id_sha256: str
    source_row_count: int
    meets_minimum: bool = True
    notes: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    schema_sha256: str = ""
    modality: str = "unknown"
    feature_families: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    group_levels: dict[str, list[str]] = Field(default_factory=dict)


class ExternalEvidenceSetRecord(BaseModel):
    """Predeclared external discovery/replication pair and compatibility policy."""

    model_config = ConfigDict(extra="forbid")

    evidence_set_id: str
    target_family: str
    modality: str
    feature_family: str
    discovery_partition_id: str
    replication_partition_ids: list[str]
    supported_predictors: list[str] = Field(default_factory=list)
    supported_group_vars: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=0)
    confirmation_role: Literal["primary", "secondary"] = "primary"
    optional: bool = False
    units: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class EvidencePartitionManifest(BaseModel):
    """Auditable manifest for split and external evaluation cohorts."""

    model_config = ConfigDict(extra="forbid")

    created_by: str = "confirm.evidence_partitions"
    seed: int
    records: list[EvidencePartitionRecord]
    external_evidence_sets: list[ExternalEvidenceSetRecord] = Field(default_factory=list)

    def partition_ids(self) -> set[str]:
        return {record.partition_id for record in self.records}

    def record_for_partition(
        self,
        partition_id: str,
        target_family: str | None = None,
    ) -> EvidencePartitionRecord | None:
        for record in self.records:
            if record.partition_id == partition_id and (target_family is None or record.target_family == target_family):
                return record
        return None

    def holdout_for_base(self, base_dataset: str, target_family: str) -> EvidencePartitionRecord | None:
        base = canonical_base_cohort(base_dataset)
        for record in self.records:
            if (
                record.base_dataset == base
                and record.target_family == target_family
                and record.role == "holdout"
                and record.evaluation_role == "holdout"
                and record.meets_minimum
            ):
                return record
        return None

    def holdout_eval_pair_for_base(self, base_dataset: str, target_family: str) -> tuple[EvidencePartitionRecord, EvidencePartitionRecord] | None:
        """Return non-overlapping holdout discovery/replication records for one source base."""

        base = canonical_base_cohort(base_dataset)
        records = [
            record
            for record in self.records
            if record.base_dataset == base
            and record.target_family == target_family
            and record.role == "holdout"
            and record.meets_minimum
        ]
        discovery = next((record for record in records if record.evaluation_role == "discovery"), None)
        replication = next((record for record in records if record.evaluation_role == "replication"), None)
        if discovery is None or replication is None or discovery.path == replication.path:
            return None
        return discovery, replication

    def holdout_evaluation_pair_for_contract(self, contract: ClaimContract) -> tuple[EvidencePartitionRecord, list[EvidencePartitionRecord]] | None:
        """Map a source contract onto excluded holdout evaluation evidence.

        Same-base source contracts require distinct ``*_HOLDOUT_DISC`` and
        ``*_HOLDOUT_REP`` partitions. Cross-base contracts prefer those
        evaluation subpartitions but can use ordinary holdout partitions when
        the discovery and replication bases are already distinct.
        """

        target_family = infer_target_family(contract)
        discovery_base = canonical_base_cohort(contract.discovery_cohort)
        replication_bases = [canonical_base_cohort(cohort) for cohort in contract.replication_cohorts]
        discovery_record = self._holdout_record_for_evaluation(discovery_base, target_family, "discovery")
        if discovery_record is None:
            return None
        replication_records: list[EvidencePartitionRecord] = []
        for base in replication_bases:
            record = self._holdout_record_for_evaluation(base, target_family, "replication")
            if record is None:
                return None
            replication_records.append(record)
        if any(record.path == discovery_record.path for record in replication_records):
            return None
        return discovery_record, replication_records

    def _holdout_record_for_evaluation(
        self,
        base_dataset: str,
        target_family: str,
        evaluation_role: Literal["discovery", "replication"],
    ) -> EvidencePartitionRecord | None:
        pair = self.holdout_eval_pair_for_base(base_dataset, target_family)
        if pair is not None:
            return pair[0] if evaluation_role == "discovery" else pair[1]
        return self.holdout_for_base(base_dataset, target_family)

    def external_pair_for_target(self, target_family: str) -> tuple[EvidencePartitionRecord, EvidencePartitionRecord] | None:
        """Backward-compatible target lookup for manifests with one unambiguous set."""

        sets = sorted(
            (
                item
                for item in self.external_evidence_sets
                if item.target_family == target_family and item.confirmation_role == "primary"
            ),
            key=lambda item: (item.priority, item.evidence_set_id),
        )
        for evidence_set in sets:
            pair = self.external_pair_for_set(evidence_set)
            if pair is not None:
                return pair[0], pair[1][0]
        return None

    def external_pair_for_set(
        self,
        evidence_set: ExternalEvidenceSetRecord,
    ) -> tuple[EvidencePartitionRecord, list[EvidencePartitionRecord]] | None:
        discovery = self.record_for_partition(evidence_set.discovery_partition_id, evidence_set.target_family)
        replications = [
            self.record_for_partition(item, evidence_set.target_family)
            for item in evidence_set.replication_partition_ids
        ]
        if discovery is None or any(item is None for item in replications):
            return None
        resolved = [item for item in replications if item is not None]
        if not discovery.meets_minimum or any(not item.meets_minimum for item in resolved):
            return None
        if discovery.role != "external_eval" or any(item.role != "external_eval" for item in resolved):
            return None
        if any(item.path == discovery.path for item in resolved):
            return None
        return discovery, resolved

    def external_sets_for_contract(self, contract: ClaimContract) -> list[ExternalEvidenceSetRecord]:
        """Return predeclared, schema-compatible sets without inspecting results."""

        target_family = infer_target_family(contract)
        modality, feature_family = contract_feature_scope(contract)
        compatible: list[ExternalEvidenceSetRecord] = []
        for evidence_set in self.external_evidence_sets:
            if evidence_set.target_family != target_family or evidence_set.modality != modality:
                continue
            family_matches = evidence_set.feature_family in {"any", feature_family} or (
                feature_family == "fc" and evidence_set.feature_family in {"network_fc", "global_fc"}
            )
            if not family_matches:
                continue
            if evidence_set.supported_predictors and contract.estimand.predictor not in evidence_set.supported_predictors:
                continue
            if (
                contract.estimand.group is not None
                and evidence_set.supported_group_vars
                and contract.estimand.group.var not in evidence_set.supported_group_vars
            ):
                continue
            pair = self.external_pair_for_set(evidence_set)
            if pair is None or not _pair_supports_contract(pair, contract):
                continue
            compatible.append(evidence_set)
        return sorted(
            compatible,
            key=lambda item: (
                0 if item.confirmation_role == "primary" else 1,
                item.priority,
                item.evidence_set_id,
            ),
        )

    def primary_external_set_for_contract(self, contract: ClaimContract) -> ExternalEvidenceSetRecord | None:
        sets = self.external_sets_for_contract(contract)
        return next((item for item in sets if item.confirmation_role == "primary"), None)

    def external_pair_for_contract(
        self,
        contract: ClaimContract,
        *,
        evidence_set_id: str | None = None,
    ) -> tuple[EvidencePartitionRecord, list[EvidencePartitionRecord], ExternalEvidenceSetRecord] | None:
        sets = self.external_sets_for_contract(contract)
        if evidence_set_id is not None:
            sets = [item for item in sets if item.evidence_set_id == evidence_set_id]
            evidence_set = sets[0] if sets else None
        else:
            evidence_set = next((item for item in sets if item.confirmation_role == "primary"), None)
        if evidence_set is None:
            return None
        pair = self.external_pair_for_set(evidence_set)
        if pair is None:
            return None
        return pair[0], pair[1], evidence_set

    def has_excluded_evidence_for_contract(self, contract: ClaimContract) -> bool:
        if self.holdout_evaluation_pair_for_contract(contract) is not None:
            return True
        return self.external_pair_for_contract(contract) is not None

    def external_only_bases(self) -> set[str]:
        roles_by_base: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            roles_by_base[record.base_dataset].add(record.role)
        return {
            base
            for base, roles in roles_by_base.items()
            if roles == {"external_eval"}
        }

    def validation_catalog_for_contract(self, contract: ClaimContract) -> dict[str, Any]:
        target_family = infer_target_family(contract)
        holdout_pair = self.holdout_evaluation_pair_for_contract(contract)
        external_sets = self.external_sets_for_contract(contract)
        external_catalog: list[dict[str, Any]] = []
        for evidence_set in external_sets:
            pair = self.external_pair_for_set(evidence_set)
            if pair is None:
                continue
            external_catalog.append(
                {
                    "evidence_set": evidence_set.model_dump(mode="json"),
                    "discovery": pair[0].model_dump(mode="json"),
                    "replication": [item.model_dump(mode="json") for item in pair[1]],
                }
            )
        primary = next(
            (item for item in external_catalog if item["evidence_set"]["confirmation_role"] == "primary"),
            None,
        )
        return {
            "target_family": target_family,
            "holdout_partitions": (
                [holdout_pair[0].model_dump(mode="json"), *[record.model_dump(mode="json") for record in holdout_pair[1]]]
                if holdout_pair
                else []
            ),
            "holdout_evaluation_pair": (
                {
                    "discovery": holdout_pair[0].model_dump(mode="json"),
                    "replication": [record.model_dump(mode="json") for record in holdout_pair[1]],
                }
                if holdout_pair
                else None
            ),
            "external_partitions": (
                [primary["discovery"], *primary["replication"]] if primary is not None else []
            ),
            "external_evidence_sets": external_catalog,
            "primary_external_evidence_set_id": (
                primary["evidence_set"]["evidence_set_id"] if primary is not None else None
            ),
        }


def contract_feature_scope(contract: ClaimContract) -> tuple[str, str]:
    outcomes = _outcomes(contract)
    if all(str(outcome).startswith("smri_") for outcome in outcomes):
        return "sMRI", "regional_volume"
    if all(str(outcome).startswith("pet_") for outcome in outcomes):
        return "PET", "regional_pet"
    if all(str(outcome).startswith("fc_fc_") for outcome in outcomes):
        return "fMRI", "network_fc"
    global_fc = {"fc_mean_abs", "fc_mean_positive", "fc_within_network", "fc_between_network"}
    if all(str(outcome) in global_fc for outcome in outcomes):
        return "fMRI", "global_fc"
    if all(str(outcome).startswith("fc_") for outcome in outcomes):
        return "fMRI", "fc"
    return "unknown", "unknown"


def _pair_supports_contract(
    pair: tuple[EvidencePartitionRecord, list[EvidencePartitionRecord]],
    contract: ClaimContract,
) -> bool:
    discovery, replications = pair
    return all(_record_supports_contract(record, contract) for record in [discovery, *replications])


def _record_supports_contract(record: EvidencePartitionRecord, contract: ClaimContract) -> bool:
    if not record.columns:
        return True
    available = set(columns_with_virtuals(record.partition_id, columns_with_canonical_aliases(record.columns)))
    required = {contract.estimand.predictor, *contract.covariates}
    if contract.estimand.group is not None:
        required.add(contract.estimand.group.var)
    if not required.issubset(available):
        return False
    if contract.estimand.group is not None:
        levels = record.group_levels.get(contract.estimand.group.var)
        if levels and not {contract.estimand.group.case, contract.estimand.group.control}.issubset(set(levels)):
            return False
    for outcome in _outcomes(contract):
        pattern = str(outcome)
        if pattern.endswith("_"):
            pattern = f"{pattern}*"
        if any(token in pattern for token in "*?["):
            if not any(fnmatch.fnmatch(column, pattern) for column in available):
                return False
        elif pattern not in available:
            return False
    return True


def canonical_base_cohort(cohort: str) -> str:
    """Strip split suffixes while preserving ordinary cohort names."""

    base = str(cohort)
    suffixes = (
        "_EXTERNAL_DISC",
        "_EXTERNAL_REP",
        "_HOLDOUT_DISC",
        "_HOLDOUT_REP",
        "_DISC_SITES",
        "_REP_SITES",
        "_HOLDOUT",
        "_DISC",
        "_REP",
        "_CN",
    )
    for suffix in suffixes:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    for split in ("_DISC_s", "_REP_s", "_HOLDOUT_s"):
        if split in base:
            return base.split(split, 1)[0]
    return base


def is_excluded_evidence_cohort(cohort: str) -> bool:
    """Return true for holdout/external partitions reserved from initial claims."""

    name = str(cohort).upper()
    return "_HOLDOUT" in name or "_EXTERNAL_" in name


def infer_target_family(contract: ClaimContract, row: dict[str, Any] | None = None) -> str:
    """Infer a coarse target family for evidence-policy routing."""

    text_parts = [
        contract.claim_id,
        contract.question,
        str(contract.estimand.outcome),
        contract.estimand.predictor,
        contract.discovery_cohort,
        *contract.replication_cohorts,
    ]
    if contract.estimand.group is not None:
        text_parts.extend([contract.estimand.group.var, contract.estimand.group.case, contract.estimand.group.control])
    if row:
        text_parts.extend(str(row.get(key) or "") for key in ("source_modality", "source_scoring_label", "source_label_class"))
    text = " ".join(text_parts).lower()
    if any(term in text for term in ("adhd", "had_adhd")):
        return "adhd"
    if any(term in text for term in ("asd", "autism", "abide")):
        return "asd"
    if any(term in text for term in ("dementia", "mci", "adni", "oasis", "nacc", "hippocamp", "entorhinal")):
        return "ad_aging"
    if any(term in text for term in ("schiz", "psychosis", "cobre", "fbirn", "bsnip", "ds000030", "cnp")):
        return "psychosis"
    if any(str(outcome).startswith(("fc_", "raw_", "beh_")) for outcome in _outcomes(contract)):
        return "normative_fmri"
    return "unknown"


def load_evidence_manifest(path: str | Path | None) -> EvidencePartitionManifest | None:
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return EvidencePartitionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def validate_manifest_no_overlap(manifest: EvidencePartitionManifest) -> list[str]:
    """Return overlap violations between partitions of the same base dataset."""

    by_base: dict[str, list[EvidencePartitionRecord]] = defaultdict(list)
    bases_with_holdout_pairs = {
        record.base_dataset
        for record in manifest.records
        if record.role == "holdout" and record.evaluation_role in {"discovery", "replication"}
    }
    for record in manifest.records:
        if record.role in {"discovery", "replication", "holdout"}:
            if (
                record.base_dataset in bases_with_holdout_pairs
                and record.role == "holdout"
                and record.evaluation_role == "holdout"
            ):
                continue
            by_base[record.base_dataset].append(record)
    violations: list[str] = []
    for base, records in by_base.items():
        subject_sets: dict[str, set[str]] = {}
        for record in records:
            try:
                table = pd.read_parquet(record.path, columns=["subject_id"])
            except Exception as exc:  # noqa: BLE001
                violations.append(f"{record.partition_id}: unreadable partition: {exc}")
                continue
            subject_sets[record.partition_id] = set(table["subject_id"].astype(str))
        ids = list(subject_sets)
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                overlap = subject_sets[left] & subject_sets[right]
                if overlap:
                    violations.append(f"{base}: {left} overlaps {right} by {len(overlap)} subjects")
    return violations


def build_evidence_partitions(config_path: str | Path, out_root: str | Path | None = None) -> EvidencePartitionManifest:
    """Materialize evidence partitions from a YAML config."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 20260630))
    output_root = Path(out_root or config.get("output_root", "data/prepared_data/evidence_partitions"))
    cohort_root = output_root / "cohorts"
    cohort_root.mkdir(parents=True, exist_ok=True)
    default_split = config.get("default_split") or {"discovery": 0.6, "replication": 0.2, "holdout": 0.2}
    min_rows = config.get("min_rows") or {}
    records: list[EvidencePartitionRecord] = []

    for dataset in config.get("datasets", []):
        base_dataset = str(dataset["dataset"])
        source_path = Path(dataset["source"])
        if not source_path.exists() and bool(dataset.get("optional", False)):
            continue
        target_families = [str(item) for item in dataset.get("target_families", ["unknown"])]
        dataset_seed = int(dataset.get("seed", seed))
        df = pd.read_parquet(source_path)
        source_row_count = len(df)
        if bool(dataset.get("external_eval", False)):
            split_tables = _split_table(df, {"discovery": 0.5, "replication": 0.5}, dataset_seed, method=str(dataset.get("split_method", "site_or_stratified")))
            external_roles = {"discovery": "EXTERNAL_DISC", "replication": "EXTERNAL_REP"}
            for role_name, suffix in external_roles.items():
                table = _filter_valid_evaluation_rows(split_tables[role_name].copy())
                partition_id = f"{base_dataset}_{suffix}"
                table["cohort"] = partition_id
                path = cohort_root / f"{partition_id}.parquet"
                table.to_parquet(path)
                for target_family in target_families:
                    records.append(
                        _record(
                            partition_id=partition_id,
                            base_dataset=base_dataset,
                            target_family=target_family,
                            role="external_eval",
                            evaluation_role=role_name,
                            path=path,
                            source_path=source_path,
                            split_method="external_site_or_stratified",
                            seed=dataset_seed,
                            source_row_count=source_row_count,
                            df=table,
                            min_rows=min_rows,
                        )
                    )
            continue

        split = dataset.get("split") or default_split
        split_tables = _split_table(df, split, dataset_seed, method=str(dataset.get("split_method", "site_or_stratified")))
        for role_name in ("discovery", "replication", "holdout"):
            if role_name not in split_tables:
                continue
            table = split_tables[role_name].copy()
            if role_name == "holdout":
                table = _filter_valid_evaluation_rows(table)
            partition_id = f"{base_dataset}_{ROLE_SUFFIX[role_name]}"
            table["cohort"] = partition_id
            path = cohort_root / f"{partition_id}.parquet"
            table.to_parquet(path)
            for target_family in target_families:
                records.append(
                    _record(
                        partition_id=partition_id,
                        base_dataset=base_dataset,
                        target_family=target_family,
                        role=role_name,
                        evaluation_role=role_name if role_name != "holdout" else "holdout",
                        path=path,
                        source_path=source_path,
                        split_method=str(dataset.get("split_method", "site_or_stratified")),
                        seed=dataset_seed,
                        source_row_count=source_row_count,
                        df=table,
                        min_rows=min_rows,
                    )
                )
            if role_name == "holdout":
                eval_split = _split_table(
                    table,
                    {"discovery": 0.5, "replication": 0.5},
                    dataset_seed + 101,
                    method=str(dataset.get("split_method", "site_or_stratified")),
                )
                for eval_role in ("discovery", "replication"):
                    eval_table = _filter_valid_evaluation_rows(eval_split[eval_role].copy())
                    eval_partition_id = f"{base_dataset}_HOLDOUT_{ROLE_SUFFIX[eval_role]}"
                    eval_table["cohort"] = eval_partition_id
                    eval_path = cohort_root / f"{eval_partition_id}.parquet"
                    eval_table.to_parquet(eval_path)
                    for target_family in target_families:
                        records.append(
                            _record(
                                partition_id=eval_partition_id,
                                base_dataset=base_dataset,
                                target_family=target_family,
                                role="holdout",
                                evaluation_role=eval_role,
                                path=eval_path,
                                source_path=source_path,
                                split_method=f"{dataset.get('split_method', 'site_or_stratified')}:holdout_eval_pair",
                                seed=dataset_seed + 101,
                                source_row_count=source_row_count,
                                df=eval_table,
                                min_rows=min_rows,
                            )
                        )

    external_evidence_sets = _build_external_evidence_sets(config, records)
    manifest = EvidencePartitionManifest(
        seed=seed,
        records=records,
        external_evidence_sets=external_evidence_sets,
    )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return manifest


def _filter_valid_evaluation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot satisfy universal evaluation covariates."""

    table = df.copy()
    mask = pd.Series(True, index=table.index)
    if "age" in table.columns:
        age = pd.to_numeric(table["age"], errors="coerce")
        mask &= age.notna()
        table["age"] = age
    if "sex" in table.columns:
        sex = normalize_sex(table["sex"])
        mask &= sex.notna()
        table["sex"] = sex
    return table.loc[mask].copy()


def _record(
    *,
    partition_id: str,
    base_dataset: str,
    target_family: str,
    role: PartitionRole,
    evaluation_role: str,
    path: Path,
    source_path: Path,
    split_method: str,
    seed: int,
    source_row_count: int,
    df: pd.DataFrame,
    min_rows: dict[str, Any],
) -> EvidencePartitionRecord:
    subject_ids = df["subject_id"].astype(str).tolist() if "subject_id" in df.columns else []
    digest = hashlib.sha256("\n".join(sorted(subject_ids)).encode("utf-8")).hexdigest()
    site_count = int(df["site"].astype(str).nunique()) if "site" in df.columns else 0
    minimum = int(min_rows.get("default_partition_rows", 20))
    if target_family == "normative_fmri":
        minimum = int(min_rows.get("continuous_rows", 100))
    meets_minimum = len(df) >= minimum
    notes = [] if meets_minimum else [f"partition rows {len(df)} below minimum {minimum}"]
    columns = list(map(str, df.columns))
    schema_sha256 = hashlib.sha256("\n".join(sorted(columns)).encode("utf-8")).hexdigest()
    modality, feature_families, units = _table_feature_metadata(columns)
    group_levels = _group_levels(df, partition_id)
    return EvidencePartitionRecord(
        partition_id=partition_id,
        base_dataset=base_dataset,
        target_family=target_family,
        role=role,
        evaluation_role=str(evaluation_role),
        path=str(path),
        source_path=str(source_path),
        split_method=split_method,
        seed=seed,
        n_rows=int(len(df)),
        site_count=site_count,
        exclusion_role="excluded_evaluation" if role in {"holdout", "external_eval"} else "claim_source",
        subject_id_sha256=digest,
        source_row_count=int(source_row_count),
        meets_minimum=meets_minimum,
        notes=notes,
        columns=columns,
        schema_sha256=schema_sha256,
        modality=modality,
        feature_families=feature_families,
        units=units,
        group_levels=group_levels,
    )


def _group_levels(df: pd.DataFrame, cohort: str) -> dict[str, list[str]]:
    table = add_virtual_columns(df, cohort)
    levels: dict[str, list[str]] = {}
    for column in ("confirm_dx", "dx"):
        if column not in table.columns:
            continue
        values = sorted(table[column].dropna().astype(str).unique().tolist())
        if len(values) <= 100:
            levels[column] = values
    return levels


def _build_external_evidence_sets(
    config: dict[str, Any],
    records: list[EvidencePartitionRecord],
) -> list[ExternalEvidenceSetRecord]:
    requested = [ExternalEvidenceSetRecord.model_validate(item) for item in config.get("external_evidence_sets", [])]
    configured: list[ExternalEvidenceSetRecord] = []
    partition_targets = {(record.partition_id, record.target_family) for record in records}
    for evidence_set in requested:
        referenced = [evidence_set.discovery_partition_id, *evidence_set.replication_partition_ids]
        missing = [
            partition_id
            for partition_id in referenced
            if (partition_id, evidence_set.target_family) not in partition_targets
        ]
        if missing:
            if evidence_set.optional:
                continue
            raise ValueError(f"External evidence set {evidence_set.evidence_set_id!r} references missing partitions: {missing}")
        configured.append(evidence_set)

    ids = [item.evidence_set_id for item in configured]
    if len(ids) != len(set(ids)):
        raise ValueError("External evidence set IDs must be unique")
    return configured


def _table_feature_metadata(columns: list[str]) -> tuple[str, list[str], dict[str, str]]:
    column_set = set(columns)
    families: list[str] = []
    units: dict[str, str] = {}
    if any(column.startswith("fc_fc_") for column in column_set):
        families.append("network_fc")
    if any(column in {"fc_mean_abs", "fc_mean_positive", "fc_within_network", "fc_between_network"} for column in column_set):
        families.append("global_fc")
    if any(column.startswith("smri_") for column in column_set):
        families.append("regional_volume")
        units["smri_*"] = "mm3"
        if "eTIV" in column_set or "smri_icv" in column_set:
            units["eTIV"] = "mm3"
    if any(column.startswith("pet_") for column in column_set):
        families.append("regional_pet")
    modalities = {
        "fMRI" if family in {"network_fc", "global_fc"} else
        "sMRI" if family == "regional_volume" else
        "PET" if family == "regional_pet" else
        "unknown"
        for family in families
    }
    modality = next(iter(modalities)) if len(modalities) == 1 else ("multimodal" if modalities else "unknown")
    return modality, families, units


def _split_table(df: pd.DataFrame, split: dict[str, float], seed: int, *, method: str) -> dict[str, pd.DataFrame]:
    roles = [role for role in ("discovery", "replication", "holdout") if float(split.get(role, 0.0)) > 0.0]
    if "holdout" not in roles and set(split) == {"discovery", "replication"}:
        roles = ["discovery", "replication"]
    if method.startswith("site") and "site" in df.columns and df["site"].astype(str).nunique() >= len(roles):
        site_split = _site_split(df, roles, split, seed)
        if not _has_subject_overlap(site_split):
            return site_split
    return _stratified_subject_split(df, roles, split, seed)


def _site_split(df: pd.DataFrame, roles: list[str], split: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sites = sorted(df["site"].dropna().astype(str).unique())
    rng.shuffle(sites)
    targets = {role: float(split[role]) * len(df) for role in roles}
    counts = {role: 0 for role in roles}
    assigned: dict[str, list[str]] = {role: [] for role in roles}
    site_counts = df.groupby(df["site"].astype(str)).size().to_dict()
    for site in sites:
        role = min(roles, key=lambda item: counts[item] / max(targets[item], 1.0))
        assigned[role].append(site)
        counts[role] += int(site_counts[site])
    return {
        role: df[df["site"].astype(str).isin(role_sites)].copy()
        for role, role_sites in assigned.items()
    }


def _stratified_subject_split(df: pd.DataFrame, roles: list[str], split: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    table = df.copy()
    unit_column = "subject_id" if "subject_id" in table.columns else "_row_unit"
    if unit_column == "_row_unit":
        table[unit_column] = [str(index) for index in table.index]
    table[unit_column] = table[unit_column].astype(str)
    unit_table = table.drop_duplicates(subset=[unit_column]).copy()
    strata_columns = [col for col in ("dx", "sex") if col in table.columns]
    if "age" in unit_table.columns:
        unit_table["_age_bin"] = pd.qcut(
            pd.to_numeric(unit_table["age"], errors="coerce"),
            q=min(4, max(1, len(unit_table) // 10)),
            duplicates="drop",
        )
        strata_columns.append("_age_bin")
    if strata_columns:
        strata = unit_table[strata_columns].astype(str).agg("|".join, axis=1)
    else:
        strata = pd.Series(["all"] * len(unit_table), index=unit_table.index)
    role_units = {role: [] for role in roles}
    weights = np.array([float(split[role]) for role in roles], dtype=float)
    weights = weights / weights.sum()
    for _, indices in unit_table.groupby(strata, dropna=False).groups.items():
        shuffled = np.array(list(indices))
        rng.shuffle(shuffled)
        boundaries = np.cumsum(weights[:-1] * len(shuffled)).astype(int)
        chunks = np.split(shuffled, boundaries)
        for role, chunk in zip(roles, chunks):
            units = unit_table.loc[list(chunk), unit_column].astype(str).tolist()
            role_units[role].extend(units)
    return {
        role: df.loc[table[unit_column].isin(set(units))].copy()
        for role, units in role_units.items()
    }


def _has_subject_overlap(splits: dict[str, pd.DataFrame]) -> bool:
    subject_sets = []
    for table in splits.values():
        if "subject_id" not in table.columns:
            return False
        subject_sets.append(set(table["subject_id"].astype(str)))
    for i, left in enumerate(subject_sets):
        for right in subject_sets[i + 1 :]:
            if left & right:
                return True
    return False


def _outcomes(contract: ClaimContract) -> list[str]:
    outcome = contract.estimand.outcome
    return list(outcome) if isinstance(outcome, list) else [outcome]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evidence_partitions.yml")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--check-overlap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_evidence_partitions(args.config, args.out_root)
    violations = validate_manifest_no_overlap(manifest) if args.check_overlap else []
    summary = {
        "records": len(manifest.records),
        "partition_ids": sorted(manifest.partition_ids()),
        "overlap_violations": violations,
    }
    print(json.dumps(summary, indent=2))
    if violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
