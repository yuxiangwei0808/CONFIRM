"""Build the frozen source snapshot used to construct NeuroClaimBench v2.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from confirm.contract import ClaimContract

from bench.neuroclaimbench_v21_compat import (
    AdjudicationRecord,
    BenchmarkItem,
    EvaluationTask,
    SourceReference,
    exact_contract_hash,
    semantic_contract_hash,
    sha256_payload,
    unresolved_semantic_hash,
)

DEFAULT_STAGE2 = Path("review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json")
DEFAULT_LEGACY = Path("data/claims/fixed_literature_claims.csv")
DEFAULT_LEGACY_SYNTHETIC = Path("data/claims/synthetic_stress_claims.csv")
DEFAULT_SYNTHETIC = Path("review-stage/claim-search-safety-gpt55-r10-c10-v7/gates/known_negative_results.json")
DEFAULT_NACC = Path("data/external_benchmark/nacc_claims.csv")
DEFAULT_CNP = Path("data/external_benchmark/ds000030_claims.csv")
DEFAULT_EVIDENCE_MANIFEST = Path("data/prepared_data/evidence_partitions/manifest.json")
DEFAULT_OUT_DIR = Path("data/neuroclaimbench/v2.1-source")
RANDOM_CONTROL_SEED = 20260722


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: str(value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    payloads = [row.model_dump(mode="json") if hasattr(row, "model_dump") else row for row in rows]
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in payloads))


def _manifest_partitions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    return {str(row["partition_id"]): row for row in payload.get("records", [])}


def _modality(contract: ClaimContract) -> str:
    outcomes = contract.estimand.outcome if isinstance(contract.estimand.outcome, list) else [contract.estimand.outcome]
    text = " ".join(outcomes).lower()
    if "smri" in text:
        return "sMRI"
    if "pet" in text:
        return "PET"
    if "fc" in text:
        return "fMRI-FC"
    return "other"


def _dataset_id(cohort: str) -> str:
    for suffix in ("_EXTERNAL_DISC", "_EXTERNAL_REP", "_HOLDOUT_DISC", "_HOLDOUT_REP", "_DISC", "_REP"):
        if cohort.endswith(suffix):
            return cohort[: -len(suffix)]
    return cohort


def _evaluation_task(
    item_id: str,
    contract: ClaimContract,
    *,
    role: str,
    partitions: dict[str, dict[str, Any]],
    freshness: str,
    generator_spec: dict[str, Any] | None = None,
) -> EvaluationTask:
    cohorts = [contract.discovery_cohort, *contract.replication_cohorts]
    hashes = {
        cohort: str(partitions.get(cohort, {}).get("content_sha256") or "")
        for cohort in cohorts
        if partitions.get(cohort, {}).get("content_sha256")
    }
    identity = {
        "item": item_id,
        "role": role,
        "cohorts": cohorts,
        "hashes": hashes,
        "generator_spec": generator_spec,
    }
    return EvaluationTask(
        task_id=f"ncb-task-{sha256_payload(identity)[:16]}",
        benchmark_item_id=item_id,
        contract=contract,
        evidence_role=role,
        dataset_id=_dataset_id(contract.discovery_cohort),
        discovery_cohort=contract.discovery_cohort,
        replication_cohorts=list(contract.replication_cohorts),
        partition_hashes=hashes,
        evidence_freshness=freshness,
        generator_spec=generator_spec,
    )


class _CorpusBuilder:
    def __init__(self, partitions: dict[str, dict[str, Any]]) -> None:
        self.partitions = partitions
        self.items: dict[str, BenchmarkItem] = {}
        self.key_to_id: dict[tuple[str, str], str] = {}
        self.tasks: dict[str, EvaluationTask] = {}
        self.adjudications: list[AdjudicationRecord] = []
        self.crosswalk: list[dict[str, str]] = []

    def add_ready(
        self,
        *,
        track: str,
        target_family: str,
        question: str,
        contract: ClaimContract,
        source: SourceReference,
        label_class: str = "candidate_unknown",
        construction_derived: bool = False,
        evidence_role: str = "source",
        freshness: str = "unknown",
        generator_spec: dict[str, Any] | None = None,
    ) -> BenchmarkItem:
        exact_hash = exact_contract_hash(contract)
        semantic_hash = semantic_contract_hash(contract)
        execution_hash = sha256_payload({"contract": exact_hash, "generator_spec": generator_spec}) if generator_spec else exact_hash
        key = (track, execution_hash)
        item_id = self.key_to_id.get(key)
        if item_id is not None:
            item = self.items[item_id]
            if source.source_id not in item.aliases:
                item.aliases.append(source.source_id)
                item.source_references.append(source)
            self._crosswalk(source, item, "exact_contract_alias")
            return item

        item_id = f"ncb-{track.replace('_', '-')}-{execution_hash[:16]}"
        task = _evaluation_task(
            item_id,
            contract,
            role=evidence_role,
            partitions=self.partitions,
            freshness=freshness,
            generator_spec=generator_spec,
        )
        item = BenchmarkItem(
            benchmark_item_id=item_id,
            claim_uid=f"ncb-claim-{semantic_hash[:16]}",
            semantic_cluster_id=f"ncb-sem-{semantic_hash[:16]}",
            benchmark_track=track,
            target_family=target_family,
            modality=_modality(contract),
            question=question,
            contract=contract,
            exact_contract_sha256=exact_hash,
            semantic_claim_sha256=semantic_hash,
            source_references=[source],
            aliases=[source.source_id],
            evaluation_task_ids=[task.task_id],
            label_class=label_class if construction_derived else "candidate_unknown",
            reference_disposition="abstain" if construction_derived else "unresolved",
            adjudication_status="construction_derived" if construction_derived else "pending",
            score_eligible=construction_derived,
        )
        self.items[item_id] = item
        self.key_to_id[key] = item_id
        self.tasks[task.task_id] = task
        self._crosswalk(source, item, "canonical")
        if construction_derived:
            self.adjudications.append(
                AdjudicationRecord(
                    benchmark_item_id=item_id,
                    scientific_question_sha256=item.scientific_question_sha256,
                    vote_models=[],
                    final_label=label_class,
                    reference_disposition="abstain",
                    adjudication_status="construction_derived",
                    consensus_rule="label_fixed_by_deterministic_control_construction",
                    score_eligible=True,
                )
            )
        return item

    def add_pending(self, row: dict[str, str], source_path: Path) -> BenchmarkItem:
        source = SourceReference(
            source_collection="legacy_scientific",
            source_id=row["claim_id"],
            source_path=str(source_path),
            source_mode=row.get("source_mode", ""),
            target_family=row.get("target_family", ""),
            prior_label=row.get("label_class", ""),
            source_citation=row.get("source_citation", ""),
        )
        semantic_hash = unresolved_semantic_hash(
            question=row.get("question", ""),
            target_family=row.get("target_family", ""),
            source_id=row["claim_id"],
        )
        item_id = f"ncb-source-{semantic_hash[:16]}"
        item = BenchmarkItem(
            benchmark_item_id=item_id,
            claim_uid=f"ncb-claim-{semantic_hash[:16]}",
            semantic_cluster_id=f"ncb-sem-{semantic_hash[:16]}",
            benchmark_track="scientific",
            target_family=row.get("target_family", ""),
            modality="unknown",
            question=row.get("question", ""),
            semantic_claim_sha256=semantic_hash,
            source_references=[source],
            aliases=[row["claim_id"]],
            migration_status="pending_contract",
        )
        self.items[item_id] = item
        self._crosswalk(source, item, "pending_legacy_redraft")
        return item

    def _crosswalk(self, source: SourceReference, item: BenchmarkItem, status: str) -> None:
        self.crosswalk.append(
            {
                "source_collection": source.source_collection,
                "source_id": source.source_id,
                "source_path": source.source_path,
                "benchmark_item_id": item.benchmark_item_id,
                "claim_uid": item.claim_uid,
                "mapping_status": status,
            }
        )


def _stage2_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("claims") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Stage 2 payload has no claims list: {path}")
    return rows


def _load_legacy_contracts(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            claim_id = str(row.get("claim_id") or "")
            payload = row.get("drafted_contract") or row.get("contract")
            if claim_id and isinstance(payload, dict):
                out[claim_id] = row
    return out


def _external_contract(row: dict[str, str], dataset: str) -> ClaimContract:
    if dataset == "NACC":
        discovery, replication = "NACC_EXTERNAL_DISC", "NACC_EXTERNAL_REP"
        target_family, covariates = "ad_aging", ["age", "sex", "eTIV"]
    else:
        discovery, replication = "ds000030_EXTERNAL_DISC", "ds000030_EXTERNAL_REP"
        target_family, covariates = "psychosis", ["age", "sex"]
    case, control = row["case"], row["control"]
    direction = {"increase": "positive", "decrease": "negative"}.get(row.get("expected_sign", ""), row.get("expected_sign", "two_sided"))
    if direction not in {"positive", "negative", "two_sided"}:
        direction = "two_sided"
    question = f"In {dataset}, is {row['outcome']} {direction} for {case} compared with {control}?"
    return ClaimContract.model_validate(
        {
            "claim_id": f"external_{dataset.lower()}_{row['claim_id'].lower()}",
            "question": question,
            "estimand": {
                "type": "group_diff",
                "outcome": row["outcome"],
                "predictor": "dx",
                "group": {"var": "dx", "case": case, "control": control},
                "direction": direction,
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": covariates,
            "inclusion": None,
            "discovery_cohort": discovery,
            "replication_cohorts": [replication],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": covariates, "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "combat",
                    "pattern_corr_min": 0.5,
                    "region_replication_frac_min": 0.5,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def _random_control_contract(parent: ClaimContract, seed: int) -> ClaimContract:
    payload = parent.model_dump(mode="json")
    payload["claim_id"] = f"{parent.claim_id}_random_label_s{seed}"
    payload["question"] = f"Random-label control matched to {parent.claim_id}."
    payload["estimand"]["predictor"] = "ncb_random_group"
    payload["estimand"]["group"] = {"var": "ncb_random_group", "case": "A", "control": "B"}
    payload["estimand"]["direction"] = "two_sided"
    return ClaimContract.model_validate(payload)


def _balanced_random_assignments(subject_ids: list[str], seed: int) -> dict[str, str]:
    ranked = sorted(
        set(subject_ids),
        key=lambda subject_id: hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).hexdigest(),
    )
    midpoint = len(ranked) // 2
    return {subject_id: "A" if index < midpoint else "B" for index, subject_id in enumerate(ranked)}


def _materialize_random_assignments(
    specs: list[dict[str, Any]],
    *,
    partitions: dict[str, dict[str, Any]],
    out_dir: Path,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for spec in specs:
        control_id = str(spec["control_id"])
        control_rows: list[dict[str, Any]] = []
        for cohort, seed in ((spec["discovery_cohort"], spec["discovery_seed"]), (spec["replication_cohort"], spec["replication_seed"])):
            source_path = Path(str(partitions.get(cohort, {}).get("path") or ""))
            if not source_path.exists():
                raise FileNotFoundError(f"Cannot materialize random control; missing partition {cohort}: {source_path}")
            frame = pd.read_parquet(source_path, columns=["subject_id"])
            assignments = _balanced_random_assignments(frame["subject_id"].astype(str).tolist(), int(seed))
            control_rows.extend(
                {
                    "control_id": control_id,
                    "cohort": cohort,
                    "subject_id": subject_id,
                    "ncb_random_group": group,
                    "seed": int(seed),
                }
                for subject_id, group in sorted(assignments.items())
            )
        rows.extend(control_rows)
        metadata[control_id] = {
            "assignment_sha256": sha256_payload(control_rows),
            "assignment_count": len(control_rows),
        }
    path = out_dir / "external_random_control_assignments.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_parquet(temp, index=False)
    os.replace(temp, path)
    return path, metadata


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "stage2": Path(args.stage2_results),
        "legacy": Path(args.legacy_claims),
        "legacy_synthetic": Path(args.legacy_synthetic),
        "synthetic": Path(args.synthetic_results),
        "nacc": Path(args.nacc_claims),
        "cnp": Path(args.cnp_claims),
        "evidence_manifest": Path(args.evidence_manifest),
    }
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} input: {path}")
    partitions = _manifest_partitions(paths["evidence_manifest"])
    builder = _CorpusBuilder(partitions)

    stage2_rows = _stage2_rows(paths["stage2"])
    for row in stage2_rows:
        contract = ClaimContract.model_validate(row["contract"])
        source = SourceReference(
            source_collection="stage2_current",
            source_id=str(row["claim_id"]),
            source_path=str(paths["stage2"]),
            source_mode=str(row.get("source_mode") or ""),
            target_family=str(row.get("target_family") or ""),
            prior_label=str(row.get("label_class") or ""),
            source_citation=str(row.get("source_citation") or ""),
        )
        builder.add_ready(
            track="scientific",
            target_family=source.target_family,
            question=contract.question,
            contract=contract,
            source=source,
        )

    legacy_contracts = _load_legacy_contracts([Path(value) for value in args.legacy_contracts])
    for row in _read_csv(paths["legacy"]):
        drafted = legacy_contracts.get(row["claim_id"])
        if drafted is None:
            builder.add_pending(row, paths["legacy"])
            continue
        contract = ClaimContract.model_validate(drafted.get("drafted_contract") or drafted["contract"])
        source = SourceReference(
            source_collection="legacy_scientific",
            source_id=row["claim_id"],
            source_path=str(paths["legacy"]),
            source_mode=row.get("source_mode", ""),
            target_family=row.get("target_family", ""),
            prior_label=row.get("label_class", ""),
            source_citation=row.get("source_citation", ""),
        )
        builder.add_ready(
            track="scientific",
            target_family=source.target_family,
            question=contract.question,
            contract=contract,
            source=source,
        )

    synthetic_payload = _read_json(paths["synthetic"])
    synthetic_items_by_family: dict[str, list[str]] = defaultdict(list)
    for row in synthetic_payload.get("claims", []):
        contract = ClaimContract.model_validate(row["contract"])
        source = SourceReference(
            source_collection="synthetic_stress_v7",
            source_id=str(row["claim_id"]),
            source_path=str(paths["synthetic"]),
            source_mode="synthetic_stress",
            target_family=str(row.get("target_family") or "synthetic_stress"),
            prior_label=str(row.get("scoring_label") or "known_null"),
            source_citation=str(row.get("source_citation") or "synthetic construction"),
        )
        item = builder.add_ready(
            track="synthetic_stress",
            target_family="synthetic_stress",
            question=contract.question,
            contract=contract,
            source=source,
            label_class=str(row.get("scoring_label") or "known_null"),
            construction_derived=True,
            freshness="previously_queried",
        )
        synthetic_items_by_family[str(row.get("synthetic_failure_family") or row.get("family") or "unknown")].append(item.benchmark_item_id)

    for row in _read_csv(paths["legacy_synthetic"]):
        text = f"{row.get('claim_id')} {row.get('question')}".lower()
        family = next(
            (name for token, name in (("random", "random_label"), ("fishing", "p_fishing"), ("site", "site_confound"), ("underpowered", "underpowered")) if token in text),
            "unknown",
        )
        candidates = sorted(synthetic_items_by_family.get(family, []))
        builder.crosswalk.append(
            {
                "source_collection": "legacy_synthetic",
                "source_id": row["claim_id"],
                "source_path": str(paths["legacy_synthetic"]),
                "benchmark_item_id": candidates[0] if candidates else "",
                "claim_uid": builder.items[candidates[0]].claim_uid if candidates else "",
                "mapping_status": "superseded_family_reference" if candidates else "superseded_unmapped",
            }
        )

    external_rows: list[tuple[str, Path, dict[str, str]]] = []
    external_rows.extend(("NACC", paths["nacc"], row) for row in _read_csv(paths["nacc"]))
    external_rows.extend(("CNP", paths["cnp"], row) for row in _read_csv(paths["cnp"]))
    random_assignment_specs: list[dict[str, Any]] = []
    for index, (dataset, path, row) in enumerate(external_rows):
        contract = _external_contract(row, dataset)
        target = "ad_aging" if dataset == "NACC" else "psychosis"
        source = SourceReference(
            source_collection=f"external_{dataset.lower()}",
            source_id=row["claim_id"],
            source_path=str(path),
            source_mode="external_transfer",
            target_family=target,
            prior_label=row.get("label_class", ""),
            source_citation=row.get("basis", ""),
        )
        builder.add_ready(
            track="external_transfer",
            target_family=target,
            question=contract.question,
            contract=contract,
            source=source,
            evidence_role="external",
            freshness="previously_queried",
        )

        seed = RANDOM_CONTROL_SEED + index
        random_contract = _random_control_contract(contract, seed)
        random_source = SourceReference(
            source_collection=f"external_{dataset.lower()}_random_controls",
            source_id=random_contract.claim_id,
            source_path=str(path),
            source_mode="external_random_control",
            target_family=target,
            prior_label="known_null",
            source_citation=f"deterministic random-label control seed={seed}",
        )
        random_item = builder.add_ready(
            track="external_transfer",
            target_family=target,
            question=random_contract.question,
            contract=random_contract,
            source=random_source,
            label_class="known_null",
            construction_derived=True,
            evidence_role="synthetic_control",
            freshness="fresh",
            generator_spec={
                "type": "independent_random_group_labels",
                "seed": seed,
                "parent_external_claim_id": row["claim_id"],
                "group_column": "ncb_random_group",
                "discovery_seed": seed,
                "replication_seed": seed + 1_000_000,
            },
        )
        random_assignment_specs.append(
            {
                "control_id": random_item.benchmark_item_id,
                "discovery_cohort": random_contract.discovery_cohort,
                "replication_cohort": random_contract.replication_cohorts[0],
                "discovery_seed": seed,
                "replication_seed": seed + 1_000_000,
            }
        )

    out_dir = Path(args.out_dir)
    assignment_path, assignment_metadata = _materialize_random_assignments(
        random_assignment_specs,
        partitions=partitions,
        out_dir=out_dir,
    )
    for task in builder.tasks.values():
        if not task.generator_spec:
            continue
        details = assignment_metadata[task.benchmark_item_id]
        task.generator_spec.update(
            {
                "assignment_path": str(assignment_path),
                "assignment_sha256": details["assignment_sha256"],
                "assignment_count": details["assignment_count"],
            }
        )

    items = sorted(builder.items.values(), key=lambda row: row.benchmark_item_id)
    tasks = sorted(builder.tasks.values(), key=lambda row: row.task_id)
    adjudications = sorted(builder.adjudications, key=lambda row: row.benchmark_item_id)
    crosswalk = sorted(builder.crosswalk, key=lambda row: (row["source_collection"], row["source_id"]))
    _write_jsonl(out_dir / "benchmark_items.jsonl", items)
    _write_jsonl(out_dir / "evaluation_tasks.jsonl", tasks)
    _write_jsonl(out_dir / "adjudications.jsonl", adjudications)
    _write_jsonl(out_dir / "evidence_records.jsonl", [])
    _write_jsonl(out_dir / "label_votes.jsonl", [])
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=out_dir, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_collection", "source_id", "source_path", "benchmark_item_id", "claim_uid", "mapping_status"])
        writer.writeheader()
        writer.writerows(crosswalk)
        crosswalk_temp = Path(handle.name)
    os.chmod(crosswalk_temp, 0o644)
    os.replace(crosswalk_temp, out_dir / "legacy_crosswalk.csv")

    splits = {
        "version": "2.0.0",
        "tracks": {
            track: [item.benchmark_item_id for item in items if item.benchmark_track == track]
            for track in ("scientific", "synthetic_stress", "external_transfer")
        },
        "adjudication_candidates": [
            item.benchmark_item_id
            for item in items
            if item.migration_status == "ready" and not item.score_eligible and item.benchmark_track in {"scientific", "external_transfer"}
        ],
    }
    _atomic_text(out_dir / "benchmark_splits.json", json.dumps(splits, indent=2, sort_keys=True) + "\n")

    output_names = [
        "benchmark_items.jsonl",
        "evidence_records.jsonl",
        "label_votes.jsonl",
        "adjudications.jsonl",
        "evaluation_tasks.jsonl",
        "legacy_crosswalk.csv",
        "benchmark_splits.json",
        "external_random_control_assignments.parquet",
    ]
    manifest = {
        "benchmark_name": "NeuroClaimBench",
        "version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_files": {key: {"path": str(path), "sha256": _sha256_file(path)} for key, path in paths.items()},
        "optional_legacy_contracts": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in map(Path, args.legacy_contracts) if path.exists()
        ],
        "source_counts": {
            "stage2_current": len(stage2_rows),
            "legacy_scientific": len(_read_csv(paths["legacy"])),
            "legacy_synthetic": len(_read_csv(paths["legacy_synthetic"])),
            "synthetic_stress_v7": len(synthetic_payload.get("claims", [])),
            "external_scientific": len(external_rows),
            "external_random_controls": len(external_rows),
        },
        "canonical_counts": {
            "benchmark_items": len(items),
            "evaluation_tasks": len(tasks),
            "pending_contracts": sum(item.migration_status == "pending_contract" for item in items),
            "track_counts": dict(Counter(item.benchmark_track for item in items)),
            "exact_alias_count": sum(max(0, len(item.source_references) - 1) for item in items),
        },
        "output_files": {name: {"path": str(out_dir / name), "sha256": _sha256_file(out_dir / name)} for name in output_names},
        "interpretation_restrictions": [
            "multi-model labels are not human expert ground truth",
            "unresolved claims are excluded from scored-label denominators",
            "scientific, synthetic, NACC, and CNP denominators must remain separate",
            "existing feedback-search artifacts are not modified by this build",
        ],
    }
    _atomic_text(out_dir / "build_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-results", default=str(DEFAULT_STAGE2))
    parser.add_argument("--legacy-claims", default=str(DEFAULT_LEGACY))
    parser.add_argument("--legacy-synthetic", default=str(DEFAULT_LEGACY_SYNTHETIC))
    parser.add_argument("--legacy-contracts", action="append", default=[])
    parser.add_argument("--synthetic-results", default=str(DEFAULT_SYNTHETIC))
    parser.add_argument("--nacc-claims", default=str(DEFAULT_NACC))
    parser.add_argument("--cnp-claims", default=str(DEFAULT_CNP))
    parser.add_argument("--evidence-manifest", default=str(DEFAULT_EVIDENCE_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    manifest = build_package(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", "canonical_counts": manifest["canonical_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
