"""Build and evaluate the local synthetic known-negative safety benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bench.injected_nulls import NegativeStressTask, generate_negative_stress_tasks
from bench.labels import label_authority, label_provenance, scoring_bucket
from bench.metrics import DEFAULT_RUNGS, exact_binomial_ci, summarize_rows
from confirm.analysis import audit_confound_completeness, directionally_consistent, fit_effect, multiplicity_threshold
from confirm.contract import ClaimContract
from confirm.evidence_partitions import EvidencePartitionManifest, EvidencePartitionRecord, infer_target_family
from confirm.multiverse import run_multiverse
from confirm.power import power_check
from confirm.replication import replicate
from confirm.verdict import decide

RUNGS = DEFAULT_RUNGS


def _json_safe(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_json_safe(value) for value in data]
    if isinstance(data, tuple):
        return [_json_safe(value) for value in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


def _contract_with_covariates(contract: ClaimContract, covariates: list[str]) -> ClaimContract:
    return contract.model_copy(update={"covariates": covariates})


def _complete_required_rows(df: pd.DataFrame, contract: ClaimContract) -> pd.DataFrame:
    outcome = contract.estimand.outcome
    outcomes = [outcome] if isinstance(outcome, str) else list(outcome)
    needed = ["subject_id", "cohort", "site", "age", "sex", *outcomes, *contract.covariates]
    if contract.estimand.group is not None:
        needed.append(contract.estimand.group.var)
    needed = [col for col in dict.fromkeys(needed) if col in df.columns]
    out = df.dropna(subset=needed).copy()
    if contract.estimand.group is not None:
        group = contract.estimand.group
        out = out[out[group.var].isin([group.case, group.control])].copy()
    return out


def _effect_significant(effect: Any, contract: ClaimContract) -> bool:
    return bool(effect.p <= multiplicity_threshold(contract) and directionally_consistent(effect.beta, contract))


def evaluate_negative_task(task: NegativeStressTask) -> dict[str, Any]:
    """Execute one generated synthetic negative through the scalar gate ladder."""

    disc = _complete_required_rows(task.discovery, task.contract)
    rep = _complete_required_rows(task.replication, task.contract)
    if len(disc) < 20 or len(rep) < 20:
        raise ValueError(f"Too few complete rows: discovery={len(disc)}, replication={len(rep)}")

    min_contract = _contract_with_covariates(task.contract, task.covariates_min)
    min_primary = fit_effect(disc, min_contract, covariates=min_contract.covariates, model="ols")
    primary = fit_effect(disc, task.contract, covariates=task.contract.covariates, model="ols")
    confound_audit = audit_confound_completeness(disc, task.contract)
    power = power_check(primary, task.contract, ref_effect=task.contract.gates.power.ref_effect)
    multiverse = run_multiverse(disc, task.contract)
    replication = replicate(primary, disc, [rep], task.contract)
    verdict = decide(primary, multiverse, power, replication, task.contract, confound_audit=confound_audit)

    primary_admissible = bool(verdict.gates.get("search_provenance", True)) and bool(verdict.gates.get("multiplicity", False))
    confound_pass = (
        primary_admissible
        and bool(verdict.gates.get("confound", False))
        and bool(verdict.gates.get("confound_completeness", False))
    )
    power_pass = confound_pass and not power.under_powered
    multiverse_pass = power_pass and multiverse.passed
    replication_pass = multiverse_pass and replication.passed
    label_row = dict(task.label_row)
    label_row["label_authority"] = label_authority(label_row)
    scoring_label = task.label_class

    return {
        "claim_id": task.claim_id,
        "family": task.family,
        "synthetic_failure_family": task.family,
        "expected_gate": task.expected_gate,
        "modality": label_row["modality"],
        "claim_type": "synthetic_negative",
        "target_family": "synthetic_stress",
        "source_mode": "synthetic_stress",
        "ground_truth": scoring_label,
        "scoring_label": scoring_label,
        "scoring_bucket": scoring_bucket(scoring_label),
        "label_provenance": label_provenance(scoring_label),
        "label_basis": label_row["label_basis"],
        "adjudication_status": label_row["adjudication_status"],
        "label_authority": label_row["label_authority"],
        "label_confidence": label_row["label_confidence"],
        "source_citation": label_row["source_citation"],
        "label_metadata": label_row,
        "discovery_cohort": str(disc["cohort"].iloc[0]),
        "replication_cohort": str(rep["cohort"].iloc[0]),
        "n_discovery": int(len(disc)),
        "n_replication": int(len(rep)),
        "label_counts_discovery": _group_counts(disc, task.contract),
        "label_counts_replication": _group_counts(rep, task.contract),
        "outcome": task.contract.estimand.outcome,
        "n_features": int(task.contract.search_provenance.family_size),
        "reported_features": [task.contract.estimand.outcome],
        "reported_feature_count": 1,
        "covariates_full": list(task.contract.covariates),
        "covariates_min": list(task.covariates_min),
        "search_provenance": task.contract.search_provenance.model_dump(mode="json"),
        "synthetic_metadata": {key: value for key, value in task.metadata.items() if not key.startswith("_")},
        "best_region": task.contract.estimand.outcome,
        "best_beta": float(primary.beta),
        "best_p": float(primary.p),
        "best_standardized_effect": float(primary.standardized_effect),
        "primary_effect": primary.to_dict(),
        "min_primary_effect": min_primary.to_dict(),
        "primary_region_table": None,
        "primary_sig_regions": int(bool(verdict.gates.get("multiplicity", False))),
        "confound_completeness": confound_audit,
        "confound_valid": bool(confound_audit.get("passed", True)),
        "multiverse_fraction_consistent": float(multiverse.fraction_consistent),
        "multiverse_specs": [item.to_dict() for item in multiverse.specs],
        "exec_only": _effect_significant(min_primary, min_contract),
        "+confound": bool(confound_pass),
        "+power": bool(power_pass),
        "+multiverse": bool(multiverse_pass),
        "+replication": bool(replication_pass),
        "final_label": verdict.label,
        "confirmation_subtype": verdict.confirmation_subtype,
        "heterogeneity_i2": verdict.heterogeneity_i2,
        "abstained": bool(verdict.label != "confirmed"),
        "rationale": verdict.rationale,
        "gate_state": verdict.gates,
        "power": power.to_dict(),
        "replication": replication.to_dict(),
        "contract": task.contract.model_dump(mode="json"),
    }


def materialize_negative_task(task: NegativeStressTask, data_root: Path) -> NegativeStressTask:
    """Persist current and subject-disjoint holdout pairs for claim-search replay."""

    data_root.mkdir(parents=True, exist_ok=True)
    prefix = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in task.claim_id)
    discovery_cohort = f"{prefix}_DISC"
    replication_cohort = f"{prefix}_REP"
    holdout_discovery_cohort = f"{prefix}_HOLDOUT_DISC"
    holdout_replication_cohort = f"{prefix}_HOLDOUT_REP"
    discovery = task.discovery.copy()
    replication = task.replication.copy()
    holdout_discovery = task.holdout_discovery.copy()
    holdout_replication = task.holdout_replication.copy()
    discovery["cohort"] = discovery_cohort
    replication["cohort"] = replication_cohort
    if not holdout_discovery.empty:
        holdout_discovery["cohort"] = holdout_discovery_cohort
    if not holdout_replication.empty:
        holdout_replication["cohort"] = holdout_replication_cohort

    contract_payload = task.contract.model_dump(mode="json")
    contract_payload["discovery_cohort"] = discovery_cohort
    contract_payload["replication_cohorts"] = [replication_cohort]
    contract = ClaimContract.model_validate(contract_payload)

    label_row = dict(task.label_row)
    label_row["cohorts"] = f"{discovery_cohort};{replication_cohort}"
    label_row["discovery_cohort"] = discovery_cohort
    label_row["replication_cohort"] = replication_cohort

    discovery.to_parquet(data_root / f"{discovery_cohort}.parquet", index=False)
    replication.to_parquet(data_root / f"{replication_cohort}.parquet", index=False)
    if not holdout_discovery.empty:
        holdout_discovery.to_parquet(data_root / f"{holdout_discovery_cohort}.parquet", index=False)
    if not holdout_replication.empty:
        holdout_replication.to_parquet(data_root / f"{holdout_replication_cohort}.parquet", index=False)

    total_rows = len(discovery) + len(replication) + len(holdout_discovery) + len(holdout_replication)
    target_family = infer_target_family(contract)
    source_path = Path(str(task.metadata.get("cohort_path") or "synthetic"))
    seed = int(task.metadata.get("seed") or 0)
    partition_specs = [
        (discovery_cohort, discovery, "discovery", "discovery", seed),
        (replication_cohort, replication, "replication", "replication", seed + 1),
    ]
    if not holdout_discovery.empty:
        partition_specs.append(
            (holdout_discovery_cohort, holdout_discovery, "holdout", "discovery", seed + 100_000)
        )
    if not holdout_replication.empty:
        partition_specs.append(
            (holdout_replication_cohort, holdout_replication, "holdout", "replication", seed + 100_001)
        )
    partition_records = [
        _synthetic_partition_record(
            partition_id=partition_id,
            base_dataset=prefix,
            target_family=target_family,
            role=role,
            evaluation_role=evaluation_role,
            path=data_root / f"{partition_id}.parquet",
            source_path=source_path,
            seed=partition_seed,
            source_row_count=total_rows,
            frame=frame,
        ).model_dump(mode="json")
        for partition_id, frame, role, evaluation_role, partition_seed in partition_specs
    ]
    metadata = dict(task.metadata)
    metadata["_evidence_partition_records"] = partition_records
    return replace(
        task,
        discovery=discovery,
        replication=replication,
        holdout_discovery=holdout_discovery,
        holdout_replication=holdout_replication,
        contract=contract,
        label_row=label_row,
        metadata=metadata,
    )


def _synthetic_partition_record(
    *,
    partition_id: str,
    base_dataset: str,
    target_family: str,
    role: str,
    evaluation_role: str,
    path: Path,
    source_path: Path,
    seed: int,
    source_row_count: int,
    frame: pd.DataFrame,
) -> EvidencePartitionRecord:
    subject_ids = sorted(frame["subject_id"].astype(str).tolist())
    columns = list(map(str, frame.columns))
    idp_columns = [column for column in columns if column.startswith(("smri_", "pet_", "fc_"))]
    if idp_columns and all(column.startswith("smri_") for column in idp_columns):
        modality = "sMRI"
        feature_families = ["regional_volume"]
    else:
        modality = "fMRI"
        feature_families = ["fc"]
    group_levels = {
        column: sorted(frame[column].dropna().astype(str).unique().tolist())
        for column in ("bench_group", "sex")
        if column in frame.columns
    }
    return EvidencePartitionRecord(
        partition_id=partition_id,
        base_dataset=base_dataset,
        target_family=target_family,
        role=role,
        evaluation_role=evaluation_role,
        path=str(path.resolve()),
        source_path=str(source_path),
        split_method="synthetic_subject_disjoint",
        seed=seed,
        n_rows=len(frame),
        site_count=int(frame["site"].dropna().astype(str).nunique()) if "site" in frame.columns else 0,
        exclusion_role="excluded_evaluation" if role == "holdout" else "claim_source",
        subject_id_sha256=hashlib.sha256("\n".join(subject_ids).encode("utf-8")).hexdigest(),
        source_row_count=source_row_count,
        meets_minimum=len(frame) >= 20,
        notes=[] if len(frame) >= 20 else [f"partition rows {len(frame)} below minimum 20"],
        columns=columns,
        schema_sha256=hashlib.sha256("\n".join(sorted(columns)).encode("utf-8")).hexdigest(),
        content_sha256=_file_content_sha256(path),
        modality=modality,
        feature_families=feature_families,
        units={},
        group_levels=group_levels,
    )


def _file_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_counts(df: pd.DataFrame, contract: ClaimContract) -> dict[str, int]:
    if contract.estimand.group is None:
        return {}
    group_col = contract.estimand.group.var
    if group_col not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[group_col].value_counts(dropna=False).items()}


def _family_fcr(rows: list[dict[str, Any]], rung: str = "+replication") -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for family, family_rows in sorted(by_family.items()):
        denominator = len(family_rows)
        count = int(sum(bool(row.get(rung, False)) for row in family_rows))
        ci = exact_binomial_ci(count, denominator)
        out[family] = {
            "FCR": float(count / denominator) if denominator else math.nan,
            "FCR_count": count,
            "FCR_denominator": denominator,
            "FCR_ci95_exact": list(ci),
        }
    return out


def _false_confirms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if bool(row.get("+replication", False)):
            out.append(
                {
                    "claim_id": row["claim_id"],
                    "family": row["family"],
                    "expected_gate": row["expected_gate"],
                    "final_label": row["final_label"],
                    "best_p": row["best_p"],
                    "replication_reason": (row.get("replication") or {}).get("reason"),
                    "rationale": row["rationale"],
                }
            )
    return out


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    tasks, generation_skipped = generate_negative_stress_tasks(
        args.root,
        fishing_feature_limit=args.fishing_feature_limit,
        underpowered_cohort_limit=args.underpowered_cohort_limit,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    partition_records: list[dict[str, Any]] = []
    materialize_root = Path(args.materialize_data_root) if args.materialize_data_root else None
    for i, task in enumerate(tasks, start=1):
        try:
            if materialize_root is not None:
                task = materialize_negative_task(task, materialize_root)
                partition_records.extend(task.metadata.get("_evidence_partition_records", []))
            row = evaluate_negative_task(task)
            rows.append(row)
            print(f"[ok {i:03d}/{len(tasks):03d}] {row['claim_id']} final={row['final_label']} +replication={row['+replication']}")
        except Exception as exc:
            errors.append({"claim_id": task.claim_id, "family": task.family, "error": str(exc)})
            print(f"[error {i:03d}/{len(tasks):03d}] {task.claim_id}: {exc}")

    metrics = summarize_rows(rows, RUNGS)
    false_confirms = _false_confirms(rows)
    evidence_manifest_path = None
    if materialize_root is not None:
        manifest = EvidencePartitionManifest(
            seed=20260615,
            records=[EvidencePartitionRecord.model_validate(record) for record in partition_records],
        )
        evidence_manifest_path = materialize_root.parent / "manifest.json"
        evidence_manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": {
            "root": str(args.root),
            "out_dir": str(args.out_dir),
            "limit": args.limit,
            "fishing_feature_limit": args.fishing_feature_limit,
            "underpowered_cohort_limit": args.underpowered_cohort_limit,
            "materialize_data_root": str(materialize_root) if materialize_root is not None else None,
            "evidence_manifest": str(evidence_manifest_path) if evidence_manifest_path is not None else None,
        },
        "n_generated": len(rows),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "metrics": metrics,
        "per_family_FCR": _family_fcr(rows),
        "false_confirms": false_confirms,
        "claims": rows,
        "errors": errors,
        "generation_skipped": generation_skipped,
    }


def write_outputs(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "known_negative_results.json"
    csv_path = out_dir / "known_negative_claims.csv"
    audit_path = out_dir / "known_negative_audit.csv"
    report_path = out_dir / "known_negative_report.md"

    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    pd.DataFrame(payload["claims"]).to_csv(csv_path, index=False)
    pd.DataFrame(_audit_rows(payload["claims"])).to_csv(audit_path, index=False)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "audit": str(audit_path),
        "report": str(report_path),
    }


def _audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit = []
    for row in rows:
        rep = row.get("replication") or {}
        audit.append(
            {
                "claim_id": row["claim_id"],
                "family": row["family"],
                "expected_gate": row["expected_gate"],
                "scoring_label": row["scoring_label"],
                "label_authority": row["label_authority"],
                "final_label": row["final_label"],
                "exec_only": row["exec_only"],
                "+confound": row["+confound"],
                "+power": row["+power"],
                "+multiverse": row["+multiverse"],
                "+replication": row["+replication"],
                "n_discovery": row["n_discovery"],
                "n_replication": row["n_replication"],
                "n_features": row["n_features"],
                "outcome": row["outcome"],
                "best_beta": row["best_beta"],
                "best_p": row["best_p"],
                "best_standardized_effect": row["best_standardized_effect"],
                "achieved_power": (row.get("power") or {}).get("achieved_power"),
                "under_powered": (row.get("power") or {}).get("under_powered"),
                "multiverse_fraction_consistent": row["multiverse_fraction_consistent"],
                "replication_passed": rep.get("passed"),
                "replication_reason": rep.get("reason"),
                "search_family_size": (row.get("search_provenance") or {}).get("family_size"),
                "search_selection": (row.get("search_provenance") or {}).get("selection"),
                "confound_completeness_passed": (row.get("confound_completeness") or {}).get("passed"),
                "confound_completeness_reason": (row.get("confound_completeness") or {}).get("reason"),
                "synthetic_metadata": json.dumps(_json_safe(row.get("synthetic_metadata")), sort_keys=True),
                "rationale": row["rationale"],
            }
        )
    return audit


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _render_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    full = metrics["summary_full"]["+replication"]
    main = metrics["summary_main"]["+replication"]
    lines = [
        "# Negatives Expansion FCR Report",
        "",
        f"Generated negatives scored: {payload['n_generated']}",
        "",
        "## Per-family counts",
    ]
    for family, count in sorted(payload["family_counts"].items()):
        lines.append(f"- {family}: {count}")
    lines.extend(
        [
            "",
            "## Full-gate FCR",
            f"- Full set: {full['FCR_count']}/{full['FCR_denominator']} = {full['FCR']:.4f}, exact 95% CI {_fmt_ci(full['FCR_ci95_exact'])}",
            f"- MAIN labels: {main['FCR_count']}/{main['FCR_denominator']} = {main['FCR']:.4f}, exact 95% CI {_fmt_ci(main['FCR_ci95_exact'])}",
            "",
            "## Per-family FCR",
        ]
    )
    for family, stats in sorted(payload["per_family_FCR"].items()):
        lines.append(
            f"- {family}: {stats['FCR_count']}/{stats['FCR_denominator']} = {stats['FCR']:.4f}, "
            f"exact 95% CI {_fmt_ci(stats['FCR_ci95_exact'])}"
        )
    lines.extend(["", "## False confirms"])
    if payload["false_confirms"]:
        for item in payload["false_confirms"]:
            lines.append(
                f"- {item['claim_id']} ({item['family']}): expected catch gate = {item['expected_gate']}; "
                f"replication_reason={item['replication_reason']}; best_p={item['best_p']:.3g}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Ladder FCR"])
    for rung in RUNGS:
        row = metrics["summary_full"][rung]
        lines.append(
            f"- {rung}: {row['FCR_count']}/{row['FCR_denominator']} = {row['FCR']:.4f}, "
            f"exact 95% CI {_fmt_ci(row['FCR_ci95_exact'])}"
        )
    if payload["errors"] or payload["generation_skipped"]:
        lines.extend(["", "## Skips and errors"])
        for item in payload["generation_skipped"]:
            lines.append(f"- generation skip {item.get('builder')}: {item.get('reason')}")
        for item in payload["errors"]:
            lines.append(f"- execution error {item.get('claim_id')}: {item.get('error')}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fishing-feature-limit", type=int, default=24)
    parser.add_argument("--underpowered-cohort-limit", type=int, default=7)
    parser.add_argument(
        "--materialize-data-root",
        default=None,
        help="Optional directory for unique synthetic discovery/replication parquet files used by claim-search replay.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(args)
    paths = write_outputs(payload, Path(args.out_dir))
    print(f"wrote {paths['json']}")
    print(f"wrote {paths['csv']}")
    print(f"wrote {paths['audit']}")
    print(f"wrote {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
