"""Run label-aware multimodal benchmark adapters over staged local tables.

This runner adds the first non-fMRI-descriptor benchmark layer:

- NACC MRI + CSF tables: aging and CSF/MRI association claims.

It deliberately avoids disease claims where diagnosis mappings are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bench.labels import claim_label_for_claim, label_provenance, scoring_bucket
from bench.run_benchmark_ready import (
    RUNGS,
    _any_region_significant,
    _best_effect,
    _complete_required_rows,
    _confound_violation,
    _contract_with_covariates,
    _json_safe,
    summarize,
)
from confirm.brainwide import run_brainwide
from confirm.contract import (
    ClaimContract,
    ConfoundGate,
    Estimand,
    Gates,
    GroupSpec,
    MultiplicityGate,
    MultiverseGate,
    PowerGate,
    ReplicationGate,
)
from confirm.multiverse import run_brainwide_multiverse
from confirm.power import power_check
from confirm.replication import replicate_brainwide
from confirm.verdict import decide_brainwide

SEED = 20260616


@dataclass(frozen=True)
class Task:
    claim_id: str
    modality: str
    scoring_label: str
    notes: str
    discovery: pd.DataFrame
    replication: pd.DataFrame
    outcomes: list[str]
    estimand_type: str
    predictor: str
    group: GroupSpec | None
    direction: str
    covariates_full: list[str]
    covariates_min: list[str]
    ref_effect: float | None = None


def _read_misc(root: Path, filename: str) -> pd.DataFrame:
    return pd.read_parquet(root / "misc_tables" / filename)


def _clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    sentinels = [88.8888, 888.8888, 8888.8888, 99.9999, 999.9999, 9999.9999]
    for value in sentinels:
        out = out.mask(np.isclose(out, value, rtol=0, atol=1e-3))
    out = out.mask(out.isin([-4, -9, 888, 999, 8888, 9999]))
    return out


def _split(df: pd.DataFrame, seed: int, discovery: str, replication: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df))
    np.random.default_rng(seed).shuffle(idx)
    half = len(idx) // 2
    disc = df.iloc[idx[:half]].copy()
    rep = df.iloc[idx[half:]].copy()
    disc["cohort"] = discovery
    rep["cohort"] = replication
    return disc, rep


def _limit_features(cols: list[str], limit: int | None) -> list[str]:
    if limit and limit > 0:
        return cols[:limit]
    return cols


def _load_nacc_mri(root: Path) -> pd.DataFrame:
    mri = _read_misc(root, "data_qneuromark_Data_NACC_data_investigator_mri_nacc65.parquet")
    feature_map = {
        "NACCBRNV": "smri_nacc_brain_volume",
        "GRAYVOL": "smri_nacc_gray_volume",
        "WHITEVOL": "smri_nacc_white_volume",
        "HIPPOVOL": "smri_nacc_hippocampus_total",
        "LHIPPO": "smri_nacc_hippocampus_lh",
        "RHIPPO": "smri_nacc_hippocampus_rh",
        "FRCORT": "smri_nacc_frontal_cortex",
        "TEMPCOR": "smri_nacc_temporal_cortex",
        "PARCORT": "smri_nacc_parietal_cortex",
        "OCCCORT": "smri_nacc_occipital_cortex",
    }
    out = pd.DataFrame(
        {
            "subject_id": mri["NACCID"].astype(str),
            "session": "ses-01",
            "cohort": "NACC",
            "site": mri["NACCADC"].astype(str),
            "age": _clean_numeric(mri["NACCMRIA"]),
            "eTIV": _clean_numeric(mri["NACCICV"]),
        }
    )
    for src, dst in feature_map.items():
        out[dst] = _clean_numeric(mri[src])
    out = out.sort_values(["subject_id", "age"]).groupby("subject_id", as_index=False).first()
    return out


def _load_nacc_csf(root: Path) -> pd.DataFrame:
    csf = _read_misc(root, "data_qneuromark_Data_NACC_data_investigator_fcsf_nacc65.parquet")
    out = pd.DataFrame(
        {
            "subject_id": csf["NACCID"].astype(str),
            "csf_abeta": _clean_numeric(csf["CSFABETA"]),
            "csf_ptau": _clean_numeric(csf["CSFPTAU"]),
            "csf_ttau": _clean_numeric(csf["CSFTTAU"]),
        }
    )
    out = out.groupby("subject_id", as_index=False).median(numeric_only=True)
    for col in ["csf_abeta", "csf_ptau", "csf_ttau"]:
        out[f"{col}_log"] = np.log(out[col].where(out[col] > 0))
    return out


def _nacc_tasks(root: Path, feature_limit: int | None, seed: int) -> list[Task]:
    mri = _load_nacc_mri(root)
    outcomes = _limit_features([col for col in mri.columns if col.startswith("smri_nacc_")], feature_limit)
    disc, rep = _split(mri.dropna(subset=["age"]), seed, "NACC_DISC", "NACC_REP")
    tasks = [
        Task(
            claim_id="nacc_age_smri_split",
            modality="NACC/sMRI",
            scoring_label="positive_stability",
            notes="NACC MRI regional volume aging association; diagnosis labels not required.",
            discovery=disc,
            replication=rep,
            outcomes=outcomes,
            estimand_type="association",
            predictor="age",
            group=None,
            direction="negative",
            covariates_full=["site", "eTIV"],
            covariates_min=["eTIV"],
            ref_effect=0.1,
        )
    ]

    csf = _load_nacc_csf(root)
    joined = mri.merge(csf, on="subject_id", how="inner")
    for predictor, direction in [("csf_ptau_log", "negative"), ("csf_ttau_log", "negative"), ("csf_abeta_log", "positive")]:
        data = joined.dropna(subset=[predictor]).copy()
        if len(data) < 80:
            continue
        disc, rep = _split(data, seed + len(tasks), "NACC_CSF_DISC", "NACC_CSF_REP")
        tasks.append(
            Task(
                claim_id=f"nacc_{predictor.replace('_log', '')}_smri",
                modality="NACC/sMRI+CSF",
                scoring_label="small_positive_candidate",
                notes=f"NACC CSF biomarker association with MRI regional volumes using {predictor}.",
                discovery=disc,
                replication=rep,
                outcomes=outcomes,
                estimand_type="association",
                predictor=predictor,
                group=None,
                direction=direction,
                covariates_full=["age", "site", "eTIV"],
                covariates_min=["age", "eTIV"],
                ref_effect=0.1,
            )
        )
    return tasks


def _build_contract(task: Task, harmonize: str) -> ClaimContract:
    estimand = Estimand(
        type=task.estimand_type,  # type: ignore[arg-type]
        outcome=task.outcomes,
        predictor=task.predictor,
        group=task.group,
        direction=task.direction,  # type: ignore[arg-type]
        unit="brainwide",
        region_set=task.modality,
    )
    gates = Gates(
        multiplicity=MultiplicityGate(method="fdr_bh", alpha=0.05, family_size=max(1, len(task.outcomes))),
        confound=ConfoundGate(require_covariates=task.covariates_full, motion_check=False),
        power=PowerGate(min_power=0.8, ref_effect=task.ref_effect),
        multiverse=MultiverseGate(min_fraction_consistent=0.6),
        replication=ReplicationGate(
            alpha=0.05,
            require_same_sign=True,
            require_ci_overlap=False,
            harmonize=harmonize,  # type: ignore[arg-type]
            pattern_corr_min=0.25,
            region_replication_frac_min=0.05,
            dice_min=0.0,
        ),
    )
    return ClaimContract(
        claim_id=task.claim_id,
        question=task.notes,
        estimand=estimand,
        covariates=task.covariates_full,
        inclusion=None,
        discovery_cohort=str(task.discovery["cohort"].iloc[0]),
        replication_cohorts=[str(task.replication["cohort"].iloc[0])],
        gates=gates,
        reporting_language_allowed=["confirmed", "non_replicated", "under_powered", "fragile"],
    )


def _label_counts(df: pd.DataFrame, group: GroupSpec | None) -> dict[str, int]:
    if group is None or group.var not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[group.var].value_counts(dropna=False).items()}


def evaluate_task(task: Task, harmonize: str) -> dict[str, Any]:
    contract = _build_contract(task, harmonize)
    min_contract = _contract_with_covariates(contract, task.covariates_min)
    disc = _complete_required_rows(task.discovery, task.outcomes, task)  # type: ignore[arg-type]
    rep = _complete_required_rows(task.replication, task.outcomes, task)  # type: ignore[arg-type]
    if len(disc) < 20 or len(rep) < 20:
        raise ValueError(f"Too few complete rows after filtering: discovery={len(disc)}, replication={len(rep)}")

    min_regions = run_brainwide(disc, min_contract)
    full_regions = run_brainwide(disc, contract)
    power = power_check(_best_effect(full_regions), contract)
    multiverse = run_brainwide_multiverse(disc, full_regions, contract, min_covariates=task.covariates_min)
    replication = replicate_brainwide(full_regions, disc, [rep], contract)
    verdict = decide_brainwide(full_regions, multiverse, power, replication, contract)

    confound_valid = not _confound_violation(disc, task)  # type: ignore[arg-type]
    exec_only = _any_region_significant(min_regions)
    confound = _any_region_significant(full_regions) and confound_valid
    power_pass = confound and not power.under_powered
    multiverse_pass = power_pass and multiverse.passed
    replication_pass = multiverse_pass and replication.passed
    final_label = verdict.label
    rationale = verdict.rationale
    if not confound_valid:
        final_label = "fragile"
        rationale = "Failed gates: confound; predictor is nested in a declared confound."
    best = _best_effect(full_regions)
    label_row = claim_label_for_claim({"claim_id": task.claim_id}) or {}
    scoring_label = label_row.get("label_class", task.scoring_label)
    bucket = scoring_bucket(scoring_label)

    return {
        "claim_id": task.claim_id,
        "modality": task.modality,
        "claim_type": "multimodal_adapter",
        "ground_truth": scoring_label,
        "scoring_label": scoring_label,
        "scoring_bucket": bucket,
        "label_provenance": label_provenance(scoring_label),
        "label_basis": label_row.get("label_basis"),
        "adjudication_status": label_row.get("adjudication_status"),
        "label_authority": label_row.get("label_authority", "supplementary"),
        "label_confidence": label_row.get("label_confidence"),
        "source_citation": label_row.get("source_citation"),
        "label_metadata": label_row,
        "expected_label": scoring_label,
        "outcome_family": task.modality,
        "source_kind": task.modality,
        "discovery_cohort": str(disc["cohort"].iloc[0]),
        "replication_cohort": str(rep["cohort"].iloc[0]),
        "n_discovery": int(len(disc)),
        "n_replication": int(len(rep)),
        "label_counts_discovery": _label_counts(disc, task.group),
        "label_counts_replication": _label_counts(rep, task.group),
        "n_features": int(len(task.outcomes)),
        "covariates_full": task.covariates_full,
        "covariates_min": task.covariates_min,
        "best_region": next(region.region for region in sorted(full_regions.regions, key=lambda item: item.effect.p)),
        "best_beta": float(best.beta),
        "best_p": float(best.p),
        "best_standardized_effect": float(best.standardized_effect),
        "primary_effect": best.to_dict(),
        "primary_region_table": full_regions.to_dict(),
        "primary_sig_regions": int(sum(region.significant for region in full_regions.regions)),
        "confound_valid": bool(confound_valid),
        "multiverse_fraction_consistent": float(multiverse.fraction_consistent),
        "multiverse_specs": [item.to_dict() for item in multiverse.specs],
        "exec_only": bool(exec_only),
        "+confound": bool(confound),
        "+power": bool(power_pass),
        "+multiverse": bool(multiverse_pass),
        "+replication": bool(replication_pass),
        "final_label": final_label,
        "abstained": bool(final_label != "confirmed"),
        "rationale": rationale,
        "power": power.to_dict(),
        "replication": replication.to_dict(),
        "contract": contract.model_dump(mode="json"),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _lockfile(root: Path, claims: list[dict[str, Any]], skipped: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    source_files = [
        root / "misc_table_manifest.csv",
        Path("data/labels/claim_label_table.csv"),
        Path("docs/data_manifests/remote_peek_20260616.md"),
        Path("docs/literature_labels/fmri_claim_label_ledger.md"),
    ]
    code_files = [
        Path("src/bench/labels.py"),
        Path("src/bench/metrics.py"),
        Path("src/bench/run_multimodal_benchmark.py"),
        Path("src/confirm/brainwide.py"),
        Path("src/confirm/multiverse.py"),
        Path("src/confirm/replication.py"),
        Path("src/confirm/power.py"),
        Path("src/confirm/verdict.py"),
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "harmonize": args.harmonize,
        "feature_limit": args.feature_limit,
        "source_files": [{"path": str(path), "sha256": _file_sha256(path)} for path in source_files if path.exists()],
        "code_files": [{"path": str(path), "sha256": _file_sha256(path)} for path in code_files if path.exists()],
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "scoring_label": claim["scoring_label"],
                "scoring_bucket": claim["scoring_bucket"],
                "label_authority": claim.get("label_authority"),
                "label_provenance": claim["label_provenance"],
                "modality": claim["modality"],
                "discovery_cohort": claim["discovery_cohort"],
                "replication_cohort": claim["replication_cohort"],
                "n_features": claim["n_features"],
                "covariates_full": claim["covariates_full"],
                "covariates_min": claim["covariates_min"],
            }
            for claim in claims
        ],
        "skipped": skipped,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"multimodal_benchmark_results_{timestamp}.json"
    csv_path = out_dir / f"multimodal_benchmark_claims_{timestamp}.csv"
    audit_path = out_dir / f"multimodal_benchmark_audit_{timestamp}.csv"
    risk_path = out_dir / f"multimodal_benchmark_risk_coverage_{timestamp}.csv"
    lock_path = out_dir / f"multimodal_benchmark_lockfile_{timestamp}.json"
    latest_json = out_dir / "multimodal_benchmark_results.json"
    latest_csv = out_dir / "multimodal_benchmark_claims.csv"
    latest_audit = out_dir / "multimodal_benchmark_audit.csv"
    latest_risk = out_dir / "multimodal_benchmark_risk_coverage.csv"
    latest_lock = out_dir / "multimodal_benchmark_lockfile.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    pd.DataFrame(payload["claims"]).to_csv(csv_path, index=False)
    audit_cols = [
        "claim_id",
        "modality",
        "scoring_label",
        "scoring_bucket",
        "label_authority",
        "final_label",
        *RUNGS,
        "n_discovery",
        "n_replication",
        "n_features",
        "best_region",
        "best_beta",
        "best_p",
        "best_standardized_effect",
        "multiverse_fraction_consistent",
        "rationale",
    ]
    pd.DataFrame(payload["claims"])[audit_cols].to_csv(audit_path, index=False)
    pd.DataFrame(payload.get("risk_coverage", [])).to_csv(risk_path, index=False)
    lock_path.write_text(json.dumps(_json_safe(payload["lockfile"]), indent=2), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(csv_path, latest_csv)
    shutil.copyfile(audit_path, latest_audit)
    shutil.copyfile(risk_path, latest_risk)
    shutil.copyfile(lock_path, latest_lock)
    return json_path, csv_path, latest_json


def build_tasks(root: Path, feature_limit: int | None, seed: int) -> list[Task]:
    return _nacc_tasks(root, feature_limit, seed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.data_root)
    tasks = build_tasks(root, args.feature_limit, args.seed)
    if args.claim_id:
        wanted = set(args.claim_id)
        tasks = [task for task in tasks if task.claim_id in wanted]
    if args.limit:
        tasks = tasks[: args.limit]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for task in tasks:
        try:
            if not task.outcomes:
                skipped.append({"claim_id": task.claim_id, "reason": "no_feature_columns"})
                print(f"[skip] {task.claim_id}: no feature columns")
                continue
            result = evaluate_task(task, args.harmonize)
            rows.append(result)
            print(f"[ok] {result['claim_id']} final={result['final_label']} features={result['n_features']}")
        except Exception as exc:
            errors.append({"claim_id": task.claim_id, "error": str(exc)})
            print(f"[error] {task.claim_id}: {exc}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": {
            "data_root": str(root),
            "feature_limit": args.feature_limit,
            "limit": args.limit,
            "claim_id": args.claim_id,
            "seed": args.seed,
            "harmonize": args.harmonize,
        },
        **summarize(rows),
        "claims": rows,
        "errors": errors,
        "skipped": skipped,
    }
    payload["lockfile"] = _lockfile(root, rows, skipped, args)
    json_path, csv_path, latest_json = write_outputs(Path(args.out_dir), payload)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"latest {latest_json}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/prepared_data/benchmark_ready")
    parser.add_argument("--out-dir", default="review-stage/multimodal-label-aware-combat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--claim-id", action="append", default=None)
    parser.add_argument("--feature-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--harmonize", choices=["none", "combat"], default="combat")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
