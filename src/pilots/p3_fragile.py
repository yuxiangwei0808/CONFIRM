"""Pilot 3: small single-site ASD case-control should be fragile or non-replicated."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from confirm.agent import run_claim
from confirm.ingest.abide import AbideAdapter
from confirm.schema import idp_columns, validate_canonical


def _ensure_abide(data_dir: Path) -> Path:
    path = data_dir / "ABIDE.parquet"
    if not path.exists():
        AbideAdapter(data_dir=data_dir.parent / "raw" / "abide").write(data_dir)
    return path


def _choose_site(df: pd.DataFrame) -> str:
    candidates = []
    for site, group in df.groupby("site"):
        counts = group["dx"].value_counts()
        if counts.get("ASD", 0) >= 5 and counts.get("TC", 0) >= 5:
            candidates.append((len(group), str(site)))
    if not candidates:
        raise ValueError("No ABIDE site has at least 5 ASD and 5 TC rows")
    return sorted(candidates)[0][1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/canonical")
    parser.add_argument("--out", default="runs/p3_fragile")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    pilot_data = out_dir / "canonical"
    pilot_data.mkdir(parents=True, exist_ok=True)
    source = validate_canonical(pd.read_parquet(_ensure_abide(data_dir)))
    source = source[source["dx"].isin(["ASD", "TC"])].copy()
    outcome = "smri_meanthickness" if "smri_meanthickness" in source.columns else idp_columns(source.columns)[0]
    site = _choose_site(source)
    disc = source[source["site"].astype(str).eq(site)].copy()
    rep = source[~source["site"].astype(str).eq(site)].copy()
    disc["cohort"] = "P3_DISC"
    rep["cohort"] = "P3_REP"
    covars = ["age", "sex"]
    if "eTIV" in source.columns and source["eTIV"].notna().mean() > 0.8:
        covars.append("eTIV")
    contract = {
        "claim_id": "asd_single_site_fragile",
        "question": "Does ASD differ from controls on the selected neuroimaging phenotype in one small site?",
        "estimand": {
            "type": "group_diff",
            "outcome": outcome,
            "predictor": "dx",
            "group": {"var": "dx", "case": "ASD", "control": "TC"},
            "direction": "two_sided",
        },
        "covariates": covars,
        "inclusion": None,
        "discovery_cohort": "P3_DISC",
        "replication_cohorts": ["P3_REP"],
        "gates": {
            "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
            "confound": {"require_covariates": ["age", "sex"], "motion_check": False},
            "power": {"min_power": 0.8, "ref_effect": None},
            "multiverse": {"min_fraction_consistent": 0.6},
            "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "combat"},
        },
        "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
    }
    validate_canonical(disc).to_parquet(pilot_data / "P3_DISC.parquet", index=False)
    validate_canonical(rep).to_parquet(pilot_data / "P3_REP.parquet", index=False)
    contract_path = out_dir / "p3_contract.yaml"
    with contract_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(contract, handle, sort_keys=False)
    verdict = run_claim(contract_path, pilot_data, out_dir, command=sys.argv)
    print(f"expected=fragile|non_replicated actual={verdict.label}")
    return 0 if verdict.label in {"fragile", "non_replicated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
