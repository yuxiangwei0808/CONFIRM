"""Build legacy claim-source CSVs from the old label table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "claim_id",
    "target_family",
    "source_mode",
    "question",
    "label_class",
    "label_basis",
    "source_citation",
    "notes",
    "include_in_main",
]


def _target_family(row: dict[str, str]) -> str:
    claim_id = row["claim_id"].lower()
    phenotype = row["phenotype"].lower()
    modality = row["modality"].lower()
    if "adhd" in claim_id or "adhd" in phenotype:
        return "adhd"
    if "asd" in claim_id or "autism" in phenotype:
        return "asd"
    if claim_id.startswith("sz_") or "schizophrenia" in phenotype or "psychosis" in modality:
        return "psychosis"
    if (
        claim_id.startswith(("ad_", "nacc_", "aibl_", "miriad_", "fdg_"))
        or "dementia" in phenotype
        or "alzheimer" in phenotype
        or "brain aging" in phenotype
    ):
        return "ad_aging"
    return "normative_fmri"


def _source_mode(row: dict[str, str]) -> str:
    if row["label_basis"] == "synthetic_stress":
        return "synthetic_stress"
    if row["label_basis"] in {"canonical_literature", "meta_analysis", "large_cohort_replication"}:
        return "literature"
    return "inventory"


def _question(row: dict[str, str]) -> str:
    discovery = row["discovery_cohort"]
    replication = row["replication_cohort"]
    covariates = row["confound_set"]
    covariate_text = covariates if "available" in covariates.lower() else f"{covariates} when available"
    direction = row["expected_direction"]
    phenotype = row["phenotype"]
    modality = row["modality"]
    if replication and replication.lower() != "none":
        evidence = f"using {discovery} as discovery and {replication} as replication"
    else:
        evidence = f"using {discovery} as discovery"
    return (
        f"For {phenotype} in {modality}, {evidence}, test the claim with expected direction "
        f"{direction} and adjust for {covariate_text}."
    )


def _convert(row: dict[str, str]) -> dict[str, str]:
    mode = _source_mode(row)
    return {
        "claim_id": row["claim_id"],
        "target_family": _target_family(row),
        "source_mode": mode,
        "question": _question(row),
        "label_class": row["label_class"],
        "label_basis": row["label_basis"],
        "source_citation": row["source_citation"],
        "notes": row["construct_validity_notes"],
        "include_in_main": "false" if mode == "synthetic_stress" else "true",
    }


def run(args: argparse.Namespace) -> None:
    src = Path(args.label_table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as handle:
        rows = [{key: str(value).strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    converted = [_convert(row) for row in rows]
    fixed = [row for row in converted if row["source_mode"] != "synthetic_stress"]
    synthetic = [row for row in converted if row["source_mode"] == "synthetic_stress"]
    for path, payload in [
        (out_dir / "legacy_inventory_claims.csv", fixed),
        (out_dir / "synthetic_stress_claims.csv", synthetic),
    ]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(payload)
        print(f"wrote {path} ({len(payload)} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-table", default="data/labels/claim_label_table.csv")
    parser.add_argument("--out-dir", default="data/claims")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
