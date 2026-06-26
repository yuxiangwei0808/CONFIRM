"""Build the benchmark-ready data layer from prepared local tables.

Inputs:
  - data/prepared_data/fmri_descriptors/*.parquet
  - data/prepared_data/misc_tables/*.parquet
  - docs/data_manifests/remote_benchmark_claim_inventory.csv

Outputs:
  - data/prepared_data/benchmark_ready/cohorts/*.parquet
  - data/prepared_data/benchmark_ready/misc_tables/*.parquet
  - data/prepared_data/benchmark_ready/feature_dictionary.csv
  - data/prepared_data/benchmark_ready/cohort_manifest.csv
  - data/prepared_data/benchmark_ready/claim_inventory_ready.csv
  - data/prepared_data/benchmark_ready/README.md

This script only cleans/recreates the generated benchmark_ready directory.
It never modifies copied raw files, remote data, or experiment outputs.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PREPARED = Path("data/prepared_data")
SOURCE_FMRI = PREPARED / "fmri_descriptors"
SOURCE_MISC = PREPARED / "misc_tables"
OUT = PREPARED / "benchmark_ready"
COHORTS_OUT = OUT / "cohorts"
MISC_OUT = OUT / "misc_tables"
CLAIM_INVENTORY = Path("docs/data_manifests/remote_benchmark_claim_inventory.csv")

ID_COLUMNS = ["subject_id", "session", "cohort", "site", "age", "sex", "dx"]
FEATURE_SOURCE_KINDS = {"fc_self_descriptors", "region_self_descriptors", "ica_dyno_descriptors"}


@dataclass(frozen=True)
class CohortSummary:
    cohort: str
    path: str
    rows: int
    columns: int
    feature_columns: int
    rows_with_any_feature: int
    age_nonmissing: int
    sex_nonmissing: int
    dx_nonmissing: int
    modalities: str


def clean_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    COHORTS_OUT.mkdir(parents=True, exist_ok=True)
    MISC_OUT.mkdir(parents=True, exist_ok=True)


def normalize_sex_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if text in {"M", "MALE", "1", "1.0", "TRUE"}:
        return "M"
    if text in {"F", "FEMALE", "0", "0.0", "2", "2.0", "FALSE"}:
        return "F"
    return str(value).strip()


def normalize_canonical_columns(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    out = df.copy()
    for col in ID_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["subject_id"] = out["subject_id"].astype(str)
    out["session"] = out["session"].astype(str)
    out["cohort"] = cohort
    out["site"] = out["site"].fillna(cohort).astype(str)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["sex"] = out["sex"].map(normalize_sex_value).astype("string")
    out["dx"] = out["dx"].astype("string")
    return out


def feature_family(column: str) -> tuple[str, str, str]:
    """Return (modality, family, source_kind) for a feature column."""
    if column.startswith("fc_fc_"):
        return "fMRI", "FC-network", "fc_self_descriptors"
    if column.startswith("raw_fmri_region_"):
        return "fMRI", "region", "region_self_descriptors"
    if column.startswith("raw_fmri_dyno_"):
        return "fMRI", "dynamics", "ica_dyno_descriptors"
    if column.startswith("raw_fmri_"):
        return "fMRI", "other", "descriptor"
    if column.startswith("fc_"):
        return "fMRI", "FC", "descriptor"
    if column.startswith("smri_"):
        return "sMRI", "regional", "canonical"
    if column.startswith("pet_"):
        return "PET", "regional", "canonical"
    if column.startswith("phen_"):
        return "phenotype", "covariate_or_label", "metadata"
    return "unknown", "unknown", "unknown"


def pretty_feature_name(column: str) -> str:
    for prefix in ["fc_fc_", "raw_fmri_region_", "raw_fmri_dyno_", "raw_fmri_", "fc_", "phen_"]:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def prepare_cohort(path: Path) -> tuple[pd.DataFrame, list[dict[str, object]], CohortSummary]:
    cohort = path.stem
    df = pd.read_parquet(path)
    df = normalize_canonical_columns(df, cohort)

    feature_cols = [
        col
        for col in df.columns
        if col not in ID_COLUMNS and col.startswith(("fc_", "raw_fmri_", "smri_", "pet_"))
    ]
    phenotype_cols = [col for col in df.columns if col.startswith("phen_")]

    ordered_cols = ID_COLUMNS + phenotype_cols + feature_cols
    out = df[ordered_cols].copy()

    feature_rows = []
    for col in feature_cols + phenotype_cols:
        modality, family, source_kind = feature_family(col)
        values = pd.to_numeric(out[col], errors="coerce") if col not in phenotype_cols else out[col]
        nonmissing = int(values.notna().sum())
        numeric = pd.to_numeric(out[col], errors="coerce")
        feature_rows.append(
            {
                "cohort": cohort,
                "column": col,
                "feature_name": pretty_feature_name(col),
                "modality": modality,
                "family": family,
                "source_kind": source_kind,
                "is_feature": col in feature_cols,
                "is_phenotype": col in phenotype_cols,
                "nonmissing": nonmissing,
                "missing_fraction": float(1.0 - nonmissing / max(len(out), 1)),
                "mean": float(numeric.mean()) if numeric.notna().any() and col in feature_cols else pd.NA,
                "std": float(numeric.std()) if numeric.notna().any() and col in feature_cols else pd.NA,
            }
        )

    rows_with_any_feature = int(out[feature_cols].notna().any(axis=1).sum()) if feature_cols else 0
    modalities = sorted({feature_family(col)[0] for col in feature_cols})
    summary = CohortSummary(
        cohort=cohort,
        path=str(COHORTS_OUT / f"{cohort}.parquet"),
        rows=len(out),
        columns=len(out.columns),
        feature_columns=len(feature_cols),
        rows_with_any_feature=rows_with_any_feature,
        age_nonmissing=int(out["age"].notna().sum()),
        sex_nonmissing=int(out["sex"].notna().sum()),
        dx_nonmissing=int(out["dx"].notna().sum()),
        modalities=";".join(modalities),
    )
    return out, feature_rows, summary


def copy_misc_tables() -> pd.DataFrame:
    rows = []
    if not SOURCE_MISC.exists():
        return pd.DataFrame(rows)
    for source in sorted(SOURCE_MISC.glob("*.parquet")):
        target = MISC_OUT / source.name
        shutil.copy2(source, target)
        try:
            df = pd.read_parquet(target)
            rows.append(
                {
                    "table": source.stem,
                    "path": str(target),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "columns_preview": "|".join(map(str, df.columns[:40])),
                }
            )
        except Exception as exc:
            rows.append({"table": source.stem, "path": str(target), "error": str(exc)})
    return pd.DataFrame(rows)


def write_claim_inventory(feature_dict: pd.DataFrame, cohort_manifest: pd.DataFrame) -> None:
    claims = pd.read_csv(CLAIM_INVENTORY)
    available_cohorts = set(cohort_manifest["cohort"])
    prepared_rows = []
    for _, row in claims.iterrows():
        discovery_ready = row["discovery_cohort"] in available_cohorts or row["prepared_status"].startswith("already")
        replication_ready = (
            row["replication_cohort"] in available_cohorts
            or row["replication_cohort"] == "ADNI split"
            or row["prepared_status"].startswith("already")
        )
        row = row.to_dict()
        source_kind = str(row.get("outcome_family", ""))
        shared_feature_count: int | None = None
        if (
            source_kind in FEATURE_SOURCE_KINDS
            and row["discovery_cohort"] in available_cohorts
            and row["replication_cohort"] in available_cohorts
        ):
            disc_features = set(
                feature_dict[
                    (feature_dict["cohort"] == row["discovery_cohort"])
                    & (feature_dict["source_kind"] == source_kind)
                    & (feature_dict["is_feature"].astype(bool))
                ]["column"].astype(str)
            )
            rep_features = set(
                feature_dict[
                    (feature_dict["cohort"] == row["replication_cohort"])
                    & (feature_dict["source_kind"] == source_kind)
                    & (feature_dict["is_feature"].astype(bool))
                ]["column"].astype(str)
            )
            shared_feature_count = len(disc_features & rep_features)
        feature_ready = shared_feature_count is None or shared_feature_count > 0
        row["discovery_ready"] = bool(discovery_ready)
        row["replication_ready"] = bool(replication_ready)
        row["feature_ready"] = bool(feature_ready)
        row["shared_feature_count"] = shared_feature_count if shared_feature_count is not None else pd.NA
        row["benchmark_ready"] = bool(
            discovery_ready
            and replication_ready
            and feature_ready
            and row["prepared_status"] in {"prepared", "already_working", "already_working_within_adni"}
        )
        prepared_rows.append(row)
    pd.DataFrame(prepared_rows).to_csv(OUT / "claim_inventory_ready.csv", index=False)


def write_readme(cohort_manifest: pd.DataFrame, misc_manifest: pd.DataFrame) -> None:
    text = f"""# Benchmark-Ready Data Layer

Generated by `scripts/build_benchmark_ready_layer.py`.

This layer is generated locally from copied remote tables. It is safe to delete
and rebuild. The script does not modify remote files or copied raw files.

## Cohort Bundles

- Directory: `data/prepared_data/benchmark_ready/cohorts/`
- Cohorts: {', '.join(cohort_manifest['cohort'].tolist())}
- Total rows: {int(cohort_manifest['rows'].sum())}
- Total feature columns across cohorts: {int(cohort_manifest['feature_columns'].sum())}

## Core Files

- `cohort_manifest.csv`: row/feature/covariate coverage by cohort
- `feature_dictionary.csv`: feature modality/family/source and missingness
- `claim_inventory_ready.csv`: candidate benchmark claims annotated with readiness
- `misc_table_manifest.csv`: copied disease/multimodal tables not yet canonicalized

## Notes

- This layer currently emphasizes fMRI descriptor cohorts.
- HDF5/zarr/raw-image data were intentionally not copied into this layer.
- Some miscellaneous tables are staged for later adapters, especially AIBL,
  NACC, MIRIAD, COBRE, FBIRN, and SZ_JH.
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    clean_output()
    feature_rows: list[dict[str, object]] = []
    summaries: list[CohortSummary] = []

    for source in sorted(SOURCE_FMRI.glob("*.parquet")):
        cohort_df, rows, summary = prepare_cohort(source)
        cohort_df.to_parquet(COHORTS_OUT / f"{source.stem}.parquet", index=False)
        feature_rows.extend(rows)
        summaries.append(summary)

    feature_dict = pd.DataFrame(feature_rows)
    cohort_manifest = pd.DataFrame([summary.__dict__ for summary in summaries])
    misc_manifest = copy_misc_tables()

    feature_dict.to_csv(OUT / "feature_dictionary.csv", index=False)
    cohort_manifest.to_csv(OUT / "cohort_manifest.csv", index=False)
    misc_manifest.to_csv(OUT / "misc_table_manifest.csv", index=False)
    write_claim_inventory(feature_dict, cohort_manifest)
    write_readme(cohort_manifest, misc_manifest)

    print("cohorts:", len(cohort_manifest))
    print(cohort_manifest[["cohort", "rows", "feature_columns", "rows_with_any_feature", "modalities"]].to_string(index=False))
    print("features:", len(feature_dict))
    print("misc tables:", len(misc_manifest))
    print("out:", OUT)


if __name__ == "__main__":
    main()
