"""Prepare copied remote tables for later CONFIRM ingestion.

This script only reads local copies under ``data/raw_remote`` and writes new
prepared artifacts under ``data/prepared_data``. It does not touch the remote.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAW = Path("data/raw_remote/arcdev")
OUT = Path("data/prepared_data")
MANIFEST_DIR = Path("docs/data_manifests")


@dataclass(frozen=True)
class FmriCohortSpec:
    cohort: str
    metadata: str
    descriptors: tuple[tuple[str, str], ...]


FMRI_SPECS = [
    FmriCohortSpec(
        "ABCD",
        "data/users1/ywei/data/ABCD/fmri/metadata.csv",
        (
            ("fc", "data/users1/ywei/data/ABCD/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/ABCD/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/ABCD/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
    FmriCohortSpec(
        "HCP",
        "data/users1/ywei/data/HCP/fmri/metadata_with_text_medical.csv",
        (
            ("fc", "data/users1/ywei/data/HCP/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/HCP/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/HCP/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
    FmriCohortSpec(
        "HCP_Aging",
        "data/users1/ywei/data/HCP_Aging/fmri/metadata.csv",
        (
            ("fc", "data/users1/ywei/data/HCP_Aging/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/HCP_Aging/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/HCP_Aging/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
    FmriCohortSpec(
        "UKB",
        "data/users1/ywei/data/UKB/fmri/metadata.csv",
        (
            ("fc", "data/users1/ywei/data/UKB/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/UKB/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/UKB/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
    FmriCohortSpec(
        "ADNI_fMRI",
        "data/users1/ywei/data/ADNI/fmri/metadata.csv",
        (
            ("fc", "data/users1/ywei/data/ADNI/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/ADNI/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/ADNI/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
    FmriCohortSpec(
        "OASIS3_fMRI",
        "data/users1/ywei/data/OASIS3/fmri/metadata.csv",
        (
            ("fc", "data/users1/ywei/data/OASIS3/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/OASIS3/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/OASIS3/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
    FmriCohortSpec(
        "ABIDE2",
        "data/users1/ywei/data/ABIDE2/fmri/metadata.csv",
        (
            ("fc", "data/users1/ywei/data/ABIDE2/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/ABIDE2/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/ABIDE2/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
    FmriCohortSpec(
        "ADHD200",
        "data/users1/ywei/data/ADHD200/fmri/metadata_with_text_medical_all.csv",
        (
            ("fc", "data/users1/ywei/data/ADHD200/fmri/descriptors/fc_self_descriptors.csv"),
            ("region", "data/users1/ywei/data/ADHD200/fmri/descriptors/region_self_descriptors.csv"),
            ("dyno", "data/users1/ywei/data/ADHD200/fmri/descriptors/ica_dyno_descriptors.csv"),
        ),
    ),
]


def safe_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] or "unnamed"


def normalize_sex(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if text in {"M", "MALE", "1"}:
        return "M"
    if text in {"F", "FEMALE", "2", "0"}:
        return "F"
    return text


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, low_memory=False)


def file_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        row = {
            "local_path": str(path),
            "remote_path": "/" + str(path.relative_to(RAW)),
            "suffix": suffix,
            "size_bytes": path.stat().st_size,
        }
        if suffix in {".csv", ".tsv"}:
            try:
                sep = "\t" if suffix == ".tsv" else ","
                with path.open(newline="", errors="replace") as handle:
                    reader = csv.reader(handle, delimiter=sep)
                    header = next(reader)
                row["n_columns"] = len(header)
                row["columns_preview"] = "|".join(header[:40])
                with path.open("rb") as handle:
                    row["n_lines"] = sum(1 for _ in handle)
            except Exception as exc:
                row["read_error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def best_column_by_coverage(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the candidate with the most non-missing values."""

    found = [col for col in (first_column(df, [candidate]) for candidate in candidates) if col]
    if not found:
        return None
    return max(found, key=lambda col: int(df[col].notna().sum()))


def adni_session_id(df: pd.DataFrame) -> pd.Series | None:
    """Construct ADNI descriptor-style session IDs from metadata paths."""

    needed = {"subject_id", "session_path", "time_path", "fmri_path"}
    if not needed.issubset(df.columns):
        return None
    session_name = df["session_path"].astype(str).str.rstrip("/").str.rsplit("/", n=1).str[-1]
    scan_id = df["fmri_path"].astype(str).str.extract(r"/(S[0-9]+)/rest/", expand=False)
    if scan_id.notna().sum() == 0:
        return None
    return df["subject_id"].astype(str) + "-" + session_name + "-" + df["time_path"].astype(str) + "-" + scan_id.astype(str)


def metadata_frame(spec: FmriCohortSpec) -> pd.DataFrame:
    path = RAW / spec.metadata
    df = read_table(path)
    out = pd.DataFrame()
    subj = first_column(df, ["subject_id", "subject", "Subject", "subID", "id", "eid"])
    sess = first_column(df, ["session_id", "session", "Session", "visit", "VISCODE"])
    if subj is None:
        raise ValueError(f"{spec.cohort}: no subject id column in {path}")
    out["subject_id"] = df[subj].astype(str)
    parsed_session = adni_session_id(df)
    if sess:
        out["session"] = df[sess].astype(str)
    elif parsed_session is not None:
        out["session"] = parsed_session
    else:
        out["session"] = "ses-01"
    out["cohort"] = spec.cohort
    site = first_column(df, ["site", "SITE_ID", "scanner", "scanner_id"])
    out["site"] = df[site].astype(str) if site else spec.cohort
    age = best_column_by_coverage(df, ["age", "Age", "AGE", "ageAtEntry", "Age_in_Yrs", "Age_MRI"])
    out["age"] = pd.to_numeric(df[age], errors="coerce") if age else pd.NA
    sex = first_column(df, ["sex", "Sex", "SEX", "gender", "Gender", "MF", "M/F"])
    out["sex"] = df[sex].map(normalize_sex).astype("string") if sex else pd.NA
    dx = first_column(df, ["dx", "DX", "diagnosis", "Diagnosis", "ADHD", "ASD", "had_adhd", "had_asd", "had_schizophrenia"])
    out["dx"] = df[dx].astype(str) if dx else pd.NA

    for col in df.columns:
        low = col.lower()
        if col in {subj, sess, site, age, sex, dx}:
            continue
        if any(token in low for token in ["fluid", "flanker", "iq", "mmse", "cdr", "bmi", "cholesterol", "blood_pressure", "height", "weight", "apoe", "av45", "had_", "diagnosis"]):
            out[f"phen_{safe_name(col)}"] = df[col]

    out = out.drop_duplicates(["subject_id", "session"], keep="first")
    return out


def descriptor_frame(path: Path, family: str) -> pd.DataFrame:
    df = read_table(path)
    subj = first_column(df, ["subject_id", "subject", "Subject", "subID", "id", "eid"])
    sess = first_column(df, ["session_id", "session", "Session", "visit", "VISCODE"])
    if subj is None:
        raise ValueError(f"no subject id column in {path}")

    out = pd.DataFrame()
    out["subject_id"] = df[subj].astype(str)
    out["session"] = df[sess].astype(str) if sess else "ses-01"

    skip = {subj, sess, "summary", "Unnamed: 0", ""}
    for col in df.columns:
        if col in skip:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        prefix = "fc" if family == "fc" else "raw_fmri"
        out[f"{prefix}_{family}_{safe_name(col)}"] = values

    out = out.drop_duplicates(["subject_id", "session"], keep="first")
    return out


def prepare_fmri(spec: FmriCohortSpec) -> dict[str, object]:
    meta = metadata_frame(spec)
    merged = meta
    descriptor_summaries = []

    for family, rel in spec.descriptors:
        path = RAW / rel
        if not path.exists():
            descriptor_summaries.append({"family": family, "status": "missing", "path": str(path)})
            continue
        desc = descriptor_frame(path, family)
        before = len(merged)
        merged = merged.merge(desc, on=["subject_id", "session"], how="left")
        feature_count = len([c for c in desc.columns if c not in {"subject_id", "session"}])
        descriptor_summaries.append(
            {
                "family": family,
                "rows": len(desc),
                "features": feature_count,
                "matched_rows_before": before,
                "matched_nonnull_any": int(merged[[c for c in desc.columns if c not in {"subject_id", "session"}]].notna().any(axis=1).sum()) if feature_count else 0,
            }
        )

    cohort_dir = OUT / "fmri_descriptors"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    out_path = cohort_dir / f"{spec.cohort}.parquet"
    dict_path = cohort_dir / f"{spec.cohort}.columns.csv"
    merged.to_parquet(out_path, index=False)
    pd.DataFrame(
        {
            "column": merged.columns,
            "role": ["id" if c in {"subject_id", "session", "cohort", "site", "age", "sex", "dx"} else "feature_or_covariate" for c in merged.columns],
        }
    ).to_csv(dict_path, index=False)

    return {
        "cohort": spec.cohort,
        "rows": len(merged),
        "columns": len(merged.columns),
        "feature_columns": len([c for c in merged.columns if c.startswith(("fc_", "raw_fmri_"))]),
        "out_path": str(out_path),
        "descriptor_summaries": descriptor_summaries,
    }


def convert_misc_tables() -> pd.DataFrame:
    rows = []
    misc_dir = OUT / "misc_tables"
    misc_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        if "/data/users1/ywei/data/" in str(path) and "/fmri/descriptors/" in str(path):
            continue
        if "/data/users1/ywei/data/" in str(path) and "/fmri/metadata" in str(path):
            continue
        try:
            df = read_table(path)
            rel = path.relative_to(RAW)
            name = "__".join(rel.parts).replace(".csv", "").replace(".tsv", "")
            out_path = misc_dir / f"{safe_name(name)}.parquet"
            df.to_parquet(out_path, index=False)
            rows.append({"source": str(path), "rows": len(df), "columns": len(df.columns), "out_path": str(out_path)})
        except Exception as exc:
            rows.append({"source": str(path), "error": str(exc)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = file_manifest()
    manifest.to_csv(MANIFEST_DIR / "raw_remote_wave1_manifest.csv", index=False)

    summaries = []
    for spec in FMRI_SPECS:
        try:
            summaries.append(prepare_fmri(spec))
        except Exception as exc:
            summaries.append({"cohort": spec.cohort, "error": str(exc)})

    fmri_summary = pd.DataFrame(summaries)
    fmri_summary.to_csv(MANIFEST_DIR / "prepared_fmri_wave1_summary.csv", index=False)

    misc_summary = convert_misc_tables()
    misc_summary.to_csv(MANIFEST_DIR / "prepared_misc_wave1_summary.csv", index=False)

    print("raw files:", len(manifest))
    print(fmri_summary[["cohort", "rows", "columns", "feature_columns", "out_path"]].to_string(index=False))
    print("misc tables converted:", len(misc_summary))


if __name__ == "__main__":
    main()
