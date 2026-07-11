"""Authoritative phenotype adapters for external evidence datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nbs_data.external_dataset_registry import ExternalDatasetSpec


CANONICAL_METADATA = ["subject_id", "session", "site", "age", "sex", "dx"]


def load_metadata(dataset: ExternalDatasetSpec) -> pd.DataFrame:
    adapter = dataset.metadata.adapter
    if adapter == "none":
        return pd.DataFrame(columns=CANONICAL_METADATA)
    if adapter == "ehbs":
        return _ehbs(dataset)
    if adapter == "cnp":
        return _cnp(dataset)
    if adapter == "aibl":
        return _aibl(dataset)
    if adapter == "blsa":
        return _blsa(dataset)
    if adapter == "pk_mprc":
        return _pk_mprc(dataset)
    raise ValueError(f"Unsupported metadata adapter: {adapter}")


def _ehbs(dataset: ExternalDatasetSpec) -> pd.DataFrame:
    source = pd.read_csv(dataset.metadata.paths[0], low_memory=False)
    out = pd.DataFrame(
        {
            "subject_id": source["subject_id"].map(_clean_id),
            "session": _series_or_default(source, "session_id", "ses-01").astype(str),
            "site": dataset.dataset_id,
            "age": pd.to_numeric(source["age"], errors="coerce"),
            "sex": source["sex"].map(_sex),
            "dx": pd.NA,
        }
    )
    return _dedupe(out)


def _cnp(dataset: ExternalDatasetSpec) -> pd.DataFrame:
    source = pd.read_csv(dataset.metadata.paths[0], sep="\t", low_memory=False)
    diagnosis = source["diagnosis"].astype(str).str.strip().str.upper()
    dx = diagnosis.map(
        {
            "CONTROL": "HC",
            "HC": "HC",
            "SCHZ": "SZ",
            "SCHIZOPHRENIA": "SZ",
            "PSYCHOSIS": "SZ",
        }
    )
    out = pd.DataFrame(
        {
            "subject_id": source["participant_id"].map(_clean_id),
            "session": "ses_01",
            "site": dataset.dataset_id,
            "age": pd.to_numeric(source["age"], errors="coerce"),
            "sex": source["gender"].map(_sex),
            "dx": dx,
        }
    )
    return _dedupe(out)


def _aibl(dataset: ExternalDatasetSpec) -> pd.DataFrame:
    master = pd.read_csv(dataset.metadata.paths[0], low_memory=False)
    diagnosis = pd.read_csv(dataset.metadata.paths[1], low_memory=False)
    if "Modality" in master.columns:
        master = master[master["Modality"].astype(str).str.upper().eq("MRI")]
    if "Description" in master.columns:
        master = master[master["Description"].astype(str).str.contains("MPRAGE", case=False, na=False)]
    master = master.copy()
    master["subject_id"] = master["Subject"].map(_clean_id)
    master["acquisition_date"] = pd.to_datetime(master["Acq Date"], errors="coerce")
    master["session"] = master["acquisition_date"].dt.strftime("%Y-%m-%d")
    visit_number = pd.to_numeric(master["Visit"], errors="coerce")
    master["VISCODE"] = visit_number.map({1: "bl", 2: "m18", 3: "m36", 4: "m54"})

    diagnosis = diagnosis.copy()
    diagnosis["subject_id"] = diagnosis["RID"].map(_clean_id)
    diagnosis["dx"] = "other"
    diagnosis.loc[pd.to_numeric(diagnosis.get("DXNORM"), errors="coerce").eq(1), "dx"] = "CN"
    diagnosis.loc[pd.to_numeric(diagnosis.get("DXMCI"), errors="coerce").eq(1), "dx"] = "MCI"
    diagnosis.loc[pd.to_numeric(diagnosis.get("DXAD"), errors="coerce").eq(1), "dx"] = "AD"
    dx_columns = ["subject_id", "VISCODE", "dx"]
    if "SITEID" in diagnosis.columns:
        dx_columns.append("SITEID")
    merged = master.merge(diagnosis[dx_columns], on=["subject_id", "VISCODE"], how="left")
    site = merged["SITEID"].map(lambda value: f"AIBL_{_clean_id(value)}") if "SITEID" in merged else dataset.dataset_id
    out = pd.DataFrame(
        {
            "subject_id": merged["subject_id"],
            "session": merged["session"],
            "site": site,
            "age": pd.to_numeric(merged["Age"], errors="coerce"),
            "sex": merged["Sex"].map(_sex),
            "dx": merged["dx"].fillna("other"),
        }
    )
    return _dedupe(out)


def _blsa(dataset: ExternalDatasetSpec) -> pd.DataFrame:
    source = pd.read_excel(dataset.metadata.paths[0])
    source = source[pd.to_numeric(source.get("MPRAGE"), errors="coerce").eq(1)].copy()
    diagnosis = pd.to_numeric(source.get("dxfinal"), errors="coerce")
    dx = pd.Series("other", index=source.index, dtype="object")
    dx.loc[diagnosis.eq(0)] = "CN"
    subject = source["session"].astype(str).str.extract(r"^(BLSA_\d+)", expand=False)
    out = pd.DataFrame(
        {
            "subject_id": subject.map(_clean_id),
            "session": source["session"].astype(str),
            "site": _series_or_default(source, "scanner", dataset.dataset_id).map(
                lambda value: f"BLSA_{_clean_id(value)}"
            ),
            "age": pd.to_numeric(source["Age"], errors="coerce"),
            "sex": source["sex"].map(_sex),
            "dx": dx,
        }
    )
    return _dedupe(out)


def _pk_mprc(dataset: ExternalDatasetSpec) -> pd.DataFrame:
    source = pd.read_excel(dataset.metadata.paths[0])
    records: list[dict[str, object]] = []
    for _, row in source.iterrows():
        ids = [_clean_id(row.get("zid")), _clean_id(row.get("aid"))]
        for subject_id in dict.fromkeys(value for value in ids if value):
            diagnosis = pd.to_numeric(pd.Series([row.get("dx(0=crl,1=scz)")]), errors="coerce").iloc[0]
            records.append(
                {
                    "subject_id": subject_id,
                    "session": "ses-01",
                    "site": f"Scanner{_clean_id(row.get('class (scanner)'))}",
                    "age": pd.to_numeric(pd.Series([row.get("Age")]), errors="coerce").iloc[0],
                    "sex": _sex(row.get("sex (0=F,1=M)")),
                    "dx": "SZ" if diagnosis == 1 else ("HC" if diagnosis == 0 else pd.NA),
                }
            )
    return _dedupe(pd.DataFrame(records, columns=CANONICAL_METADATA))


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CANONICAL_METADATA:
        if column not in out.columns:
            out[column] = pd.NA
    out["subject_id"] = out["subject_id"].map(_clean_id)
    out["session"] = out["session"].fillna("ses-01").astype(str)
    return out.dropna(subset=["subject_id"]).drop_duplicates(["subject_id", "session"], keep="first")[CANONICAL_METADATA]


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _series_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _sex(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if text in {"M", "MALE", "1", "1.0"}:
        return "M"
    if text in {"F", "FEMALE", "0", "0.0", "2", "2.0"}:
        return "F"
    return pd.NA
