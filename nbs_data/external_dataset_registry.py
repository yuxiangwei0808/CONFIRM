"""Typed registry and deterministic scan selection for external evidence data."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


MetadataAdapter = Literal["none", "ehbs", "cnp", "aibl", "blsa", "pk_mprc"]
StructuralLayout = Literal[
    "bids_single_session",
    "ehbs_single",
    "olin_acquisition",
    "blsa_dated_visit",
    "aibl_dated_visit",
    "shile_bids",
    "pk_mprc",
]
FmriBackend = Literal["descriptor_csv", "gift_timecourses", "static_fc_vector", "audit_only"]


class MetadataSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: MetadataAdapter
    paths: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)


class StructuralSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str
    pattern: str
    layout: StructuralLayout
    preferred_tokens: list[str] = Field(default_factory=list)
    existing_subjects_dir: Optional[str] = None


class FmriSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: FmriBackend
    descriptor_path: Optional[str] = None
    descriptor_index_path: Optional[str] = None
    descriptor_subject_column: str = "subject_id"
    descriptor_session_column: Optional[str] = "session"
    metadata_path: Optional[str] = None
    ica_dir: Optional[str] = None
    ica_prefix: Optional[str] = None
    subject_order_path: Optional[str] = None
    subject_id_regex: Optional[str] = None
    path_contains: Optional[str] = None
    session_label: str = "ses_01"
    component_count: int = Field(default=100, ge=2)
    parity_reference: Optional[str] = None
    static_root: Optional[str] = None
    static_pattern: Optional[str] = None
    static_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_backend_paths(self) -> "FmriSpec":
        if self.backend == "descriptor_csv" and not self.descriptor_path:
            raise ValueError("descriptor_csv requires descriptor_path")
        if self.backend == "gift_timecourses":
            required = {
                "ica_dir": self.ica_dir,
                "ica_prefix": self.ica_prefix,
                "subject_order_path": self.subject_order_path,
                "subject_id_regex": self.subject_id_regex,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"gift_timecourses requires {missing}")
        if self.backend == "static_fc_vector":
            required = {
                "static_root": self.static_root,
                "static_pattern": self.static_pattern,
                "static_key": self.static_key,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(f"static_fc_vector requires {missing}")
        return self


class ExternalDatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    target_families: list[str]
    quarantine: bool = False
    quarantine_reasons: list[str] = Field(default_factory=list)
    metadata: MetadataSpec
    structural: Optional[StructuralSpec] = None
    fmri: Optional[FmriSpec] = None


class ExternalDatasetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    remote_output_root: str
    datasets: list[ExternalDatasetSpec]

    @model_validator(mode="after")
    def unique_dataset_ids(self) -> "ExternalDatasetRegistry":
        ids = [dataset.dataset_id for dataset in self.datasets]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset_id values must be unique")
        return self

    def selected(self, values: str | list[str]) -> list[ExternalDatasetSpec]:
        if values == "all" or values == ["all"]:
            return list(self.datasets)
        requested = {
            item.strip()
            for item in (values.split(",") if isinstance(values, str) else values)
            if item.strip()
        }
        found = {dataset.dataset_id: dataset for dataset in self.datasets}
        missing = sorted(requested - set(found))
        if missing:
            raise ValueError(f"Unknown external datasets: {missing}")
        return [dataset for dataset in self.datasets if dataset.dataset_id in requested]


class StructuralCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    subject_id: str
    source_subject_id: str
    session: str
    site: str
    t1_path: str
    selection_order: tuple[int, str]
    acquisition_date: Optional[date] = None


class SubjectManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    subject_id: str
    source_subject_id: str
    session: str
    site: str
    t1_path: str
    t1_sha256: str
    acquisition_date: Optional[date] = None
    selection_reason: Literal["baseline_first_valid", "baseline_qc_fallback"]
    site_selection_reason: Literal["single_site", "metadata_site_match"] = "single_site"
    rejected_earlier_candidates: int = 0
    nifti_shape: list[int]
    evidence_eligible: bool
    quarantine_reasons: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "complete", "failed", "quarantined"] = "pending"


class ManifestBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    candidates_found: int
    subjects_found: int
    selected_rows: list[SubjectManifestRow]
    imaging_qc_failures: list[dict[str, str]] = Field(default_factory=list)
    selection_exclusions: list[dict[str, object]] = Field(default_factory=list)


def load_registry(path: str | Path) -> ExternalDatasetRegistry:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExternalDatasetRegistry.model_validate(data)


def build_subject_manifest(
    dataset: ExternalDatasetSpec,
    *,
    metadata: pd.DataFrame | None = None,
    progress: bool = False,
) -> ManifestBuildResult:
    if dataset.structural is None:
        return ManifestBuildResult(dataset_id=dataset.dataset_id, candidates_found=0, subjects_found=0, selected_rows=[])

    structural = dataset.structural
    root = Path(structural.root)
    paths = sorted(path for path in root.glob(structural.pattern) if path.is_file()) if root.exists() else []
    candidates = [_structural_candidate(dataset, path) for path in paths]
    grouped: dict[str, list[StructuralCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.subject_id].append(candidate)

    rows: list[SubjectManifestRow] = []
    failures: list[dict[str, str]] = []
    exclusions: list[dict[str, object]] = []
    metadata_sites = _metadata_site_assignments(metadata)
    total_subjects = len(grouped)
    for index, (subject_id, subject_candidates) in enumerate(sorted(grouped.items()), start=1):
        candidate_sites = sorted({candidate.site for candidate in subject_candidates})
        site_selection_reason = "single_site"
        if len(candidate_sites) > 1:
            assigned_sites = metadata_sites.get(subject_id, set())
            matching_sites = sorted(set(candidate_sites) & assigned_sites)
            if len(assigned_sites) != 1 or len(matching_sites) != 1:
                exclusions.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "subject_id": subject_id,
                        "reason": "ambiguous_site_assignment",
                        "candidate_sites": candidate_sites,
                        "metadata_sites": sorted(assigned_sites),
                    }
                )
                _manifest_progress(dataset.dataset_id, index, total_subjects, len(rows), len(failures), progress)
                continue
            subject_candidates = [
                candidate for candidate in subject_candidates if candidate.site == matching_sites[0]
            ]
            site_selection_reason = "metadata_site_match"
        ordered = sorted(subject_candidates, key=lambda item: item.selection_order)
        selected: StructuralCandidate | None = None
        shape: list[int] = []
        rejected = 0
        for candidate in ordered:
            ok, reason, candidate_shape = nifti_preflight(candidate.t1_path)
            if ok:
                selected = candidate
                shape = candidate_shape
                break
            rejected += 1
            failures.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "subject_id": subject_id,
                    "t1_path": candidate.t1_path,
                    "reason": reason,
                }
            )
        if selected is None:
            _manifest_progress(dataset.dataset_id, index, total_subjects, len(rows), len(failures), progress)
            continue
        rows.append(
            SubjectManifestRow(
                dataset_id=dataset.dataset_id,
                subject_id=selected.subject_id,
                source_subject_id=selected.source_subject_id,
                session=selected.session,
                site=selected.site,
                t1_path=selected.t1_path,
                t1_sha256=sha256_file(selected.t1_path),
                acquisition_date=selected.acquisition_date,
                selection_reason="baseline_qc_fallback" if rejected else "baseline_first_valid",
                site_selection_reason=site_selection_reason,
                rejected_earlier_candidates=rejected,
                nifti_shape=shape,
                evidence_eligible=not dataset.quarantine,
                quarantine_reasons=list(dataset.quarantine_reasons),
            )
        )
        _manifest_progress(dataset.dataset_id, index, total_subjects, len(rows), len(failures), progress)

    return ManifestBuildResult(
        dataset_id=dataset.dataset_id,
        candidates_found=len(candidates),
        subjects_found=len(grouped),
        selected_rows=rows,
        imaging_qc_failures=failures,
        selection_exclusions=exclusions,
    )


def write_subject_manifest(result: ManifestBuildResult, out_dir: str | Path) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{result.dataset_id}_subjects.json"
    tsv_path = target / f"{result.dataset_id}_subjects.tsv"
    json_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    pd.DataFrame([row.model_dump(mode="json") for row in result.selected_rows]).to_csv(tsv_path, sep="\t", index=False)
    return json_path, tsv_path


def nifti_preflight(path: str | Path) -> tuple[bool, str, list[int]]:
    try:
        image = nib.load(str(path))
        shape = [int(value) for value in image.shape]
        if len(shape) < 3 or any(value <= 0 for value in shape[:3]):
            return False, f"invalid NIfTI shape {shape}", shape
        affine = np.asarray(image.affine, dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            return False, "invalid NIfTI affine", shape
        zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
        if zooms.size != 3 or not np.isfinite(zooms).all() or (zooms <= 0).any():
            return False, "invalid NIfTI voxel sizes", shape
        return True, "ok", shape
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable NIfTI: {exc}", []


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def registry_audit(registry: ExternalDatasetRegistry, datasets: list[ExternalDatasetSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        structural = dataset.structural
        fmri = dataset.fmri
        structural_root = Path(structural.root) if structural else None
        structural_count = (
            sum(1 for path in structural_root.glob(structural.pattern) if path.is_file())
            if structural_root is not None and structural_root.exists()
            else 0
        )
        metadata_exists = [str(Path(path).exists()) for path in dataset.metadata.paths]
        metadata_checksums = [_asset_checksum(path) for path in dataset.metadata.paths]
        fmri_assets = _fmri_asset_paths(fmri)
        rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "target_families": "|".join(dataset.target_families),
                "quarantine": dataset.quarantine,
                "quarantine_reasons": "|".join(dataset.quarantine_reasons),
                "metadata_adapter": dataset.metadata.adapter,
                "metadata_paths": "|".join(dataset.metadata.paths),
                "metadata_paths_exist": "|".join(metadata_exists),
                "metadata_path_sha256": "|".join(metadata_checksums),
                "structural_root": str(structural_root) if structural_root else "",
                "structural_root_exists": bool(structural_root and structural_root.exists()),
                "structural_candidate_count": structural_count,
                "fmri_backend": fmri.backend if fmri else "",
                "fmri_assets": "|".join(fmri_assets),
                "fmri_assets_exist": "|".join(str(Path(path).exists()) for path in fmri_assets),
                "fmri_asset_sha256": "|".join(_asset_checksum(path) for path in fmri_assets),
                "remote_output_root": registry.remote_output_root,
            }
        )
    return rows


def _asset_checksum(path: str) -> str:
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else ""


def _fmri_asset_paths(spec: FmriSpec | None) -> list[str]:
    if spec is None:
        return []
    return [
        value
        for value in (
            spec.descriptor_path,
            spec.descriptor_index_path,
            spec.metadata_path,
            spec.ica_dir,
            spec.subject_order_path,
            spec.static_root,
        )
        if value
    ]


def _structural_candidate(dataset: ExternalDatasetSpec, path: Path) -> StructuralCandidate:
    assert dataset.structural is not None
    root = Path(dataset.structural.root)
    rel = path.relative_to(root)
    layout = dataset.structural.layout
    parts = rel.parts
    acquisition_date: date | None = None

    if layout == "ehbs_single":
        source_subject = parts[0]
        return _candidate(dataset, source_subject, source_subject, "ses-01", dataset.dataset_id, path, (0, str(rel)))
    if layout == "bids_single_session":
        source_subject = next((part for part in parts if part.lower().startswith("sub")), parts[0])
        session = next((part for part in parts if part.lower().startswith("ses")), "ses_01")
        return _candidate(dataset, source_subject, source_subject, session, dataset.dataset_id, path, (0, str(rel)))
    if layout == "shile_bids":
        site, source_subject = parts[0], parts[1]
        session = next((part for part in parts if part.lower().startswith("ses")), "ses_01")
        subject_id = f"{site}_{source_subject}"
        return _candidate(dataset, subject_id, source_subject, session, site, path, (0, str(rel)))
    if layout == "pk_mprc":
        site, source_subject = parts[0], parts[1]
        return _candidate(dataset, source_subject, source_subject, "ses-01", site, path, (0, str(rel)))
    if layout == "olin_acquisition":
        source_folder, acquisition = parts[0], parts[1]
        source_subject = source_folder
        preferred = dataset.structural.preferred_tokens
        preference = next((index for index, token in enumerate(preferred) if token.lower() in acquisition.lower()), len(preferred))
        series_match = re.match(r"(\d+)", acquisition)
        series = int(series_match.group(1)) if series_match else 10**9
        return _candidate(dataset, source_subject, source_subject, source_folder, dataset.dataset_id, path, (preference * 10**6 + series, str(rel)))
    if layout == "blsa_dated_visit":
        source_subject, visit_folder = parts[0], parts[1]
        visit_match = re.match(r"(\d+)", visit_folder)
        visit = int(visit_match.group(1)) if visit_match else 10**9
        session = f"{source_subject}_{visit_folder}"
        return _candidate(dataset, source_subject, source_subject, session, dataset.dataset_id, path, (visit, str(rel)))
    if layout == "aibl_dated_visit":
        source_subject, date_text = parts[0], parts[1]
        acquisition_date = datetime.strptime(date_text, "%Y-%m-%d").date()
        candidate = _candidate(dataset, source_subject, source_subject, date_text, dataset.dataset_id, path, (acquisition_date.toordinal(), str(rel)))
        return candidate.model_copy(update={"acquisition_date": acquisition_date})
    raise ValueError(f"Unsupported structural layout: {layout}")


def _metadata_site_assignments(metadata: pd.DataFrame | None) -> dict[str, set[str]]:
    if metadata is None or metadata.empty or not {"subject_id", "site"}.issubset(metadata.columns):
        return {}
    assignments: dict[str, set[str]] = defaultdict(set)
    for subject_id, site in metadata[["subject_id", "site"]].dropna().itertuples(index=False, name=None):
        assignments[str(subject_id)].add(str(site))
    return assignments


def _manifest_progress(
    dataset_id: str,
    index: int,
    total: int,
    selected: int,
    qc_failures: int,
    enabled: bool,
) -> None:
    if enabled and (index == total or index % max(1, total // 20) == 0):
        print(
            f"[{dataset_id}] structural manifest {index}/{total} "
            f"selected={selected} qc_failures={qc_failures}",
            flush=True,
        )


def _candidate(
    dataset: ExternalDatasetSpec,
    subject_id: str,
    source_subject_id: str,
    session: str,
    site: str,
    path: Path,
    order: tuple[int, str],
) -> StructuralCandidate:
    return StructuralCandidate(
        dataset_id=dataset.dataset_id,
        subject_id=str(subject_id),
        source_subject_id=str(source_subject_id),
        session=str(session),
        site=str(site),
        t1_path=str(path.resolve()),
        selection_order=order,
    )
