"""Auditable fMRI feature preparation for external evidence cohorts."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from confirm.schema import validate_canonical
from nbs_data.external_dataset_registry import ExternalDatasetSpec
from nbs_data.external_metadata import load_metadata


NEUROMARK_DOMAIN_SIZES = (5, 2, 9, 9, 17, 7, 4)
GLOBAL_FC_METHOD = "pearson_fisher_z_scaled_neuromark53_domain_partition_v1"
STATIC_FC_METHOD = "existing_upper_triangle_neuromark53_domain_partition_v1"


def prepare_fmri_dataset(dataset: ExternalDatasetSpec, out_root: str | Path, *, max_workers: int = 1) -> dict[str, Any]:
    if dataset.fmri is None:
        return {"dataset_id": dataset.dataset_id, "status": "no_fmri_spec"}
    if dataset.fmri.backend == "audit_only":
        return {
            "dataset_id": dataset.dataset_id,
            "status": "audit_only",
            "quarantine": dataset.quarantine,
            "quarantine_reasons": dataset.quarantine_reasons,
        }
    if dataset.fmri.backend == "descriptor_csv":
        frame, provenance = _prepare_descriptor_csv(dataset)
    elif dataset.fmri.backend == "gift_timecourses":
        frame, provenance = _prepare_gift_timecourses(dataset, max_workers=max_workers)
    elif dataset.fmri.backend == "static_fc_vector":
        frame, provenance = _prepare_static_fc_vectors(dataset, max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported fMRI backend: {dataset.fmri.backend}")

    output_root = Path(out_root)
    destination = output_root / ("quarantine" if dataset.quarantine else "canonical")
    destination.mkdir(parents=True, exist_ok=True)
    output_dataset_id = f"{dataset.dataset_id}_fMRI"
    frame = frame.copy()
    frame["cohort"] = output_dataset_id
    output_path = destination / f"{output_dataset_id}.parquet"
    if dataset.quarantine:
        prepared = frame.copy()
        status = "quarantined"
    else:
        prepared = validate_canonical(frame)
        status = "ready"
    prepared.to_parquet(output_path, index=False)

    feature_columns = [column for column in prepared.columns if column.startswith("fc_")]
    feature_manifest = {
        "dataset_id": dataset.dataset_id,
        "output_dataset_id": output_dataset_id,
        "status": status,
        "quarantine_reasons": list(dataset.quarantine_reasons),
        "rows": len(prepared),
        "feature_columns": feature_columns,
        "schema_sha256": _schema_hash(prepared),
        "units": provenance.get("feature_units", {}),
        "provenance": provenance,
        "output_path": str(output_path),
    }
    manifest_path = destination / f"{output_dataset_id}.features.json"
    manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    return feature_manifest


def _prepare_descriptor_csv(dataset: ExternalDatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert dataset.fmri is not None and dataset.fmri.descriptor_path is not None
    source = pd.read_csv(dataset.fmri.descriptor_path, low_memory=False)
    metadata = load_metadata(dataset)
    subject_column = _first_column(source, "subject_id", "subject", "participant_id")
    session_column = _first_column(source, "session_id", "session")
    if subject_column is None:
        if not dataset.fmri.descriptor_index_path:
            raise ValueError(f"{dataset.dataset_id}: descriptor table has no subject identifier or index table")
        index = pd.read_csv(dataset.fmri.descriptor_index_path, low_memory=False)
        if len(index) != len(source):
            raise ValueError(
                f"{dataset.dataset_id}: descriptor/index row mismatch "
                f"({len(source)} != {len(index)})"
            )
        subject_column = dataset.fmri.descriptor_subject_column
        session_column = dataset.fmri.descriptor_session_column
        if subject_column not in index.columns:
            raise ValueError(f"{dataset.dataset_id}: descriptor index lacks {subject_column!r}")
        identifiers = pd.DataFrame(
            {
                "subject_id": index[subject_column].astype(str),
                "session": index[session_column].astype(str) if session_column in index.columns else "ses-01",
            }
        )
        join_method = "explicit_descriptor_index"
    else:
        identifiers = pd.DataFrame(
            {
                "subject_id": source[subject_column].astype(str),
                "session": source[session_column].astype(str) if session_column else "ses-01",
            }
        )
        join_method = "embedded_subject_keys"
    features: dict[str, pd.Series] = {}
    for column in source.columns:
        if column in {subject_column, session_column, "summary", "Unnamed: 0", ""}:
            continue
        values = pd.to_numeric(source[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        features[_descriptor_feature_name(column)] = values
    descriptors = pd.concat([identifiers, pd.DataFrame(features)], axis=1)
    merged = metadata.merge(descriptors, on=["subject_id", "session"], how="inner")
    merged["cohort"] = dataset.dataset_id
    return merged, {
        "backend": "descriptor_csv",
        "source": dataset.fmri.descriptor_path,
        "source_sha256": _sha256(dataset.fmri.descriptor_path),
        "descriptor_index_path": dataset.fmri.descriptor_index_path,
        "descriptor_index_sha256": (
            _sha256(dataset.fmri.descriptor_index_path) if dataset.fmri.descriptor_index_path else None
        ),
        "join_method": join_method,
        "descriptor_rows": len(source),
        "metadata_matched_rows": len(merged),
        "feature_units": {"fc_*": "provider_descriptor_scale"},
        "invented_z_columns": False,
    }


def _prepare_gift_timecourses(dataset: ExternalDatasetSpec, *, max_workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert dataset.fmri is not None
    spec = dataset.fmri
    paths = load_gift_subject_order(spec.subject_order_path)
    regex = re.compile(str(spec.subject_id_regex))
    tasks: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for index, source_path in enumerate(paths, start=1):
        if spec.path_contains and spec.path_contains not in source_path:
            continue
        match = regex.search(source_path)
        if match is None:
            continue
        subject_id = match.group(1)
        if subject_id in seen:
            continue
        seen.add(subject_id)
        ica_path = Path(str(spec.ica_dir)) / f"{spec.ica_prefix}_ica_br{index}.mat"
        if ica_path.exists():
            tasks.append((subject_id, str(ica_path), spec.component_count))

    records: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            records.append(_gift_feature_worker(task))
            if completed == len(tasks) or completed % max(1, len(tasks) // 20) == 0:
                print(f"[{dataset.dataset_id}] fMRI {completed}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks) or 1)) as pool:
            futures = {pool.submit(_gift_feature_worker, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if completed == len(futures) or completed % max(1, len(futures) // 20) == 0:
                    print(f"[{dataset.dataset_id}] fMRI {completed}/{len(futures)}", flush=True)

    features = pd.DataFrame(records)
    if features.empty:
        raise ValueError(f"{dataset.dataset_id}: no readable GIFT time-course records")
    features["session"] = spec.session_label
    metadata = load_metadata(dataset)
    if metadata.empty:
        merged = features.copy()
        merged["site"] = dataset.dataset_id
        merged["age"] = pd.NA
        merged["sex"] = pd.NA
        merged["dx"] = pd.NA
    else:
        merged = metadata.merge(features, on=["subject_id", "session"], how="inner")
    merged["cohort"] = dataset.dataset_id
    if not dataset.quarantine and dataset.dataset_id == "LA5C":
        merged = merged[merged["dx"].isin(["SZ", "HC"])].copy()
    provenance = {
        "backend": "gift_timecourses",
        "method": GLOBAL_FC_METHOD,
        "subject_order_path": spec.subject_order_path,
        "subject_order_sha256": _sha256(str(spec.subject_order_path)),
        "component_count": spec.component_count,
        "candidate_subjects": len(tasks),
        "prepared_subjects": len(merged),
        "network_fc_parity_reference": spec.parity_reference,
        "feature_units": {"fc_*": "fisher_z"},
        "network_fc_columns_emitted": False,
        "invented_z_columns": False,
    }
    return merged, provenance


def _prepare_static_fc_vectors(
    dataset: ExternalDatasetSpec,
    *,
    max_workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert dataset.fmri is not None
    spec = dataset.fmri
    root = Path(str(spec.static_root))
    paths = sorted(path for path in root.glob(str(spec.static_pattern)) if path.is_file())
    by_subject: dict[str, list[tuple[int, str, str]]] = {}
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            continue
        subject_id, visit = relative.parts[0], relative.parts[1]
        by_subject.setdefault(subject_id, []).append((_visit_order(visit), visit, str(path)))
    tasks = [
        (subject_id, sorted(candidates), str(spec.static_key), spec.component_count)
        for subject_id, candidates in sorted(by_subject.items())
    ]
    results: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            results.append(_static_fc_worker(task))
            if completed == len(tasks) or completed % max(1, len(tasks) // 20) == 0:
                print(f"[{dataset.dataset_id}] static FC {completed}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks) or 1)) as pool:
            futures = {pool.submit(_static_fc_worker, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed == len(futures) or completed % max(1, len(futures) // 20) == 0:
                    print(f"[{dataset.dataset_id}] static FC {completed}/{len(futures)}", flush=True)
    failures = [item for item in results if item.get("status") != "ready"]
    features = pd.DataFrame([{key: value for key, value in item.items() if key != "status"} for item in results if item.get("status") == "ready"])
    if features.empty:
        raise ValueError(f"{dataset.dataset_id}: no readable static-FC records")
    metadata = load_metadata(dataset)
    merged = metadata.merge(features, on=["subject_id", "session"], how="inner")
    merged["cohort"] = dataset.dataset_id
    return merged, {
        "backend": "static_fc_vector",
        "method": STATIC_FC_METHOD,
        "source_root": str(root),
        "source_pattern": spec.static_pattern,
        "matrix_key": spec.static_key,
        "component_count": spec.component_count,
        "candidate_subjects": len(tasks),
        "prepared_subjects": len(merged),
        "failed_subjects": failures,
        "feature_units": {"fc_*": "provider_sFNC_scale"},
        "network_fc_columns_emitted": False,
        "invented_z_columns": False,
    }


def load_gift_subject_order(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    import scipy.io as sio

    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        files = np.atleast_1d(data["files"])
        return [str(np.atleast_1d(getattr(item, "name")).ravel()[0]).split(",")[0].strip() for item in files.ravel()]
    except NotImplementedError:
        import h5py

        output: list[str] = []
        with h5py.File(path, "r") as handle:
            references = np.asarray(handle["files"]["name"]).ravel()
            for reference in references:
                chars = np.asarray(handle[reference]).ravel(order="F")
                text = "".join(chr(int(value)) for value in chars if 0 < int(value) < 0x110000)
                output.append(text.split(",")[0].strip())
        return output


def load_gift_timecourses(path: str | Path, component_count: int) -> np.ndarray:
    import scipy.io as sio

    try:
        data = sio.loadmat(path)
        timecourses = np.asarray(data["compSet"]["tc"][0, 0], dtype=float)
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:
            timecourses = np.asarray(handle["compSet"]["tc"][()], dtype=float)
    if timecourses.ndim != 2:
        raise ValueError(f"Expected two-dimensional time courses, got {timecourses.shape}")
    if timecourses.shape[1] != component_count and timecourses.shape[0] == component_count:
        timecourses = timecourses.T
    if timecourses.shape[1] != component_count:
        raise ValueError(f"Expected {component_count} components, got {timecourses.shape}")
    if not np.isfinite(timecourses).all():
        raise ValueError("GIFT time courses contain non-finite values")
    return timecourses


def global_fc_features(timecourses: np.ndarray) -> dict[str, float]:
    component_count = timecourses.shape[1]
    correlation = np.corrcoef(timecourses.T)
    upper = np.triu_indices(component_count, k=1)
    values = np.arctanh(np.clip(correlation[upper], -0.999999, 0.999999))
    if not np.isfinite(values).all():
        raise ValueError("Functional-connectivity matrix contains non-finite values")
    return upper_triangle_fc_features(values, component_count)


def upper_triangle_fc_features(values: np.ndarray, component_count: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    expected = component_count * (component_count - 1) // 2
    if values.size != expected:
        raise ValueError(f"Expected {expected} upper-triangle FC values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Functional-connectivity vector contains non-finite values")
    upper = np.triu_indices(component_count, k=1)
    domains = _scaled_domain_partition(component_count)
    within = domains[upper[0]] == domains[upper[1]]
    positive = values[values > 0]
    return {
        "fc_mean_abs": float(np.mean(np.abs(values))),
        "fc_mean_positive": float(np.mean(positive)) if positive.size else float("nan"),
        "fc_within_network": float(np.mean(values[within])),
        "fc_between_network": float(np.mean(values[~within])),
    }


def _static_fc_worker(task: tuple[str, list[tuple[int, str, str]], str, int]) -> dict[str, Any]:
    subject_id, candidates, key, component_count = task
    errors: list[str] = []
    for _, visit, path in candidates:
        try:
            values = _load_mat_vector(path, key)
            return {
                "status": "ready",
                "subject_id": subject_id,
                "session": f"{subject_id}_{visit}",
                "source_fc_path": path,
                "source_fc_sha256": _sha256(path),
                **order_invariant_fc_features(values, component_count),
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
    return {"status": "failed", "subject_id": subject_id, "errors": errors}


def order_invariant_fc_features(values: np.ndarray, component_count: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    expected = component_count * (component_count - 1) // 2
    if values.size != expected:
        raise ValueError(f"Expected {expected} upper-triangle FC values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Functional-connectivity vector contains non-finite values")
    positive = values[values > 0]
    return {
        "fc_mean_abs": float(np.mean(np.abs(values))),
        "fc_mean_positive": float(np.mean(positive)) if positive.size else float("nan"),
    }


def _load_mat_vector(path: str | Path, key: str) -> np.ndarray:
    import scipy.io as sio

    try:
        data = sio.loadmat(path)
        values = np.asarray(data[key], dtype=float)
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as handle:
            values = np.asarray(handle[key][()], dtype=float)
    return values.reshape(-1)


def _visit_order(visit: str) -> int:
    match = re.match(r"(\d+)", visit)
    return int(match.group(1)) if match else 10**9


def _gift_feature_worker(task: tuple[str, str, int]) -> dict[str, Any]:
    subject_id, path, component_count = task
    timecourses = load_gift_timecourses(path, component_count)
    return {
        "subject_id": subject_id,
        "source_ica_path": path,
        "source_ica_sha256": _sha256(path),
        **global_fc_features(timecourses),
    }


def _scaled_domain_partition(component_count: int) -> np.ndarray:
    weights = np.asarray(NEUROMARK_DOMAIN_SIZES, dtype=float)
    raw = component_count * weights / weights.sum()
    sizes = np.floor(raw).astype(int)
    remainder = component_count - int(sizes.sum())
    order = np.argsort(-(raw - sizes))
    for index in order[:remainder]:
        sizes[index] += 1
    return np.concatenate([np.full(size, index, dtype=int) for index, size in enumerate(sizes)])


def _first_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _safe_name(value: object) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip())
    return re.sub(r"_+", "_", text).strip("_")


def _descriptor_feature_name(value: object) -> str:
    name = _safe_name(value)
    return name if name.startswith("fc_fc_") else f"fc_fc_{name}"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_hash(frame: pd.DataFrame) -> str:
    schema = "\n".join(f"{column}:{frame[column].dtype}" for column in frame.columns)
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()
