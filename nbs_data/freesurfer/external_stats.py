"""Completion receipts and canonical feature extraction for FreeSurfer outputs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


DKT_STATS = {
    "lh": ("stats/lh.aparc.DKTatlas.stats", "stats/lh.aparc.DKTatlas.mapped.stats"),
    "rh": ("stats/rh.aparc.DKTatlas.stats", "stats/rh.aparc.DKTatlas.mapped.stats"),
}
REQUIRED_SURFACES = ("surf/lh.white", "surf/rh.white")
LEGACY_SUCCESS_MARKERS = ("scripts/recon-all.done", "scripts/recon-surf.done")
RECEIPT_NAME = ".confirm_complete.json"


class CompletionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    status: Literal["complete", "failed"]
    exit_code: int
    engine: Literal["fastsurfer", "recon-all"]
    freesurfer_version: str
    fastsurfer_version: Optional[str] = None
    fastsurfer_python: Optional[str] = None
    fastsurfer_python_version: Optional[str] = None
    t1_path: str
    t1_sha256: str
    started_at: str
    finished_at: str
    required_artifacts: list[str] = Field(default_factory=list)
    adopted_legacy_output: bool = False


class CompletionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    complete: bool
    subject_dir: str
    receipt_path: Optional[str] = None
    required_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    reason: str


def completion_check(subject_dir: str | Path, *, allow_legacy: bool = False) -> CompletionCheck:
    root = Path(subject_dir)
    receipt_path = root / RECEIPT_NAME
    required = required_artifacts(root)
    missing = [path for path in required if not (root / path).is_file() or (root / path).stat().st_size == 0]

    if receipt_path.exists():
        try:
            receipt = CompletionReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return CompletionCheck(
                complete=False,
                subject_dir=str(root),
                receipt_path=str(receipt_path),
                required_artifacts=required,
                missing_artifacts=missing,
                reason=f"invalid completion receipt: {exc}",
            )
        ok = receipt.status == "complete" and receipt.exit_code == 0 and not missing
        return CompletionCheck(
            complete=ok,
            subject_dir=str(root),
            receipt_path=str(receipt_path),
            required_artifacts=required,
            missing_artifacts=missing,
            reason="complete" if ok else "receipt or required artifact check failed",
        )

    if allow_legacy:
        legacy_markers = [marker for marker in LEGACY_SUCCESS_MARKERS if (root / marker).is_file()]
        ok = len(legacy_markers) == len(LEGACY_SUCCESS_MARKERS) and not missing
        return CompletionCheck(
            complete=ok,
            subject_dir=str(root),
            required_artifacts=required,
            missing_artifacts=missing,
            reason="legacy output complete" if ok else "legacy success markers or required artifacts missing",
        )

    return CompletionCheck(
        complete=False,
        subject_dir=str(root),
        required_artifacts=required,
        missing_artifacts=missing,
        reason="completion receipt missing",
    )


def required_artifact_check(subject_dir: str | Path) -> CompletionCheck:
    """Check the complete anatomical artifact set without requiring a receipt."""

    root = Path(subject_dir)
    required = required_artifacts(root)
    missing = [path for path in required if not (root / path).is_file() or (root / path).stat().st_size == 0]
    return CompletionCheck(
        complete=not missing,
        subject_dir=str(root),
        required_artifacts=required,
        missing_artifacts=missing,
        reason="required artifacts complete" if not missing else "required artifacts missing",
    )


def write_completion_receipt(subject_dir: str | Path, receipt: CompletionReceipt) -> Path:
    target = Path(subject_dir) / RECEIPT_NAME
    target.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2), encoding="utf-8")
    return target


def adopt_legacy_completion(
    subject_dir: str | Path,
    *,
    subject_id: str,
    t1_path: str,
    t1_sha256: str,
    freesurfer_version: str = "legacy_unknown",
    fastsurfer_version: str = "legacy_unknown",
) -> CompletionReceipt:
    check = completion_check(subject_dir, allow_legacy=True)
    if not check.complete:
        raise ValueError(f"Cannot adopt incomplete legacy output: {check.reason}; missing={check.missing_artifacts}")
    now = datetime.now().isoformat(timespec="seconds")
    receipt = CompletionReceipt(
        subject_id=subject_id,
        status="complete",
        exit_code=0,
        engine="fastsurfer",
        freesurfer_version=freesurfer_version,
        fastsurfer_version=fastsurfer_version,
        t1_path=t1_path,
        t1_sha256=t1_sha256,
        started_at=now,
        finished_at=now,
        required_artifacts=check.required_artifacts,
        adopted_legacy_output=True,
    )
    write_completion_receipt(subject_dir, receipt)
    return receipt


def canonical_features(subject_dir: str | Path) -> dict[str, float]:
    root = Path(subject_dir)
    check = completion_check(root, allow_legacy=True)
    if not check.complete:
        raise ValueError(f"Incomplete FreeSurfer output: {check.reason}; missing={check.missing_artifacts}")

    measures, structures = parse_aseg_stats(root / "stats/aseg.stats")
    lh = parse_aparc_stats(root / _find_dkt_stats(root, "lh"))
    rh = parse_aparc_stats(root / _find_dkt_stats(root, "rh"))

    features = {
        "eTIV": _require_measure(measures, "EstimatedTotalIntraCranialVol", "eTIV"),
        "smri_hippocampus": _sum_structures(structures, "Left-Hippocampus", "Right-Hippocampus"),
        "smri_entorhinal": _sum_cortical_volume(lh, rh, "entorhinal"),
        "smri_fusiform": _sum_cortical_volume(lh, rh, "fusiform"),
        "smri_midtemp": _sum_cortical_volume(lh, rh, "middletemporal"),
        "smri_ventricles": _sum_structures(structures, "Left-Lateral-Ventricle", "Right-Lateral-Ventricle"),
        "smri_wholebrain": _require_measure(measures, "BrainSegNotVent", "BrainSegVolNotVent"),
        "smri_thickness_entorhinal": _mean_cortical_thickness(lh, rh, "entorhinal"),
        "smri_thickness_fusiform": _mean_cortical_thickness(lh, rh, "fusiform"),
        "smri_thickness_midtemp": _mean_cortical_thickness(lh, rh, "middletemporal"),
    }
    return {name: float(value) for name, value in features.items()}


def parse_aseg_stats(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    measures: dict[str, float] = {}
    structures: dict[str, float] = {}
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("# Measure "):
            fields = [field.strip() for field in line[len("# Measure ") :].split(",")]
            if len(fields) >= 4:
                value = float(fields[-2])
                measures[fields[0]] = value
                measures[fields[1]] = value
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 5 and fields[0].isdigit():
            structures[fields[4]] = float(fields[3])
    return measures, structures


def parse_aparc_stats(path: str | Path) -> dict[str, dict[str, float]]:
    columns: list[str] | None = None
    rows: dict[str, dict[str, float]] = {}
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("# ColHeaders "):
            columns = line[len("# ColHeaders ") :].split()
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if columns is None or len(fields) < len(columns):
            continue
        values: dict[str, float] = {}
        for name, value in zip(columns[1:], fields[1:]):
            values[name] = float(value)
        rows[fields[0]] = values
    return rows


def required_artifacts(root: str | Path) -> list[str]:
    root = Path(root)
    required = ["stats/aseg.stats", *REQUIRED_SURFACES]
    for hemi in ("lh", "rh"):
        existing = next((path for path in DKT_STATS[hemi] if (root / path).is_file()), DKT_STATS[hemi][0])
        required.append(existing)
    return required


def _find_dkt_stats(root: Path, hemi: str) -> str:
    found = next((path for path in DKT_STATS[hemi] if (root / path).is_file()), None)
    if found is None:
        raise FileNotFoundError(f"Missing {hemi} DKT stats under {root}")
    return found


def _require_measure(measures: dict[str, float], *names: str) -> float:
    for name in names:
        if name in measures:
            return measures[name]
    raise ValueError(f"Missing aseg measure; expected one of {names}")


def _sum_structures(structures: dict[str, float], *names: str) -> float:
    missing = [name for name in names if name not in structures]
    if missing:
        raise ValueError(f"Missing aseg structures: {missing}")
    return sum(structures[name] for name in names)


def _sum_cortical_volume(lh: dict[str, dict[str, float]], rh: dict[str, dict[str, float]], region: str) -> float:
    return _cortical_value(lh, region, "GrayVol") + _cortical_value(rh, region, "GrayVol")


def _mean_cortical_thickness(lh: dict[str, dict[str, float]], rh: dict[str, dict[str, float]], region: str) -> float:
    return (_cortical_value(lh, region, "ThickAvg") + _cortical_value(rh, region, "ThickAvg")) / 2.0


def _cortical_value(table: dict[str, dict[str, float]], region: str, measure: str) -> float:
    if region not in table or measure not in table[region]:
        raise ValueError(f"Missing cortical value {region}.{measure}")
    return table[region][measure]
