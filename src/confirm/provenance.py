"""Re-runnable provenance receipts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def file_sha256(path: str | Path) -> str:
    """Compute a SHA256 digest for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    """Return the current git SHA when available."""

    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return proc.stdout.strip()
    except Exception:
        return None


def package_versions(packages: list[str] | None = None) -> dict[str, str]:
    """Collect package versions for the provenance receipt."""

    names = packages or [
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "nibabel",
        "nilearn",
        "neuroHarmonize",
        "pyyaml",
        "pydantic",
        "pyarrow",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def make_receipt(
    contract_path: str | Path,
    cohort_paths: list[str | Path],
    command: list[str] | None,
    seed: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Build a provenance receipt dictionary."""

    return {
        "contract": {"path": str(contract_path), "sha256": file_sha256(contract_path)},
        "cohorts": [{"path": str(path), "sha256": file_sha256(path)} for path in cohort_paths],
        "code": {"git_sha": git_sha()},
        "environment": {"python": sys.version, "packages": package_versions()},
        "seed": seed,
        "command": command or sys.argv,
        "results": _jsonable(results),
    }


def write_receipt(out_dir: str | Path, receipt: dict[str, Any]) -> Path:
    """Write a JSON provenance receipt."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "receipt.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(receipt), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path

