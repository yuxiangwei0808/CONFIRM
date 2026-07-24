"""Shared, read-only utilities for claim-search paper analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


RESULT_HEADER_KEYS = (
    "created_at",
    "status",
    "llm_model",
    "max_workers",
    "parallel_backend",
    "config",
    "candidate_evaluation",
    "provenance",
    "searchable_claim_count",
    "completed_search_count",
    "skipped_search_count",
    "summary",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_result_header(path: str | Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Read top-level metadata without deserializing large ``rows`` or ``states`` arrays."""

    source = Path(path)
    data = bytearray()
    boundary = b'\n  "rows":'
    with source.open("rb") as handle:
        while len(data) < max_bytes:
            chunk = handle.read(min(1024 * 1024, max_bytes - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            marker = data.find(boundary)
            if marker >= 0:
                data = data[:marker]
                break
    text = data.decode("utf-8")
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {}
    for key in RESULT_HEADER_KEYS:
        match = re.search(rf'(?m)^  {re.escape(json.dumps(key))}:\s*', text)
        if not match:
            continue
        try:
            value, _ = decoder.raw_decode(text[match.end() :])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Result header field {key!r} is incomplete within {max_bytes} bytes: {source}"
            ) from exc
        result[key] = value
    required = {"status", "config", "provenance", "summary"}
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"Result header is missing fields {missing}: {source}")
    return result


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def write_csv_atomic(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def arm_sort_key(arm_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"r(\d+)_c(\d+)", arm_id)
    if not match:
        raise ValueError(f"Invalid arm ID: {arm_id}")
    return int(match.group(1)), int(match.group(2))


def clustered_binary_interval(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 20260721,
) -> tuple[float, float, float]:
    """Parent-clustered percentile interval for one binary outcome per parent."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def output_manifest(
    *,
    inputs: Sequence[str | Path],
    outputs: Sequence[str | Path],
    restrictions: Sequence[str],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "inputs": [
            {"path": str(Path(path)), "sha256": sha256_file(path)}
            for path in inputs
            if Path(path).exists() and Path(path).is_file()
        ],
        "outputs": [
            {"path": str(Path(path)), "sha256": sha256_file(path)}
            for path in outputs
            if Path(path).exists() and Path(path).is_file()
        ],
        "parameters": parameters,
        "interpretation_restrictions": list(restrictions),
    }


def merge_analysis_manifest(
    manifest_path: str | Path,
    *,
    section_name: str,
    section_payload: dict[str, Any],
    inputs: Sequence[str | Path],
    outputs: Sequence[str | Path],
    restrictions: Sequence[str],
) -> dict[str, Any]:
    """Merge one analysis phase while refreshing all recorded file hashes."""

    destination = Path(manifest_path)
    manifest = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {}
    manifest[section_name] = section_payload
    input_paths = {
        str(Path(item["path"]))
        for item in manifest.get("inputs") or []
        if isinstance(item, dict) and item.get("path")
    }
    output_paths = {
        str(Path(item["path"]))
        for item in manifest.get("outputs") or []
        if isinstance(item, dict) and item.get("path")
    }
    input_paths.update(str(Path(path)) for path in inputs)
    output_paths.update(str(Path(path)) for path in outputs)
    manifest["inputs"] = [
        {"path": path, "sha256": sha256_file(path)}
        for path in sorted(input_paths)
        if Path(path).exists() and Path(path).is_file()
    ]
    manifest["outputs"] = [
        {"path": path, "sha256": sha256_file(path)}
        for path in sorted(output_paths)
        if Path(path).exists() and Path(path).is_file()
    ]
    manifest["interpretation_restrictions"] = sorted(
        set(manifest.get("interpretation_restrictions") or []) | set(restrictions)
    )
    write_json_atomic(destination, manifest)
    return manifest


def configure_matplotlib(out_dir: str | Path) -> None:
    cache = Path(out_dir) / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
