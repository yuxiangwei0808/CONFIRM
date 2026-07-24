"""Minimal JSON/JSONL I/O shared by benchmark commands."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON objects."""

    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def atomic_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """Atomically write models or mappings as newline-delimited JSON."""

    payloads = [
        row.model_dump(mode="json") if hasattr(row, "model_dump") else row
        for row in rows
    ]
    atomic_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payloads),
    )
