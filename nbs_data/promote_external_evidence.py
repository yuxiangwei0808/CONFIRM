#!/usr/bin/env python3
"""Validate and explicitly promote staged external evidence parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from confirm.schema import validate_canonical


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = Path(args.run_dir)
    canonical_dir = run_dir / "canonical"
    active_dir = Path(args.active_dir)
    active_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    promoted: list[dict[str, object]] = []
    for source in sorted(canonical_dir.glob("*.parquet")):
        sidecar = source.with_suffix(".features.json")
        if not sidecar.exists():
            raise ValueError(f"Missing feature manifest for {source}")
        feature_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if feature_manifest.get("status") != "ready":
            raise ValueError(f"Refusing to promote non-ready dataset {source.stem}: {feature_manifest.get('status')}")
        frame = validate_canonical(pd.read_parquet(source))
        if len(frame) != int(feature_manifest.get("rows", feature_manifest.get("output_row_count", len(frame)))):
            raise ValueError(f"Row-count mismatch for {source}")
        destination = active_dir / source.name
        archived = None
        if destination.exists():
            archived = active_dir / "_archive" / timestamp / destination.name
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, archived)
        shutil.copy2(source, destination)
        promoted.append(
            {
                "dataset_id": source.stem,
                "source": str(source),
                "destination": str(destination),
                "archived_previous": str(archived) if archived else None,
                "rows": len(frame),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    result = {
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "active_dir": str(active_dir),
        "promoted": promoted,
    }
    manifest_path = active_dir / f"promotion_{timestamp}.json"
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--active-dir", default="data/prepared_data/external")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
