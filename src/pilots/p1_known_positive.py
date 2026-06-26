"""Pilot 1: age-related atrophy should confirm across open cohorts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from confirm.agent import run_claim
from confirm.ingest.abide import AbideAdapter
from confirm.ingest.adhd200 import Adhd200Adapter


def _ensure_open_canonical(data_dir: Path) -> None:
    if not (data_dir / "ABIDE.parquet").exists():
        AbideAdapter(data_dir=data_dir.parent / "raw" / "abide").write(data_dir)
    if not (data_dir / "ADHD200.parquet").exists():
        Adhd200Adapter(data_dir=data_dir.parent / "raw" / "adhd200").write(data_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/canonical")
    parser.add_argument("--contract", default="configs/contracts/example_age_atrophy.yaml")
    parser.add_argument("--out", default="runs/p1_known_positive")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    _ensure_open_canonical(data_dir)
    verdict = run_claim(args.contract, data_dir, args.out, command=sys.argv)
    expected = "confirmed"
    print(f"expected={expected} actual={verdict.label}")
    return 0 if verdict.label == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())

