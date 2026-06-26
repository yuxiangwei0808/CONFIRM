"""Pilot 2: injected null must not confirm."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

from bench.injected_nulls import inject_site_confound, random_label_null
from confirm.agent import run_claim
from confirm.ingest.abide import AbideAdapter
from confirm.schema import validate_canonical


def _ensure_abide(data_dir: Path) -> Path:
    path = data_dir / "ABIDE.parquet"
    if not path.exists():
        AbideAdapter(data_dir=data_dir.parent / "raw" / "abide").write(data_dir)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/canonical")
    parser.add_argument("--out", default="runs/p2_injected_null")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    pilot_data = out_dir / "canonical"
    pilot_data.mkdir(parents=True, exist_ok=True)
    source = validate_canonical(pd.read_parquet(_ensure_abide(data_dir)))
    source = source[source["dx"].isna() | source["dx"].eq("TC")].copy()
    split = source["subject_id"].astype(str).map(lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % 2 == 0)
    disc, contract, expected = inject_site_confound(source.loc[split].copy())
    rep, _, _ = random_label_null(source.loc[~split].copy())
    disc["cohort"] = "P2_DISC"
    rep["cohort"] = "P2_REP"
    contract_data = contract.model_dump()
    contract_data["discovery_cohort"] = "P2_DISC"
    contract_data["replication_cohorts"] = ["P2_REP"]
    validate_canonical(disc).to_parquet(pilot_data / "P2_DISC.parquet", index=False)
    validate_canonical(rep).to_parquet(pilot_data / "P2_REP.parquet", index=False)
    contract_path = out_dir / "p2_contract.yaml"
    with contract_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(contract_data, handle, sort_keys=False)
    verdict = run_claim(contract_path, pilot_data, out_dir, command=sys.argv)
    print(f"expected_not=confirmed expected_class={expected} actual={verdict.label}")
    return 0 if verdict.label != "confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
