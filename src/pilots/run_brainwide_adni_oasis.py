"""Brain-wide ADNI -> OASIS-3 AD-vs-CN pattern replication demo.

Run: python -m pilots.run_brainwide_adni_oasis
Optional agent path: CONFIRM_LLM=openai python -m pilots.run_brainwide_adni_oasis --agent
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from confirm.agent import run_question
from confirm.brainwide import run_brainwide
from confirm.contract import ClaimContract
from confirm.env import load_env
from confirm.replication import replicate_brainwide
from confirm.verdict import decide_brainwide
from confirm.agent import _brainwide_multiverse_equivalent, _best_region_effect
from confirm.power import power_check
from pilots.run_adni import load_adni
from pilots.run_adni_oasis import SHARED, load_oasis3


def _contract(shared_regions: list[str]) -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "brainwide_adni_oasis_ad_signature",
            "question": "Does the AD-vs-CN regional atrophy pattern discovered in ADNI replicate in OASIS-3?",
            "estimand": {
                "type": "group_diff",
                "outcome": shared_regions,
                "predictor": "dx",
                "group": {"var": "dx", "case": "Dementia", "control": "CN"},
                "direction": "negative",
                "unit": "brainwide",
                "region_set": "shared_ad_signature",
            },
            "covariates": ["age", "sex", "eTIV"],
            "inclusion": None,
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": len(shared_regions)},
                "confound": {"require_covariates": ["age", "sex", "eTIV"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "combat",
                    "pattern_corr_min": 0.5,
                    "region_replication_frac_min": 0.5,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def _write_canonical_for_agent(adni: pd.DataFrame, oasis: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    adni.to_parquet(out_dir / "ADNI.parquet")
    oasis.to_parquet(out_dir / "OASIS3.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="Run the LLM-drafted variant if a key/backend is configured")
    parser.add_argument("--agent-data-dir", default="runs/brainwide_adni_oasis/canonical")
    parser.add_argument("--agent-out", default="runs/brainwide_adni_oasis/agent")
    args = parser.parse_args(argv)

    adni = load_adni().copy()
    adni["cohort"] = "ADNI"
    oasis = load_oasis3()
    shared_regions = [region for region, _, _ in SHARED]
    contract = _contract(shared_regions)

    regions = run_brainwide(adni, contract)
    replication = replicate_brainwide(regions, adni, [oasis], contract)
    multiverse = _brainwide_multiverse_equivalent(regions, contract)
    power = power_check(_best_region_effect(regions), contract, ref_effect=None)
    verdict = decide_brainwide(regions, multiverse, power, replication, contract)

    print("==== Brain-wide ADNI -> OASIS-3 AD-vs-CN pattern replication ====")
    print(f"verdict={verdict.label}")
    print(f"pattern_corr={replication.pattern_corr:.3f}")
    print(f"dice={replication.dice:.3f}")
    print(f"region_replication_fraction={replication.region_replication_fraction:.3f}")
    print(f"rationale={verdict.rationale}")

    if args.agent:
        load_env()
        backend = os.getenv("CONFIRM_LLM", "openai").lower()
        has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or backend in {"standin", "offline", "manual"})
        if not has_key:
            print("Agent variant skipped: no LLM key/backend available.")
            return 0
        data_dir = Path(args.agent_data_dir)
        _write_canonical_for_agent(adni, oasis, data_dir)
        question = "Does the AD-vs-CN regional atrophy pattern discovered in ADNI replicate in OASIS-3?"
        agent_verdict = run_question(question, data_dir, args.agent_out, approve=False)
        print(f"agent_verdict={agent_verdict.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
