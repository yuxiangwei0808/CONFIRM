"""Small synthetic examples for the public quickstart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from confirm.agent import run_claim
from confirm.contract import ClaimContract
from confirm.verdict import classify_support


def make_example_data(root: Path) -> dict[str, Path]:
    """Create independent discovery/replication samples and three fixed plans."""
    root.mkdir(parents=True, exist_ok=False)
    data = root / "cohorts"
    plans = root / "contracts"
    data.mkdir()
    plans.mkdir()
    for cohort, seed in (("DEMO_DISC", 1701), ("DEMO_REP", 1702)):
        rng = np.random.default_rng(seed)
        n = 600
        site_b = rng.integers(0, 2, n)
        age = rng.uniform(20, 65, n) + 15 * site_b
        sex = rng.choice(["M", "F"], n)
        etiv = rng.normal(1500, 100, n)
        df = pd.DataFrame({
            "subject_id": [f"{cohort}_{i}" for i in range(n)],
            "session": "ses-01", "cohort": cohort,
            "site": np.where(site_b, "B", "A"), "age": age,
            "sex": sex, "eTIV": etiv,
            "smri_signal": 4 - .025 * age + .0002 * etiv + rng.normal(0, .15, n),
            "smri_site_only": 2 + 2 * site_b + rng.normal(0, .15, n),
            "smri_discovery_only": 4 + (-.025 * age if cohort == "DEMO_DISC" else 0)
                + rng.normal(0, .15, n),
        })
        df.to_parquet(data / f"{cohort}.parquet", index=False)
    specs = {
        "supported": ("smri_signal", "negative", ["sex", "eTIV", "site"]),
        "site_confounded": ("smri_site_only", "positive", ["sex", "eTIV"]),
        "nonreplicating": ("smri_discovery_only", "negative", ["sex", "eTIV", "site"]),
    }
    paths = {}
    for name, (outcome, direction, covariates) in specs.items():
        contract = ClaimContract.model_validate({
            "claim_id": f"demo_{name}",
            "question": f"Is age associated with {outcome} in these synthetic samples?",
            "estimand": {"type": "association", "outcome": outcome,
                         "predictor": "age", "group": None, "direction": direction},
            "covariates": covariates, "inclusion": None,
            "discovery_cohort": "DEMO_DISC", "replication_cohorts": ["DEMO_REP"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": .05, "family_size": 1},
                "confound": {"require_covariates": covariates, "motion_check": False},
                "power": {"min_power": .8, "ref_effect": .5},
                "multiverse": {"min_fraction_consistent": .6},
                "replication": {"alpha": .05, "require_same_sign": True,
                                "require_ci_overlap": False, "harmonize": "none"},
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        })
        paths[name] = plans / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(contract.model_dump(mode="json"), sort_keys=False))
    return paths


def run_demo(root: Path) -> dict:
    """Run fixed plans; this example makes no model calls or dataset downloads."""
    paths = make_example_data(root)
    results = {}
    for name, path in paths.items():
        verdict = run_claim(path, root / "cohorts", root / "results" / name,
                            command=["confirm", "demo", "--out", str(root)])
        results[name] = {
            "label": verdict.label, "rationale": verdict.rationale,
            "support": {label: classify_support(verdict.gates, tier).supported
                        for label, tier in (("screen", "discovery"), ("replicate", "replicated"), ("full", "confirmed"))},
        }
    summary = {"synthetic": True, "llm_calls": 0, "results": results}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
