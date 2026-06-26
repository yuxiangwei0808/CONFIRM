import numpy as np
import pandas as pd

from confirm.brainwide import run_brainwide
from confirm.contract import ClaimContract
from confirm.power import power_check
from confirm.replication import replicate_brainwide
from confirm.results import MultiverseResult, MultiverseSpecResult
from confirm.schema import validate_canonical
from confirm.verdict import decide_brainwide


REGIONS = [f"smri_region_{i:02d}" for i in range(8)]
PATTERN = np.array([-1.2, -1.0, -0.8, -0.6, -0.45, -0.3, -0.2, -0.1])


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "synthetic_brainwide_ad",
            "question": "Synthetic brain-wide pattern.",
            "estimand": {
                "type": "group_diff",
                "outcome": REGIONS,
                "predictor": "dx",
                "group": {"var": "dx", "case": "AD", "control": "CN"},
                "direction": "negative",
                "unit": "brainwide",
                "region_set": "synthetic",
            },
            "covariates": ["age", "sex", "eTIV", "site"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": len(REGIONS)},
                "confound": {"require_covariates": ["age", "sex", "eTIV"], "motion_check": False},
                "power": {"min_power": 0.1, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": True,
                    "harmonize": "combat",
                    "pattern_corr_min": 0.5,
                    "region_replication_frac_min": 0.5,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def _cohort(cohort: str, pattern: np.ndarray, seed: int, n: int = 220, noise: float = 0.55) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dx = np.array(["AD"] * (n // 2) + ["CN"] * (n - n // 2))
    rng.shuffle(dx)
    age = rng.normal(72, 7, n) + np.where(dx == "AD", 2.0, 0.0)
    sex = rng.choice(["M", "F"], n)
    site = rng.choice(["S1", "S2"], n)
    etiv = rng.normal(1500, 120, n)
    data = {
        "subject_id": [f"{cohort}_{i}" for i in range(n)],
        "session": "ses-01",
        "cohort": cohort,
        "site": site,
        "age": age,
        "sex": sex,
        "dx": dx,
        "eTIV": etiv,
    }
    for region, beta in zip(REGIONS, pattern):
        data[region] = (
            5.0
            + beta * (dx == "AD").astype(float)
            - 0.01 * age
            + 0.0001 * (etiv - 1500)
            + np.where(sex == "M", 0.03, -0.03)
            + rng.normal(0, noise, n)
        )
    return validate_canonical(pd.DataFrame(data))


def _multiverse_equivalent(regions):
    significant = any(region.significant for region in regions.regions)
    best = sorted(regions.regions, key=lambda region: region.effect.p)[0].effect
    return MultiverseResult(
        fraction_consistent=1.0 if significant else 0.0,
        passed=significant,
        specs=[
            MultiverseSpecResult(
                spec_id="brainwide_primary",
                same_sign=significant,
                significant=significant,
                beta=best.beta,
                p=best.p,
                n=best.n,
            )
        ],
    )


def _run_all(disc: pd.DataFrame, rep: pd.DataFrame, contract: ClaimContract):
    regions = run_brainwide(disc, contract)
    replication = replicate_brainwide(regions, disc, [rep], contract)
    best = sorted(regions.regions, key=lambda region: region.effect.p)[0].effect
    power = power_check(best, contract, ref_effect=None)
    return decide_brainwide(regions, _multiverse_equivalent(regions), power, replication, contract), replication


def test_planted_spatial_pattern_replicates_confirmed():
    contract = _contract()
    verdict, replication = _run_all(_cohort("DISC", PATTERN, 1), _cohort("REP", PATTERN, 2), contract)
    assert verdict.label == "confirmed", verdict.rationale
    assert replication.pattern_corr >= 0.5
    assert replication.region_replication_fraction >= 0.5


def test_pattern_present_only_in_discovery_non_replicated():
    contract = _contract()
    verdict, replication = _run_all(_cohort("DISC", PATTERN, 3), _cohort("REP", np.zeros_like(PATTERN), 4, noise=1.0), contract)
    assert verdict.label == "non_replicated", verdict.rationale
    assert replication.region_replication_fraction < 0.5


def test_pure_noise_region_profile_not_confirmed():
    contract = _contract()
    verdict, _ = _run_all(_cohort("DISC", np.zeros_like(PATTERN), 5, noise=1.0), _cohort("REP", np.zeros_like(PATTERN), 6, noise=1.0), contract)
    assert verdict.label != "confirmed"
