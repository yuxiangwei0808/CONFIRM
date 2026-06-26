import numpy as np
import pandas as pd

from confirm.agent import _run_scalar_contract
from confirm.analysis import fit_effect, run_primary
from confirm.contract import ClaimContract
from confirm.harmonize import combat_harmonize
from confirm.multiverse import run_multiverse
from confirm.power import power_check
from confirm.replication import replicate
from confirm.schema import validate_canonical
from confirm.verdict import decide


SEED = 20260615


def _contract(
    claim_id="synthetic_age",
    outcome="smri_meanthickness",
    predictor="age",
    direction="negative",
    discovery="DISC",
    rep="REP",
    min_power=0.8,
    min_fraction=0.6,
    ref_effect=None,
):
    return ClaimContract.model_validate(
        {
            "claim_id": claim_id,
            "question": "Synthetic test claim.",
            "estimand": {"type": "association", "outcome": outcome, "predictor": predictor, "group": None, "direction": direction},
            "covariates": ["sex", "eTIV", "site"],
            "inclusion": None,
            "discovery_cohort": discovery,
            "replication_cohorts": [rep],
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["sex", "eTIV"], "motion_check": False},
                "power": {"min_power": min_power, "ref_effect": ref_effect},
                "multiverse": {"min_fraction_consistent": min_fraction},
                "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": True, "harmonize": "combat"},
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def _cohort(cohort, n=160, beta_age=-0.06, noise=0.25, seed=SEED, site_shift=0.0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 75, n)
    sex = rng.choice(["M", "F"], n)
    site = rng.choice(["S1", "S2"], n)
    etiv = rng.normal(1500, 120, n)
    sex_effect = np.where(sex == "M", 0.05, -0.05)
    site_effect = np.where(site == "S2", site_shift, 0.0)
    y = 3.0 + beta_age * age + 0.0002 * (etiv - 1500) + sex_effect + site_effect + rng.normal(0, noise, n)
    return validate_canonical(
        pd.DataFrame(
            {
                "subject_id": [f"{cohort}_{i}" for i in range(n)],
                "session": "ses-01",
                "cohort": cohort,
                "site": site,
                "age": age,
                "sex": sex,
                "dx": "TC",
                "eTIV": etiv,
                "smri_meanthickness": y,
            }
        )
    )


def _run_all(disc, rep, contract):
    primary = run_primary(disc, contract)
    multiverse = run_multiverse(disc, contract)
    power = power_check(primary, contract, contract.gates.power.ref_effect)
    replication = replicate(primary, disc, [rep], contract)
    return decide(primary, multiverse, power, replication, contract)


def test_planted_strong_effect_replicates_confirmed():
    contract = _contract()
    verdict = _run_all(_cohort("DISC", seed=1), _cohort("REP", seed=2), contract)
    assert verdict.label == "confirmed"


def test_planted_effect_in_discovery_only_non_replicated():
    contract = _contract()
    disc = _cohort("DISC", seed=3, beta_age=-0.07)
    rep = _cohort("REP", seed=4, beta_age=0.0, noise=1.2)
    verdict = _run_all(disc, rep, contract)
    assert verdict.label == "non_replicated"


def test_tiny_n_planted_effect_under_powered():
    contract = _contract(min_power=0.95, ref_effect=0.2)
    disc = _cohort("DISC", n=14, beta_age=-0.08, noise=0.35, seed=5)
    rep = _cohort("REP", n=80, beta_age=-0.08, noise=0.35, seed=6)
    verdict = _run_all(disc, rep, contract)
    assert verdict.label == "under_powered"


def test_fork_dependent_effect_fragile():
    # Suppressor design: the age effect is only visible AFTER adjusting for eTIV.
    # The primary spec (eTIV included) is strong and well-powered -> it passes the
    # power gate; but every fork that drops eTIV loses the effect -> multiverse
    # instability -> "fragile". (Engine gate order is power -> multiverse, so a
    # genuine fragile case must stay well-powered in the primary spec.)
    rng = np.random.default_rng(7)
    n = 220
    age = rng.uniform(20, 75, n)
    sex = rng.choice(["M", "F"], n)
    site = rng.choice(["S1", "S2"], n)
    etiv = 1500.0 + 4.0 * age + rng.normal(0, 120, n)
    # marginal age slope ~ -0.06 + 0.015*4 = 0 (omitting eTIV); partial ~ -0.06 (with eTIV)
    y = (3.0 - 0.06 * age + 0.015 * (etiv - 1500.0)
         + np.where(sex == "M", 0.05, -0.05) + rng.normal(0, 0.18, n))
    disc = validate_canonical(pd.DataFrame({
        "subject_id": [f"DISC_{i}" for i in range(n)], "session": "ses-01", "cohort": "DISC",
        "site": site, "age": age, "sex": sex, "dx": "TC", "eTIV": etiv, "smri_meanthickness": y,
    }))
    rep = _cohort("REP", n=200, beta_age=-0.05, noise=0.3, seed=8)
    contract = _contract(min_fraction=0.8)
    verdict = _run_all(disc, rep, contract)
    assert verdict.label == "fragile", verdict.rationale


def test_pure_noise_not_confirmed():
    contract = _contract()
    disc = _cohort("DISC", beta_age=0.0, noise=1.0, seed=9)
    rep = _cohort("REP", beta_age=0.0, noise=1.0, seed=10)
    verdict = _run_all(disc, rep, contract)
    assert verdict.label != "confirmed"


def test_combat_removes_site_shift_preserves_age_effect():
    df = _cohort("DISC", n=220, beta_age=-0.04, noise=0.15, seed=11, site_shift=2.0)
    before = abs(df.groupby("site")["smri_meanthickness"].mean().diff().dropna().iloc[0])
    harmonized = combat_harmonize(df, ["smri_meanthickness"], batch_col="site", covars=["age", "sex", "eTIV"])
    after = abs(harmonized.groupby("site")["smri_meanthickness"].mean().diff().dropna().iloc[0])
    assert after < before * 0.5
    contract = _contract(rep="DISC")
    effect = run_primary(harmonized, contract)
    assert effect.beta < 0
    assert effect.p < 0.05


def _sex_contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "synthetic_sex",
            "question": "Synthetic sex contrast.",
            "estimand": {
                "type": "group_diff",
                "outcome": "smri_meanthickness",
                "predictor": "sex",
                "group": {"var": "sex", "case": "F", "control": "M"},
                "direction": "two_sided",
            },
            "covariates": ["age", "eTIV"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["age", "eTIV"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "combat"},
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def test_sex_predictor_covariate_override_dedups_and_fits():
    df = _cohort("DISC", n=180, seed=12)
    contract = _sex_contract()
    legacy_contract = contract.model_copy(update={"covariates": ["age", "sex", "eTIV"]})
    effect = fit_effect(df, legacy_contract, covariates=legacy_contract.covariates)
    assert effect.n > 0
    assert np.isfinite(effect.beta)
    assert np.isfinite(effect.p)


def test_group_by_site_without_site_covariate_abstains_confound_incomplete():
    rng = np.random.default_rng(13)
    n = 180
    site = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    sex = rng.choice(["M", "F"], n)
    df = validate_canonical(
        pd.DataFrame(
            {
                "subject_id": [f"DISC_{i}" for i in range(n)],
                "session": "ses-01",
                "cohort": "DISC",
                "site": site,
                "age": rng.uniform(20, 75, n),
                "sex": sex,
                "dx": "TC",
                "smri_meanthickness": 3.0 + (site == "S1").astype(float) * 0.4 + rng.normal(0, 0.2, n),
            }
        )
    )
    contract = ClaimContract.model_validate(
        {
            "claim_id": "site_confound_incomplete",
            "question": "Synthetic omitted-site confound.",
            "estimand": {
                "type": "group_diff",
                "outcome": "smri_meanthickness",
                "predictor": "site",
                "group": {"var": "site", "case": "S1", "control": "S2"},
                "direction": "two_sided",
            },
            "covariates": ["age", "sex"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["DISC"],
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["age", "sex"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "combat"},
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )
    verdict, results = _run_scalar_contract(contract, df, [df], ref_effect=None)
    assert verdict.label == "fragile"
    assert verdict.abstained is True
    assert verdict.gates["reason"] == "confound_incomplete"
    assert results["confound_completeness"]["reason"] == "confound_incomplete"
