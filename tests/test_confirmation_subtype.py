from confirm.contract import ClaimContract
from confirm.results import (
    CohortEffectSummary,
    CohortReplicationResult,
    EffectResult,
    HeterogeneityResult,
    MultiverseResult,
    PowerResult,
    ReplicationResult,
    scalar_confirmation_subtype,
)
from confirm.verdict import decide


def _heterogeneity(i2: float, effects: list[tuple[str, float, float]]) -> HeterogeneityResult:
    return HeterogeneityResult(
        cohort_effects=[
            CohortEffectSummary(cohort=cohort, standardized_effect=effect, se_standardized_effect=se, sign="negative", p=1e-6, n=100)
            for cohort, effect, se in effects
        ],
        random_effect=-0.5,
        random_effect_se=0.1,
        ci_low=-0.7,
        ci_high=-0.3,
        q=10.0,
        tau2=0.1,
        i2=i2,
        high_i2=i2 >= 75.0,
    )


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "scalar_subtype_test",
            "question": "Synthetic scalar subtype test.",
            "estimand": {
                "type": "association",
                "outcome": "smri_x",
                "predictor": "age",
                "group": None,
                "direction": "negative",
            },
            "covariates": ["sex"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["sex"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "none"},
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def _effect() -> EffectResult:
    return EffectResult(
        beta=-1.0,
        se=0.1,
        ci_low=-1.2,
        ci_high=-0.8,
        p=1e-6,
        n=100,
        dof=96.0,
        standardized_effect=-0.5,
    )


def test_scalar_confirmation_subtype_uses_heterogeneity_thresholds():
    low = _heterogeneity(10.0, [("DISC", -0.5, 0.1), ("REP", -0.55, 0.1)])
    moderate = _heterogeneity(55.0, [("DISC", -0.5, 0.1), ("REP", -0.8, 0.1)])
    high = _heterogeneity(95.0, [("DISC", -1.7, 0.06), ("REP", -1.3, 0.07)])

    assert scalar_confirmation_subtype(low) == "transportable_confirmed"
    assert scalar_confirmation_subtype(moderate) == "magnitude_confirmed"
    assert scalar_confirmation_subtype(high) == "direction_confirmed"


def test_confirmed_verdict_reports_direction_only_for_high_i2():
    effect = _effect()
    heterogeneity = _heterogeneity(95.0, [("DISC", -1.7, 0.06), ("REP", -1.3, 0.07)])
    replication = ReplicationResult(
        passed=True,
        reason="passed",
        cohort_results=[CohortReplicationResult("REP", True, "passed", effect)],
        harmonized=False,
        heterogeneity=heterogeneity,
        replicated_but_heterogeneous=True,
        confirmation_subtype=scalar_confirmation_subtype(heterogeneity),
        confirmation_i2=heterogeneity.i2,
    )
    verdict = decide(
        effect,
        MultiverseResult(fraction_consistent=1.0, passed=True, specs=[]),
        PowerResult(0.99, 20.0, 1.0, -0.5, False, None, "power ok"),
        replication,
        _contract(),
    )

    assert verdict.label == "confirmed"
    assert verdict.confirmation_subtype == "direction_confirmed"
    assert verdict.heterogeneity_i2 == 95.0
    assert verdict.gates["confirmation_subtype"] == "direction_confirmed"
    assert "confirmation_subtype=direction_confirmed" in verdict.rationale
