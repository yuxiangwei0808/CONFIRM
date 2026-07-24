from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from confirm.analysis import AnalysisNonIdentifiableError, _diagnostics, build_analysis_design, fit_effect
from confirm.contract import ClaimContract
from confirm.multiverse import run_multiverse
from confirm.results import EffectResult


def _contract(covariates: list[str]) -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "design_test",
            "question": "Is x associated with y?",
            "estimand": {
                "type": "association",
                "outcome": "smri_y",
                "predictor": "x",
                "group": None,
                "direction": "positive",
                "unit": "scalar",
            },
            "covariates": covariates,
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": covariates, "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {"alpha": 0.05, "require_same_sign": True, "harmonize": "none"},
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def test_exactly_singular_design_is_blocked_before_statsmodels():
    x = np.linspace(-1.0, 1.0, 80)
    frame = pd.DataFrame({"x": x, "duplicate_x": x, "smri_y": x + 0.1})

    with pytest.raises(AnalysisNonIdentifiableError) as error:
        fit_effect(frame, _contract(["duplicate_x"]))

    assert error.value.reason == "design matrix is rank deficient"
    assert error.value.diagnostics["full_rank"] is False


def test_near_singular_design_is_full_rank_but_warned():
    rng = np.random.default_rng(9)
    x = rng.normal(size=200)
    frame = pd.DataFrame(
        {
            "x": x,
            "almost_x": x + rng.normal(scale=1e-8, size=len(x)),
            "smri_y": 0.4 * x + rng.normal(size=len(x)),
        }
    )

    design = build_analysis_design(frame, _contract(["almost_x"]))

    assert design.diagnostics["full_rank"] is True
    assert design.diagnostics["condition_number_warning"] is True
    assert design.diagnostics["condition_number_standardized"] > 1e8


def test_robust_fit_suppresses_tall_matrix_pinv_noise():
    rng = np.random.default_rng(17)
    x = rng.normal(size=1200)
    frame = pd.DataFrame({"x": x, "smri_y": 0.3 * x + rng.normal(size=len(x))})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        effect = fit_effect(frame, _contract([]), model="robust")

    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert all(math.isfinite(value) for value in (effect.beta, effect.se, effect.p))


def test_optional_influence_warnings_are_recorded_without_leaking():
    class FakeInfluence:
        @property
        def cooks_distance(self):
            warnings.warn("invalid value encountered in sqrt", RuntimeWarning)
            return np.array([0.1, np.nan, 0.3]), None

    class FakeFitted:
        resid = np.array([0.2, -0.1, -0.1])

        @staticmethod
        def get_influence():
            return FakeInfluence()

    y = pd.Series([1.0, 2.0, 3.0])
    x = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = _diagnostics(y, x, FakeFitted())

    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert diagnostics["cooks_distance_warning_count"] == 1
    assert diagnostics["cooks_distance_nonfinite_count"] == 1
    assert diagnostics["cooks_distance_top"] == [
        {"row": "2", "value": 0.3},
        {"row": "0", "value": 0.1},
    ]


def test_multiverse_errors_remain_in_consistency_denominator(monkeypatch):
    calls = 0

    def fake_fit_effect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AnalysisNonIdentifiableError("singular fork")
        return EffectResult(
            beta=1.0,
            se=0.1,
            ci_low=0.8,
            ci_high=1.2,
            p=0.001,
            n=100,
            dof=98.0,
            standardized_effect=0.5,
        )

    monkeypatch.setattr("confirm.multiverse.fit_effect", fake_fit_effect)
    result = run_multiverse(
        pd.DataFrame({"x": [0.0], "smri_y": [0.0]}),
        _contract([]),
        forks={"model": ["ols", "robust"]},
    )

    assert len(result.specs) == 2
    assert sum(item.status == "error" for item in result.specs) == 1
    assert result.fraction_consistent == 0.5
    assert not result.passed
