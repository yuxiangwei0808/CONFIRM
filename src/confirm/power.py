"""Power and winner's-curse checks."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower

from confirm.contract import ClaimContract
from confirm.results import EffectResult, PowerResult


# Pre-declared minimal effect of interest, used when no external/literature
# reference effect is supplied. Power is assessed against this MDE (or a supplied
# reference), NEVER against the observed effect, so a lucky large estimate in a
# small sample cannot certify its own power.
DEFAULT_MDE = 0.3


def _reference_effect(effect: EffectResult, ref_effect: float | None) -> float:
    if ref_effect is not None and not math.isfinite(float(ref_effect)):
        raise ValueError("Reference effect must be finite")
    if ref_effect is not None:
        return abs(float(ref_effect))
    return DEFAULT_MDE


def _achieved_power(effect_size: float, n: int, alpha: float) -> float:
    if effect_size <= 0 or n <= 3:
        return 0.0
    try:
        value = float(TTestPower().power(effect_size=effect_size, nobs=n, alpha=alpha, alternative="two-sided"))
        if math.isfinite(value):
            return value
    except Exception:
        pass
    # The same two-sided t-test, using symmetry to avoid an unstable lower CDF.
    df = n - 1
    ncp = effect_size * math.sqrt(n)
    crit = stats.t.isf(alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.sf(crit, df, -ncp))


def _n_needed(effect_size: float, alpha: float, target_power: float) -> float:
    if effect_size <= 0:
        return float("inf")
    try:
        return float(TTestPower().solve_power(effect_size=effect_size, power=target_power, alpha=alpha, alternative="two-sided"))
    except Exception:
        for n in range(4, 100000):
            if _achieved_power(effect_size, n, alpha) >= target_power:
                return float(n)
    return float("inf")


def power_check(effect: EffectResult, contract: ClaimContract, ref_effect: float | None = None) -> PowerResult:
    """Compute power gate quantities using ref effect when available."""

    effect_size = _reference_effect(effect, ref_effect)
    alpha = contract.gates.multiplicity.alpha
    achieved = _achieved_power(effect_size, effect.n, alpha)
    n80 = _n_needed(effect_size, alpha, 0.8)
    z = abs(effect.beta / effect.se) if effect.se > 0 else 0.0
    shrinkage = max(0.0, 1.0 - 1.0 / (z * z)) if z > 1.0 else 0.0
    shrunken = effect.beta * shrinkage
    # Fail closed: power is judged against a pre-declared reference/MDE, never the
    # observed effect, so an inflated small-sample estimate cannot pass this gate.
    under_powered = not math.isfinite(achieved) or achieved < contract.gates.power.min_power
    ref_kind = "ref" if ref_effect is not None else f"MDE={DEFAULT_MDE}"
    rationale = (
        f"power={achieved:.3f} vs {ref_kind} effect={effect_size:.2f}, "
        f"n_needed_80={n80:.1f}"
    )
    return PowerResult(
        achieved_power=float(achieved),
        n_needed_80=float(n80),
        shrinkage_factor=float(shrinkage),
        shrunken_effect=float(shrunken),
        under_powered=bool(under_powered),
        ref_effect=ref_effect,
        rationale=rationale,
    )

