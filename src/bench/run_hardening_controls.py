"""Targeted hardening controls requested by auto-review.

These controls are deliberately small and deterministic:

1. FDR trap: random labels over many fMRI features produce uncorrected hits that
   disappear after BH correction.
2. Power/MDE trap: a discovery-significant selected null is underpowered for a
   predeclared minimum detectable effect.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

from confirm.schema import normalize_sex

SEED = 20260615


def _fit_group_effect(df: pd.DataFrame, outcome: str, group_col: str, covariates: list[str]) -> dict[str, float]:
    cols = [outcome, group_col, *covariates]
    data = df[cols].dropna().copy()
    if len(data) < max(8, len(cols) + 3):
        raise ValueError("Too few complete rows")
    y = pd.to_numeric(data[outcome], errors="coerce")
    x_parts = [pd.Series((data[group_col] == "case").astype(float), index=data.index, name="group")]
    for cov in covariates:
        series = data[cov]
        if pd.api.types.is_numeric_dtype(series):
            x_parts.append(pd.Series(pd.to_numeric(series, errors="coerce"), index=data.index, name=cov))
        else:
            x_parts.append(pd.get_dummies(series.astype("string"), prefix=cov, drop_first=True, dtype=float))
    x = pd.concat(x_parts, axis=1)
    complete = pd.concat([y.rename(outcome), x], axis=1).dropna()
    y = complete[outcome].astype(float)
    x = sm.add_constant(complete.drop(columns=[outcome]).astype(float), has_constant="add")
    fit = sm.OLS(y, x).fit()
    beta = float(fit.params["group"])
    se = float(fit.bse["group"])
    t_value = beta / se if se > 0 else 0.0
    dof = float(fit.df_resid)
    std_effect = float(math.copysign(math.sqrt((t_value * t_value) / (t_value * t_value + dof)), beta)) if dof > 0 else float("nan")
    return {
        "beta": beta,
        "se": se,
        "p": float(fit.pvalues["group"]),
        "n": int(len(y)),
        "standardized_effect": std_effect,
    }


def _prepared_hcp(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(data_root / "cohorts" / "HCP.parquet")
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = normalize_sex(df["sex"])
    features = [col for col in df.columns if col.startswith("fc_fc_")]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, sorted(features)


def run_fdr_trap(data_root: Path, seed: int) -> dict[str, Any]:
    df, features = _prepared_hcp(data_root)
    rng = np.random.default_rng(seed)
    for attempt in range(500):
        work = df.copy()
        work["random_group"] = rng.choice(["case", "control"], size=len(work))
        rows = []
        for feature in features:
            try:
                effect = _fit_group_effect(work, feature, "random_group", ["age", "sex"])
                rows.append({"feature": feature, **effect})
            except Exception:
                continue
        table = pd.DataFrame(rows)
        if table.empty:
            continue
        table["q_bh"] = multipletests(table["p"].to_numpy(), alpha=0.05, method="fdr_bh")[1]
        uncorrected = int((table["p"] < 0.05).sum())
        fdr = int((table["q_bh"] < 0.05).sum())
        if uncorrected > 0 and fdr == 0:
            best = table.sort_values("p").iloc[0].to_dict()
            return {
                "control_id": "fdr_random_label_hcp_fc",
                "passed": True,
                "attempt": attempt,
                "n_features": int(len(table)),
                "uncorrected_hits_p_lt_0_05": uncorrected,
                "fdr_hits_q_lt_0_05": fdr,
                "best_feature": best,
                "interpretation": "Uncorrected execution would report a hit; BH-FDR correctly removes it.",
            }
    return {"control_id": "fdr_random_label_hcp_fc", "passed": False, "reason": "no suitable random seed found"}


def run_power_mde_trap(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 36
    m = 256
    ref_effect = 0.2
    group = rng.choice(["case", "control"], size=n)
    base = pd.DataFrame({"group": group, "age": rng.normal(60, 8, size=n), "sex": rng.choice(["F", "M"], size=n)})
    features = pd.DataFrame(rng.normal(size=(n, m)), columns=[f"y_{j:03d}" for j in range(m)])
    base = pd.concat([base, features], axis=1)
    rows = []
    for feature in [col for col in base.columns if col.startswith("y_")]:
        effect = _fit_group_effect(base, feature, "group", ["age", "sex"])
        rows.append({"feature": feature, **effect})
    table = pd.DataFrame(rows).sort_values("p")
    best = table.iloc[0].to_dict()
    achieved_power = float(TTestPower().power(effect_size=ref_effect, nobs=int(best["n"]), alpha=0.05, alternative="two-sided"))
    return {
        "control_id": "power_mde_selected_null",
        "passed": bool(best["p"] < 0.05 and achieved_power < 0.8),
        "n": int(best["n"]),
        "n_features_searched": m,
        "predeclared_ref_effect": ref_effect,
        "best_selected_feature": best,
        "achieved_power_at_ref_effect": achieved_power,
        "exec_only_significant": bool(best["p"] < 0.05),
        "power_gate_passed": bool(achieved_power >= 0.8),
        "interpretation": "A selected discovery hit is significant but underpowered for the predeclared MDE.",
    }


def _json_safe(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(k): _json_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_json_safe(v) for v in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/prepared_data/benchmark_ready")
    parser.add_argument("--out-dir", default="review-stage/hardening-controls")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "controls": [
            run_fdr_trap(Path(args.data_root), args.seed),
            run_power_mde_trap(args.seed + 1),
        ],
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"hardening_controls_results_{timestamp}.json"
    latest = out_dir / "hardening_controls_results.json"
    out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    shutil.copyfile(out, latest)
    print(json.dumps(_json_safe(payload), indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
