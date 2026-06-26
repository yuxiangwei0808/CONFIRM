"""Benchmark metric summaries with exact binomial intervals and risk coverage."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd

from bench.labels import LABEL_TABLE_COLUMNS, label_authority, load_claim_label_table, scoring_bucket, validate_claim_label_table

DEFAULT_RUNGS = ["exec_only", "+confound", "+power", "+multiverse", "+replication"]


def exact_binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval."""

    if n <= 0:
        return (float("nan"), float("nan"))
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].fillna(False).astype(bool)


def _rate(k: int, n: int) -> float:
    return float(k / n) if n else math.nan


def _rate_fields(k: int, n: int, prefix: str) -> dict[str, Any]:
    ci = exact_binomial_ci(k, n) if n else (math.nan, math.nan)
    return {
        prefix: _rate(k, n),
        f"{prefix}_ci95_exact": list(ci),
        f"{prefix}_count": int(k),
        f"{prefix}_denominator": int(n),
    }


def _prepare_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "claim_id" in df.columns:
        _validate_claim_label_coverage(rows)
    if "scoring_bucket" not in df.columns:
        df["scoring_bucket"] = df["scoring_label"].map(scoring_bucket)
    if "label_authority" not in df.columns:
        df["label_authority"] = [_row_label_authority(row) for row in rows]
    else:
        df["label_authority"] = df.apply(lambda row: _row_label_authority(row.to_dict()), axis=1)
    return df


def _validate_claim_label_coverage(rows: list[dict[str, Any]]) -> None:
    """Validate claim IDs against the static table plus generated row labels."""

    table_ids = set(load_claim_label_table())
    generated_rows: list[dict[str, str]] = []
    generated_ids: set[str] = set()
    claim_ids = {str(row.get("claim_id", "")).strip() for row in rows if str(row.get("claim_id", "")).strip()}
    for row in rows:
        claim_id = str(row.get("claim_id", "")).strip()
        metadata = row.get("label_metadata")
        if not claim_id or not isinstance(metadata, dict):
            continue
        metadata_claim_id = str(metadata.get("claim_id", "")).strip()
        if metadata_claim_id != claim_id:
            continue
        if all(col in metadata and str(metadata.get(col, "")).strip() for col in LABEL_TABLE_COLUMNS):
            generated = {col: str(metadata.get(col, "")).strip() for col in LABEL_TABLE_COLUMNS}
            generated_rows.append(generated)
            generated_ids.add(claim_id)
    if generated_rows:
        validate_claim_label_table(generated_rows)
    missing = sorted(claim_ids - table_ids - generated_ids)
    if missing:
        raise ValueError(f"Missing claim-label table rows for benchmark metrics: {missing}")


def _row_label_authority(row: dict[str, Any]) -> str:
    value = str(row.get("label_authority", "") or "").strip().lower()
    if value in {"main", "supplementary"}:
        return value
    metadata = row.get("label_metadata")
    if isinstance(metadata, dict) and metadata:
        return label_authority({key: str(value) for key, value in metadata.items()})
    required = {"label_basis", "adjudication_status", "source_citation"}
    if required.issubset(row):
        return label_authority({key: str(row.get(key, "")) for key in required})
    return "supplementary"


def _summarize_frame(df: pd.DataFrame, rungs: Iterable[str]) -> tuple[dict[str, int], dict[str, Any]]:
    pos = df[df["scoring_bucket"] == "positive"]
    neg = df[df["scoring_bucket"] == "negative"]
    small = df[df["scoring_bucket"] == "small_positive"]
    candidate = df[df["scoring_bucket"] == "candidate"]
    summary: dict[str, Any] = {}

    for rung in rungs:
        confirmed = _bool_series(df, rung)
        pos_confirmed = _bool_series(pos, rung)
        neg_confirmed = _bool_series(neg, rung)
        small_confirmed = _bool_series(small, rung)
        candidate_confirmed = _bool_series(candidate, rung)

        coverage_count = int(confirmed.sum())
        abstention_count = int((~confirmed).sum())
        tp = int(pos_confirmed.sum())
        fp = int(neg_confirmed.sum())
        small_hits = int(small_confirmed.sum())
        candidate_hits = int(candidate_confirmed.sum())

        rung_summary = {
            **_rate_fields(tp, len(pos), "known_positive_recall"),
            **_rate_fields(fp, len(neg), "FCR"),
            **_rate_fields(small_hits, len(small), "small_positive_recovery"),
            **_rate_fields(candidate_hits, len(candidate), "candidate_yield"),
            **_rate_fields(coverage_count, len(df), "coverage"),
            **_rate_fields(abstention_count, len(df), "overall_abstention_rate"),
        }
        # Backward-compatible aliases used by existing reports.
        rung_summary["TPR"] = rung_summary["known_positive_recall"]
        rung_summary["TPR_ci95"] = rung_summary["known_positive_recall_ci95_exact"]
        rung_summary["TPR_count"] = rung_summary["known_positive_recall_count"]
        rung_summary["TPR_denominator"] = rung_summary["known_positive_recall_denominator"]
        rung_summary["FCR_ci95"] = rung_summary["FCR_ci95_exact"]
        rung_summary["small_positive_recovery_ci95"] = rung_summary["small_positive_recovery_ci95_exact"]
        rung_summary["small_positive_count"] = rung_summary["small_positive_recovery_count"]
        rung_summary["small_positive_denominator"] = rung_summary["small_positive_recovery_denominator"]
        rung_summary["candidate_confirmation_rate"] = rung_summary["candidate_yield"]
        rung_summary["candidate_confirmation_ci95"] = rung_summary["candidate_yield_ci95_exact"]
        rung_summary["candidate_confirmation_count"] = rung_summary["candidate_yield_count"]
        rung_summary["candidate_confirmation_denominator"] = rung_summary["candidate_yield_denominator"]
        summary[rung] = rung_summary

    counts = {
        "n_claims": int(len(df)),
        "n_positive": int(len(pos)),
        "n_negative_or_fragile": int(len(neg)),
        "n_small_positive": int(len(small)),
        "n_candidate": int(len(candidate)),
    }
    return counts, summary


def _main_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _row_label_authority(row) == "main"]


def summarize_rows(rows: list[dict[str, Any]], rungs: Iterable[str] = DEFAULT_RUNGS) -> dict[str, Any]:
    """Summarize gate-ladder rates using the registered scoring buckets."""

    df = _prepare_frame(rows)
    if df.empty:
        return {
            "n_claims": 0,
            "n_claims_main": 0,
            "label_authority_counts": {},
            "summary": {},
            "summary_full": {},
            "summary_main": {},
            "risk_coverage": [],
            "risk_coverage_full": [],
            "risk_coverage_main": [],
        }

    full_counts, full_summary = _summarize_frame(df, rungs)
    main_df = df[df["label_authority"] == "main"].copy()
    main_counts, main_summary = _summarize_frame(main_df, rungs)
    risk_coverage = risk_coverage_rows(rows)
    risk_coverage_main = risk_coverage_rows(_main_rows(rows))
    return {
        **full_counts,
        "n_claims_main": main_counts["n_claims"],
        "n_positive_main": main_counts["n_positive"],
        "n_negative_or_fragile_main": main_counts["n_negative_or_fragile"],
        "n_small_positive_main": main_counts["n_small_positive"],
        "n_candidate_main": main_counts["n_candidate"],
        "label_authority_counts": {str(k): int(v) for k, v in df["label_authority"].value_counts().items()},
        "summary": full_summary,
        "summary_full": full_summary,
        "summary_main": main_summary,
        "metrics_by_label_authority": {
            "full": {**full_counts, "summary": full_summary},
            "main": {**main_counts, "summary": main_summary},
        },
        "risk_coverage": risk_coverage,
        "risk_coverage_full": risk_coverage,
        "risk_coverage_main": risk_coverage_main,
    }


def risk_coverage_rows(rows: list[dict[str, Any]], alphas: Iterable[float] = (0.10, 0.05, 0.01)) -> list[dict[str, Any]]:
    """Compute full-ladder admissibility under alpha sweeps from stored effects."""

    df = _prepare_frame(rows)
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for alpha in alphas:
        confirmed = pd.Series([_confirmed_at_alpha(row, float(alpha)) for row in rows], index=df.index)
        pos = df["scoring_bucket"] == "positive"
        neg = df["scoring_bucket"] == "negative"
        small = df["scoring_bucket"] == "small_positive"
        candidate = df["scoring_bucket"] == "candidate"
        tp = int(confirmed[pos].sum())
        fp = int(confirmed[neg].sum())
        small_hits = int(confirmed[small].sum())
        candidate_hits = int(confirmed[candidate].sum())
        coverage = int(confirmed.sum())
        out.append(
            {
                "alpha": float(alpha),
                "confirmed_count": coverage,
                "n_claims": int(len(df)),
                **_rate_fields(coverage, len(df), "coverage"),
                **_rate_fields(int((~confirmed).sum()), len(df), "overall_abstention_rate"),
                **_rate_fields(tp, int(pos.sum()), "known_positive_recall"),
                **_rate_fields(fp, int(neg.sum()), "FCR"),
                **_rate_fields(small_hits, int(small.sum()), "small_positive_recovery"),
                **_rate_fields(candidate_hits, int(candidate.sum()), "candidate_yield"),
            }
        )
    return out


def _confirmed_at_alpha(row: dict[str, Any], alpha: float) -> bool:
    if not _search_provenance_admissible(row):
        return False
    if not bool(row.get("+confound", False)) or not bool(row.get("+power", False)):
        return False
    if not _primary_pass_at_alpha(row, alpha):
        return False
    if not _multiverse_pass_at_alpha(row, alpha):
        return False
    return _replication_pass_at_alpha(row, alpha)


def _primary_pass_at_alpha(row: dict[str, Any], alpha: float) -> bool:
    table = row.get("primary_region_table")
    if isinstance(table, dict) and table.get("regions"):
        regions = table["regions"]
        p_values = [
            _effect_p(region.get("effect", {})) if isinstance(region, dict) else math.inf
            for region in regions
        ]
        q_values = _bh_q_values(p_values, _effective_family_size(row, len(regions)))
        return any(q <= alpha for q in q_values)
    effect = row.get("primary_effect")
    if isinstance(effect, dict):
        return _effect_p(effect) <= alpha / _effective_family_size(row, 1)
    return float(row.get("best_p", math.inf)) <= alpha / _effective_family_size(row, 1)


def _multiverse_pass_at_alpha(row: dict[str, Any], alpha: float) -> bool:
    specs = row.get("multiverse_specs")
    if not isinstance(specs, list) or not specs:
        return bool(row.get("+multiverse", False))
    ok = [spec for spec in specs if spec.get("status", "ok") == "ok"]
    if not ok:
        return False
    consistent = [spec for spec in ok if bool(spec.get("same_sign", False)) and float(spec.get("p", math.inf)) < alpha]
    threshold = _contract_multiverse_threshold(row)
    return (len(consistent) / len(ok)) >= threshold


def _replication_pass_at_alpha(row: dict[str, Any], alpha: float) -> bool:
    replication = row.get("replication")
    if not isinstance(replication, dict):
        return bool(row.get("+replication", False))
    if "pattern_corr" in replication:
        return _brainwide_replication_pass_at_alpha(row, replication, alpha)
    return _scalar_replication_pass_at_alpha(row, replication, alpha)


def _scalar_replication_pass_at_alpha(row: dict[str, Any], replication: dict[str, Any], alpha: float) -> bool:
    primary = row.get("primary_effect")
    primary_beta = float(primary.get("beta", row.get("best_beta", math.nan))) if isinstance(primary, dict) else float(row.get("best_beta", math.nan))
    if not _beta_direction_ok(primary_beta, row):
        return False
    results = replication.get("cohort_results", [])
    if not isinstance(results, list) or not results:
        return False
    for result in results:
        effect = result.get("effect") if isinstance(result, dict) else None
        if not isinstance(effect, dict):
            return False
        beta = float(effect.get("beta", math.nan))
        if not _beta_direction_ok(beta, row):
            return False
        if _require_same_sign(row) and not (beta * primary_beta > 0):
            return False
        if _effect_p(effect) >= alpha:
            return False
    return True


def _brainwide_replication_pass_at_alpha(row: dict[str, Any], replication: dict[str, Any], alpha: float) -> bool:
    primary = row.get("primary_region_table")
    if not isinstance(primary, dict) or not primary.get("regions"):
        return bool(row.get("+replication", False)) if alpha == 0.05 else False
    primary_regions = {
        region["region"]: region
        for region in primary["regions"]
        if float(region.get("q", math.inf)) <= alpha
    }
    if not primary_regions:
        return False
    gate = _replication_gate(row)
    if float(replication.get("pattern_corr", math.nan)) < float(gate.get("pattern_corr_min", 0.0)):
        return False
    if float(replication.get("dice", math.nan)) < float(gate.get("dice_min", 0.0)):
        return False
    min_frac = float(gate.get("region_replication_frac_min", 0.0))
    for result in replication.get("cohort_results", []):
        if not isinstance(result, dict):
            return False
        table = result.get("region_table")
        if not isinstance(table, dict) or not table.get("regions"):
            return False
        replicated = 0
        rep_by_region = {region["region"]: region for region in table["regions"]}
        for region_name, disc_region in primary_regions.items():
            rep_region = rep_by_region.get(region_name)
            if rep_region is None:
                continue
            rep_effect = rep_region.get("effect", {})
            disc_effect = disc_region.get("effect", {})
            rep_beta = float(rep_effect.get("beta", math.nan))
            disc_beta = float(disc_effect.get("beta", math.nan))
            if not _beta_direction_ok(rep_beta, row):
                continue
            if _require_same_sign(row) and not (rep_beta * disc_beta > 0):
                continue
            if _effect_p(rep_effect) < alpha:
                replicated += 1
        if replicated / len(primary_regions) < min_frac:
            return False
    return True


def _effect_p(effect: dict[str, Any]) -> float:
    try:
        return float(effect.get("p", math.inf))
    except (TypeError, ValueError):
        return math.inf


def _bh_q_values(p_values: list[float], family_size: int) -> list[float]:
    finite = [(i, p) for i, p in enumerate(p_values) if math.isfinite(p)]
    out = [math.nan for _ in p_values]
    if not finite:
        return out
    ordered = sorted(finite, key=lambda item: item[1])
    m = float(max(family_size, len(ordered), 1))
    adjusted = [p * m / rank for rank, (_, p) in enumerate(ordered, start=1)]
    for i in range(len(adjusted) - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    for (original_idx, _), q in zip(ordered, adjusted):
        out[original_idx] = min(max(float(q), 0.0), 1.0)
    return out


def _search_provenance(row: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("search_provenance")
    if isinstance(direct, dict):
        return direct
    contract = _contract(row)
    provenance = contract.get("search_provenance", {})
    return provenance if isinstance(provenance, dict) else {}


def _search_provenance_admissible(row: dict[str, Any]) -> bool:
    provenance = _search_provenance(row)
    declared = bool(provenance.get("declared", True))
    selection = str(provenance.get("selection", "preregistered"))
    return bool(declared and selection not in {"unknown", "full_data"})


def _effective_family_size(row: dict[str, Any], observed: int) -> int:
    contract = _contract(row)
    gates = contract.get("gates", {})
    multiplicity = gates.get("multiplicity", {}) if isinstance(gates, dict) else {}
    provenance = _search_provenance(row)
    values = [observed, 1]
    for source in [multiplicity, provenance]:
        if isinstance(source, dict):
            try:
                values.append(int(source.get("family_size", 1)))
            except (TypeError, ValueError):
                values.append(1)
    return max(values)


def _contract(row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("contract", {})
    return contract if isinstance(contract, dict) else {}


def _contract_multiverse_threshold(row: dict[str, Any]) -> float:
    gates = _contract(row).get("gates", {})
    multiverse = gates.get("multiverse", {}) if isinstance(gates, dict) else {}
    try:
        return float(multiverse.get("min_fraction_consistent", 0.6))
    except (TypeError, ValueError):
        return 0.6


def _replication_gate(row: dict[str, Any]) -> dict[str, Any]:
    gates = _contract(row).get("gates", {})
    gate = gates.get("replication", {}) if isinstance(gates, dict) else {}
    return gate if isinstance(gate, dict) else {}


def _require_same_sign(row: dict[str, Any]) -> bool:
    return bool(_replication_gate(row).get("require_same_sign", True))


def _beta_direction_ok(beta: float, row: dict[str, Any]) -> bool:
    direction = _contract(row).get("estimand", {}).get("direction", "two_sided")
    if direction == "negative":
        return beta < 0
    if direction == "positive":
        return beta > 0
    return math.isfinite(beta)
