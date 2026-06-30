"""Data-aware executable preflight checks for generated claim candidates."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from confirm.contract import ClaimContract
from confirm.schema import idp_columns, normalize_sex


class CandidatePreflightResult(BaseModel):
    """Deterministic executability check for a candidate contract."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_data_paths: dict[str, str] = Field(default_factory=dict)


class CohortPreflightInfo(BaseModel):
    """Lightweight catalog information for one resolved cohort table."""

    model_config = ConfigDict(extra="forbid")

    cohort: str
    path: str
    columns: list[str]
    idps: list[str]


class CandidatePreflightContext:
    """Data-root-backed validator for generated candidate contracts."""

    def __init__(self, cohorts: dict[str, CohortPreflightInfo]) -> None:
        self.cohorts = cohorts
        self._df_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}

    @classmethod
    def from_roots(cls, data_roots: Iterable[str | Path]) -> "CandidatePreflightContext":
        cohorts: dict[str, CohortPreflightInfo] = {}
        for root_value in data_roots:
            root = Path(root_value)
            for path in sorted(root.glob("*.parquet")):
                columns = _parquet_columns(path)
                info = CohortPreflightInfo(
                    cohort=path.stem,
                    path=str(path),
                    columns=columns,
                    idps=idp_columns(columns),
                )
                for alias in _cohort_aliases(path.stem):
                    cohorts.setdefault(alias, info)
        return cls(cohorts)

    def prompt_catalog(self, contract: ClaimContract, *, outcome_limit: int = 160) -> dict[str, Any]:
        """Build a compact executable catalog slice for the LLM prompt."""

        cohorts = [contract.discovery_cohort, *contract.replication_cohorts]
        resolved: dict[str, Any] = {}
        all_idps: list[set[str]] = []
        for cohort in cohorts:
            info = self.resolve(cohort)
            if info is None:
                resolved[cohort] = {"available": False}
                continue
            idps = set(info.idps)
            all_idps.append(idps)
            columns = set(info.columns)
            relevant_columns = sorted(
                col
                for col in info.columns
                if col in {"subject_id", "cohort", "site", "age", "sex", "dx", "eTIV"}
                or col == contract.estimand.predictor
                or col in contract.covariates
                or (contract.estimand.group is not None and col == contract.estimand.group.var)
            )
            resolved[cohort] = {
                "available": True,
                "path": info.path,
                "columns": relevant_columns,
                "outcome_columns_sample": sorted(info.idps)[:outcome_limit],
                "outcome_column_count": len(info.idps),
            }
        common_outcomes = sorted(set.intersection(*all_idps)) if all_idps else []
        group_levels: dict[str, list[str]] = {}
        if contract.estimand.group is not None:
            group_var = contract.estimand.group.var
            levels = self.levels(contract.discovery_cohort, group_var)
            group_levels[group_var] = levels[:50]
        return {
            "allowed_cohorts": sorted(self.cohorts),
            "resolved_parent_cohorts": resolved,
            "common_outcome_columns_sample": common_outcomes[:outcome_limit],
            "common_outcome_column_count": len(common_outcomes),
            "original_predictor": contract.estimand.predictor,
            "original_group": contract.estimand.group.model_dump(mode="json") if contract.estimand.group else None,
            "observed_group_levels": group_levels,
            "original_covariates": list(contract.covariates),
            "allowed_inclusion_examples": [None, 'sex == "F"', 'sex == "M"', "age >= 65", "age <= 30"],
            "contract_rules": [
                "Use only cohorts and columns present in this executable catalog.",
                "For same-data adaptive candidates, preserve the original predictor and group contrast.",
                "Do not introduce synthetic variables unless they are present as columns.",
                "Use Python/pandas-query-compatible inclusion strings with quoted string levels.",
            ],
        }

    def resolve(self, cohort: str) -> CohortPreflightInfo | None:
        for alias in _cohort_aliases(cohort):
            info = self.cohorts.get(alias)
            if info is not None:
                return info
        return None

    def levels(self, cohort: str, column: str) -> list[str]:
        info = self.resolve(cohort)
        if info is None or column not in info.columns:
            return []
        df = self._read_columns(info, [column])
        return sorted(str(value) for value in df[column].dropna().unique())

    def validate_contract(self, contract: ClaimContract, *, min_complete_rows: int = 20) -> CandidatePreflightResult:
        violations: list[str] = []
        warnings: list[str] = []
        resolved: dict[str, str] = {}

        cohort_infos: dict[str, CohortPreflightInfo] = {}
        for cohort in [contract.discovery_cohort, *contract.replication_cohorts]:
            info = self.resolve(cohort)
            if info is None:
                violations.append(f"Preflight: cohort {cohort!r} was not found in configured data roots.")
                continue
            cohort_infos[cohort] = info
            resolved[cohort] = info.path

        outcomes = _outcomes(contract)
        analysis_columns = _analysis_columns(contract)
        inclusion_names, inclusion_error = _inclusion_identifiers(contract.inclusion)
        if inclusion_error is not None:
            violations.append(f"Preflight: invalid inclusion query {contract.inclusion!r}: {inclusion_error}")
        analysis_columns.extend(inclusion_names)
        analysis_columns = list(dict.fromkeys(analysis_columns))

        for cohort, info in cohort_infos.items():
            columns = set(info.columns)
            missing_outcomes = [col for col in outcomes if col not in columns]
            missing_analysis = [col for col in analysis_columns if col not in columns]
            if missing_outcomes:
                violations.append(f"Preflight: cohort {cohort!r} is missing outcome columns: {missing_outcomes[:10]}")
            if missing_analysis:
                violations.append(f"Preflight: cohort {cohort!r} is missing analysis columns: {missing_analysis[:10]}")

        if violations:
            return CandidatePreflightResult(ok=False, violations=violations, warnings=warnings, resolved_data_paths=resolved)

        for cohort, info in cohort_infos.items():
            needed = list(dict.fromkeys([*outcomes, *analysis_columns, "age", "sex"]))
            try:
                df = self._read_columns(info, needed)
                table_violations, table_warnings = _validate_table_slice(df, contract, cohort, min_complete_rows)
                violations.extend(table_violations)
                warnings.extend(table_warnings)
            except Exception as exc:  # noqa: BLE001
                violations.append(f"Preflight: cohort {cohort!r} could not be read for executable checks: {exc}")

        return CandidatePreflightResult(
            ok=not violations,
            violations=violations,
            warnings=warnings,
            resolved_data_paths=resolved,
        )

    def _read_columns(self, info: CohortPreflightInfo, columns: list[str]) -> pd.DataFrame:
        cols = tuple(sorted(dict.fromkeys(col for col in columns if col in set(info.columns))))
        key = (info.path, cols)
        cached = self._df_cache.get(key)
        if cached is not None:
            return cached.copy()
        df = pd.read_parquet(info.path, columns=list(cols))
        self._df_cache[key] = df
        return df.copy()


def _validate_table_slice(
    df: pd.DataFrame,
    contract: ClaimContract,
    cohort: str,
    min_complete_rows: int,
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    warnings: list[str] = []
    table = df.copy()
    if "age" in table:
        age = pd.to_numeric(table["age"], errors="coerce")
        if age.isna().any():
            violations.append(f"Preflight: cohort {cohort!r} has missing or non-numeric age values.")
    if "sex" in table:
        sex = normalize_sex(table["sex"])
        if sex.isna().any():
            violations.append(f"Preflight: cohort {cohort!r} has invalid sex encodings.")
        table["sex"] = sex

    if contract.inclusion:
        try:
            table = table.query(contract.inclusion, engine="python")
        except Exception as exc:  # noqa: BLE001
            violations.append(f"Preflight: inclusion query {contract.inclusion!r} failed on cohort {cohort!r}: {exc}")
            return violations, warnings

    if contract.estimand.group is not None:
        group = contract.estimand.group
        levels = {str(value) for value in table[group.var].dropna().unique()}
        missing_levels = [level for level in [group.case, group.control] if level not in levels]
        if missing_levels:
            violations.append(f"Preflight: cohort {cohort!r} missing group levels for {group.var!r}: {missing_levels}")
        table = table[table[group.var].astype(str).isin([group.case, group.control])].copy()

    complete_columns = [*_outcomes(contract), *_analysis_columns(contract)]
    complete_columns = [col for col in dict.fromkeys(complete_columns) if col in table.columns]
    complete = table.dropna(subset=complete_columns)
    if len(complete) < min_complete_rows:
        violations.append(
            f"Preflight: cohort {cohort!r} has too few complete rows after filters "
            f"({len(complete)} < {min_complete_rows})."
        )
    elif len(complete) < 50:
        warnings.append(f"Preflight: cohort {cohort!r} has only {len(complete)} complete rows after filters.")
    return violations, warnings


def _outcomes(contract: ClaimContract) -> list[str]:
    outcome = contract.estimand.outcome
    return list(outcome) if isinstance(outcome, list) else [outcome]


def _analysis_columns(contract: ClaimContract) -> list[str]:
    columns = [contract.estimand.predictor, *contract.covariates, *contract.gates.confound.require_covariates]
    if contract.estimand.group is not None:
        columns.append(contract.estimand.group.var)
    return list(dict.fromkeys(columns))


def _inclusion_identifiers(inclusion: str | None) -> tuple[list[str], str | None]:
    if not inclusion:
        return [], None
    try:
        tree = ast.parse(inclusion, mode="eval")
    except SyntaxError as exc:
        return [], str(exc)
    names = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
    names = [name for name in names if name not in {"True", "False", "None", "and", "or", "not"}]
    return names, None


def _cohort_aliases(cohort: str) -> list[str]:
    aliases = [cohort]
    suffixes = (
        "_DISC_SITES",
        "_REP_SITES",
        "_DISC",
        "_REP",
        "_CN",
    )
    for suffix in suffixes:
        if cohort.endswith(suffix):
            aliases.append(cohort[: -len(suffix)])
    for split in ("_DISC_s", "_REP_s"):
        if split in cohort:
            aliases.append(cohort.split(split, 1)[0])
    return list(dict.fromkeys(item for item in aliases if item))


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        return list(pd.read_parquet(path).columns)
