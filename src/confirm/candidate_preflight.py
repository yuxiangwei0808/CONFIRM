"""Data-aware executable preflight checks for generated claim candidates."""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from confirm.analysis import AnalysisNonIdentifiableError, build_analysis_design
from confirm.contract import ClaimContract
from confirm.derived_columns import CONFIRM_DX, add_virtual_columns, columns_with_virtuals
from confirm.evidence_partitions import load_evidence_manifest
from confirm.schema import (
    columns_with_canonical_aliases,
    harmonize_canonical_columns,
    idp_columns,
    normalize_sex,
    physical_column_for,
)

_CATEGORICAL_ASSOCIATION_PREDICTORS = {"sex", "dx", "diagnosis", "site", "group", "cohort"}


class CandidatePreflightResult(BaseModel):
    """Deterministic executability check for a candidate contract."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_data_paths: dict[str, str] = Field(default_factory=dict)
    resolved_outcome_columns: dict[str, list[str]] = Field(default_factory=dict)
    design_diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CohortPreflightInfo(BaseModel):
    """Lightweight catalog information for one resolved cohort table."""

    model_config = ConfigDict(extra="forbid")

    cohort: str
    path: str
    columns: list[str]
    idps: list[str]
    source_columns: list[str] = Field(default_factory=list)
    base_dataset: Optional[str] = None
    target_family: Optional[str] = None
    partition_role: Optional[str] = None
    evaluation_role: Optional[str] = None


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
            manifest_path = root / "manifest.json"
            if not manifest_path.exists():
                manifest_path = root.parent / "manifest.json"
            manifest = load_evidence_manifest(manifest_path)
            if manifest is not None:
                for record in manifest.records:
                    path = Path(record.path)
                    if not path.exists():
                        continue
                    source_columns = _parquet_columns(path)
                    columns = columns_with_canonical_aliases(source_columns)
                    columns = columns_with_virtuals(record.partition_id, columns)
                    cohorts.setdefault(
                        record.partition_id,
                        CohortPreflightInfo(
                            cohort=record.partition_id,
                            path=str(path),
                            columns=columns,
                            idps=idp_columns(columns),
                            source_columns=source_columns,
                            base_dataset=record.base_dataset,
                            target_family=record.target_family,
                            partition_role=record.role,
                            evaluation_role=record.evaluation_role,
                        ),
                    )
            for path in sorted(root.glob("*.parquet")):
                source_columns = _parquet_columns(path)
                columns = columns_with_canonical_aliases(source_columns)
                columns = columns_with_virtuals(path.stem, columns)
                info = CohortPreflightInfo(
                    cohort=path.stem,
                    path=str(path),
                    columns=columns,
                    idps=idp_columns(columns),
                    source_columns=source_columns,
                )
                cohorts.setdefault(path.stem, info)
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
        inclusion_examples = self._feasible_inclusion_examples(contract, cohorts)
        return {
            "allowed_cohorts": cohorts,
            "resolved_parent_cohorts": resolved,
            "common_outcome_columns_sample": common_outcomes[:outcome_limit],
            "common_outcome_column_count": len(common_outcomes),
            "original_predictor": contract.estimand.predictor,
            "original_group": contract.estimand.group.model_dump(mode="json") if contract.estimand.group else None,
            "observed_group_levels": group_levels,
            "original_covariates": list(contract.covariates),
            "allowed_inclusion_examples": inclusion_examples,
            "contract_rules": [
                "Use only cohorts and columns present in this executable catalog.",
                "The allowed_cohorts list is limited to the parent discovery/replication contract.",
                "Do not place holdout or external evaluation cohorts in proposed_contract for patch-like follow-ups.",
                "For same-data adaptive candidates, preserve the original predictor and group contrast.",
                "Do not introduce synthetic variables unless they are present as columns.",
                "Use Python/pandas-query-compatible inclusion strings with quoted string levels.",
            ],
        }

    def _feasible_inclusion_examples(self, contract: ClaimContract, cohorts: list[str]) -> list[str | None]:
        """Return parent-data-feasible filters without exposing outcome statistics."""

        tables: list[pd.DataFrame] = []
        for cohort in cohorts:
            info = self.resolve(cohort)
            if info is None:
                return [None]
            matched_outcomes = _matched_outcome_columns(contract, info)
            if not matched_outcomes:
                return [None]
            required = list(dict.fromkeys([matched_outcomes[0], *_analysis_columns(contract), "age", "sex"]))
            try:
                table = self._read_columns(info, required)
            except Exception:  # noqa: BLE001
                return [None]
            if "age" in table:
                table["age"] = pd.to_numeric(table["age"], errors="coerce")
            if "sex" in table:
                table["sex"] = normalize_sex(table["sex"])
            tables.append(table)

        base_inclusion = contract.inclusion
        examples: list[str | None] = [base_inclusion]
        predicates = ['sex == "F"', 'sex == "M"']
        discovery_age = tables[0]["age"].dropna() if tables and "age" in tables[0] else pd.Series(dtype=float)
        if len(discovery_age) >= 40:
            lower, upper = discovery_age.quantile([0.25, 0.75])
            predicates.extend([f"age <= {float(lower):.3g}", f"age >= {float(upper):.3g}"])

        for predicate in predicates:
            candidate_predicate = (
                f"({base_inclusion}) and ({predicate})"
                if base_inclusion
                else predicate
            )
            feasible = True
            for table in tables:
                try:
                    subset = table.query(candidate_predicate, engine="python")
                except Exception:  # noqa: BLE001
                    feasible = False
                    break
                complete_columns = [column for column in table.columns if column in _analysis_columns(contract)]
                outcome = next((column for column in table.columns if column in _outcomes(contract)), None)
                if outcome is None and contract.estimand.unit == "brainwide":
                    outcome = next((column for column in table.columns if column.startswith(tuple(("smri_", "pet_", "fc_")))), None)
                required = [column for column in [outcome, *complete_columns] if column]
                if len(subset.dropna(subset=required)) < 20:
                    feasible = False
                    break
            if feasible:
                examples.append(candidate_predicate)
        return list(dict.fromkeys(examples))

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
        design_diagnostics: dict[str, dict[str, Any]] = {}

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
        matched_outcomes_by_cohort: dict[str, list[str]] = {}

        for cohort, info in cohort_infos.items():
            columns = set(info.columns)
            matched_outcomes = _matched_outcome_columns(contract, info)
            matched_outcomes_by_cohort[cohort] = matched_outcomes
            missing_outcomes = outcomes if not matched_outcomes else []
            missing_analysis = [col for col in analysis_columns if col not in columns]
            if missing_outcomes:
                violations.append(f"Preflight: cohort {cohort!r} is missing outcome columns: {missing_outcomes[:10]}")
            if missing_analysis:
                violations.append(f"Preflight: cohort {cohort!r} is missing analysis columns: {missing_analysis[:10]}")

        if violations:
            return CandidatePreflightResult(
                ok=False,
                violations=violations,
                warnings=warnings,
                resolved_data_paths=resolved,
                resolved_outcome_columns=matched_outcomes_by_cohort,
                design_diagnostics=design_diagnostics,
            )

        for cohort, info in cohort_infos.items():
            outcome_columns = matched_outcomes_by_cohort.get(cohort, [])
            needed = list(dict.fromkeys([*outcome_columns, *analysis_columns, "age", "sex"]))
            try:
                df = self._read_columns(info, needed)
                table_violations, table_warnings, table_diagnostics = _validate_table_slice(
                    df,
                    contract,
                    cohort,
                    min_complete_rows,
                    outcome_columns=outcome_columns,
                )
                violations.extend(table_violations)
                warnings.extend(table_warnings)
                if table_diagnostics is not None:
                    design_diagnostics[cohort] = table_diagnostics
            except Exception as exc:  # noqa: BLE001
                violations.append(f"Preflight: cohort {cohort!r} could not be read for executable checks: {exc}")

        return CandidatePreflightResult(
            ok=not violations,
            violations=violations,
            warnings=warnings,
            resolved_data_paths=resolved,
            resolved_outcome_columns=matched_outcomes_by_cohort,
            design_diagnostics=design_diagnostics,
        )

    def _read_columns(self, info: CohortPreflightInfo, columns: list[str]) -> pd.DataFrame:
        requested = list(dict.fromkeys(columns))
        available_source = info.source_columns or info.columns
        read_columns = [
            physical
            for column in requested
            if column != CONFIRM_DX
            for physical in [physical_column_for(column, available_source)]
            if physical is not None
        ]
        if CONFIRM_DX in requested and "dx" not in read_columns:
            read_columns.append("dx")
        cols = tuple(sorted(dict.fromkeys(read_columns)))
        key = (info.path, cols)
        cached = self._df_cache.get(key)
        if cached is not None:
            df = cached.copy()
            return df[[col for col in requested if col in df.columns]].copy()
        df = pd.read_parquet(info.path, columns=list(cols))
        df = harmonize_canonical_columns(df)
        df = add_virtual_columns(df, info.cohort)
        self._df_cache[key] = df
        return df[[col for col in requested if col in df.columns]].copy()


def _validate_table_slice(
    df: pd.DataFrame,
    contract: ClaimContract,
    cohort: str,
    min_complete_rows: int,
    *,
    outcome_columns: list[str],
) -> tuple[list[str], list[str], dict[str, Any] | None]:
    violations: list[str] = []
    warnings: list[str] = []
    table = df.copy()
    required_columns = set(_analysis_columns(contract))
    inclusion_columns, _ = _inclusion_identifiers(contract.inclusion)
    required_columns.update(inclusion_columns)
    if "age" in table:
        age = pd.to_numeric(table["age"], errors="coerce")
        missing_age = int(age.isna().sum())
        if missing_age and "age" in required_columns:
            if age.notna().any():
                warnings.append(
                    f"Preflight: cohort {cohort!r} excludes {missing_age} rows with missing or non-numeric age "
                    "during complete-case analysis."
                )
            else:
                violations.append(f"Preflight: cohort {cohort!r} has no numerically usable age values.")
        table["age"] = age
    if "sex" in table:
        sex = normalize_sex(table["sex"])
        missing_sex = int(sex.isna().sum())
        if missing_sex and "sex" in required_columns:
            if sex.notna().any():
                warnings.append(
                    f"Preflight: cohort {cohort!r} excludes {missing_sex} rows with missing or invalid sex "
                    "during complete-case analysis."
                )
            else:
                violations.append(f"Preflight: cohort {cohort!r} has no usable sex values.")
        table["sex"] = sex

    if contract.inclusion:
        try:
            table = table.query(contract.inclusion, engine="python")
        except Exception as exc:  # noqa: BLE001
            violations.append(f"Preflight: inclusion query {contract.inclusion!r} failed on cohort {cohort!r}: {exc}")
            return violations, warnings, None

    if contract.estimand.group is not None:
        group = contract.estimand.group
        levels = {str(value) for value in table[group.var].dropna().unique()}
        missing_levels = [level for level in [group.case, group.control] if level not in levels]
        if missing_levels:
            violations.append(f"Preflight: cohort {cohort!r} missing group levels for {group.var!r}: {missing_levels}")
        table = table[table[group.var].astype(str).isin([group.case, group.control])].copy()
    elif contract.estimand.type == "association":
        predictor = contract.estimand.predictor
        if predictor in table.columns:
            predictor_violations = _validate_association_predictor(table[predictor], predictor, cohort, min_complete_rows)
            violations.extend(predictor_violations)

    analysis_columns = [col for col in _analysis_columns(contract) if col in table.columns]
    usable_counts: list[tuple[int, str]] = []
    group_complete_missing: dict[str, list[str]] = {}
    for outcome in outcome_columns:
        if outcome not in table.columns:
            continue
        complete_columns = [outcome, *analysis_columns]
        complete_columns = [col for col in dict.fromkeys(complete_columns) if col in table.columns]
        complete = table.dropna(subset=complete_columns)
        if contract.estimand.group is not None:
            group = contract.estimand.group
            levels = {str(value) for value in complete[group.var].dropna().unique()}
            missing_levels = [level for level in [group.case, group.control] if level not in levels]
            if missing_levels:
                group_complete_missing[outcome] = missing_levels
                continue
        usable_counts.append((len(complete), outcome))
    best_complete = max((count for count, _ in usable_counts), default=0)
    if usable_counts and best_complete < min_complete_rows:
        violations.append(
            f"Preflight: cohort {cohort!r} has too few complete rows after filters "
            f"({best_complete} < {min_complete_rows})."
        )
    elif not usable_counts:
        if group_complete_missing:
            examples = list(group_complete_missing.items())[:5]
            violations.append(
                f"Preflight: cohort {cohort!r} missing group levels after complete-case filtering "
                f"for {contract.estimand.group.var!r}: {examples}"
            )
        else:
            violations.append(
                f"Preflight: cohort {cohort!r} has too few complete rows after filters "
                f"(0 < {min_complete_rows})."
            )
    elif best_complete < 50:
        warnings.append(f"Preflight: cohort {cohort!r} has only {best_complete} complete rows after filters.")

    diagnostics: dict[str, Any] | None = None
    if not violations and usable_counts:
        _, best_outcome = max(usable_counts, key=lambda item: item[0])
        estimand = contract.estimand.model_copy(update={"outcome": best_outcome, "unit": "scalar"})
        design_contract = contract.model_copy(update={"estimand": estimand})
        try:
            design = build_analysis_design(table, design_contract, design_contract.covariates)
            diagnostics = dict(design.diagnostics)
            diagnostics["representative_outcome"] = best_outcome
            if diagnostics.get("condition_number_warning"):
                warnings.append(
                    f"Preflight: cohort {cohort!r} design has a high standardized condition number "
                    f"({diagnostics['condition_number_standardized']:.3g})."
                )
        except AnalysisNonIdentifiableError as exc:
            diagnostics = dict(exc.diagnostics)
            diagnostics["representative_outcome"] = best_outcome
            violations.append(f"Preflight: cohort {cohort!r} {exc}")
    return violations, warnings, diagnostics


def _outcomes(contract: ClaimContract) -> list[str]:
    outcome = contract.estimand.outcome
    return list(outcome) if isinstance(outcome, list) else [outcome]


def _matched_outcome_columns(contract: ClaimContract, info: CohortPreflightInfo) -> list[str]:
    outcomes = _outcomes(contract)
    idps = list(info.idps)
    columns = set(info.columns)
    if contract.estimand.unit == "scalar":
        return [outcome for outcome in outcomes if outcome in columns]

    matched: list[str] = []
    for outcome in outcomes:
        text = str(outcome)
        if text in {"*", "brainwide", "all"}:
            matched.extend(idps)
        elif any(char in text for char in "*?["):
            matched.extend(col for col in idps if fnmatch.fnmatch(col, text))
        elif text in columns:
            matched.append(text)
        else:
            prefix = text if text.endswith("_") else f"{text}_"
            matched.extend(col for col in idps if col.startswith(prefix))
    return list(dict.fromkeys(matched))


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
    tautology = _tautological_comparison(tree)
    if tautology is not None:
        return names, tautology
    return names, None


def _validate_association_predictor(series: pd.Series, predictor: str, cohort: str, min_complete_rows: int) -> list[str]:
    violations: list[str] = []
    predictor_key = predictor.lower()
    if predictor_key in _CATEGORICAL_ASSOCIATION_PREDICTORS or predictor_key.startswith("dx_"):
        violations.append(
            f"Preflight: association predictor {predictor!r} in cohort {cohort!r} is categorical; "
            "use estimand.type='group_diff' with estimand.group for sex, diagnosis, site, or case/control contrasts."
        )
        return violations
    numeric = pd.to_numeric(series, errors="coerce")
    n_numeric = int(numeric.notna().sum())
    if n_numeric < min_complete_rows:
        violations.append(
            f"Preflight: association predictor {predictor!r} in cohort {cohort!r} is not numerically usable "
            f"after filters ({n_numeric} numeric rows < {min_complete_rows}); use group_diff for categorical contrasts."
        )
        return violations
    if numeric.dropna().nunique() < 2:
        violations.append(f"Preflight: association predictor {predictor!r} in cohort {cohort!r} has fewer than two numeric values after filters.")
    return violations


def _tautological_comparison(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left_name = _simple_name(node.left)
        if left_name is None:
            continue
        for op, comparator in zip(node.ops, node.comparators):
            right_name = _simple_name(comparator)
            if right_name == left_name and isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                return f"tautological inclusion comparison {left_name!r} to itself is not allowed"
    return None


def _simple_name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _cohort_aliases(cohort: str) -> list[str]:
    aliases = [cohort]
    suffixes = (
        "_DISC_SITES",
        "_REP_SITES",
        "_HOLDOUT_SITES",
        "_HOLDOUT_DISC",
        "_HOLDOUT_REP",
        "_EXTERNAL_DISC",
        "_EXTERNAL_REP",
        "_HOLDOUT",
        "_DISC",
        "_REP",
        "_CN",
    )
    for suffix in suffixes:
        if cohort.endswith(suffix):
            aliases.append(cohort[: -len(suffix)])
    for split in ("_DISC_s", "_REP_s", "_HOLDOUT_s"):
        if split in cohort:
            aliases.append(cohort.split(split, 1)[0])
    return list(dict.fromkeys(item for item in aliases if item))


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        return list(pd.read_parquet(path).columns)
