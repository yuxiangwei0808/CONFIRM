"""Generic YAML-driven CSV adapter for user-provided cohorts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from confirm.ingest.base import CohortAdapter
from confirm.schema import CANONICAL_COLUMNS, normalize_sex, validate_canonical

LOGGER = logging.getLogger(__name__)


class GenericCsvAdapter(CohortAdapter):
    """Normalize an arbitrary CSV using a mapping YAML.

    The mapping schema is documented in ``configs/cohort_maps/_TEMPLATE.yaml``.
    Unknown raw columns that look like numeric IDPs can be explicitly listed in
    ``idps``; unlisted columns are not guessed into the analysis surface.
    """

    def __init__(self, raw_csv: str | Path, mapping_yaml: str | Path):
        self.raw_csv = Path(raw_csv)
        self.mapping_yaml = Path(mapping_yaml)
        with self.mapping_yaml.open("r", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle) or {}
        cohort = str(mapping.get("cohort", self.raw_csv.stem))
        super().__init__(cohort=cohort, raw_paths={"csv": self.raw_csv}, column_map=mapping)

    def _source(self, raw: pd.DataFrame, name: str, default: Any = pd.NA) -> pd.Series:
        columns = self.column_map.get("columns", {})
        defaults = self.column_map.get("defaults", {})
        if name in columns and columns[name] in raw.columns:
            return raw[columns[name]]
        value = defaults.get(name, default)
        return pd.Series([value] * len(raw), index=raw.index)

    def _map_values(self, series: pd.Series, mapping_name: str) -> pd.Series:
        value_map = self.column_map.get(mapping_name, {}) or {}
        if not value_map:
            return series
        normalized = {str(key): value for key, value in value_map.items()}
        return series.map(lambda value: normalized.get(str(value), value))

    def to_canonical(self) -> pd.DataFrame:
        raw = pd.read_csv(self.raw_csv)
        out = pd.DataFrame(index=raw.index)
        out["subject_id"] = self._source(raw, "subject_id").astype(str)
        out["session"] = self._source(raw, "session", "ses-01").fillna("ses-01")
        out["cohort"] = self.cohort
        out["site"] = self._source(raw, "site", "unknown").fillna("unknown")
        out["field_strength"] = self._source(raw, "field_strength")
        out["fs_version"] = self._source(raw, "fs_version")
        out["age"] = self._source(raw, "age")
        out["sex"] = normalize_sex(self._map_values(self._source(raw, "sex"), "sex_map"))
        out["dx"] = self._map_values(self._source(raw, "dx"), "dx_map")
        out["eTIV"] = self._source(raw, "eTIV")

        for canonical, source in (self.column_map.get("idps", {}) or {}).items():
            if source not in raw.columns:
                raise ValueError(f"Mapped IDP source column {source!r} not found in {self.raw_csv}")
            target = canonical if canonical.startswith(("smri_", "pet_", "fc_")) else f"raw_{canonical}"
            if target.startswith("raw_"):
                LOGGER.warning("Passing through unmapped IDP %s as %s", canonical, target)
            out[target] = raw[source]

        for canonical, source in (self.column_map.get("behavioral", {}) or {}).items():
            if source not in raw.columns:
                raise ValueError(f"Mapped behavioral source column {source!r} not found in {self.raw_csv}")
            target = canonical if canonical.startswith("beh_") else f"beh_{canonical}"
            out[target] = raw[source]

        missing_required_sources = [
            name for name in ("subject_id", "age", "sex") if self.column_map.get("columns", {}).get(name) not in raw.columns
        ]
        if missing_required_sources:
            raise ValueError(f"Mapping is missing required source columns: {missing_required_sources}")

        return validate_canonical(out[[col for col in out.columns if col in CANONICAL_COLUMNS or col.startswith(("smri_", "pet_", "fc_", "beh_", "raw_"))]])

