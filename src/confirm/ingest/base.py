"""Base class for cohort adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from confirm.schema import CANONICAL_IDPS, idp_columns, validate_canonical


class CohortAdapter(ABC):
    """Normalize one cohort into the CONFIRM canonical subject-level schema."""

    def __init__(self, cohort: str, raw_paths: dict[str, str | Path] | None = None, column_map: dict[str, Any] | None = None):
        self.cohort = cohort
        self.raw_paths = raw_paths or {}
        self.column_map = column_map or {}

    @abstractmethod
    def to_canonical(self) -> pd.DataFrame:
        """Return a validated canonical dataframe."""

    def column_dictionary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a small column dictionary emitted next to each canonical file."""

        rows: list[dict[str, str]] = []
        for col in df.columns:
            if col in CANONICAL_IDPS:
                meaning = CANONICAL_IDPS[col]
            elif col in idp_columns([col]):
                meaning = "Imaging-derived phenotype."
            elif col.startswith("beh_"):
                meaning = "Behavioral or clinical score."
            elif col.startswith("raw_"):
                meaning = "Unmapped raw IDP passed through with raw_ prefix."
            else:
                meaning = "Canonical subject-level metadata."
            rows.append({"column": col, "meaning": meaning, "dtype": str(df[col].dtype)})
        return pd.DataFrame(rows)

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        """Write ``<cohort>.parquet`` and ``<cohort>.dict.csv``."""

        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        df = validate_canonical(self.to_canonical())
        parquet_path = target / f"{self.cohort}.parquet"
        dict_path = target / f"{self.cohort}.dict.csv"
        df.to_parquet(parquet_path, index=False)
        self.column_dictionary(df).to_csv(dict_path, index=False)
        return parquet_path, dict_path

