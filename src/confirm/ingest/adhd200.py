"""ADHD-200 adapter using nilearn's ADHD fetcher."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from confirm.ingest.base import CohortAdapter
from confirm.ingest.abide import AbideAdapter
from confirm.schema import normalize_sex, validate_canonical

LOGGER = logging.getLogger(__name__)


class Adhd200Adapter(CohortAdapter):
    """Normalize ADHD-200 phenotype and compact fMRI derivative summaries."""

    def __init__(self, data_dir: str | Path | None = None, n_subjects: int | None = None):
        super().__init__("ADHD200", raw_paths={"data_dir": data_dir} if data_dir else {})
        self.data_dir = Path(data_dir) if data_dir else None
        self.n_subjects = n_subjects

    def _fetch(self) -> Any:
        from nilearn.datasets import fetch_adhd

        return fetch_adhd(data_dir=str(self.data_dir) if self.data_dir else None, n_subjects=self.n_subjects)

    def to_canonical(self) -> pd.DataFrame:
        data = self._fetch()
        phen_obj = getattr(data, "phenotypic", None)
        if phen_obj is None and hasattr(data, "get"):
            phen_obj = data.get("phenotypic")
        phen = pd.DataFrame(phen_obj)
        out = pd.DataFrame(index=phen.index)
        first = AbideAdapter._first_existing
        subject = first(phen, ["Subject", "SUB_ID", "subject_id", "participant_id"])
        out["subject_id"] = subject.astype(str) if subject is not None else [f"ADHD200_{i:05d}" for i in range(len(phen))]
        out["session"] = "ses-01"
        out["cohort"] = "ADHD200"
        site = first(phen, ["site", "SITE", "SITE_ID"])
        out["site"] = site.fillna("unknown") if site is not None else "unknown"
        out["field_strength"] = first(phen, ["field_strength", "FIELD_STRENGTH"])
        out["fs_version"] = first(phen, ["fs_version", "FS_VERSION"])
        age = first(phen, ["age", "Age", "AGE"])
        if age is None:
            raise ValueError("ADHD-200 phenotype lacks age")
        out["age"] = age
        sex = first(phen, ["sex", "Sex", "SEX", "Gender"])
        if sex is None:
            raise ValueError("ADHD-200 phenotype lacks sex")
        out["sex"] = normalize_sex(sex)
        dx = first(phen, ["dx", "DX", "adhd", "ADHD", "diagnosis"])
        out["dx"] = dx.map({0: "TC", 1: "ADHD", "0": "TC", "1": "ADHD"}).fillna(dx) if dx is not None else pd.NA
        out["eTIV"] = first(phen, ["eTIV", "ICV", "EstimatedTotalIntraCranialVol"])

        thickness_cols = [col for col in phen.columns if "thick" in col.lower()]
        if "smri_meanthickness" in phen.columns:
            out["smri_meanthickness"] = phen["smri_meanthickness"]
        elif thickness_cols:
            out["smri_meanthickness"] = phen[thickness_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            func_obj = getattr(data, "func", None)
            if func_obj is None and hasattr(data, "get"):
                func_obj = data.get("func", [])
            func_paths = list(func_obj or [])
            summaries = []
            for path in func_paths:
                try:
                    summaries.append(AbideAdapter._fc_summary(path))
                except Exception as exc:  # pragma: no cover - depends on nilearn dataset shape
                    LOGGER.warning("Could not summarize ADHD-200 functional file %s: %s", path, exc)
                    summaries.append({"fc_mean_abs": np.nan, "fc_mean_positive": np.nan})
            for key in ("fc_mean_abs", "fc_mean_positive"):
                out[key] = [summary.get(key, np.nan) for summary in summaries]
            # TODO(confirm): Add richer fc_* network summaries and pet_* feature
            # extraction beyond these seed scalar summaries.
            LOGGER.info("ADHD-200 structural IDPs absent; computed fc_* summaries from fetched functional arrays")

        return validate_canonical(out)
