"""ABIDE adapter using nilearn's ABIDE PCP fetcher."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from confirm.ingest.base import CohortAdapter
from confirm.schema import normalize_sex, validate_canonical

LOGGER = logging.getLogger(__name__)


class AbideAdapter(CohortAdapter):
    """Normalize ABIDE phenotypic data and available derivatives.

    If structural phenotype columns are absent, the adapter computes compact
    ``fc_*`` summary features from CC200 time-series files. This keeps B0 purely
    tabular after ingestion while avoiding raw image preprocessing.
    """

    def __init__(self, data_dir: str | Path | None = None, n_subjects: int | None = None):
        super().__init__("ABIDE", raw_paths={"data_dir": data_dir} if data_dir else {})
        self.data_dir = Path(data_dir) if data_dir else None
        self.n_subjects = n_subjects

    def _fetch(self) -> Any:
        from nilearn.datasets import fetch_abide_pcp

        return fetch_abide_pcp(
            data_dir=str(self.data_dir) if self.data_dir else None,
            n_subjects=self.n_subjects,
            pipeline="cpac",
            derivatives=["rois_cc200"],
        )

    @staticmethod
    def _phenotypic(data: Any) -> pd.DataFrame:
        phen = getattr(data, "phenotypic", None)
        if phen is None:
            phen = data.get("phenotypic")
        return pd.DataFrame(phen)

    @staticmethod
    def _first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
        lowered = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate in df.columns:
                return df[candidate]
            if candidate.lower() in lowered:
                return df[lowered[candidate.lower()]]
        return None

    @staticmethod
    def _fc_summary(source: Any) -> dict[str, float]:
        # nilearn's fetch_abide_pcp(derivatives=["rois_cc200"]) returns a list of
        # (timepoints, n_rois) numpy arrays — NOT file paths. Handle arrays directly.
        if isinstance(source, np.ndarray):
            arr = np.asarray(source, dtype=float)
            if arr.ndim != 2 or min(arr.shape) < 2:
                return {"fc_mean_abs": float("nan"), "fc_mean_positive": float("nan")}
            corr = np.corrcoef(arr.T)  # rows=timepoints, cols=ROIs -> ROI x ROI corr
            upper = corr[np.triu_indices_from(corr, k=1)]
            return {
                "fc_mean_abs": float(np.nanmean(np.abs(upper))),
                "fc_mean_positive": float(np.nanmean(upper[upper > 0])) if np.any(upper > 0) else float("nan"),
            }
        try:
            arr = np.loadtxt(source)
        except Exception:
            import nibabel as nib

            img = nib.load(str(source))
            data = np.asanyarray(img.dataobj, dtype=float)
            if data.ndim != 4 or data.shape[-1] < 3:
                return {"fc_mean_abs": float(np.nanstd(data)), "fc_mean_positive": float(np.nanmean(data))}
            voxels = data.reshape(-1, data.shape[-1])
            keep = np.nanstd(voxels, axis=1) > 0
            voxels = voxels[keep]
            if len(voxels) < 2:
                return {"fc_mean_abs": np.nan, "fc_mean_positive": np.nan}
            # NOTE: For image-only fetches, use a deterministic voxel subsample
            # as a compact CPU-only seed FC summary.
            take = np.linspace(0, len(voxels) - 1, min(200, len(voxels))).astype(int)
            arr = voxels[take].T
        if arr.ndim != 2 or min(arr.shape) < 2:
            return {"fc_mean_abs": np.nan, "fc_mean_positive": np.nan}
        if arr.shape[0] < arr.shape[1]:
            time_by_feature = arr
        else:
            time_by_feature = arr.T if arr.shape[1] < 5 and arr.shape[0] > 20 else arr
        corr = np.corrcoef(time_by_feature.T)
        upper = corr[np.triu_indices_from(corr, k=1)]
        return {
            "fc_mean_abs": float(np.nanmean(np.abs(upper))),
            "fc_mean_positive": float(np.nanmean(upper[upper > 0])) if np.any(upper > 0) else np.nan,
        }

    def to_canonical(self) -> pd.DataFrame:
        data = self._fetch()
        phen = self._phenotypic(data)
        out = pd.DataFrame(index=phen.index)
        subject = self._first_existing(phen, ["SUB_ID", "subject_id", "participant_id"])
        out["subject_id"] = subject.astype(str) if subject is not None else [f"ABIDE_{i:05d}" for i in range(len(phen))]
        out["session"] = "ses-01"
        out["cohort"] = "ABIDE"
        site = self._first_existing(phen, ["SITE_ID", "site"])
        out["site"] = site.fillna("unknown") if site is not None else "unknown"
        out["field_strength"] = self._first_existing(phen, ["field_strength", "FIELD_STRENGTH"])
        out["fs_version"] = self._first_existing(phen, ["fs_version", "FS_VERSION"])
        age = self._first_existing(phen, ["AGE_AT_SCAN", "age"])
        if age is None:
            raise ValueError("ABIDE phenotype lacks AGE_AT_SCAN")
        out["age"] = age
        sex = self._first_existing(phen, ["SEX", "sex"])
        if sex is None:
            raise ValueError("ABIDE phenotype lacks SEX")
        out["sex"] = normalize_sex(sex)
        dx = self._first_existing(phen, ["DX_GROUP", "dx"])
        out["dx"] = dx.map({1: "ASD", 2: "TC", "1": "ASD", "2": "TC"}).fillna(dx) if dx is not None else pd.NA
        etiv = self._first_existing(phen, ["eTIV", "EstimatedTotalIntraCranialVol", "ICV"])
        out["eTIV"] = etiv if etiv is not None else pd.NA

        thickness_candidates = [col for col in phen.columns if "thick" in col.lower()]
        if "smri_meanthickness" in phen.columns:
            out["smri_meanthickness"] = phen["smri_meanthickness"]
        elif thickness_candidates:
            out["smri_meanthickness"] = phen[thickness_candidates].apply(pd.to_numeric, errors="coerce").mean(axis=1)

        hippocampus_l = self._first_existing(phen, ["smri_hippocampus_lh", "Left-Hippocampus", "lh_hippocampus"])
        hippocampus_r = self._first_existing(phen, ["smri_hippocampus_rh", "Right-Hippocampus", "rh_hippocampus"])
        if hippocampus_l is not None:
            out["smri_hippocampus_lh"] = hippocampus_l
        if hippocampus_r is not None:
            out["smri_hippocampus_rh"] = hippocampus_r
        if "smri_meanthickness" not in out.columns:
            rois = getattr(data, "rois_cc200", None) or data.get("rois_cc200", [])
            summaries = [self._fc_summary(path) for path in rois]
            for key in ("fc_mean_abs", "fc_mean_positive"):
                out[key] = [summary.get(key, np.nan) for summary in summaries]
            # TODO(confirm): Add richer fc_* network summaries and pet_* feature
            # extraction beyond these seed scalar summaries.
            LOGGER.info("ABIDE structural IDPs absent; computed fc_* summaries from CC200 time series")

        return validate_canonical(out)
