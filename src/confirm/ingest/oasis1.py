"""OASIS-1 VBM adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from confirm.ingest.base import CohortAdapter
from confirm.schema import normalize_sex, validate_canonical


class Oasis1Adapter(CohortAdapter):
    """Normalize OASIS-1 VBM metadata and derive a rough gray-matter total."""

    def __init__(self, data_dir: str | Path | None = None):
        super().__init__("OASIS1", raw_paths={"data_dir": data_dir} if data_dir else {})
        self.data_dir = Path(data_dir) if data_dir else None

    def _fetch(self) -> Any:
        from nilearn.datasets import fetch_oasis_vbm

        return fetch_oasis_vbm(data_dir=str(self.data_dir) if self.data_dir else None)

    @staticmethod
    def _gm_total(path: str | Path) -> float:
        import nibabel as nib

        # TODO(confirm): OASIS-1 GM-total is a rough proxy from VBM probability
        # maps, not a curated anatomical volume. Replace with validated cohort
        # derivatives when available.
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj, dtype=float)
        return float(np.nansum(data[data > 0]))

    def to_canonical(self) -> pd.DataFrame:
        data = self._fetch()
        ext_obj = getattr(data, "ext_vars", None)
        if ext_obj is None and hasattr(data, "get"):
            ext_obj = data.get("ext_vars")
        gm_obj = getattr(data, "gray_matter_maps", None)
        if gm_obj is None and hasattr(data, "get"):
            gm_obj = data.get("gray_matter_maps", [])
        ext = pd.DataFrame(ext_obj)
        gm_paths = list(gm_obj or [])
        if len(gm_paths) != len(ext):
            raise ValueError("OASIS-1 metadata and gray-matter maps have different lengths")

        out = pd.DataFrame(index=ext.index)
        # nilearn OASIS-1 ext_vars columns: id, mf, hand, age, educ, ses, mmse, cdr, etiv, nwbv, asf, delay
        out["subject_id"] = ext["id"].astype(str)
        out["session"] = "ses-01"
        out["cohort"] = "OASIS1"
        out["site"] = "OASIS"
        out["field_strength"] = pd.NA
        out["fs_version"] = pd.NA
        out["age"] = ext["age"]
        out["sex"] = normalize_sex(ext["mf"])
        out["dx"] = ext.get("cdr", pd.Series(pd.NA, index=ext.index)).map(lambda cdr: "CN" if cdr == 0 else "AD" if pd.notna(cdr) else pd.NA)
        out["eTIV"] = pd.to_numeric(ext["etiv"], errors="coerce")
        out["smri_gm_total"] = [self._gm_total(path) for path in gm_paths]
        # nWBV = normalized whole-brain volume: a curated, validated atrophy measure that
        # declines strongly with age. Preferred over the rough gm_total VBM-sum proxy.
        if "nwbv" in ext.columns:
            out["smri_nwbv"] = pd.to_numeric(ext["nwbv"], errors="coerce")
        if "mmse" in ext.columns:
            out["beh_mmse"] = ext["mmse"]
        if "cdr" in ext.columns:
            out["beh_cdr"] = ext["cdr"]
        return validate_canonical(out)
