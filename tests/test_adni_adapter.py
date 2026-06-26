import pandas as pd

from confirm.ingest.adni import AdniAdapter
from confirm.ingest.oasis3 import _cdr_to_dx
from confirm.schema import idp_columns, validate_canonical


def test_adni_adapter_canonicalizes_fixture():
    raw = pd.DataFrame(
        {
            "PTID": ["011_S_0002", "011_S_0003", "011_S_0004", "011_S_0005"],
            "VISCODE": ["bl", "bl", "m06", "bl"],
            "SITE": [11, 11, 12, 13],
            "AGE": [74.3, 81.3, 70.0, 76.0],
            "PTGENDER": ["Male", "Female", "Male", "Female"],
            "DX": ["CN", "Dementia", "MCI", "MCI"],
            "ICV": [1984660, 1920690, 1500000, 1600000],
            "Hippocampus": [8336, 5319, 7200, 6800],
            "WholeBrain": [1000000, 900000, 950000, 925000],
            "Entorhinal": [4200, 3100, 3900, 3700],
            "Fusiform": [18000, 15000, 17000, 16000],
            "MidTemp": [19000, 15000, 17500, 16500],
            "Ventricles": [30000, 55000, 40000, 45000],
            "FDG": [1.36, 1.09, 1.22, 1.18],
            "AV45": [1.1, 1.4, 1.2, 1.3],
            "MMSE": [30, 23, 28, 27],
            "CDRSB": [0.0, 5.0, 0.5, 1.0],
            "ADAS13": [8, 28, 12, 14],
            "MOCA": [28, 18, 24, 23],
            "PTEDUCAT": [16, 14, 18, 12],
            "APOE4": [0, 1, 0, 2],
        }
    )

    canonical = AdniAdapter(raw).to_canonical()

    assert {"subject_id", "cohort", "site", "age", "sex"}.issubset(canonical.columns)
    assert set(canonical["session"]) == {"bl"}
    assert set(canonical["sex"]) == {"M", "F"}
    assert set(canonical["dx"].dropna()).issubset({"CN", "MCI", "Dementia"})
    assert any(column.startswith("smri_") for column in idp_columns(canonical.columns))
    assert any(column.startswith("pet_") for column in idp_columns(canonical.columns))
    validate_canonical(canonical)


def test_oasis3_cdr_threshold_can_exclude_mild_cdr():
    assert _cdr_to_dx(0.0, dementia_cdr_min=1.0) == "CN"
    assert pd.isna(_cdr_to_dx(0.5, dementia_cdr_min=1.0))
    assert _cdr_to_dx(1.0, dementia_cdr_min=1.0) == "Dementia"
    assert _cdr_to_dx(0.5, dementia_cdr_min=0.0) == "Dementia"
