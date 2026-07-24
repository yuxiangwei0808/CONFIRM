from __future__ import annotations

import pandas as pd

from confirm.derived_columns import (
    add_virtual_columns,
    columns_with_virtuals,
    confirm_dx_levels,
    confirm_dx_mapping,
)


def test_cnp_schz_maps_to_case_without_mapping_other_diagnoses():
    mapping = confirm_dx_mapping("ds000030_EXTERNAL_DISC")

    assert mapping["schz"] == "case"
    assert mapping["control"] == "control"
    assert "bipolar" not in mapping
    assert "adhd" not in mapping
    assert confirm_dx_levels(
        "ds000030_EXTERNAL_REP",
        ["SCHZ", "CONTROL", "BIPOLAR", "ADHD"],
    ) == ["case", "control"]

    table = add_virtual_columns(
        pd.DataFrame({"dx": ["SCHZ", "CONTROL", "BIPOLAR", "ADHD"]}),
        "ds000030_EXTERNAL_DISC",
    )
    assert table["confirm_dx"].tolist()[:2] == ["case", "control"]
    assert table["confirm_dx"].isna().tolist()[2:] == [True, True]


def test_nacc_measurement_aliases_are_available_as_virtual_columns():
    columns = columns_with_virtuals(
        "NACC_EXTERNAL_DISC",
        ["smri_midtemp", "smri_ventricles"],
    )
    table = add_virtual_columns(
        pd.DataFrame(
            {
                "smri_midtemp": [1.0],
                "smri_ventricles": [2.0],
            }
        ),
        "NACC_EXTERNAL_REP",
    )

    assert "smri_midtemporal" in columns
    assert "smri_lateralventricle" in columns
    assert table["smri_midtemporal"].tolist() == [1.0]
    assert table["smri_lateralventricle"].tolist() == [2.0]
