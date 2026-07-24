# Data Preparation Utilities

Canonical external-data policy and preparation are implemented by
`external_dataset_registry.py`, `prepare_external_evidence.py`, and
`promote_external_evidence.py`. Remote structural and fMRI processing is
launched only through `scripts/data_processing/`.

`prepare_nacc_external.py` and `prepare_ds000030_external.py` are retained as
dataset-specific provenance utilities for the currently prepared NACC and CNP
tables. They are not called by the Stage 0-4 experiment launchers.
