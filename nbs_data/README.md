# Data Preparation Utilities

Canonical external-data policy and preparation are implemented by
`external_dataset_registry.py`, `prepare_external_evidence.py`, and
`promote_external_evidence.py`. Remote structural and fMRI processing is
launched only through `scripts/data_processing/`.

Copy `configs/external_datasets.example.yml` to the ignored
`configs/external_datasets.local.yml` before running an external-data command.
The local file holds lab- and machine-specific paths. Remote launchers require
`REMOTE_ROOT` and `REMOTE_PYTHON`; they never supply user-specific defaults.

`prepare_nacc_external.py` and `prepare_ds000030_external.py` are retained as
dataset-specific provenance utilities for the currently prepared NACC and CNP
tables. They are not called by the Stage 0-4 experiment launchers.

The one-off remote table-discovery converter is archived under
`_archive_20260730_remote_discovery/`; it is not part of the active stack.
