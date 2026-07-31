# Lab handoff and local layout

## Working directories

| Directory | Status | Use |
|---|---|---|
| `src/confirm/` | active | Package implementation and CLI. |
| `tests/` | active | Default project test suite. |
| `scripts/` | active | Reproducible stage launchers and data-preparation entry points. |
| `nbs/` | active | Deterministic analysis and paper-figure builders. |
| `nbs_data/` | active | Cohort preparation and external-evidence utilities. |
| `benchmark/neuroclaimbench-v2.1/` | active | Lean, checksummed benchmark package. |
| `data/` | local, active | Restricted raw data, prepared cohorts, and caches; ignored by Git. |
| `review-stage/` | local, active | Frozen run outputs; ignored by Git except its index. |
| `review-stage/_archive_*/` | local archive | Superseded or exploratory runs. |
| `external/` | local baseline | Pinned third-party comparison repositories; excluded from default tests. |
| `paper/` | active submodule | Manuscript source and paper assets. |
| `docs/archive/` | archive | Historical planning, discovery, and ideation records. |

## Local configuration

`configs/external_datasets.local.yml` is the only location for site- or
machine-specific paths. Begin from `configs/external_datasets.example.yml`.
Do not put personal home directories, SSH host aliases, credentials, or
machine-specific paths in tracked configuration.

The remote data-preparation scripts require `REMOTE_ROOT` and `REMOTE_PYTHON`.
They optionally accept `EXTERNAL_DATASET_CONFIG`; otherwise they use the local
configuration file in the deployed workspace.

## Result handling

1. Write new runs to a new descriptive directory under `review-stage/`.
2. Promote a run to active only after adding it to `MANIFEST.md`,
   `RESULTS_MANIFEST.md`, and `RESULTS_SHA256SUMS` as appropriate.
3. Before archiving a run, record its original path, replacement, and SHA-256
   manifest in the destination archive index.
4. Never delete the only local copy of a raw input or a result used by a paper
   figure/table.

The archive index is [`ARCHIVE_INDEX.md`](ARCHIVE_INDEX.md). Existing archives
are retained for provenance; new archives use `_archive_YYYYMMDD_<reason>`.

## External baselines

The current comparison checkouts are:

| Baseline | Path | Pinned revision |
|---|---|---|
| NeuroClaw | `external/NeuroClaw/` | `b9e3833a795b0f3a5d6348ffab814b0b4c904c3e` |
| Veritas | `external/veritas/` | `17dbdc96cef23c29a4efbf0291de6d6295908a17` |

Update the table and the corresponding result manifest whenever a baseline is
re-pinned. Their test suites have separate dependencies; run them only from
their own checkout.
