# CONFIRM lab workspace

This repository contains the code, data layout, frozen experiment receipts, and
paper sources for CONFIRM (statistical claim governance for neuroimaging
discovery). It is a shared lab workspace: large cohort data and generated runs
remain local, while this Git repository records the code and small, reusable
metadata needed to navigate and reproduce the work.

## Start here

| Need | Location |
|---|---|
| Project map and active evidence | `MANIFEST.md` |
| Experimental provenance and result hashes | `RESULTS_MANIFEST.md`, `RESULTS_SHA256SUMS` |
| Method and end-to-end commands | `IMPLEMENTATION_NOTES.md` |
| Benchmark package | `benchmark/neuroclaimbench-v2.1/` |
| Analysis/figure builders | `nbs/` |
| Cohort preparation | `nbs_data/` and `scripts/data_processing/` |
| Paper source | `paper/` (Git submodule) |
| Archived material | `docs/archive/` and `review-stage/_archive_*/` |

## Setup

Use Python 3.9 or newer. Install the package and the test tools with:

```bash
python -m pip install -e '.[dev]'
pytest
```

The default test suite is deliberately scoped to `tests/`. External baselines
under `external/` have their own environments and are not part of this suite.

## Data and configuration

Raw and prepared cohort data are intentionally ignored by Git and live under
`data/`. The expected layout is documented in `MANIFEST.md` and
`docs/LAB_HANDOFF.md`.

External dataset locations are lab-specific. Copy
`configs/external_datasets.example.yml` to
`configs/external_datasets.local.yml`, then replace the placeholders with the
paths available on the machine or cluster. The local file is ignored by Git.
Remote preparation launchers require explicit `REMOTE_ROOT` and
`REMOTE_PYTHON` settings; see `nbs_data/README.md`.

## Results and archives

Only the directories listed as **active** in `MANIFEST.md` support the current
paper. Superseded runs remain recoverable under dated archive directories. Do
not remove an active result or raw input until a replacement is recorded in the
manifest and its hashes have been checked.

## Paper

`paper/` is maintained as a separate repository because it is synced with the
manuscript workflow. After cloning the workspace, initialize it with:

```bash
git submodule update --init --recursive
cd paper && make all
```

Paper changes are staged and committed inside `paper/` first; the root
repository then records the resulting submodule commit.
