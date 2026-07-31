# External Structural MRI Preparation

The active structural pipeline is driven by `configs/external_datasets.local.yml` and
the arcdev launcher
`scripts/data_processing/launch_external_freesurfer_arcdev.sh`.

The previous numbered shell workflow was retired because it inferred subject
identity from paths, treated `aseg.stats` as completion, silently skipped
missing subjects during aggregation, defaulted to older FreeSurfer modules,
and deleted partial outputs before retrying.

The replacement provides:

- deterministic one-scan-per-subject manifests with NIfTI preflight;
- explicit FreeSurfer 7.4.1 validation and FastSurfer GPU execution;
- one sequential subject queue per GPU (`GPU_IDS=0,1` by default);
- receipt-backed completion checks over segmentation, bilateral DKT stats, and
  cortical surfaces;
- preserved and timestamped failed attempts;
- canonical regional volumes in mm3 and separately named thickness measures;
- quarantine for imaging whose authoritative phenotype join is unavailable.

Use a canary before a full run:

```bash
SSH_HOST=arcdev \
REMOTE_FASTSURFER_HOME=/path/to/FastSurfer \
DATASETS=AIBL,BLSA LIMIT=2 \
scripts/data_processing/launch_external_freesurfer_arcdev.sh
```

`SSH_HOST` may be any SSH host name or alias. It defaults to `arcdev`;
`ARCDEV_HOST` remains accepted for older commands.

Omit `LIMIT` only after the canary receipts and canonical aggregation pass.

Set `RECON_ENGINE=recon-all` for the explicit CPU override. The launcher never
falls back to another version. The remote worker runs
`module load freesurfer/7.4.1` by default; set `FREESURFER_MODULE` to select
another module name. The selected installation must report FreeSurfer 7.4.1.
Set `REMOTE_FASTSURFER_HOME` manually to the remote FastSurfer checkout when
using the default FastSurfer engine. Failed/partial outputs are left untouched
unless `RETRY_FAILED=1` is supplied; a retry moves the prior attempt into the
run's `failed_attempts/` directory.

The runner defaults `REMOTE_FASTSURFER_PYTHON` to
`$REMOTE_FASTSURFER_HOME/.venv/bin/python`, verifies Python 3.10 or newer plus
the required `nibabel`, `torch`, and `yacs` imports, and prepends that virtual
environment to `PATH` for the FastSurfer shell wrapper.

Use a unique `RUN_ID` for every launch. Existing logs and concurrent workers
for the same run/stage are rejected by default. A failed subject makes the
stage exit nonzero even if FastSurfer's wrapper incorrectly returns zero.

`EXPECTED_FREESURFER_VERSION` defaults to `7.4.1`. Override it explicitly only
when intentionally running an entire evidence set with another installed
version; the runtime validates and records that choice.

Structural outputs are written as `*_sMRI.parquet`. Completion receipts are
strict inputs to aggregation, and their hashes plus source-T1 checksums are
carried into the prepared table. Existing Shile outputs are reused only after
bilateral DKT and surface checks and receipt adoption.
