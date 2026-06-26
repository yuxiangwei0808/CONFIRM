# Phase 1 Plan — ADNI sMRI/PET/Cognition Adapter + AD Claims

Generated: 2026-06-17. Owner-reviewed plan; implemented by Codex, reviewed by Claude.

## Goal
Add real Alzheimer's disease claims to the CONFIRM benchmark using local ADNI tabular derivatives (no raw images, no GPU), with a real cross-cohort replication partner. This is the first scientifically-meaningful disease domain beyond the existing age/sex stability controls.

## Why this unblocks everything
The AD known-positives (hippocampal atrophy, FDG hypometabolism) have published effect-size priors, so they are the cleanest "must-confirm" anchors for the faithfulness benchmark. They also exercise the headline replication gate (ADNI → OASIS3) on a real disease, not a synthetic trap.

## Data (confirmed local)
- `data/raw/ADNIMERGE.xlsx` — one row per subject-visit, 114 cols. Confirmed fields:
  - IDs/visit: `PTID`, `VISCODE`, `SITE`, `EXAMDATE`
  - demo/dx: `AGE`, `PTGENDER`, `PTEDUCAT`, `APOE4`, `DX` (CN/MCI/Dementia)
  - FreeSurfer sMRI: `Hippocampus`, `WholeBrain`, `Entorhinal`, `Fusiform`, `MidTemp`, `Ventricles`, `ICV`
  - PET: `FDG`, `AV45`, `ABETA`, `TAU`, `PTAU`
  - cognition: `MMSE`, `CDRSB`, `ADAS11`, `ADAS13`, `RAVLT_immediate`, `FAQ`, `MOCA`
- `data/raw/ADNI_document-...zip` — cleaner per-instrument CSVs available if ADNIMERGE fields prove insufficient (selective small-CSV extraction only; do NOT unpack nested `Imaging.zip`).
- OASIS3 (replication partner): investigate `data/raw/oasis3_extracted/` and `data/raw/oasis3_data_info*.tar.gz` for FreeSurfer regional volumes + diagnosis.

## Deliverables
1. `src/confirm/ingest/adni.py` — `AdniAdapter(CohortAdapter)` producing a canonical ADNI sMRI/PET/cognition table from ADNIMERGE.
2. `src/confirm/ingest/oasis3.py` — OASIS3 sMRI adapter **if** local FreeSurfer + diagnosis is available; otherwise report exactly what is missing and leave replication as a marked TODO (do NOT fabricate).
3. Canonical outputs in `data/prepared_data/smri_disease/` (kept separate from the fMRI benchmark so the existing staged results are untouched).
4. Claim contracts in `configs/contracts/`:
   - AD hippocampal atrophy (ADNI → OASIS3), expected `confirmed`.
   - AD FDG hypometabolism (ADNI; replication only if a real 2nd FDG cohort exists, else single-cohort candidate).
5. `tests/test_adni_adapter.py` using a small fixture (not the full xlsx).
6. A short run receipt under `review-stage/adni-phase1/`.

## Acceptance
- `validate_canonical` passes on the ADNI table; report N rows, DX counts, non-null IDP coverage.
- Existing suite still green: `PYTHONPATH=src ./.venv/bin/python -m pytest -q`.
- AD hippocampal-atrophy claim runs end-to-end and emits a verdict + receipt from executed code (no hardcoded numbers).

## Guardrails
CPU-only, no network, no raw-image processing. Surgical: new files + minimal wiring, no refactors of existing modules. Replication cohorts must be real or explicitly marked TODO.
```
