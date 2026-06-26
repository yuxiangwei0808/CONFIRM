# Experiment Code Review — Benchmark-Ready Runner

Date: 2026-06-16

Scope: `src/bench/run_benchmark_ready.py`

Review mode: local-only fallback. A secondary Codex review was attempted, but the first request used an unavailable model and the second request timed out before returning findings.

## Blocking

None after fixes.

Issues found during sanity/smoke and fixed before full run:

- Claims with no shared discovery/replication feature columns now skip cleanly instead of surfacing as run errors.
- `underpowered_asd_site_abide2` and `underpowered_adhd_site_adhd200` now route through disease-label logic and balanced downsampling.
- Injected site-confound traps now include a confound-validity check that abstains when the group label is nested in a declared confound.
- CSV ground-truth label now uses `known_null` instead of `null`, because default pandas CSV parsing treats `null` as missing.

## Non-Blocking

- Brain-wide multiverse remains a primary-spec-equivalent placeholder, matching the existing v2 TODO.
- The fMRI descriptor benchmark currently runs with `harmonize=none` by default. A ComBat sensitivity pass should be run as a separate ablation before paper claims.
- Three claim-inventory entries are marked ready but do not share feature columns across the requested cohorts:
  - `age_region_ukb_hcpa`
  - `age_dyno_ukb_hcpa`
  - `asd_region_abide2_abcd`
- Some fragile brain-behavior claims still confirm, which is scientifically useful but means the gate policy is not yet tuned to eliminate every fragile false confirmation.

## Checks

- Compile check passed with bytecode cache redirected to `/private/tmp`.
- Sanity known-positive run passed: `age_fc_ukb_hcpaging` confirmed.
- Sanity injected-null run passed: `injected_null_random_hcp` abstained as fragile.
- Full fMRI benchmark-ready run completed: 20 runnable claims, 3 skipped, 0 errors.
- Test suite passed: 10/10.
