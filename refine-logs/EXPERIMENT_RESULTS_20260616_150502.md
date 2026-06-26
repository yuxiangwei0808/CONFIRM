# Initial Experiment Results — Benchmark-Ready fMRI Wave

Date: 2026-06-16

Plan: `refine-logs/EXPERIMENT_PLAN.md`

Runner: `python -m bench.run_benchmark_ready --out-dir review-stage/benchmark-ready-full`

## Status

Full fMRI benchmark-ready wave completed locally on CPU.

- Runnable claims: 20
- Skipped claims: 3
- Execution errors: 0
- Known-positive claims: 4
- Known-null / fragile claims: 16
- Tests after implementation: 10/10 passed

Skipped because discovery/replication cohorts did not share the requested feature family:

- `age_region_ukb_hcpa`
- `age_dyno_ukb_hcpa`
- `asd_region_abide2_abcd`

## Gate-Ladder Result

| Rung | TPR | FCR |
|---|---:|---:|
| `exec_only` | 1.0000 | 0.5625 |
| `+confound` | 1.0000 | 0.4375 |
| `+power` | 1.0000 | 0.4375 |
| `+multiverse` | 1.0000 | 0.4375 |
| `+replication` | 1.0000 | 0.1875 |

Interpretation: the full CONFIRM gate chain preserved all known-positive fMRI stability claims in this prepared wave while reducing false confirmations from 56.25% to 18.75%. This is enough to proceed to full-scale iteration, but not enough to claim the benchmark is final: several fragile brain-behavior/disease claims still confirm and should drive the next ablation/policy-tuning pass.

## Outputs

- JSON: `review-stage/benchmark-ready-full/benchmark_ready_results.json`
- CSV: `review-stage/benchmark-ready-full/benchmark_ready_claims.csv`
- Timestamped JSON: `review-stage/benchmark-ready-full/benchmark_ready_results_20260616_150502.json`
- Timestamped CSV: `review-stage/benchmark-ready-full/benchmark_ready_claims_20260616_150502.csv`

## Readiness Verdict

Ready to run larger benchmark iterations over the prepared derivative layer.

Not yet ready to freeze paper-level claims without:

- a ComBat/harmonization sensitivity run;
- a review of the three skipped inventory entries;
- a policy pass on fragile claims that still confirm;
- optional psychosis adapter work only after imaging features and clinical labels are aligned.
