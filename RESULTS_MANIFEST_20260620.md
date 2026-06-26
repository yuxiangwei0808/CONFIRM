# CONFIRM — Results Manifest (2026-06-20)

Single source of truth mapping every manuscript number to its artifact, SHA-256
(first 16 hex), evidence stratum, and the command that produced it. Supersedes
the numbers in `PAPER_PLAN.md` and `docs/STAGED_RESULTS_INDEX.md` (both stale).

**Provenance statement.** The external runs (NACC, ds000030) are
*frozen-configuration* evaluations: the claim table, gate configuration, and
runner code were hashed (below) and the run executed once with no post-hoc
threshold tuning. This workspace is **not** git-backed, so we cite SHA-256
hashes, **not** a git tag, and we do **not** claim formal "preregistration."

**Power gate (2026-06-20 fix).** `src/confirm/power.py` now judges power against
a pre-declared minimal effect of interest (reference value if supplied, else
$d{=}0.3$), never the observed effect. All benchmarks below that depend on the
power gate were re-run after this change; `src/confirm/power.py`
sha256=`ca52d0f91f1dea35`. 36/36 unit tests pass.

## Evidence strata
- **INT-ADJ** internal adjudicated benchmark (literature/externally-anchored labels)
- **SYN** synthetic gate-targeted stress negatives (development cohorts)
- **EXT** external unseen-cohort evaluation
- **AGENT** runnable-agent comparison / retrofit

| Manuscript number | Value | Stratum | Artifact (sha256-16) |
|---|---|---|---|
| MAIN known-positive recall (TPR) | 10/10 | INT-ADJ | round5-combat/combined_benchmark_results.json `25e52a10de149745` |
| MAIN full-gate FCR | 0/27 [0,0.128] | INT-ADJ | same |
| FULL benchmark TPR / FCR | 11/11 / 0/29 [0,0.119] | INT-ADJ | same |
| Stress-suite FCR (full gate) | **0/177** [0,0.021] | SYN | negatives_expansion_results_20260620_102803.json `dedcc71ac3bf78d7` |
| Stress per-family FCR | 0/42,0/42,0/42,0/21,0/3 | SYN | same |
| NeuroClaw TPR / FCR | 9/10 / 5/15 | AGENT | round5-neuroclaw/neuroclaw_comparison.json `dfbf94184aa0ec82` |
| CONFIRM (shared set) TPR / FCR | 10/10 / 0/15 | AGENT | same |
| CONFIRM-layer FCR | 0.33 -> 0.0 (TPR 9/10) | AGENT | confirm-layer/confirm_layer_result.json `354a3f5fb530c9c1` |
| Multi-LLM cross-model agreement | 7/9 (six models) | INT | agentic_multillm_summary_full_sweep_v2.json `9189dae87df9a04e` |
| Numeric-guard catches | 40 | INT | same |
| NACC external TPR / FCR | 9/9 / 0/28 [0,0.123] | EXT | external-nacc/nacc_external_results.json `67a2b8488e21fbe9` |
| NACC significance-only baseline FCR | 2/28 | EXT | same |
| ds000030 external TPR / FCR | 0/14 / 0/16 | EXT | external-cnp/CNP_external_results.json `f617efb155083069` |
| Combined external null FCR | 0/44 [0,0.080] | EXT | NACC + ds000030 random-label controls |

## Frozen-configuration inputs (external)
- NACC cohort `data/prepared_data/external/NACC.parquet` (built by `src/bench/prepare_nacc_external.py`); diagnosis join `data/external_benchmark/nacc_dx.csv`.
- ds000030 cohort `data/prepared_data/external/ds000030.parquet` sha256=`4a8c57eb5880e8e4` (built by `src/bench/prepare_ds000030_external.py` from legacy-S3 FreeSurfer aseg).
- ds000030 claims `data/external_benchmark/ds000030_claims.csv` sha256=`46407654fde45132`.
- Each external `*_results.json` embeds its own lockfile (claims/cohort/runner sha256, seed, gate config).

## Commands (re-run)
```
PYTHONPATH=src python -m bench.run_negatives_expansion
PYTHONPATH=src python -m bench.run_nacc_external
PYTHONPATH=src python -m bench.run_external_generic --cohort data/prepared_data/external/ds000030.parquet \
    --claims data/external_benchmark/ds000030_claims.csv --control-dx CONTROL --cohort-name CNP \
    --out-dir review-stage/external-cnp
```

## Notes on honest scope
- The 0/177 bound is a **stress-test** characteristic on synthetic, gate-targeted nulls over development cohorts — NOT a real-world false-confirmation rate.
- External null controls are **random-label / procedural** (NACC, ds000030) plus ENIGMA literature-null claims in the ds000030 set; they are not a broad literature-null suite. NACC alone (0/28) has upper bound 12.3%, above a 10% target; only the combined 0/44 reaches 8.0%.
- External positive recovery is **AD/MCI-heavy** (NACC); ds000030 psychiatric positives are correctly abstained as under-powered, not recovered.
