# Claims from Results - CONFIRM

Generated after data cleanup on 2026-06-17.

## Supported Narrow Claims

**C1 - The benchmark pipeline is label-aware and auditable.**  
CONFIRM separates paper-facing false-confirmation labels from small expected effects and candidate claims. Current buckets are positive/stability, negative/stress, small-positive, and candidate, with per-claim provenance in the audit tables.

Evidence:

- `src/bench/labels.py`
- `review-stage/combined-label-aware-combat/combined_benchmark_audit.csv`

**C2 - The full gate stack rejects staged false confirmations on the current cleaned benchmark.**  
On the 24-claim combined ComBat benchmark, execution-only confirms 4/7 negative/stress claims; the full gate stack confirms 0/7.

Evidence:

- `review-stage/combined-label-aware-combat/combined_benchmark_results.json`

Support caveat: promising but pilot-scale; `0/7` has a wide confidence interval.

**C3 - The active multimodal probe layer is NACC-only and non-diagnostic.**  
The current table-probe layer uses NACC MRI/CSF tables for aging and CSF/MRI association checks. It does not make NACC disease claims because the needed clinical diagnosis/demographic table is not local.

Evidence:

- `src/bench/run_multimodal_benchmark.py`
- `review-stage/multimodal-label-aware-combat/multimodal_benchmark_audit.csv`

**C4 - The immediate scalable backbone should be fMRI descriptors.**  
The complete near-term pipeline should use the local fMRI descriptor cohorts, then add ADNI/OASIS/OASIS3/UKB/ABCD/HCP-style claim contracts, artifact-injection tests, and replication gates.

## Current Quantitative Result

Combined ComBat benchmark:

| bucket | denominator | full-gate result |
|---|---:|---:|
| positive/stability TPR | 5 | 5/5 = 1.000 |
| paper-facing FCR | 7 | 0/7 = 0.000 |
| small-positive recovery | 6 | 2/6 = 0.333 |
| candidate confirmation | 6 | 0/6 = 0.000 |

Gate ladder:

| rung | TPR | FCR | small-positive recovery | candidate confirmation |
|---|---:|---:|---:|---:|
| exec_only | 5/5 | 4/7 | 4/6 | 3/6 |
| +confound | 5/5 | 2/7 | 4/6 | 3/6 |
| +power | 5/5 | 2/7 | 4/6 | 3/6 |
| +multiverse | 5/5 | 2/7 | 3/6 | 3/6 |
| +replication | 5/5 | 0/7 | 2/6 | 0/6 |

## Claims Not Yet Supported

- Do not claim broad neuroimaging false-discovery control.
- Do not claim multimodal disease validation.
- Do not claim NACC disease validation.
- Do not claim FCR is solved; it is stratified and currently based on 7 negative/stress claims.
- Do not claim the label ledger is authoritative; it is a first-pass literature taxonomy.

## Next Evidence Needed

1. Freeze a formal label table with DOI, cohort, modality, phenotype, direction/effect-size prior, and confidence.
2. Expand the fMRI descriptor benchmark to more adjudicated negative/stress claims.
3. Add ADNI/OASIS/OASIS3 disease and cognition claims from locally available tables.
4. Add naturally occurring tasks where confound, power, and multiverse gates independently change verdicts.
5. Add NACC disease claims only after the clinical diagnosis/demographic table is available.
