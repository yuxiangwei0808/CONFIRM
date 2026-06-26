**Score: 6/10**

**Verdict: almost**

Ready to proceed to larger prepared-derivative experiments. Not ready for paper-level claims yet.

I verified the latest JSON: 20 runnable claims, 4 positives, 16 null/fragile, 3 skipped, no errors, `harmonize: "none"` [benchmark_ready_results.json](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/review-stage/benchmark-ready-full/benchmark_ready_results.json:2). Tests pass: `10 passed`.

**Critical Weaknesses**

1. **Residual false confirmations are still material.** Final FCR is 3/16 = 18.75%, with exact 95% CI about 4.1-45.7%. The three false confirmations are `cognition_fc_ukb_abcd`, `cognition_dyno_ukb_abcd`, and `adhd_region_adhd200_abcd`, all labeled fragile but confirmed [benchmark_ready_results.json](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/review-stage/benchmark-ready-full/benchmark_ready_results.json:15366).  
Minimum fix: adjudicate these labels externally or reclassify them; report binomial CIs and do not call them “false confirmations” unless the label provenance is defensible.

2. **Ground-truth labels are too weak for paper claims.** The runner derives labels from inventory text such as `fragile_or_positive_candidate` and `fragile_or_small_effect` [run_benchmark_ready.py](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/src/bench/run_benchmark_ready.py:122). That is fine for stress testing, not enough for benchmark ground truth.  
Minimum fix: create a frozen claim manifest with label source, literature/provenance, seeds, cohorts, covariates, feature family, and generation code hash.

3. **Brain-wide multiverse is effectively a placeholder.** The prepared runner’s brain-wide multiverse is one spec, `brainwide_primary`, and passes whenever any region is significant [run_benchmark_ready.py](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/src/bench/run_benchmark_ready.py:417). This explains why `+power` and `+multiverse` metrics are identical.  
Minimum fix: implement a real brain-wide fork grid, or explicitly exclude brain-wide multiverse from governed claims.

4. **Power gate remains circular / winner’s-curse-prone.** `power_check` falls back to observed standardized effect when no `ref_effect` exists [power.py](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/src/confirm/power.py:15), and the benchmark applies it to the best discovered region [run_benchmark_ready.py](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/src/bench/run_benchmark_ready.py:508). The “underpowered” claims report power >0.90 and fail other gates instead [benchmark_ready_results.json](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/review-stage/benchmark-ready-full/benchmark_ready_results.json:63660).  
Minimum fix: use external/meta-analytic MDEs or rename this gate to winner’s-curse risk; add a task where power is the decisive failing gate.

5. **Replication/harmonization claims are ahead of the run.** Latest run used `harmonize: "none"` [benchmark_ready_results.json](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/review-stage/benchmark-ready-full/benchmark_ready_results.json:9), while the method narrative emphasizes ComBat. Replication thresholds are also permissive: pattern corr 0.25, region fraction 0.05, dice 0 [run_benchmark_ready.py](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/src/bench/run_benchmark_ready.py:380).  
Minimum fix: run ComBat sensitivity, threshold risk curves, and fail closed if harmonization fails.

6. **Skipped “ready” claims expose inventory mismatch.** Three benchmark-ready claims skipped for no shared features [benchmark_ready_results.json](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/review-stage/benchmark-ready-full/benchmark_ready_results.json:71343), despite being marked ready in the inventory [claim_inventory_ready.csv](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/data/prepared_data/benchmark_ready/claim_inventory_ready.csv:4).  
Minimum fix: either repair feature mappings or remove them from the frozen benchmark denominator.

7. **Baselines are still ablations, not real competitors.** The lite runner explicitly says real NeuroClaw/NIAgent baselines are unavailable [run_benchmark.py](/Users/yuxiang.wei/Desktop/Playground/NeuroScientistAgents/src/bench/run_benchmark.py:11).  
Minimum fix: frame claims as gate-ablation evidence, or add matched baseline runners plus blinded claim extraction.

**Resources**

No GPU is needed now. More CPU is useful for ComBat sweeps, bootstrap/permutation checks, and expanded 30-40 claim reruns. The bigger need is data/resource quality: true scanner/site/session metadata, label provenance, and more cross-disease prepared derivatives.

**Run Next**

1. Rerun the full benchmark with `--harmonize combat`, plus a no-harmonization comparison.
2. Implement true brain-wide multiverse and rerun the gate ladder, focusing on the 3 fragile confirmations.
3. Freeze a benchmark lockfile and add exact CIs to every TPR/FCR number.
4. Add an adversarial FDR task where uncorrected hits fail BH correction.
5. Add a principled power/MDE task where the power gate, not replication, causes abstention.
