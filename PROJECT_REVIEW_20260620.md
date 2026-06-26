# CONFIRM Project Review

Date: 2026-06-20  
Scope: method, benchmark/baselines, novelty, result provenance, and current WACV-style manuscript readiness.  
External reviewer trace: `.aris/traces/research-review/2026-06-20_run01/`

## Executive Verdict

CONFIRM has a real paper idea: a post-execution statistical claim-governance layer for agentic neuroimaging. The strongest contribution is not a new statistical test; it is the enforcement architecture: frozen claim contracts, code-only numeric authority, gate-based abstention, and retrofit governance over an existing agent.

The project is not yet submission-ready as written. The blocking issue is provenance: the manuscript and external-results note call NACC/CNP "preregistered", "frozen", and "run once", while the preregistration file says it is a draft and not frozen. For a paper whose thesis is governance and auditability, that contradiction is central, not cosmetic.

Approximate readiness:

- As written: 5/10 for WACV applications, likely reject on provenance/evidence mismatch.
- After fixing provenance/framing and result manifest: 6.5-7/10, credible weak-to-borderline main-conference submission.
- To reach strong accept territory: add a genuinely frozen external benchmark with real/literature nulls and stronger simple-governance baselines.

## Severity-Ranked Findings

### Fatal: External Preregistration Claim Contradicts The Repo

The manuscript says the NACC run used gates "frozen in advance" and is "the first external, frozen-gate, run-once evaluation" (`paper/sec/05_experiments.tex:172-185`). The external-results note also says "Preregistered design" and "Frozen gates, run once" (`EXTERNAL_BENCHMARK_RESULTS.md:3-6`). But the preregistration file says "DRAFT preregistration (not yet frozen)" and requires a git tag before any run (`EXTERNAL_BENCHMARK_PREREG.md:3`), while `docs/STAGED_RESULTS_INDEX.md:5` says the workspace is not a git repository.

Fix before submission:

- If the run was not actually preregistered, remove "preregistered" and "run once" language. Call it an "external frozen-configuration evaluation" only if you can prove the configuration was frozen before execution.
- If you want to preserve the stronger claim, create an immutable provenance package: claim table hash, code hash, config hash, exact command, timestamped manifest, result hash, and reviewer-readable "no rerun/no tuning" statement. Because the repo is not git-backed, do not cite a nonexistent git tag.

### Major: External Validation Is Promising But Does Not Yet Carry The Safety Claim

NACC is useful: 9/9 AD/MCI positives, 0/28 random-label controls, significance-only baseline 2/28 (`EXTERNAL_BENCHMARK_RESULTS.md:15-19`). But its FCR upper CI is 12.3%, which misses the preregistered target of upper CI <10% (`EXTERNAL_BENCHMARK_PREREG.md:33-36`), and the note itself admits the random-label controls share subjects and the CI is approximate (`EXTERNAL_BENCHMARK_RESULTS.md:32-36`). The "AD-spared" regions are not clean nulls (`EXTERNAL_BENCHMARK_RESULTS.md:21-26`).

CNP/ds000030 is even weaker as positive evidence: CONFIRM confirms 0/14 psychiatric positives and 0/16 random nulls, while significance-only also has 0/16 FCR. That is defensible as an abstention story, but it does not demonstrate broad external positive recall.

Fix before submission:

- Present NACC as "initial external positive-recall evidence plus random-label null controls", not as a passed preregistered external safety benchmark.
- Present CNP as an underpowered-abstention case study, not external validation of performance.
- Do not combine NACC+CNP into "0/44 external false confirmations" without emphasizing that the nulls are random controls and the positive result is effectively AD-only.

### Major: The Power Gate Is Statistically Vulnerable

The method says the power gate checks whether the design could detect the contracted effect (`paper/sec/03_method.tex:122-127`). In implementation, `power_check` uses an external `ref_effect` if available, but otherwise falls back to the observed standardized effect (`src/confirm/power.py:15-21`, `src/confirm/power.py:49-60`). That can convert lucky inflated estimates into "powered" claims.

The E5 false confirmation illustrates exactly this risk: an intended underpowered negative carried a large effect and achieved power around 0.99, so the gate passed (`paper/sec/05_experiments.tex:141-147`).

Fix before submission:

- Prefer predeclared minimal detectable effects or literature effect sizes for every paper-facing claim.
- If no external/reference effect exists, fail closed as `power_reference_missing` or report a sensitivity analysis instead of allowing observed-effect power to pass a claim.
- Add a table listing each paper-facing claim's power reference source.

### Major: Benchmark Labels Mix Too Many Evidence Types

The manuscript says the benchmark has "externally anchored known-positive, known-null, and fragile labels" (`paper/sec/01_intro.tex:60-65`). In reality, the benchmark combines literature positives, synthetic known nulls, fragile/stress claims, provisional/candidate labels, random relabelings, and local synthetic generators. The limitations section correctly says the 1/177 bound is synthetic and gate-targeted (`paper/sec/06_limitations.tex:4-8`), but the main contribution language still sounds too much like ground truth.

Fix before submission:

- Replace "truth status is known" language with "adjudicated benchmark label" or "paper-facing label".
- Stratify every headline result into: adjudicated internal, synthetic gate-targeted, external random-label, external literature-positive, external real/literature-null.
- Make the 1/177 number a stress-test result, not the top-level false-confirmation-rate estimate.

### Major: Baselines Are Suggestive But Thin

The NeuroClaw result is the best empirical evidence: NeuroClaw TPR 9/10 and FCR 5/15; CONFIRM TPR 10/10 and FCR 0/15; the post-hoc CONFIRM layer preserves NeuroClaw's 9/10 positive recall while converting false confirmations to abstentions (`paper/sec/05_experiments.tex:73-85`).

The weakness is that the shared set is small and many failures are exactly the synthetic/site-confound cases the gates target. Reviewers will ask whether simpler governance baselines would do the same.

Add these baselines, in priority order:

1. Significance-only, already partly present.
2. BH + required covariates, no replication.
3. Replication-only.
4. Confound-completeness-only.
5. Full CONFIRM minus each gate.
6. NeuroClaw + simple post-hoc replication requirement.
7. NeuroClaw + BH/covariate checklist.

### Major: Multi-LLM Robustness Is Overstated

The paper reports 7/9 cross-model verdict agreement across six LLMs (`paper/sec/05_experiments.tex:99-111`). That is a useful smoke test. It is not enough to claim broad LLM robustness, especially because two of nine shared claims disagree.

Fix before submission:

- Reframe as "no safety-critical false confirmation observed in a six-model smoke test."
- Avoid "largely invariant" unless the denominator grows.
- Add contract-equivalence diagnostics: do models draft the same estimand, covariates, cohorts, and search family?

### Major: Result Lineage Is Stale And Inconsistent

The current staged index still describes a 24-claim pilot and `combined-label-aware-combat` as canonical (`docs/STAGED_RESULTS_INDEX.md:7-47`), while the paper uses round5, negatives-expansion, NACC, CNP, NeuroClaw, confirm-layer, and agentic-multillm artifacts. `PAPER_PLAN.md` still says "0 observed false confirmations over 177" (`PAPER_PLAN.md:10-17`, `PAPER_PLAN.md:36-39`) and also says external preregistered real-claim benchmark is future work (`PAPER_PLAN.md:30`), while the paper now includes external sections.

Fix before paper writing:

- Create a single `RESULTS_MANIFEST_20260620.md` or JSON manifest.
- For every manuscript number, list artifact path, hash, timestamp, command, claim count, and whether it is internal/synthetic/external.
- Mark stale docs as superseded or update them.

### Major: Related Work And Citation Hygiene Need Audit

NeuroClaw, NeuroAgent, and NIAgent/Towards a Virtual Neuroscientist are real and relevant. The local bibliography names `nexus2026` but the arXiv title appears to be NIAgent/Towards a Virtual Neuroscientist (`paper/references.bib:21-28`). The `NEURA` citation could not be verified from web search and only appears as a bioRxiv DOI-like string in the bib (`paper/references.bib:30-35`).

Fix before submission:

- Verify every 2026 agentic-neuroimaging citation manually.
- Rename `nexus2026` to the actual system name used by the paper if needed.
- Remove or qualify `NEURA` unless the DOI/preprint is independently verifiable.
- In related work, explicitly state: NeuroClaw emphasizes executable/reproducible workflows and NeuroBench executability/artifact validity; NeuroAgent emphasizes multimodal preprocessing/analysis and AD classification; NIAgent emphasizes autonomous workflow/QC/predictive performance. CONFIRM is different because it governs post-execution claim admissibility.

## Method Assessment

Strengths:

- The core architecture is clean and reviewable: contract drafting, code-only numbers, and gate-based abstention (`paper/sec/03_method.tex:4-8`, `paper/sec/03_method.tex:40-54`).
- The code matches the manuscript at a high level: deterministic schema, multiplicity family size, confound audit, power, multiverse, replication, and numeric guard are present.
- The "retrofit layer" angle is strong because it makes the method useful even if other agent systems win on workflow automation.
- The abstention taxonomy is a good conceptual contribution for high-throughput agentic science.

Weaknesses:

- The gates are conventional. The novelty is systems integration/enforcement, not new statistical methodology.
- Confound-completeness is narrow: it only scans `site`, `scanner`, `field_strength`, and `fs_version` (`src/confirm/analysis.py:15-17`, `src/confirm/analysis.py:126-180`). That is fine if described as a structural-confound audit, but not enough for broad "statistical admissibility".
- The confound screen uses an uncorrected p<0.05 association test (`paper/sec/03_method.tex:111-120`). In small samples it can miss important imbalances; in large samples it can flag trivial imbalance.
- ComBat and multiverse details are underspecified for reproducibility (`paper/sec/03_method.tex:129-144`).

## Baseline Assessment

Best current baseline evidence:

- NeuroClaw head-to-head and post-hoc layer, because it addresses the core claim that execution-integrity agents can still false-confirm (`paper/sec/05_experiments.tex:73-85`).
- Gate ladder ablation, because it shows where FCR drops (`paper/sec/05_experiments.tex:20-28`, `paper/sec/05_experiments.tex:52-58`).

Insufficient current baseline evidence:

- Significance-only is used externally but should be included consistently across all internal and external sets.
- No simple rule-based governance baselines exist yet.
- No "always require replication" or "BH + covariates" baseline exists, which reviewers will see as the natural alternative.

## Novelty Assessment

The novelty is credible if framed narrowly:

> CONFIRM is a claim-admissibility layer for agentic neuroimaging that turns statistical safeguards into enforceable, auditable gates over LLM-generated claims.

Do not frame it as:

- A new statistical method.
- A solution to neuroimaging false discovery.
- A general LLM scientific truth guarantee.
- Broad LLM robustness.

The strongest novelty language is "in this setting": search-provenance and confound-completeness gates are not new statistical ideas, but enforcing them inside an LLM-agent claim lifecycle appears novel relative to NeuroClaw/NeuroAgent/NIAgent-style workflow agents.

## Paper Assessment

The abstract is directionally good and now correctly says 1/177 rather than 0/177 (`paper/sec/00_abstract.tex:18-28`). The main risk is line 24's "gates frozen in advance" for NACC, which depends on resolving the preregistration contradiction.

The intro contribution list still overstates external/preregistered evidence and broad label authority (`paper/sec/01_intro.tex:60-75`). It should be rewritten after the provenance decision.

The experiment section is mostly readable, but E6 is too strong. "The strongest test of generalization" and "first external, frozen-gate, run-once evaluation" (`paper/sec/05_experiments.tex:172-185`) should be softened unless the prereg record is fixed.

The limitations section is unusually honest and should be preserved (`paper/sec/06_limitations.tex:4-27`). It needs one added limitation: the external preregistration/provenance status and the fact that NACC random-label controls are not literature-null claims.

## Safe Claims Now

These are supportable from current artifacts:

- CONFIRM can reduce false confirmations on internal/synthetic and gate-targeted benchmarks while preserving known-positive recall on the current main set.
- On the shared NeuroClaw set, CONFIRM reduces FCR from 5/15 to 0/15, and the post-hoc layer preserves NeuroClaw positive recall at 9/10.
- The 1/177 result is a strong internal stress-test result conditional on the synthetic generator and local cohorts.
- NACC provides initial external evidence that the method recovers well-powered AD/MCI atrophy positives and does not confirm random-label controls.
- CNP/ds000030 demonstrates conservative abstention on an underpowered psychiatric external split.

## Claims To Avoid Or Reword

- Avoid "preregistered external validation" unless provenance is fixed.
- Avoid "field-level false-confirmation rate" or any unconditional interpretation of the 3.1% bound.
- Avoid "known nulls" for random-label controls or "AD-spared" regions.
- Avoid "LLM-robust" without qualification.
- Avoid "the LLM never emits numbers"; say "the final system prevents unauthorized numeric statements from reaching the report."
- Avoid "statistical admissibility" as if the gate ladder is complete. Use "admissible under explicit CONFIRM gates."

## Prioritized Next Work

P0. Fix provenance and framing.

- Decide whether NACC/CNP are truly preregistered. If not, reword immediately.
- Build a results manifest mapping every manuscript number to an artifact path and hash.
- Update or mark stale `PAPER_PLAN.md` and `docs/STAGED_RESULTS_INDEX.md`.

P1. Add simple governance baselines.

- BH + covariates.
- Replication-only.
- Confound-completeness-only.
- Full CONFIRM minus each gate.
- NeuroClaw + simple post-hoc replication.

P2. Fix the power gate story.

- Require external effect/MDE references for paper-facing claims.
- Add sensitivity showing how many claims pass under conservative MDEs.
- Add `power_reference_missing` if no reference exists.

P3. Strengthen external nulls.

- Expand beyond NACC random labels.
- Add real/literature nulls from ENIGMA-like tables where effects are reported absent or negligible.
- At minimum, for a zero-failure external random-null suite, use enough independent controls to get the upper CI below 10%; better target 100+ real/semi-real external nulls.

P4. Citation and paper audit.

- Verify NeuroClaw, NeuroAgent, NIAgent/NEXUS, and NEURA entries.
- Add reproducible neuroimaging workflow baselines and preregistration/statistical analysis plan literature.
- Run a full citation audit before submission because the novelty framing depends on recent papers.

P5. Expand multi-LLM analysis only after P0-P3.

- Report contract equivalence, not just verdict agreement.
- Increase beyond 9 claims if claiming robustness.

## Acceptance-Lift Experiments

High lift / low-to-medium effort:

- Result manifest and provenance cleanup.
- Simple governance baselines.
- Power-gate MDE sensitivity.
- Stratified claims table: internal adjudicated, synthetic, external random, external real positive, external real null.

High lift / high effort:

- Frozen external benchmark with real positives and real/literature nulls across multiple disorders.
- Independent replay of NACC/CNP after a genuinely frozen manifest.

Medium lift:

- No-ComBat versus ComBat sensitivity.
- Numeric guard precision/recall audit with examples.
- Expanded multi-LLM contract-equivalence sweep.

## External Literature Notes

Primary sources checked:

- NeuroClaw technical report: https://arxiv.org/abs/2604.24696
- NeuroAgent: https://arxiv.org/abs/2605.06584
- Towards a Virtual Neuroscientist / NIAgent: https://arxiv.org/abs/2605.09366

Search did not verify the `NEURA` citation currently listed in `paper/references.bib`. Treat it as unverified until manually confirmed.

## Final Recommendation

Do not submit the current manuscript as-is. The core work is promising, and the NeuroClaw retrofit plus gate-ladder results are genuinely compelling. But the paper-writing phase should start with a provenance cleanup and a claim-framing pass, not with polishing prose.

The shortest path to a defensible submission is:

1. Fix or remove preregistered/frozen external claims.
2. Publish a result manifest.
3. Reframe external validation conservatively.
4. Add simple governance baselines.
5. Tighten the power gate.

After those fixes, the paper becomes a credible WACV applications submission centered on claim governance for agentic neuroimaging.
