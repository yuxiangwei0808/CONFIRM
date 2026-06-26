# CONFIRM — Auto-Review Loop

Started: 2026-06-17
Backend: codex (gpt-5.5, xhigh) · Difficulty: medium · Max rounds: 4
Stop condition: score ≥ 6/10 AND verdict ∈ {ready, almost}
Task context: expand the benchmark — freeze an authoritative label table + add the ready phenotype×ground-truth domains (aging/sex positives, autism/ADHD fragile), building on the just-completed AD phase 1.

---

## Round 1 (2026-06-17)

### Assessment (Summary)
- Score: 5.5/10 (NeurIPS/ICML); ~4.5/10 (Nature Methods)
- Verdict: not ready
- Key criticisms: (1) labels still proxy/circular — freeze an external label table with provenance + independent adjudication; (2) FCR denominator too small (0/7, CI upper ~0.41) — expand to ~25–50 negatives/fragiles, exact binomial CIs; (3) make baseline hierarchy explicit (exec_only = ablation, not competitor); (4) frame as claim-state governance, not a better neuroimaging agent; (5) add replication heterogeneity audit; (6) separate power vs replication gates with predeclared thresholds; (7) center metrics on fixed-recall risk-coverage so "abstain-all" can't game FCR.
- Expansion guidance: richer label-table fields (label_class, label_basis, adjudication_status, forbidden_evidence, etc.); define sex/aging estimands precisely (eTIV-adjusted); define "fragile" defensibly (literature-weak AND fails predeclared checks while positive controls pass same pipeline; never label "false"); pull BWAS earlier; don't delay NeuroClaw baseline.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (verbatim)</summary>

See `.aris/traces/auto-review-loop/2026-06-17_run01/001-round-1-review.response.md` for the complete verbatim response. Summary inlined above; full critical-weakness text and expansion critique preserved in the trace file.

</details>

### Actions Taken (Phase C — in progress)
- Launched Codex build: authoritative label table (rich schema) + expanded positives (AD/aging/sex) and fragiles (autism/ADHD) + metrics suite (exact binomial CIs, risk-coverage curve, coverage/abstention/recall/fragile-abstain/candidate-yield) + replication heterogeneity audit.
- Deferred to Round 2 Phase C: cognition/BWAS N-dependent task; NeuroClaw real-baseline run.

### Results (Phase C complete)
- Label table: 57 claims, 100% scored-claim coverage; 26 real / 31 provisional provenance. Classes: 5 known_positive, 15 known_null, 11 fragile, 3 underpowered_small_positive, 23 candidate.
- Metrics: exact Clopper–Pearson CIs + risk-coverage sweep wired in. Heterogeneity audit (DerSimonian–Laird I²) present.
- Expanded scored set: n=41, negative/fragile denominator 23. Full-gate: TPR 5/5 [0.48,1.0], FCR 2/23=0.087 [0.011,0.280], coverage 13/41. Tests 19→ (later) green.
- Honest surprises: 2 known_null fishing controls confirmed at full gate (the 2/23 FCR); AD positive replicated_but_heterogeneous I²≈95%; aging/sex sMRI positives skipped (data gap).

### Status
- Continuing to Round 2. Difficulty: medium.

---

## Round 2 (2026-06-17)

### Assessment (Summary)
- Score: 5.8/10 (was 5.5). Verdict: not ready ("No, but closer to Almost").
- Two hard blockers: (1) fishing-null false-confirmations; (2) label authority (31/57 provisional).
- Principled fixes: search-provenance/claim-lineage gate (not alpha-tuning) for the fishing nulls; main benchmark on externally-adjudicated labels only; AD claim-scoping (direction/magnitude/transportable) + ADNI/OASIS3 harmonization for the I²≈95%; recover real site IDs or use cohort-level replication; get sMRI for more cohorts or drop sMRI aging/sex coverage.

### Reviewer Raw Response
See `.aris/traces/auto-review-loop/2026-06-17_run01/002-round-2-review.response.md` (verbatim).

### Executor Diagnosis (fishing nulls)
Root cause: nulls are discovery-selected then tested in a same-cohort replication split, with multiplicity `family_size=1` — the gate accounts only for the final claim, not the searched family (garden of forking paths). Fix = search-provenance/claim-lineage gate (family-wise correction or `unverifiable_search_provenance` abstention) + discovery/replication separation in the null builder. Converts the failure into a headline mechanism.

### Status
- Round 3 plan: search-provenance gate + rebuild fishing nulls + label-authority split + AD claim-scoping. Data gap: user granted `ssh arcdev` access to /data/qneuromark/Data and /data/neuromark2/Data; recovery agent launched.

---

## Round 3 (2026-06-17)

### Assessment (Summary)
- Score: 6.4/10 (was 5.8). Verdict: almost. **STOP CONDITION MET** (score ≥6 AND verdict ∈ {almost}). Progression 5.5 → 5.8 → 6.4.
- Both round-2 hard blockers cleared: B1 (fishing nulls) via the search-provenance gate; B2 (label authority) via the MAIN adjudicated subset.

### Actions Taken (Phase C)
1. Search-provenance / claim-lineage gate: contract field `search_provenance{declared,family_size,selection}`; multiplicity corrects over effective family = max(declared, searched); undeclared/unknown/full_data → abstain `unverifiable_search_provenance`. Rebuilt fishing-null builders (discovery-only selection, untouched replication, family=84). Fixed a real row-level split leakage bug (UKB subject overlap) → subject-level, overlap 0.
2. Label-authority split: metrics on FULL vs MAIN (adjudicated-only) subsets, exact Clopper–Pearson CIs.
3. AD claim-scoping: direction/magnitude/transportable sub-types from the heterogeneity audit; OASIS3 case definition harmonization CDR>0 vs CDR≥1.

### Results
- FULL +replication: TPR 5/5, FCR 0/23 [0,0.148], coverage 11/41, abstention 30/41. Fishing nulls all abstain.
- MAIN +replication: TPR 4/4 [0.40,1.0], FCR 0/21 [0,0.161], coverage 6/28.
- AD: CDR>0 → direction_confirmed, I²=95.0; CDR≥1 → magnitude_confirmed, I²=64.6.
- 27 tests pass.

### Reviewer Raw Response
See `.aris/traces/auto-review-loop/2026-06-17_run01/003-round-3-review.response.md` (verbatim).

### Round 4 bar for "ready" (7.0–7.4, per reviewer)
≥10–15 externally-labeled positives across >1 domain; real NeuroClaw baseline; psychosis cross-cohort replication (or honest abstention); coverage-vs-FCR curve; calibrated governance framing.

### Status
- Stop condition met at Round 3. Cluster-recovery agent running (sMRI/site/psychosis extractables). Paused for user decision: conclude the loop vs push Round 4. → User chose to push Round 4.

---

## Round 4 (2026-06-17) — FINAL

### Assessment
- Score 6.8/10 (was 6.4). Verdict: almost (borderline weak-accept). Positive stop (score≥6 AND almost) + MAX_ROUNDS reached. Progression 5.5 → 5.8 → 6.4 → 6.8.

### Actions + Results
- Recovered cluster cohorts via `ssh arcdev` (GSU VPN): COBRE/FBIRN/ABIDE1/GSP → `data/prepared_data/cluster_recovered/` (validated). ADHD200 honestly skipped (no RC_ROI on cluster).
- **Schizophrenia** within-network FC, COBRE→FBIRN: **confirmed, transportable_confirmed, I²=0** (COBRE β=−0.074 p=.001; FBIRN β=−0.083 p=1.7e-7) — new disease domain.
- **Real NeuroClaw baseline** (MIT, gpt-5-mini, ran on our derivative tables CPU-only, ~$0.90). Shared 15-claim set: NeuroClaw TPR 5/5, **FCR 3/10** (false-confirmed `injected_null_random_abcd` p=3.5e-4, `injected_null_site_abcd`, `injected_null_site_ukb`); CONFIRM TPR 5/5, **FCR 0/10** on the same set (0/22 on full MAIN). Artifacts: `review-stage/round4-neuroclaw/`.
- Autism (ABIDE1, real 19-site split) + sex sMRI: correctly abstain. Fishing/search-provenance nulls still 0 confirmed. Coverage-vs-FCR figure built. 28 tests pass.

### Reviewer (R4) — verbatim in trace `004-round-4-review.response.md`
Mostly cleared the R3 bar (NeuroClaw baseline, fishing nulls 0, psychosis replication, coverage-vs-FCR, label authority). NOT met: ≥10–15 externally-labeled positives (only 5 MAIN / 6 FULL — main remaining weakness, data-availability-bound). #1 lever: add one externally-adjudicated positive block (5–10 claims from an independent cohort) → raise MAIN positives 5→~12 while keeping 0 false-confirms. Framing: "no observed false confirmations" / "lower observed FCR"; avoid "validated broadly" / "establishes lower FCR".

## Final Summary
Over 4 review rounds CONFIRM went from a self-graded pilot (5.5, "not ready") to a borderline-competitive, externally-grounded contribution (6.8, "almost"). Both hard blockers were resolved with principled mechanisms: a **search-provenance / claim-lineage gate** (multiplicity corrected over the declared search family; undeclared provenance → abstain `unverifiable_search_provenance`) for search-induced false confirmations, and a **MAIN externally-adjudicated label subset** for label authority. Headline evidence: at matched known-positive recall (5/5), CONFIRM shows **0 observed false confirmations** where the real NeuroClaw baseline confirms a random-label null and two site-confounded nulls. Validated positives span **AD** (hippocampal atrophy; magnitude_confirmed after CDR≥1 harmonization), **schizophrenia** (within-network dysconnectivity, transportable COBRE↔FBIRN), and **aging**. Remaining weakness is empirical maturity (5–6 positives; FCR CI upper ~0.15), largely data-availability-bound; #1 next step = more externally-adjudicated positives.

## Method Description
CONFIRM is a CPU-only **claim-governance layer** over precomputed neuroimaging derivative tables. An LLM drafts a frozen `ClaimContract` (estimand, covariates, discovery/replication cohorts, gate settings, declared search provenance) from a natural-language question; it never emits numbers. Executed code computes all statistics and applies admissibility gates in order: **multiplicity** (corrected over the declared search family; undeclared/full-data provenance → abstain `unverifiable_search_provenance`), **confound** (required covariates present), **power**, **multiverse** stability, and **cross-cohort replication** (same-sign + independent significance after ComBat, with a DerSimonian–Laird heterogeneity audit yielding direction/magnitude/transportable confirmation sub-types). A finding is `confirmed` only if all gates pass, else it abstains (`non_replicated` / `under_powered` / `fragile`). The same mechanism runs as a native in-loop agent and as a modular layer wrapping an existing agent's claims. Evaluation = a cross-cohort benchmark with an externally-adjudicated label table, scored on a gate ladder (exec_only → +confound → +power → +multiverse → +replication) by known-positive recall vs false-confirmed rate, against a real NeuroClaw baseline.

### Next steps (post-loop)
1. (Highest leverage) Add one externally-adjudicated positive block (5–10 claims, independent cohort) → MAIN positives 5→~12, keep 0 false-confirms.
2. Paper-writing with the calibrated framing above.
3. Optional: `/render-html review-stage/AUTO_REVIEW.md`.

---

## Loop 2 / Round 1 (2026-06-18) — close gaps (a) agentic-at-scale + (b) modular layer

### Assessment
- Score **7.3/10**, verdict **Almost** (up from 6.8). Path to 7.5–7.8 with fixes. Reviewer via direct OpenAI API (Codex MCP kept freezing; bypassed). Full text: review-stage/loop2-round1-review.md.

### Actions (Phase C done)
- (a) Multi-LLM agentic loop: 6 LLMs (gpt-5-mini, gpt-4o, claude-sonnet-4-6, claude-haiku-4-5, deepseek-v3, qwen-2.5-72b) drive draft→gate→interpret on 9 curated claims. All draft (8–9/9); estimand-match 0.75–0.89; cross-model gate-verdict agreement 7/8; anti-hallucination guard fired 42×. Wrinkles: claude-haiku omitted a site confound → false-confirmed 1 null (the 1/8 disagreement); sex claim execution-errored across all models (predictor/covariate collision); aging errored 3/6.
- (b) CONFIRM-as-layer over NeuroClaw: FCR 5/15 (0.33) → 0/15 (0.0), TPR preserved 9/10. src/bench/run_confirm_layer.py.
- Built multi-provider LLM client (make_llm: openai/anthropic/openrouter).

### Reviewer's minimum fixes (→ ready)
1. Static contract validation (predictor∉covariates). 2. Fix predictor/covariate collision. 3. Confound-completeness audit (would catch haiku's confound omission). 4. Execution failures as governance outcomes, not crashes. 5. Rerun the 6-model sweep. 6. Framing: separate claim tiers; "eliminated observed false confirmations" not "guarantees".

### Status
- Stop condition met (7.3 / almost). Final fix-round (contract-validation layer) recommended to reach ~7.5–7.8 "ready", then writeup.

---

## Loop 2 / Round 2 (2026-06-18) — contract-validation fix-round → READY

### Assessment
- Score **7.5/10**, verdict **READY** (up from 7.3). "Ready for submission as a ~7.5 paper; 8+ needs more negatives." Review: review-stage/loop2-round2-final-review.md. Score arc 5.5→5.8→6.4→6.8→7.3→7.5.

### Actions (Phase C)
- Contract-validation layer: static validation (predictor/group-var ∉ covariates); predictor/covariate dedup (fixes sex crash); confound-completeness audit (data-aware → `confound_incomplete` abstain); execution failures → governance abstentions. 35 tests pass.

### Post-fix 6-LLM rerun
- All 6 models execute cleanly (sex no longer crashes). The site-confound NULL is abstained by ALL 6 (confound_incomplete) — the prior claude-haiku false-confirm ELIMINATED → governance LLM-robust on the false-confirm axis. Cross-model agreement 7/9; the 2 residual disagreements are on a borderline POSITIVE (aging recall), not false-confirms. Anti-hallucination guard 40 catches.

### 7.5 → 8+ path (reviewer; NOT submission blockers)
1. (#1) Expand the null/confounded benchmark to tighten the FCR bound. 2. +1–2 datasets/domains (UKB/ABCD/HCP/PD/MS). 3. More gate ablations (CONFIRM minus each gate). 4. Dataset-level required-confound schema (fail-closed `metadata_incomplete`). 5. Report stripped-hallucination examples + framing discipline.

### Status
- **LOOP COMPLETE — reached READY (7.5).** Submittable now; clear path to 8+.

---

## Loop 2 / Round 3 (2026-06-18) — negatives push → READY 8.1

### Assessment
- Score **8.1/10**, verdict **READY** (up from 7.5). Crossed 8.0 via the reviewer's #1 item. Review: review-stage/loop2-round3-negatives-review.md. Arc 5.5→5.8→6.4→6.8→7.3→7.5→8.1.

### Action
- Negatives push: 150 synthetic nulls/fragiles (42 random_label, 42 site_confound, 42 p_fishing, 21 underpowered, 3 cross_cohort_nonreplication) scored through the gate ladder. Full-gate FCR 1/150; combined w/ adjudicated negatives 1/177 = 0.6% [0.00014, 0.031] → upper bound 12.8% → **3.1%**. One honest false-confirm (`neg_underpowered_hcp_s3` — mis-constructed underpowered negative that carried a large effect; power gate correctly passed it; counted anyway). 36 tests.

### Honest framing (reviewer-mandated)
- The ~3.1% bound is CONDITIONAL on the synthetic gate-targeted generator on local cohorts — NOT a real-world FCR guarantee. Say "strong internal stress-test validation," not "≤3.1% in the wild." cross_cohort_nonreplication family tiny (n=3); 3 positive domains.

### Remaining item (future work — NOT a submission blocker)
- Independent, locked EXTERNAL validation: a preregistered third-party adversarial benchmark of ~100–200 REAL claims across UNSEEN cohorts, frozen gates, run once → moves "strong internal validation" to "field-level evidence."

### Status
- **LOOP COMPLETE — READY (8.1).** Submission-ready methods contribution. Next = paper writeup (calibrated framing); external real-claim benchmark = future work.
