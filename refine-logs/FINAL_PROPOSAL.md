# FINAL PROPOSAL — CONFIRM: Statistical Claim Governance for Agentic Neuroimaging Analysis

**Date:** 2026-06-01 · **Verdict:** READY (tightly scoped) · **Refined via** cross-model review (gpt-5.5, xhigh).

## TL;DR (plain language)
**Problem:** AI agents can now run brain-imaging analyses automatically — but they report whatever they
find, even flukes, because they only check *"did the code run"*, not *"is the conclusion trustworthy."*
Brain imaging is notorious for findings that don't replicate.
**Idea (CONFIRM):** an agent that won't label a result **"confirmed"** until it passes machine-checkable
trustworthiness gates — enough statistical power, multiple-comparison control, confounds handled,
survives reasonable analysis variations, and (the headline) **replicates in an independent cohort**.
Otherwise it **abstains** and says *non-replicated / under-powered / fragile*. Numbers come only from
executed code, never from the language model.
**Contributions:** (1) a machine-checkable **claim contract** (automatic pre-registration); (2)
**cross-cohort replication as a hard gate** — no prior agent does this; (3) honest **abstention** with four
labels; (4) re-runnable provenance **"receipts"**; (5) **NeuroDecide-Bench** — a *public-data* benchmark
with planted "trap" datasets where the correct answer is *don't report anything*.
**Why us:** it needs many cohorts (which you have, plus open ones) and runs on a laptop over precomputed tables.

## Problem Anchor (frozen — do not drift)
Agentic neuroimaging systems (NeuroAgent, NIAgent, NeuroClaw, NEURA) make analyses **executable** and
even validate *execution integrity* (artifacts exist, code ran). **Nothing stops the agent from
promoting a statistically *inadmissible* finding to a "confirmed" claim** — underpowered, unreplicated,
confounded, multiplicity-invalid, or fork-dependent. Given the documented non-replication of brain-wide
associations (Marek 2022: effects inflated >5×, need N≈thousands; NARPS: same data → divergent claims),
**autonomy amplifies false confirmations.** Execution-integrity gates (EviBound) do not catch this class.

## Method Thesis (one sentence)
**CONFIRM governs the *claim*, not just the run:** a finding earns the label **"confirmed" only if
executed results pass machine-checkable statistical-*admissibility* gates — multiplicity, confound,
power, cross-cohort replication, and multiverse stability — otherwise the agent abstains and labels it
*non-replicated / under-powered / fragile / cohort-specific*,** each with a re-executable provenance bundle.

## Dominant Contribution
**Statistical claim governance via *blocking* admissibility gates, headlined by the cross-cohort
replication gate**, shown to **reduce the false-confirmed rate while preserving recovery of established
effects** (i.e., it is not an abstain-everything machine). Numbers come only from executed,
schema-validated code; the LLM proposes the contract and interprets, but never emits numbers.

## Smallest Adequate Mechanism (CONFIRM-lite)
A CPU-only Python/CLI tool over **precomputed tabular derivatives**:
1. **Claim contract (YAML/JSON):** estimand, cohort & inclusion/exclusion, confounds, multiplicity rule,
   power rule (Marek-aware), replication rule (which 2nd cohort + harmonization), multiverse spec, and
   allowed reporting language. (LLM drafts it from the question; human can approve.)
2. **Primary analysis** on derivatives (Nilearn/statsmodels) → effect + CI.
3. **Bounded multiverse** over the contract's declared forks → stability.
4. **Power / winner's-curse** check (shrinkage) → admissibility.
5. **Independent-cohort replication** with **predeclared ComBat harmonization** → replication verdict.
6. **Verdict + provenance bundle:** one of {confirmed, non-replicated, under-powered, fragile} +
   code+seeds+env hash (DataLad/BIDS-style "finding receipt").

## Key Claims (what the paper proves)
- **C1 (anchor):** CONFIRM has a **lower false-confirmed rate** than an execution-valid runner
  (NeuroClaw/NIAgent-style) and a generic DS agent, **at fixed known-positive recall**.
- **C2 (novelty isolation):** the **replication gate is the dominant driver** of C1 (gate-ablation ladder).
- **C3 (no abstain-all):** CONFIRM **preserves known-positive effects** (AD atrophy, FDG, age/sex/ICV).
- **C4 (reproducibility):** emitted bundles **re-run deterministically** across machines/seeds.

## Must-Run Ablations
Gate-ablation ladder: execution-only → +confound → +power → +multiverse → +replication. Each rung's
effect on false-confirmed rate and known-positive recall is reported (risk-coverage curve).

## Complexity Intentionally Rejected (non-goals)
Raw DICOM→derivatives preprocessing (reuse fMRIPrep/MRIQC outputs) · cross-modal triangulation ·
NiMARE/Neurosynth interpretation · full "AI neuroscientist" framing · GPU/training · clinical/diagnostic use.

## Companion Asset
**NeuroDecide-Bench-lite** (~12–15 tasks) = the evaluation *and* a standalone community asset.
**Public-data-first** (built on open cohorts so any lab can reproduce it), structured as a **claim
library**: each entry = *(question + dataset/derivative + ground-truth status {positive / null / fragile}
+ source of that ground truth)*. Spans **multiple diseases** (AD, aging, autism, ADHD, schizophrenia,
Parkinson's) so the agent's generality is tested, not just one effect. Three task classes: adversarial
**injected-nulls** (motion leakage, site imbalance, collider bias, label leakage), **known positives**
(meta-analytic effects per disease), **real fragile** cases (NARPS / small-N brain-behavior). Led by the
*validity-under-adversarial-nulls + abstention* framing (delta vs BLADE). Curation, not new data collection.

## Risks & Preemptions (from review)
- **Gate arbitrariness** → selective-risk curves (false-confirmed vs known-positive recall), not fixed thresholds.
- **Benchmark construct validity** (toy traps) → mask trap structure in realistic derived variables, real
  cohort covariance, hide trap type from prompts, include a trivial "always adjust motion/site/age/sex" baseline.
- **Straw-man baselines** → fair matched baselines + a **blinded claim extractor** mapping all reports to
  {confirmed/qualified/abstained}.
- **Concurrent work** → verify NEURA full text; position explicitly vs EviBound (execution) & BLADE (diversity).

## Recent Literature (2023–2026): what exists, and the gap CONFIRM fills
*All arXiv items below were WebFetch-verified during novelty checking (2026-06-01), except NEURA (bioRxiv full text 403'd — abstract only).*

### A. Agentic neuroimaging analysis (closest competitors)
- **NeuroAgent** (arXiv:2605.06584, 2026) — *Did:* hierarchical multi-agent system over sMRI/fMRI/dMRI/PET; preprocessing + an AD-classification task + FDR-corrected cortical-thickness group tests on 1,470 ADNI subjects. *Did not:* cross-cohort replication, provenance/reproducibility, or any claim-admissibility gating — it runs one analysis and reports it.
- **NEXUS** (arXiv:2605.09366, 2026; "Towards a Virtual Neuroscientist" — *not* "NIAgent", which was an earlier mis-citation) — *Did:* code-centric execution + cohort-level QC on ADHD-200/ADNI; *predictive*-performance evaluation (AUC/F1), not group-level inference. *Did not:* claim contracts, replication gating, abstention, power-honesty, or multiverse robustness. Public code exists but is raw-BIDS + fMRIPrep/FreeSurfer-only and unlicensed — not usable as a derivative-table baseline.
- **NeuroClaw** (arXiv:2604.24696, 2026) — *Did:* multi-agent neuroimaging on ADNI/HCP/UKB with code execution validated by checksums + output schemas (shares CONFIRM's "LLM-never-emits-numbers" discipline). *Did not:* gate on replication, power, or multiverse — its validation is *execution integrity*, not *claim validity*.
- **NEURA** (bioRxiv 2026.04.27.721217, 2026; v2 2026-05-21 retitled "proof-carrying framework for hallucination-resistant neuroimaging automation") — *Did:* autonomous workflow planning from a natural-language question (eval = planning accuracy); case study in spinocerebellar ataxia; v2 adds a verification layer that *grounds* claims to computed outputs. *Did not:* gate on statistical admissibility — its verification is output-grounding (claims trace to numbers), not replication/power/multiplicity/multiverse gating or abstention. No public code found.
- **AD-reproducibility agent** (arXiv:2505.23852, 2025) — *Did:* autonomously reproduce 35 findings from 5 published AD studies (NACC), ~53% fidelity — closest in *spirit* to rigor. *Did not:* gate claims on reproduction (measures it post-hoc), use contracts, abstain, or work cross-cohort.
- **Agentic neuro-radiology** (arXiv:2604.16729, 2026) — *Did:* training-free LLM orchestration of skull-strip/registration/segmentation/volumetry on single brain-MRI scans. *Did not:* any group-level statistics, cohort analysis, or discovery — single-patient radiology.

### B. Autonomous data-science / claim-verification agents (general)
- **The AI Scientist** (arXiv:2408.06292, 2024) + independent audit **Beel et al.** (arXiv:2502.14297, 2025) — *Did:* end-to-end autonomous ML research (idea→code→paper). *Did not:* enforce validity — the audit found **42% experiment failures + hallucinated numbers**. Motivates CONFIRM's "numbers only from executed code + gates."
- **EviBound** (arXiv:2511.05524, 2025) — *Did:* evidence-bound execution governance with a pre-execution **Approval Gate** + post-execution **Verification Gate** (0% hallucinated claims on ML tasks); the nearest "contract" prior work. *Did not:* govern *statistical* admissibility — its gates check that artifacts/run-IDs exist, not that a claim is powered, replicated, confound-adjusted, or fork-robust; not neuroimaging. **This is the boundary CONFIRM must defend: execution integrity vs. claim admissibility.**
- **"Many AI Analysts"** (arXiv:2602.18710, 2026) — *Did:* run many AI analysts to expose analytic-decision diversity + an AI auditor (the multiverse, agentified). *Did not:* enforce robustness as a gate or abstain — it *observes/reports* variation rather than *blocking* fragile claims.
- **From Fluent to Verifiable** (arXiv:2602.13855, 2026) — *Did:* propose claim-level auditability standards (provenance coverage) for research agents. *Did not:* pre-register machine-checkable contracts or gate on replication — post-hoc audit.
- **Data Interpreter / DS-Agent / CodeAct / Reflexion / ReAct** (2023–2024) — *Did:* the reusable machinery (hierarchical planning, executable-code actions, self-critique). *Did not:* check statistical assumptions, domain QC, multiple comparisons, or replication — self-verification is runtime-error-driven, not scientific-validity-driven.
- **Bayesian Hybrid Shrinkage** (arXiv:2511.06318, 2025) — *Did:* winner's-curse effect-size shrinkage (online A/B testing). *Did not:* operate as an agent or in neuroimaging — CONFIRM's delta is operationalizing this as an enforcement gate.

### C. Reproducibility & replication science (the motivation — solved as *warnings*, not *enforcers*)
- **NARPS / Botvinik-Nezer et al.** (Nature, 2020) — 70 teams, one dataset → divergent conclusions. *Provides* a ready binary benchmark; *offers no* automated enforcer.
- **Bowring, Maullin-Sapey & Nichols** (HBM, 2019/2021) — FSL vs SPM vs AFNI → thresholded-map Dice **0.00–0.74**. *Quantifies* pipeline variability; no enforcement.
- **Marek et al.** (Nature, 2022/2024) — brain-wide association studies need **N≈thousands**; published effects inflated **>5×**. *Quantifies* non-replication; proposes no workflow.
- **Multiverse / specification-curve** (Steegen et al., 2016; Simonsohn et al., NHB 2020) — *post-hoc reporting tools*, not gatekeepers.
- **Poldrack et al. "Scanning the horizon"** (Nat Rev Neurosci, 2017); **Button et al.** (Nat Rev Neurosci, 2013) — catalog systemic problems (power, flexibility, replication) and recommend practices CONFIRM operationalizes.
- **ComBat** (Fortin et al., 2018; Pomponio et al., 2020) — site/scanner harmonization; *a precondition* CONFIRM applies inside the replication gate.
- **Known-positive ground truth** (Jack et al., 2018, AD biomarkers; Franke & Gaser, 2019, BrainAGE) — established effects (AD hippocampal atrophy d≈1.5–2.0, FDG hypometabolism) CONFIRM must recover to prove it isn't an abstain-everything machine.

→ **The gap (verbatim from the survey):** *no existing tool compels a researcher to demonstrate cross-cohort replication as a precondition for claiming a discovery.* **CONFIRM is that enforcer.**

### D. Analysis-decision benchmarks (for the companion, NeuroDecide-Bench-lite)
- **BLADE** (arXiv:2408.09667, 2024) — closest. *Did:* benchmark LLM agents on expert-labeled analytical decisions (variable/transform/model choice; decision diversity; unjustifiable-choice MCQs) on social-science data. *Did not:* include injected-null ground truth, an abstention metric, or neuroimaging — it scores *diversity/plausibility*, not *validity under adversarial confounds*.
- **DiscoveryBench** (2407.01725) · **ScienceAgentBench** (2410.05080) · **DSBench** (2409.07703) · **InfiAgent-DABench** (2401.05507) · **MLE-bench** (2410.07095) · **DABstep** (2506.23719) · **QRData** (2402.17644) · **StatQA** (2406.07815) · **AbstentionBench** (2506.09038) — *Did:* benchmark data-analysis / ML / statistical-reasoning task completion or general abstention. *Did not:* combine neuroimaging + injected-null validity traps + abstention-on-invalid-analysis — exactly the NeuroDecide-Bench delta.

## Positioning Statement
> *Existing neuroimaging agents make analyses executable; CONFIRM makes scientific **claims** executable —
> a finding is "confirmed" only when a pre-registered, machine-checkable admissibility contract passes all
> gates (incl. cross-cohort replication), else the agent abstains. We show this removes a class of false
> confirmations that execution-integrity checks alone cannot prevent, while preserving established effects.*
