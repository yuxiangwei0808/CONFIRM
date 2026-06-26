# Idea Discovery Report — Agentic Framework for Reproducible Neuroimaging Analysis

**Direction:** A general agentic (LLM-driven) framework for **reproducible neuroimaging analysis** —
an agent that takes a scientific/clinical question + a cohort and autonomously **plans → runs →
QCs → interprets** a reproducible statistical analysis on precomputed derivatives.
**Goal:** research-backed open-source tool (paper + adopted tool). **Date:** 2026-06-01.
**Pipeline:** research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline.
**Status:** ✅ Phase 1 (landscape) · ✅ Phase 2 (ideas + jury) · ✅ Phase 3 (novelty, verified) · ✅ Phase 4 (review) · ✅ Phase 4.5 (refined proposal + experiment plan).

> Scope rationale & locked decisions: see repo `RESEARCH_BRIEF.md`.

---

## 0. Executive Summary

**Best idea (RECOMMENDED): "CONFIRM" — a claim-contract-governed, replication-gated neuroimaging
analysis agent.** Existing agents make analyses *executable* and validate *execution integrity*; none
prevent an agent from promoting a statistically *inadmissible* finding (underpowered, unreplicated,
confounded, fork-dependent) to "confirmed." CONFIRM makes the **scientific claim** the governed unit:
a finding is labeled **"confirmed" only if it passes machine-checkable admissibility gates — multiplicity,
confound, power, *cross-cohort replication*, multiverse — else the agent abstains/labels it.**

**Key evidence:** cross-model jury + novelty verification (every competitor WebFetch-verified) →
**cross-cohort replication-as-a-gate is HIGH novelty and unoccupied** (NeuroClaw/NeuroAgent/NIAgent
validate execution, not claims; EviBound gates execution integrity, not statistical validity; Many-AI-
Analysts *observes* the multiverse, doesn't *gate*). Senior-reviewer score **7.5→8.5/10** if scoped to
**CONFIRM-lite** and the must-win experiment lands. Uniquely enabled by your multi-cohort, multi-modal
data; the core loop runs on **CPU over precomputed derivatives**.

**Recommended next step:** build B0 infra (claim-contract schema → executor+provenance → ComBat/replication
harness), mount data, run pilots P1–P3, then the **must-win** experiment (false-confirmed rate vs an
execution-valid runner). Full method in `refine-logs/FINAL_PROPOSAL.md`; plan in `refine-logs/EXPERIMENT_PLAN.md`.

---

## 1. Literature Landscape

Four independent searches mapped: (1.1) agentic neuroimaging systems, (1.2) general autonomous
data-science agents, (1.3) reproducibility/replication science, (1.4) existing automation tooling.

### 1.1 Agentic / LLM systems for neuroimaging — *dominated by single-patient radiology; analysis layer nearly empty*

| System | Year | Covers | Gap vs. our direction |
|---|---|---|---|
| Agentic LLMs for Neuro-Radiological Analysis (2604.16729) | 2026 | skull-strip→seg→volumetry on single scans | no stats, no cohort, no discovery |
| **NeuroAgent** (2605.06584) | 2026 | preprocessing + **1 task (AD classification)** + some FDR stats on ADNI | **no cross-cohort replication, no provenance/reproducibility**, 4–8 h/subject |
| **NEURA** (bioRxiv 2026) | 2026 | autonomous **workflow planning** (89.5% planning acc) | evaluated on planning, not scientific output |
| **AD-reproducibility agent** (2505.23852) | 2025 | autonomously reproduces AD stats (NACC) | **53.2% fidelity**, single dataset, primitive |
| TissueLab / MACRO / AURA / RadA-Bench / Medical-AI-Consensus | 2024–26 | pathology/CXR/radiology seg+report | wrong modality / no stats |
| BrainGPT (Nat Hum Behav 2025) / fMRI-LM (2511.21760) | 2024–25 | literature prediction / fMRI decoding | no data analysis / not agentic |
| GPT normative sMRI (Imaging Neuro 2024); GenBrain (medRxiv 2025) | 2024–25 | normative modeling / generative augmentation | not agentic; manual setup |

**Read:** Five sub-clusters (radiology-diagnosis agents · preprocessing agents · medical co-evolution
agents · neuro-language models · reproducibility agents). The **analysis/discovery + replication +
provenance** corner is essentially unoccupied; NeuroAgent and the AD-repro agent are the only
neighbors and both stop short of it.

### 1.2 General autonomous data-science / "AI scientist" agents — *strong machinery, no scientific rigor*

- **Transferable machinery:** hierarchical task decomposition (Data Interpreter 2402.18679, DS-Agent
  2402.17453), executable-code actions (CodeAct 2402.01030, ReAct 2210.03629), self-critique/episodic
  reflection (Reflexion 2303.11366), automated-reviewer/critic agents (AI Scientist 2408.06292),
  case-based reasoning, workflow provenance (2509.13978).
- **Cautionary tale:** The AI Scientist (Sakana) was independently audited (Beel et al. 2502.14297) —
  **42% experiment failure, hallucinated numbers, ~8% code change/iteration, 5 median citations.**
  Autonomy *without enforced rigor* produces confident garbage.
- **What ALL of them systematically lack** (directly relevant): (1) statistical-validity enforcement
  (assumptions, FWE/FDR), (2) domain priors (BOLD/HRF, autocorrelation, smoothing), (3) dataset-specific
  QC as a first-class step, (4) *scientific*-error self-verification (not just runtime errors),
  (5) reproducibility across seeds/environments, (6) multiple-comparison / p-hacking / forking-paths
  guards, (7) **replication / out-of-sample validation**, (8) deep machine-readable provenance,
  (9) confound modeling, (10) structured human checkpoints.

> **Implication:** our contribution is not "another autonomous agent" — it is **rigor-as-architecture**.
> The list above is effectively our design spec; each lacked item is a feature.

### 1.3 Reproducibility & replication science — *the field is asking for exactly this, but has no enforcer*

| Work | Finding → why it matters for us |
|---|---|
| **NARPS** (Botvinik-Nezer, Nature 2020) | 70 teams, same data → divergent conclusions. NARPS OpenNeuro set = ready **binary benchmark** |
| Bowring/Nichols (HBM 2019/21) | FSL vs SPM vs AFNI → Dice **0.00–0.74**. Gives pipeline-stability metrics |
| Multiverse / spec-curve (Steegen 2016; Simonsohn NHB 2020) | report effect across all defensible specs → **specification-curve stability score** |
| **Marek et al.** (Nature 2022/2024) | BWAS need **N≈thousands**; published effects inflated **>5×**. Cross-cohort replication = our headline metric |
| ComBat / harmonization (Fortin 2018; Pomponio 2020) | site/scanner batch removal → **precondition** for cross-cohort pooling |
| Known robust effects (Jack 2018; BrainAGE; Marek) | AD hippocampal atrophy **d≈1.5–2.0**, FDG hypometabolism, brain-age gap → **known-effect recovery** ground truth |

**The gap (verbatim from the search):** *"No existing tool compels a researcher to demonstrate
cross-cohort replication as a precondition for claiming a discovery."* Multiverse/spec-curve are
post-hoc reporting; fMRIPrep/BIDS standardize but enforce no analytic policy; Marek quantified the
problem but proposed no automated workflow. **An agent that gates findings on replication would be the
first end-to-end implementation of what this literature has demanded since 2013.**

### 1.4 Existing automation tooling — *preprocessing is solved; the "decision layer" is not*

- **Solved / turnkey:** BIDS conversion+validation (HeuDiConv, CuBIDS), preprocessing (fMRIPrep, QSIPrep,
  FastSurfer/SynthSeg), IQMs (MRIQC), confound generation (fMRIPrep→`load_confounds`), GLM fit
  (Nilearn, FitLins via BIDS-Stats-Models), connectivity (C-PAC), tractometry (pyAFQ), normative modeling
  (PCNtoolkit), UKB phenotypes (FUNPACK), meta-analysis (NiMARE).
- **NOT automated — expert judgment still required (= the agent's job):**
  1. QC pass/fail **thresholding** (study-/analysis-dependent; MRIQC only ~76% on unseen sites)
  2. **Confound strategy** selection (which `load_confounds` recipe, scrubbing thresholds, GSR on/off)
  3. **Pipeline/parameter** config (atlas, parcellation, HRF, recon algorithm)
  4. **Inclusion/exclusion** decisions (auditable, study-specific, with reasons)
  5. **Covariate/model** selection (which IDPs, age/sex/site terms, normative vs mass-univariate vs multivariate)
  6. **Multiple-comparison** strategy (FWE vs FDR, cluster-forming threshold)
  7. **Interpretation** (which clusters are meaningful; link to literature via NiMARE/Neurosynth)
  8. **Cross-cohort harmonization** decisions (ComBat/CovBat/site-as-covariate; discovery vs replication split)
- **Integration seams for a research-backed tool:** consume **BIDS-Derivatives** natively; drive Nilearn
  `load_confounds` + BIDS-Stats-Models/FitLins; ingest MRIQC IQMs; emit BIDS-Derivatives exclusion lists;
  context via NiMARE; deviations via PCNtoolkit; provenance via **DataLad + BIDS**. *Reimplement nothing;
  own the decisions between tools.*

---

## 2. Synthesis — The White Space (convergence of all four searches)

All four lenses point to the **same unoccupied intersection**:

> **An agent that owns the *decision layer* of neuroimaging analysis — operating over precomputed
> derivatives via existing tools — with rigor enforced as architecture: specification/multiverse-aware
> analysis, statistical-assumption & QC-in-the-loop checking, confound/inclusion decisioning, full
> provenance + re-executability, and — the headline — *cross-cohort replication as a gating
> precondition before any finding is reported as "confirmed."* Evaluated by known-effect recovery +
> cross-cohort/cross-modal replication + reproducibility.**

**Why uniquely ours:**
- The user's **multi-cohort, multi-modal data (UKB/HCP/ADNI/ABCD/OpenNeuro; fMRI/sMRI/PET; many diseases)**
  is exactly what a replication-gating agent needs and what no prior system had.
- Preprocessing being solved means the core loop runs on **CPU over derivatives** — feasible without GPUs.
- Ground-truth effects + NARPS + Marek power curves make the evaluation **objective and computable today**.

**Strongest single novelty hook:** *"replication-before-reporting" as an enforced gate* — absent from
both the neuroimaging-agent literature and the general DS-agent literature.

**Candidate idea seeds for Phase 2** (to be expanded, filtered, novelty-checked, reviewed):
the replication-gating discovery agent; the multiverse-native analysis agent; the QC-in-the-loop /
assumption-checking layer; the auditable inclusion/exclusion decision engine; the confound-strategy
recommender; cross-modal triangulation (does an effect hold across fMRI/sMRI/PET?); a benchmark of
*agentic analysis-decision quality* vs. expert decisions; a provenance/"re-executable finding" layer.

---

## 3. Key References

Agentic neuro: 2604.16729 · NeuroAgent 2605.06584 · NEURA (bioRxiv 2026.04.27.721217) · AD-repro
2505.23852 · TissueLab 2509.20279 · BrainGPT 2403.03230 · fMRI-LM 2511.21760 · GPT-normative
(imag_a_00204) · GenBrain (medRxiv 2025.12.19).
DS agents: AI-Scientist 2408.06292 · Beel critique 2502.14297 · Jr-AI-Scientist 2511.04583 ·
DS-Agent 2402.17453 · InfiAgent-DABench 2401.05507 · MLAgentBench 2310.03302 · CodeAct 2402.01030 ·
Data-Interpreter 2402.18679 · Reflexion 2303.11366 · ReAct 2210.03629 · Coscientist 2304.05332.
Reproducibility: NARPS (Nature 2020) · Bowring (HBM 2019/21) · Marek (Nature 2022/2024) · Multiverse
(Steegen 2016) · Spec-curve (Simonsohn 2020) · Poldrack "Scanning the horizon" (NRN 2017) · Button
(NRN 2013) · ComBat (Fortin 2018; Pomponio 2020).
Tooling: fMRIPrep · MRIQC · Nilearn/`load_confounds` · FitLins/BIDS-Stats-Models · CuBIDS · NiMARE ·
PCNtoolkit · FUNPACK · pyAFQ · QSIPrep · DataLad.

> ⚠️ Closest competitors to verify in Phase 3 before finalizing any differentiation claim:
> **NeuroAgent 2605.06584**, **NIAgent 2605.09366** (surfaced by the cross-model jury; claims
> closed-loop QC + workflow automation), **AD-repro 2505.23852**, **NEURA** (bioRxiv 2026).

---

## 4. Phase 2 — Ranked & Filtered Ideas (cross-model jury: Codex `gpt-5.5`, xhigh)

10 candidates generated across the white space, then scored by an independent cross-model jury
(N=Novelty, F=Feasibility on CPU/derivatives ~3–6mo, I=Impact/Usefulness, D=Data-fit; /10).

| # | Idea | N/F/I/D | Status | Closest prior → delta |
|---|---|---|---|---|
| **1** | **Replication-Gated Discovery Agent** | **8.5/7/9.5/9.5** | **CORE** | NeuroAgent/NIAgent automate workflows; Marek shows BWAS replication failure → *delta: no "confirmed" claim without independent replication* |
| **9** | **Power-Aware Honest Discovery** (winner's-curse shrinkage, abstention) | 6.5/8.5/8.5/9 | **merge into #1** | Button 2013 / Marek → *delta: agent refuses/shrinks under-powered claims* |
| **2** | **Multiverse-Native Robustness** (do plausible forks flip the claim?) | 6.5/6.5/8.5/8 | **core module** | NARPS, spec-curve → *delta: agent auto-enumerates forks + explains conclusion flips* |
| **7** | **NeuroDecide-Bench** (benchmark of *analysis-decision* quality) | 8.5/5.5/9/8 | **companion / backbone** | InfiAgent-DABench (shallow) → *delta: benchmark claim-validity decisions, not task accuracy* |
| **8** | Re-Executable Finding / Provenance "receipts" | 4.5/8/8/8.5 | infra module | DataLad/ReproNim/BIDS-Stats-Models → *delta: automatic finding receipt* |
| **3** | QC-in-the-Analysis-Loop (statistical-assumption checks) | 6/8/8/8 | module (NIAgent threatens novelty) | MRIQC/AFNI-QC → *delta: statistical/collider/confound diagnostics, not visual/runtime* |
| **4** | Auditable Inclusion/Exclusion engine | 5.5/7.5/7.5/8 | module / bench subtask | MRIQC labels → *delta: study-specific, downstream-sensitivity-justified* |
| **6** | Confound-Strategy Recommender | 4.5/8.5/7.5/8 | module | load_confounds recipes → *delta: chooser + sensitivity* (weak as a paper) |
| **5** | Cross-Modal Triangulation (fMRI↔sMRI↔PET) | 5.5/5.5/7/6.5 | later demo/module | ADNI multimodal lit (heavily worked) → *delta: evidence-grading by cross-modal consistency* |
| **10** | Full Hypothesis→Report orchestrator | 6.5/4.5/9/9 | **tool wrapper, NOT the paper** | NeuroAgent/NEURA/AI-Scientist → delta only if gating+replication+bench are central |

### Converged contribution (what advances)

**🏆 CORE — "CONFIRM" (working name): a claim-contract-governed, replication-gated, power-aware
reproducible-analysis agent.** The unifying mechanism (from the jury's best missed idea): a
**machine-checkable *claim contract*** declared *before* any code runs — estimand, cohort,
exclusions, confounds, multiplicity rule, **power rule (Marek-aware)**, **cross-cohort replication
rule**, **multiverse-robustness rule**, allowed reporting language. The agent may only emit a
finding as **"confirmed"** if executed, schema-validated results satisfy the contract; otherwise it
**abstains** or labels (*cohort-specific / non-replicated / under-powered / fragile*). **Numbers
never come from the LLM — only from executed code + provenance bundle.** Folds in #1+#9+#2+#8+#3.

**🤝 COMPANION — NeuroDecide-Bench + adversarial error-injection suite.** A tightly-scoped benchmark
of *claim-validity decisions*, including **injected-artifact tasks** (site imbalance, motion leakage,
age/diagnosis confounding, collider bias, label leakage, bad multiplicity) that give **ground-truth
nulls**. Doubles as the agent's evaluation *and* a standalone community asset (adoption/usefulness).

**🔌 ADOPTION MOAT (enabling layer, in the tool) — cross-cohort phenotype/derivative alignment.** A
semantic mapper from "clinical question" → comparable variables/features across UKB/HCP/ADNI/ABCD/
OpenNeuro. "Boring but likely the adoption moat" — and a precondition for replication-gating to work.

### Headline claim (the paper)
> *On precomputed neuroimaging derivatives, a replication-gated, claim-contract analysis agent
> substantially reduces false "confirmed" findings and effect-size inflation versus single-cohort
> and generic data-science agents, while preserving recovery of established neuroimaging effects.*

**The defensible delta (jury's brutal-honesty verdict):** NOT "first autonomous neuroimaging agent"
(that's taken). YES — *"first neuroimaging analysis agent whose architecture treats replication,
power, multiverse robustness, and provenance as **gates** before a scientific claim is allowed."*

### Minimum viable evaluation
8–15 tasks across ADNI/UKB/HCP/ABCD/OpenNeuro: known-positives (AD hippocampal atrophy d≈1.5–2.0,
FDG hypometabolism, age/sex/ICV effects), motion-confounded FC, NARPS-style fragile effects, and
**injected null/confound tasks**. Metrics: third-cohort **confirmation rate**, **false-confirmed
rate**, **effect-size calibration error**, **abstention quality**, **bundle reproducibility**.
Baselines: single-cohort agent · generic DS agent · NeuroAgent/NIAgent-style workflow runner.

### Pilots (status: NEEDS MANUAL PILOT — paper-only this session)
No confirmed GPU and no data mounted in-session; core loop is CPU/derivatives so pilots are cheap to
run later. Designed cheap pilots: **(P1)** ADNI→(2nd cohort) replication gate on AD hippocampal
atrophy (known strong effect — must confirm); **(P2)** an injected-null task (must abstain, not
confirm — the winner's-curse trap); **(P3)** a NARPS-fragile effect through the multiverse module
(must label fragile). To execute these, confirm data access + a small compute env.

### Top reviewer attacks → preemptions
1. *"Just glue around tools."* → formal claim contract + deterministic validators + ablation showing
   replication gating **changes scientific conclusions**.
2. *"No ground truth."* → known-positives + injected nulls + NARPS consensus + held-out-cohort
   confirmation, scored separately.
3. *"LLMs hallucinate science."* → LLM never emits numbers; all claims from executed, schema-validated
   artifacts + abstention (cite AI-Scientist failure evidence, 2502.14297).

---

## 5. Phase 3 — Deep Novelty Verification (both contributions VERIFIED; PROCEED-WITH-CAUTION)

Two parallel novelty tracks, every paper WebFetch-verified (anti-hallucination). The field moved fast
(≈5 neuroimaging agents in Apr–May 2026) and surfaced new competitors — but the deltas hold.

### 5.1 CORE — newly-found competitors & per-claim verdict
New closest works (beyond NeuroAgent/NIAgent/NEURA/AD-repro):
- **NeuroClaw** (2604.24696) — multi-agent neuroimaging on ADNI/HCP/UKB, code-execution w/ checksum/schema
  validation. *Shares "no-LLM-numbers"; lacks claim contracts, replication gate, abstention, power, multiverse.*
- **EviBound** (2511.05524) — pre-execution Approval Gate + post-execution Verification Gate, a "contract."
  *But governs EXECUTION INTEGRITY (artifact exists, run_id valid), NOT statistical validity.* ← key boundary.
- **Many AI Analysts** (2602.18710) — AI analysts run multiverse-style analyses + an AI auditor.
  *OBSERVES analytic diversity; does NOT enforce it as a gate or abstain.* ← enforcement-vs-observation delta.
- **From Fluent to Verifiable** (2602.13855) — claim-level *post-hoc* auditability for research agents.
- **Bayesian Hybrid Shrinkage** (2511.06318) — winner's-curse shrinkage (A/B testing; a stats method, not an agent).

| Mechanism | Novelty | Closest threat |
|---|---|---|
| Cross-cohort replication as a **hard confirmation gate** | **HIGH** | AD-repro (measures post-hoc, doesn't gate) |
| Domain-rigor **abstention** (underpowered/non-replicated/fragile) | **HIGH** | general LLM-abstention only |
| Machine-checkable **statistical claim contract** | **MED-HIGH** | EviBound (execution-integrity, not stats) |
| **Power-aware** winner's-curse gate | **MED-HIGH** | Bayesian shrinkage (not an agent) |
| **Multiverse robustness as a mandatory gate** | **MED-HIGH** | Many AI Analysts (observes, doesn't enforce) |
| "LLM never emits numbers" | **LOW** → **DROP as a contribution** (now baseline: NeuroClaw/EviBound) | NeuroClaw |

**Biggest attack:** *"NeuroClaw + EviBound repackaged."* **Does NOT kill us** if we hold the line:
EviBound asks *"did the artifact get created?"*; we ask *"is this scientific claim epistemically permissible?"*
(estimand + power + multiplicity + confounds + replication + multiverse). No prior work combines all six.
**Must-win experiment:** a case where NeuroClaw/NeuroAgent label a finding "confirmed" that we correctly
**abstain** on (e.g. N=200, uncorrected p=0.04, fails 2nd-cohort replication).

### 5.2 COMPANION — NeuroDecide-Bench (score 7/10)
Closest = **BLADE** (2408.09667): benchmarks LLM agents on expert-labeled analytical decisions
(variable/transform/model choice, unjustifiable-choice MCQs, decision *diversity*). Also surveyed:
DiscoveryBench, ScienceAgentBench, DSBench, DABstep, InfiAgent-DABench, QRData, StatQA, MLE-bench,
AbstentionBench, RadA-BenchPlat — **none** cover injected-null validity traps or neuroimaging decisions.

| Distinctive feature | Novelty |
|---|---|
| **Injected-artifact ground-truth NULLS** (motion leakage, site imbalance, collider bias, label leakage) | **HIGH** |
| **Neuroimaging-specific** decision points + domain | **HIGH** |
| Abstention-on-traps as a first-class metric | **MED-HIGH** |
| Analysis-decision-*quality* framing | **MED** (BLADE owns "decision quality") |

**Boundary vs BLADE (must articulate):** BLADE measures decision *diversity/plausibility* on social-science
data; NeuroDecide-Bench measures analytic *validity under adversarial conditions* — ground-truth invalidity,
injected neuroimaging confound traps, calibrated **abstention** scoring. **Lead with injected-nulls +
abstention, NOT "decision quality."**

### 5.3 Net novelty actions (carried into refinement)
1. **Drop** "LLM-no-numbers" as a contribution (frame as a prerequisite).
2. **Position explicitly vs EviBound** (execution-integrity) and **vs BLADE** (decision-diversity) — one paragraph each.
3. **Release the claim-contract JSON schema as an open standard** (community-infrastructure angle, hard to dismiss).
4. **Headline ablation:** rigor-gated agent vs NeuroClaw/NIAgent-style runner on **known non-replicable** findings → false-confirmed rate.
5. ⚠️ **Verify NEURA full text** (bioRxiv 403'd) before any submission — only unverified concurrent work.

> Verified prior-work traces: `.aris/traces/novelty-check/2026-06-01_run01/`.

---

## 6. Phase 4 — External Critical Review (Codex `gpt-5.5`, xhigh; same thread, multi-round)

**Score: 7.5/10 now → 8.5/10 if tightly scoped + the must-win experiment lands.** Core CONFIRM is
stronger than the companion benchmark. Paired story is coherent but *risky if it sprawls* into
"framework + benchmark + many demos."

**Novelty language tightened (reviewer):**
- Don't claim generic "claim contracts" → claim **"machine-checkable statistical claim *admissibility*
  for neuroimaging."** The six gates aren't individually novel; the novelty is making them **blocking
  conditions for the *claim label*.**
- **Cross-cohort replication gate = strongest delta** (not seen in NeuroClaw/NeuroAgent/NIAgent).
- Multiverse gate defensible **only if it gates/abstains** (vs Many-AI-Analysts which only *reports*).
- EviBound boundary is **real, not cosmetic — iff** the gate can reject a *perfectly executed* analysis
  because the claim is statistically inadmissible (underpowered/unreplicated/fragile/confounded).
- Benchmark: claim **"first neuroimaging benchmark for validity under adversarial nulls + abstention,"**
  not "decision quality" (BLADE owns that).

**3 most damaging weaknesses → required preemptions:**
1. **Gate arbitrariness** ("why these 6 / these thresholds? did you build an abstain-everything machine?")
   → report **selective-risk curves**: false-confirmed rate vs known-positive confirmation rate.
2. **Benchmark construct validity** (injected nulls become toy traps detectable by trivial heuristics)
   → trivial baselines, **mask trap structure in realistic derived variables**, real cohort covariance,
   don't reveal trap type in prompts.
3. **Straw-man baselines** (NeuroClaw/NIAgent aren't designed to label "confirmed")
   → **fair baselines**: same data/prompt/tool-budget/execution-validation + a **blinded claim extractor**
   mapping every system's report into {confirmed / qualified / abstained}.

**Minimum experiments to reach "accept":**
- **Must-win:** realistic discovery-cohort traps where an execution-valid runner reports a significant
  finding that **fails replication / collapses under confound+multiverse**, and CONFIRM **abstains /
  labels non-replicated**. Primary metric = **false-confirmed rate**.
- **Known positives** so abstain-all fails (AD hippocampal atrophy, FDG hypometabolism, age/sex/ICV).
- **Real fragile cases** (NARPS-style; split/cohort-unstable brain-behavior associations), not only injected.
- **Gate-ablation ladder**: execution-only → +confounds → +power → +multiverse → +replication; the
  replication gate must **visibly** reduce false confirmation.
- **Selective metric**: false-confirmed rate **@ fixed known-positive recall**, or area under the
  **risk-coverage curve** (plain accuracy rewards dumb refusal).
- Construct-validity: separate *"non-replicated because false"* from *"non-replicated because cohort
  mismatch"* via **predeclared phenotype/feature harmonization**.

**Scope cut → what to ship (the refined proposal):** **CONFIRM-lite** (see `refine-logs/FINAL_PROPOSAL.md`).

---

## 7. Phase 4.5 — Refined Proposal & Experiment Plan (links)

The top idea was refined into a problem-anchored proposal + claim-driven experiment roadmap:
- **Proposal:** `refine-logs/FINAL_PROPOSAL.md` — Problem Anchor, method thesis, dominant contribution,
  smallest mechanism (CONFIRM-lite), key claims C1–C4, rejected complexity, risks.
- **Experiment plan:** `refine-logs/EXPERIMENT_PLAN.md` — datasets, NeuroDecide-Bench-lite task suite,
  selective metrics, fair baselines + blinded claim extractor, blocks B0–B6, run order + decision gates.
- **Tracker:** `refine-logs/EXPERIMENT_TRACKER.md` · **Review:** `refine-logs/REVIEW_SUMMARY.md` ·
  **Summary:** `refine-logs/PIPELINE_SUMMARY.md`.

**Method thesis:** *CONFIRM governs the claim, not just the run — "confirmed" only if executed results
pass multiplicity + confound + power + cross-cohort-replication + multiverse gates, else abstain/label.*

## 8. Eliminated / demoted ideas (audit trail)
- **#5 Cross-modal triangulation** — demoted (ADNI multimodal heavily worked; later demo only).
- **#6 Confound-strategy recommender**, **#4 inclusion engine** — demoted to *modules* of CONFIRM (weak as standalone papers).
- **#10 Full "AI neuroscientist" orchestrator** — demoted to *tool wrapper*, not the paper claim (too broad; NeuroAgent/NEURA occupy it).
- **"LLM-never-emits-numbers"** — dropped as a *contribution* (now baseline; kept as a prerequisite).
- **"Decision-quality" benchmark framing** — dropped in favor of *validity-under-adversarial-nulls + abstention* (BLADE owns "decision quality").

## 9. Next Steps
- [ ] Build **B0 infra**: claim-contract JSON/YAML schema (release as open standard) → analysis executor +
      provenance bundle → ComBat + cross-cohort replication harness.
- [ ] Mount cohort derivatives; run **pilots P1–P3** (AD-atrophy → confirm · injected-null → abstain · NARPS → fragile).
- [ ] Run **B1 must-win** (false-confirmed rate vs execution-valid runner) → decision gate before scaling.
- [ ] Verify **NEURA** full text (only unverified concurrent work) before any submission.
- [ ] Then `/run-experiment` to deploy the plan, and `/auto-review-loop` to iterate toward submission.
