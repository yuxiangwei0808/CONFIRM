> **SUPERSEDED (2026-06-20).** This early plan is stale — it predates the external benchmark and the power-gate fix and still says "0/177 → 0 false confirmations". Authoritative numbers: [RESULTS_MANIFEST_20260620.md](RESULTS_MANIFEST_20260620.md). Current manuscript: `paper/`.

# PAPER_PLAN — CONFIRM (WACV 2027)

**Venue:** WACV 2027 (IEEE/CVF, `wacv.sty`, review mode). **Page limit:** ~8 pages + unlimited refs. **Track:** *decision pending* (algorithms / applications / datasets — see end).

## Title (proposed)
**Primary:** *CONFIRM: Statistical Claim Governance for Trustworthy Agentic Neuroimaging Discovery*
- Alt 1: *Confirming Less, Trusting More: A Claim-Admissibility Layer for Agentic Neuroimaging*
- Alt 2: *Governing What Agents Confirm: Cross-Cohort Replication Gates for Neuroimaging Discovery*

## Abstract (sketch, ~200 words)
Agentic systems can now run end-to-end neuroimaging analyses, but they validate *execution integrity* (the code ran, artifacts exist) — not whether a finding is *statistically admissible*. In a field already defined by non-replicating results, automating analysis risks scaling false discovery. We present **CONFIRM**, a claim-governance layer: an LLM drafts a *frozen* ClaimContract from a natural-language question; executed code computes every number; and a gate ladder — multiplicity (with a search-provenance correction), confound and confound-completeness, power, multiverse stability, and **cross-cohort replication** — labels a finding `confirmed` only if it passes, otherwise abstaining (`non_replicated` / `under_powered` / `fragile`). The LLM never emits numbers; a numeric guard strips any it introduces. Across Alzheimer's, schizophrenia, and aging on 13 cohorts (sMRI + fMRI connectivity), CONFIRM recovers known-positive effects at full recall while abstaining on confounded, p-fished, and non-replicating nulls: **0 observed false confirmations over 177 adversarial stress claims (95% CI ≤3.1%)**. As a modular layer over a real agent (NeuroClaw), it cuts the false-confirmation rate from 33% to 0% at matched recall, and its verdicts are robust across six LLMs. We frame CONFIRM as a governance layer with strong internal stress-test validation, and release the benchmark.

## Contributions
1. **CONFIRM** — a claim-governance layer + gate ladder for agentic neuroimaging (novel: *search-provenance* and *confound-completeness* gates; cross-cohort replication as a hard gate). LLM drafts contracts; code computes all numbers.
2. **A faithfulness benchmark** — cross-cohort claims with externally-anchored known-positive / known-null / fragile labels, plus a 177-claim adversarial stress suite; metrics = false-confirmation rate at fixed known-positive recall, and a coverage-vs-FCR curve.
3. **Governance-as-faithfulness evidence** — CONFIRM beats a real execution-integrity agent (NeuroClaw) both head-to-head and as a *retrofit layer*; verdicts robust across 6 LLMs; a numeric anti-hallucination guard.
4. **Honest negatives** — a borderline multi-cohort schizophrenia case correctly withheld; one transparent false-confirm; explicit conditional-bound + external-validation limitations.

## Section outline (CVF, 8 pp)
1. **Introduction** — agentic neuroimaging + the false-discovery risk; execution-integrity ≠ claim-admissibility; CONFIRM thesis; contributions. (Fig. 1 = architecture)
2. **Related Work** — agentic neuroimaging agents (NeuroClaw, NeuroAgent, NEXUS, NEURA: execution integrity, no admissibility gating); reproducibility/multiverse/NARPS; LLM hallucination + tool use; AI-for-science governance.
3. **Method: CONFIRM** — ClaimContract (frozen, inspectable); the gate ladder (multiplicity + search-provenance; confound + confound-completeness audit; power; multiverse; cross-cohort replication w/ ComBat + heterogeneity → direction/magnitude/transportable sub-types); verdict taxonomy; LLM-drafts-not-numbers + numeric guard; native in-loop agent vs modular layer.
4. **Faithfulness Benchmark** — cohorts & modalities; claim taxonomy + authoritative label table (provenance, leakage controls); the 177-claim adversarial stress suite (random-label / site-confound / p-fishing / underpowered / non-replication families); metrics (FCR@recall, coverage-FCR, exact Clopper–Pearson CIs).
5. **Experiments** —
   - E1 Gate ladder: TPR/FCR across exec_only→+confound→+power→+multiverse→+replication (FULL + adjudicated MAIN); coverage-FCR tradeoff. (Fig. 2, Tab. 2)
   - E2 Real baseline: NeuroClaw head-to-head (CONFIRM 10/10 & 0/15 vs 9/10 & 5/15) + **modular layer** (FCR 0.33→0.0, recall preserved). (Fig. 3)
   - E3 LLM robustness: 6 models, cross-model verdict agreement, anti-hallucination guard (40+ catches). (Fig. 4)
   - E4 Negatives stress: 177 claims → FCR upper bound 12.8%→3.1%. (Tab. 3)
   - E5 Honest cases: borderline multi-cohort SZ (withheld); the 1 false-confirm (mis-constructed negative).
6. **Limitations** — conditional (internal stress-test) FCR bound, not a real-world guarantee; 3 domains / AD-heavy; synthetic gate-targeted nulls; **external preregistered real-claim benchmark = future work**.
7. **Conclusion.**

## Claims → Evidence matrix
| # | Claim | Evidence (artifact) |
|---|---|---|
| C1 | Recovers known positives at full recall | MAIN TPR 10/10 — review-stage/round5-combat/ |
| C2 | 0 observed false-confirms, tight bound | 1/177 [0,0.031] — review-stage/negatives-expansion/ |
| C3 | Beats a real agent (compare + layer) | review-stage/round5-neuroclaw/ + confirm-layer/ |
| C4 | LLM-robust governance + anti-hallucination | review-stage/agentic-multillm/ (6 models, 7/9 agree, 40 catches) |
| C5 | Replication gate is the driver | gate-ladder FCR drop at +replication — round5-combat/ |
| C6 | Honest abstention (no over-claim) | SZ-breadth borderline — review-stage/sz-brainwide/ |

## Figures / tables (sources mostly exist)
- **F1** CONFIRM architecture + gate ladder — *to draw* (figure-spec/TikZ).
- **F2** coverage-vs-FCR per gate rung — review-stage/round5-combat/coverage_vs_fcr.* + negatives.
- **F3** NeuroClaw head-to-head + modular-layer bars — round5-neuroclaw + confirm-layer.
- **F4** multi-LLM cross-model agreement + anti-hallucination — agentic-multillm.
- **T1** confirmed positives by domain (AD/SZ/aging) + effects.
- **T2** gate-ladder TPR/FCR + exact CIs (FULL + MAIN).
- **T3** negatives per-family FCR (177).

## Honest framing (mandated; carry into every section)
"No *observed* false confirmations" (never "guaranteed"); the ≤3.1% bound is **conditional on the synthetic gate-targeted stress suite on local cohorts**, not a real-world rate; "governance layer," not "automated discovery." 3 positive domains. Cross-cohort SZ is supporting/borderline, not a confirmed multi-cohort positive.

## Decisions needed (Phase-1 checkpoint)
1. **Track** — recommend **Algorithms** (CONFIRM's governance method is the core contribution); *Applications* (neuroimaging) or *Datasets* (the benchmark) are defensible alternates.
2. **Assurance** — recommend **submission** (mandatory claim + citation audits) given this paper's central numeric/safety claims; default is draft.
3. **Authorship** — review version stays anonymized (CVF default); real author block added at camera-ready.
