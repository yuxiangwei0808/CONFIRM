# Review Summary (cross-model: Codex gpt-5.5, xhigh)

Two rounds in one thread: (R1) idea jury over 10 candidates; (R2) novelty cross-check + senior review of
the converged, novelty-verified proposal. Trace: `.aris/traces/` (idea-discovery run, 2026-06-01).

## Verdict
- **Score:** 7.5/10 now → **8.5/10** if scoped to CONFIRM-lite and the must-win experiment lands.
- Core CONFIRM **stronger** than the companion benchmark; ship the sharp claim, not the framework.

## Novelty (tightened language)
- Strongest delta = **cross-cohort replication gate** (HIGH). Domain-rigor **abstention** (HIGH).
- Claim "machine-checkable statistical claim **admissibility** for neuroimaging," **not** generic "claim contracts."
- Six gates not individually novel → novelty = **blocking conditions on the claim label.**
- **Drop** "LLM-never-emits-numbers" as a contribution (baseline: NeuroClaw/EviBound).
- Boundaries that must be defended empirically: vs **EviBound** (execution-integrity, not statistical) and
  vs **Many-AI-Analysts** (observes multiverse, doesn't gate); benchmark vs **BLADE** (decision diversity,
  no injected nulls/abstention/neuroimaging).

## Top weaknesses → required fixes
1. **Gate arbitrariness** → selective-risk (risk-coverage) curves, not fixed thresholds.
2. **Benchmark construct validity** → mask traps in realistic variables; trivial-heuristic baseline; don't
   reveal trap type; real cohort covariance.
3. **Straw-man baselines** → matched fair baselines + blinded claim extractor → {confirmed/qualified/abstained}.

## Must-win experiment
Realistic traps where an execution-valid runner reports a significant finding that fails replication /
collapses under confound+multiverse, and **CONFIRM abstains** — primary metric **false-confirmed rate**,
with **known positives** preventing an abstain-all degenerate solution. Replication gate must be the
**dominant** driver in the ablation ladder.
