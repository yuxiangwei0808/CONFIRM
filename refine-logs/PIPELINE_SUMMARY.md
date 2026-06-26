# Pipeline Summary

**Problem:** Agentic neuroimaging analysis can promote statistically *inadmissible* findings to
"confirmed"; execution-integrity gates don't catch underpowered/unreplicated/confounded/fork-dependent claims.
**Final Method Thesis:** CONFIRM governs the *claim* — a finding is "confirmed" only if it passes
machine-checkable statistical-admissibility gates (multiplicity, confound, power, **cross-cohort
replication**, multiverse), else the agent abstains/labels.
**Final Verdict:** READY (tightly scoped as CONFIRM-lite). Cross-model score 7.5→8.5/10 if must-win lands.
**Date:** 2026-06-01

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report (canonical): `idea-stage/IDEA_REPORT.md`

## Contribution Snapshot
- **Dominant contribution:** statistical claim governance via blocking admissibility gates, headlined by
  the **cross-cohort replication gate**; reduces false-confirmed rate while preserving known effects.
- **Optional supporting contribution:** NeuroDecide-Bench-lite (validity-under-adversarial-nulls + abstention).
- **Adoption moat (engineering):** cross-cohort phenotype/derivative alignment + open claim-contract schema.

## What to run next
1. Build B0 infra (schema → executor+provenance → replication/ComBat harness).
2. Mount data; run pilots P1–P3 (confirm / abstain / fragile sanity).
3. Run B1 must-win (false-confirmed rate vs execution-valid runner). Decision gate before scaling.
