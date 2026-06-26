# Loop2-Round2 FINAL external review (via openai:gpt-5.5)

## 1. Final score

**7.5 / 10 — weak accept / ready-for-submission, not a strong accept.**

Yes, the contract-validation layer meaningfully moves this from **“almost”** to **“ready”**. The prior version had a bad failure mode: LLM contract variation could produce crashes or, worse, a false confirmation on a confounded null. Your fixes address that directly:

- invalid contracts are caught;
- duplicate predictors/covariates no longer crash;
- structural confound incompleteness forces abstention;
- execution failures become governed abstentions;
- the previous LLM-specific false-confirm is eliminated across all 6 models.

That is a real improvement, not cosmetic.

But this is still not an 8+ paper for me because the empirical base remains narrow and the statistical safety claims are underpowered. The method is promising and now much less brittle, but the evaluation is still too small to support broad claims like “robust agentic neuroimaging governance” without careful qualification.

My likely review stance:

> **Weak accept if the paper is honest, releases code/contracts/logs, and avoids overclaiming. Borderline if it markets itself as generally solving LLM scientific hallucination or neuroimaging reproducibility.**

---

## 2. Is the LLM-robustness story strong enough?

**Strong enough for a constrained safety claim. Not strong enough for a broad robustness claim.**

The valid claim is:

> Across 6 LLMs and 9 curated claims, the governance layer eliminated the observed safety-critical false-confirmation failure mode: all models abstained on the site-confounded null via `confound_incomplete`; residual LLM variation affected recall on a borderline positive claim, not false confirmations.

That is a good story. It shows the contract layer is doing what it should: moving model disagreement from **false confirmation** to **abstention / fragility**.

But do not oversell it as:

> “CONFIRM is LLM-robust.”

That would be too broad. The evidence is only **9 claims × 6 models**, and the disagreements still show that the NL-to-contract stage remains semantically unstable. The robustness result is currently about **one important class of failure**, not all failures.

Best phrasing:

> “In a multi-LLM stress test, contract validation and data-aware confound-completeness auditing removed the observed LLM-dependent false-confirmation failure; remaining model dependence affected sensitivity rather than false confirmation.”

That is defensible.

---

## 3. Remaining critical weaknesses, ranked, with minimum fixes

### 1. The false-confirmation rate is still statistically weakly bounded

You have **0 observed false confirms**, which is good, but the CIs are still wide:

- MAIN: 0/27, upper CI around 0.13.
- Layer setting: 0/15, upper CI around 0.22.

That is not tight enough for a strong safety claim.

**Minimum fix:**  
Add more negative/null claims, especially realistic confounded nulls. If you want an upper bound near 5% under the rule-of-three heuristic, you need roughly **60 null/negative claims with 0 false confirms**. Ideally include:

- site-confounded nulls;
- scanner/field-strength confounds;
- sex/age imbalance;
- cohort-specific artifacts;
- plausible but biologically false associations;
- negative controls across AD, schizophrenia, aging, and at least one additional domain.

Without this, you can claim “no observed false confirms,” not “low FCR.”

---

### 2. LLM contract drafting remains a recall bottleneck

The aging-positive disagreement matters. You are right that it is less dangerous than false confirmation, but it still means scientific conclusions depend on how the LLM drafts the contract.

If one model produces `confirmed` and another produces `fragile` for the same NL question, the governance layer is not fully specification-invariant.

**Minimum fix:**  
Add a deterministic or semi-deterministic contract canonicalization step. For example:

- map NL claims into a constrained ontology of predictors, outcomes, covariates, cohorts, and tests;
- require contract equivalence checks across multiple drafts;
- if semantically non-equivalent contracts are generated, default to `ambiguous_contract` or run all valid contracts as a multiverse;
- report per-claim contract variance explicitly.

The key is to make it impossible for small LLM drafting differences to silently change the estimand.

---

### 3. Benchmark breadth is still limited

The confirmed-positive domains are useful, but still narrow:

- AD structural MRI dominates the story;
- schizophrenia FC is one functional connectivity case;
- aging hippocampus is borderline;
- only one real runnable baseline, NeuroClaw.

This is not yet a broad neuroimaging validation.

**Minimum fix:**  
Add at least one or two additional datasets/domains with different failure modes. Good candidates:

- UK Biobank aging/brain-volume associations;
- HCP behavioral/FC associations;
- ABCD developmental claims;
- Parkinson’s or MS structural claims;
- task-fMRI activation claims if feasible.

You do not need a massive benchmark, but you need enough diversity that the result does not look ADNI/OASIS-tuned.

---

### 4. Confound-completeness audit is necessary but probably incomplete

Your current rule catches structural confounds **if present in the data**, have >1 level, are associated with grouping, and are omitted from covariates.

That is good, but it leaves a major hole:

> If the confound column is missing, unavailable, renamed, or not declared in the schema, the system may not know it is incomplete.

This is especially important for site/scanner/field strength/software version.

**Minimum fix:**  
Add a dataset-level required-confound schema. For each dataset/modality, define required metadata fields. If required fields are absent, the system should fail closed:

- `metadata_incomplete`;
- `cannot_assess_confound_completeness`;
- no confirmation allowed.

This would strengthen the claim substantially.

---

### 5. Baseline comparison is still thin

NeuroClaw is a useful baseline, and the modular-layer experiment is compelling. But one runnable baseline plus an execution-only ablation is not enough for a very strong systems paper.

**Minimum fix:**  
Include stronger ablations:

- CONFIRM without confound-completeness;
- CONFIRM without multiplicity/search provenance;
- CONFIRM without cross-cohort replication;
- CONFIRM without anti-hallucination number replacement;
- NeuroClaw + only execution;
- NeuroClaw + execution + partial gates;
- full NeuroClaw + CONFIRM.

This would isolate which governance components actually reduce false confirmation.

---

### 6. Anti-hallucination guard firing 40× is both a strength and a warning

The fact that prose numbers were stripped to executed values 40 times supports your premise: LLMs hallucinate or drift numerically.

But it also means the drafting layer is not reliable. The paper should not sound like the LLM is doing scientific reasoning. The LLM is proposing a contract; the system is adjudicating.

**Minimum fix:**  
Report examples of stripped hallucinated numbers and show that final manuscripts/results contain only executed values. Make this a central safety argument, not a footnote.

---

## 4. Ready for submission?

**Yes — ready for submission, but as a 7.5-ish paper, not a clean 8+.**

The implemented fixes address the previous minimum blockers. The work is now defensible as a claim-governance layer that makes agentic neuroimaging substantially safer by converting many LLM/model failures into abstentions.

The paper must be careful to say:

- “no observed false confirms,” not “guaranteed low FCR”;
- “robust across six LLMs on this curated stress test,” not “LLM-robust in general”;
- “governance layer for claim adjudication,” not “automated neuroimaging discovery system.”

If you wanted to move from **7.5** to **8+**, the single most important remaining item would be:

> **Expand the null/negative/confounded benchmark enough to tighten the false-confirmation-rate bound.**

Right now the core safety result is promising but statistically loose. More negatives would make the safety claim much harder to dismiss.
