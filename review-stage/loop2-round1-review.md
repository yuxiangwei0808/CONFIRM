# Loop2-Round1 external review (via openai:gpt-5.5)

## 1. Re-score: NeurIPS/ICML 1–10

**Current score: ~7.3 / 10.**  
Previously I would have called this borderline-positive but not ready because two central claims were under-demonstrated: agentic LLM workflow and modular retrofit. The new evidence **does move it materially toward “ready”**, especially the NeuroClaw retrofit.

My updated view:

- **Core scientific contribution:** still strong. The claim-governance framing is useful and addresses a real failure mode in LLM-for-science systems: fluent overclaiming from noisy analyses.
- **External adjudication / false-confirm reduction:** still the strongest part. The **0/27 FCR** and **0/15 FCR after retrofitting NeuroClaw** are compelling, even with wide CIs.
- **Agentic LLM demonstration:** now present, but still small and somewhat fragile.
- **Engineering maturity:** not quite there. The sex/aging execution errors and haiku confound miss are not cosmetic; they expose brittle parts of the pipeline.

So: **better than before, not a slam-dunk.**  
If the paper is written honestly and the bugs are fixed before submission, I’d be in **weak accept / borderline accept territory** for NeurIPS/ICML. Without fixing them, I’d stay borderline.

---

## 2. Is the multi-LLM result sufficient LLM-robustness evidence?

**Partially sufficient, not fully.**

The new result answers the prior concern in the narrow sense: you are no longer only evaluating programmatic contracts. You actually ran the full LLM-draft → execute → interpret loop across multiple models. That is important.

But I would not oversell it as “LLM-robustness” yet.

### What is convincing

- **6 models, 3 providers, open/closed, small/large** is a reasonable model diversity set.
- **Valid contracts 8–9/9** suggests the contract format is learnable by LLMs.
- **7/8 unanimous gate-verdict agreement** is a good sign.
- The **anti-hallucination guard firing 42 times** is excellent evidence that naive LLM interpretation is unsafe and that your guard is doing real work.

That last point is actually one of the strongest results. It shows the guard is not decorative.

### What is not yet convincing

The evaluation is still only **9 curated claims**. That is tiny. Worse, several claims are from already-known positive domains: AD, schizophrenia, aging. The true stress test is not “can models draft contracts for canonical neuroimaging findings”; it is:

- ambiguous questions,
- missing confounders,
- poorly posed estimands,
- multiple possible outcomes,
- nonstandard covariates,
- site/scanner interactions,
- weak effects,
- subgroup analyses,
- adversarially phrased or fishing-like prompts.

Your one serious stress case — the injected site-confound null — produced the most damaging failure. That matters.

### Do I need a formal agreement statistic?

A formal statistic would help, but it is not the main missing piece.

You can report:

- Fleiss’ κ or Krippendorff’s α for gate verdicts,
- pairwise agreement matrix,
- agreement conditional on valid contracts,
- agreement including execution failures as a separate category,
- estimand-match κ if human-rated.

But with **n=8 valid verdicts**, formal agreement statistics will look performative. The denominator is too small. The minimum useful addition is not just κ; it is **more claims**.

### Minimum I would want

Before submission, I would want either:

1. **Increase to ~30–50 claims**, with prespecified categories:
   - known positives,
   - known nulls,
   - confounded nulls,
   - underpowered claims,
   - multiverse-sensitive claims,
   - ambiguous natural-language prompts,
   - duplicate/rephrased questions.

or, if time is limited:

2. Keep the 9-claim suite but add a **targeted adversarial suite** of maybe 10–15 prompt variants focused on confound omission, predictor/covariate collision, multiple outcomes, and vague estimands.

Right now, the multi-LLM result is a **credible pilot robustness demonstration**, not a definitive robustness result.

---

## 3. How damaging are the honest wrinkles?

### 3a. Haiku omitted the confound and false-confirmed a null

This is **damaging** because it directly attacks the central promise: the governance layer prevents false confirmation. In this case, a weaker LLM produced a flawed contract, the gates executed faithfully, and the system returned a false confirmation.

That means the pipeline is not robust to **contract misspecification**. This is not a minor edge case. In scientific workflows, the dangerous failures are exactly omitted confounds, wrong covariate sets, and silently changed estimands.

Your defense cannot be: “the gates worked given the contract.” That is true but insufficient if the system is sold as agentic or end-to-end.

### Does it strengthen the “frozen/human-reviewed contract” argument?

Yes, but only if you frame the system correctly.

It strengthens the argument that:

> The ClaimContract should be treated as a frozen, reviewable scientific object, not as disposable internal LLM text.

It supports human-in-the-loop governance. It does **not** support a fully autonomous LLM scientist claim.

If the paper says:

> “CONFIRM is an autonomous agent that reliably turns natural-language questions into valid scientific adjudications,”

then the haiku failure is bad.

If the paper says:

> “CONFIRM separates hypothesis specification from execution and requires the ClaimContract to be frozen, inspectable, and optionally human-reviewed; our multi-LLM results show why this separation is necessary,”

then the haiku failure becomes a useful case study.

### Minimum fix for haiku failure

You need at least one of the following:

- A **semantic contract validator** that detects when a known confound field exists and is omitted for site-sensitive claims.
- A **metadata-aware covariate recommender** that forces site/scanner/batch into candidate confounds when available.
- A **contract review checklist** with explicit “confound completeness” validation.
- A **two-model drafting + critique protocol**, where one LLM drafts and another audits omitted covariates before execution.
- A **required abstention rule**: if the natural-language question or dataset contains site/scanner/batch and the contract omits it, adjudication cannot be “confirmed.”

You cannot leave this as merely an “honest wrinkle” without mitigation.

---

### 3b. Sex claim execution error

This must be fixed before submission.

A predictor also appearing as a covariate is a basic design-matrix error. It is exactly the kind of static validation your system should catch before execution.

Minimum fix:

- Add a schema/static check: **predictor ∩ covariates must be empty**, unless explicitly modeled through interaction terms.
- Add design-matrix rank checks before model fitting.
- Return a clean gate verdict such as `invalid_contract` rather than a runtime failure.

This is not hard, but reviewers will punish it if left unfixed.

---

### 3c. Aging association errors for 3/6 models

Also should be fixed before submission, or at least clearly diagnosed.

You say “association+inclusion-filter fragility,” but that is vague. The paper needs to distinguish:

- LLM drafted invalid inclusion criteria,
- cohort subset became empty or too small,
- formula mismatch,
- missing variable,
- preprocessing incompatibility,
- execution harness bug.

Minimum fix:

- Convert these into deterministic validation failures with explicit error classes.
- Rerun the 6-model sweep after fixing the harness.
- Report both pre-fix and post-fix if you want to be transparent.

Execution errors are acceptable in a prototype paper only if they are handled as first-class abstentions. Runtime crashes are not acceptable for a governance system.

---

## 4. Remaining critical weaknesses, ranked, with minimum fixes

### Weakness 1: Contract misspecification can still cause false confirmation

This is the most serious issue.

The haiku confound omission shows that your gates protect against invalid evidence **conditional on the contract**, but they do not guarantee the contract corresponds to the intended scientific question.

**Minimum fix:**

- Add a contract-audit stage.
- Report estimand-match and confound-completeness as formal metrics.
- Introduce hard validators for common neuroimaging confounds: site, scanner, age, sex, motion, intracranial volume, diagnosis/site imbalance, batch/cohort.
- Rerun the site-confound null and show the false-confirm disappears.

---

### Weakness 2: The agentic evaluation is too small

Nine claims is not enough to support broad LLM-robustness claims.

**Minimum fix:**

Add a small but targeted stress suite:

- 10–15 additional claims,
- at least half null/confounded/ambiguous,
- prespecified before running,
- include prompt paraphrases,
- evaluate all 6 models,
- report valid contract rate, estimand-match, verdict agreement, execution-error rate, and false-confirm rate.

You do not necessarily need 100 claims, but 9 is thin.

---

### Weakness 3: Runtime failures undermine the “governance” story

A governance layer should fail closed. Right now, some cases fail by crashing.

**Minimum fix:**

- Replace runtime errors with explicit verdict classes:
  - `invalid_contract`,
  - `singular_design`,
  - `empty_subset`,
  - `missing_variable`,
  - `underpowered`,
  - `unsupported_estimand`.
- Add static validation before execution.
- Rerun the multi-LLM sweep and report the corrected failure modes.

The system does not need to confirm every claim, but it must not behave unpredictably.

---

### Weakness 4: NeuroClaw retrofit is compelling but small and possibly tailored

The retrofit result is probably the cleanest new evidence:

- NeuroClaw alone: FCR 5/15.
- NeuroClaw + CONFIRM: FCR 0/15.
- TPR preserved at 9/10.

That is strong. But reviewers will ask whether those 25 claims are enough and whether the same null types were already anticipated by CONFIRM’s gates.

**Minimum fix:**

- Clearly state whether the 25-claim set was fixed before the retrofit.
- Provide all NeuroClaw outputs, CONFIRM contracts, gates, and final adjudications.
- Add a small held-out retrofit set if possible.
- Report which exact gates converted the 5 false-confirms to abstentions.

You want to avoid the appearance that CONFIRM was tuned to NeuroClaw’s known failures.

---

### Weakness 5: Confidence intervals remain wide

Your key FCR numbers are excellent but small-sample.

Examples:

- MAIN FCR 0/27 with upper CI ~0.128.
- Retrofit FCR 0/15 with upper CI ~0.22.

Those are promising but not definitive.

**Minimum fix:**

Be honest in wording. Say:

> “In these adjudicated benchmarks, CONFIRM eliminated observed false confirmations.”

Do not say:

> “CONFIRM guarantees low false-confirmation rates.”

A few more nulls would help more than a few more positives.

---

### Weakness 6: The paper risks overclaiming “LLM safety”

The anti-hallucination guard is valuable, but it only controls numeric prose hallucination after execution. It does not solve:

- wrong estimand,
- omitted confounds,
- wrong cohort,
- variable leakage,
- inappropriate model family,
- invalid multiple-testing scope.

**Minimum fix:**

Separate claims clearly:

- **Numeric anti-hallucination:** strong evidence.
- **Gate-verdict robustness to model choice:** preliminary evidence.
- **Contract correctness:** still requires validation/human review.
- **Autonomous scientific reliability:** not established.

---

## 5. Ready for submission?

**Almost.**

Not “No,” because the two prior gaps are now substantially addressed. The modular retrofit especially strengthens the paper.

But not clean “Yes” yet, because the current version contains avoidable failures that reviewers will latch onto:

1. A weaker LLM produced a confound-omitting contract and false-confirmed a null.
2. A basic predictor/covariate collision caused execution failure across all models.
3. Aging claim failures occurred for half the models.
4. The multi-LLM suite is small.

### Minimum before submission

I would fix the following before sending to NeurIPS/ICML:

- Add static contract validation.
- Fix predictor/covariate collision.
- Fix or explicitly classify aging execution failures.
- Add a confound-completeness audit rule.
- Rerun the 6-model sweep.
- Report execution failures as governance outcomes, not crashes.
- Avoid claiming full autonomy; frame ClaimContracts as frozen, inspectable, and optionally human-reviewed scientific objects.

If those are fixed, I would score it around **7.5–7.8** depending on writing and transparency. As-is, **~7.3**, borderline-positive but still vulnerable.
