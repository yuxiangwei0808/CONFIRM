# Paper review round 2 (via openai:gpt-5.5)

## Bottom line

The revisions improve the tone, but the paper still has serious soundness and evidence-matching problems. The biggest new/remaining issue is a direct contradiction in the central quantitative claim: the abstract/conclusion/limitations say “no observed false confirmations over 177,” while E4/E5 report **1/177 false confirmations**. That is not a wording nit; it invalidates the headline as currently written.

If the WACV template/example sections shown at the end are actually in the compiled submission, this is an administrative/formatting disaster and likely an immediate reject/desk-reject. I assume they are accidental in the prompt, but if not, nothing else matters.

---

# (1) Remaining major issues

## Major 1: Headline FCR claim is internally inconsistent and statistically wrong as phrased

The paper repeatedly claims zero false confirmations over 177 claims:

> “with no observed false confirmations over 177 synthetic, gate-targeted stress claims”

> “The headline figure---no observed false confirmations, with a 3.1% exact upper bound over 177 claims---”

> “showing no observed false confirmations on a 177-claim synthetic, gate-targeted stress suite”

But E4 says:

> “The full ladder yields a combined FCR of 0.006 (1/177), exact 95% CI [0.000,0.031]”

and E5 explains the one false confirmation.

You cannot both have **0/177** and **1/177**. The 3.1% upper bound corresponds to roughly the **1/177** result, not “no observed.” For **0/177**, the upper bound would be about 2.1%, not 3.1%. This is a central claim, not a minor typo. The abstract, limitations, and conclusion are currently misleading.

A defensible phrasing would be: “one observed false confirmation over 177 intended negatives, traced to a misconstructed negative; exact 95% upper bound 3.1%.” But then the “no observed false confirmations” claim must be removed everywhere.

---

## Major 2: The benchmark remains too small and too tailored to support the claimed operating characteristics

The main real benchmark is tiny:

- Main negatives: 27
- Full negatives: 29
- Known positives: 10 or 11
- Runnable-agent comparison: 15 negatives / 10 positives
- Multi-LLM shared set: 9 claims

The 177 denominator is mostly synthetic, gate-targeted claims:

> “synthetic known-null and fragile claims … in five families that each target a specific gate”

This is useful engineering validation, but it is not enough to estimate real-world faithfulness. The paper now admits this, which is good, but many result statements still lean on the 177 count as if it were a meaningful broad stress test. It is not. It tests whether gates catch the failure modes they were built for.

Also, the confidence intervals assume independent Bernoulli trials, but these claims are likely highly correlated: same cohorts, same modalities, similar constructed mechanisms, repeated gate families. Treating 177 synthetic claims as 177 independent draws overstates the effective evidence.

---

## Major 3: “Known null” and “truth status is known” are still overclaimed

The benchmark section says:

> “Evaluating governance requires claims whose truth status is known independently of the agent.”

and:

> “known null (no credible effect, or a relationship engineered to be an artifact)”

For many neuroimaging associations, “no credible effect” is not ground truth. Absence of strong literature support is not a known null. Engineered artifacts are different from naturally occurring false claims. The paper mixes literature-anchored positives, engineered negatives, fragile cases, and “no credible effect” claims under a truth-status framing that is still too strong.

A safer framing would be “externally motivated labels” or “adjudicated benchmark labels,” not “truth status is known.”

---

## Major 4: The method is under-specified for a paper claiming an auditable governance layer

Several gates are described conceptually but not with enough operational detail to reproduce or judge them.

Examples:

### Power gate

> “checked against an achieved-power threshold”

What threshold? Based on what effect size? Contracted effect? Literature effect? Observed effect? E5 says:

> “Cohen’s d ≈ 0.63, achieved power ≈ 0.99”

This strongly suggests post-hoc/observed-effect power may be involved, which is statistically dubious and often circular. If the power gate uses observed effect size, it can pass exactly the kinds of lucky inflated effects that underpowered studies produce. If it uses a pre-specified minimal effect size, that needs to be explicit.

### Multiverse gate

> “pre-declared set of defensible analytic choices”

Which choices? How many? How were they chosen? Are they fixed per domain or per claim? What constitutes a sign/significance flip? Is “significance” corrected inside each multiverse? This is a major source of researcher degrees of freedom, yet the paper gives only prose.

### Replication gate

> “same sign and reach independent significance”

At what alpha? One-sided or two-sided? Is multiplicity accounted for across replication cohorts, regions, edges, and domains? What happens with multiple replication cohorts: all must pass, one must pass, meta-analysis must pass? The schizophrenia borderline case implies a threshold, but it is not specified.

### Multiplicity/search provenance

> “effective family, defined as the maximum of the declared comparisons and the comparisons actually searched”

How is “comparisons actually searched” determined for an external runnable agent? From logs? From code static analysis? From output tables? If provenance is absent, do you always abstain? This is central to the novelty claim, but the mechanism is vague.

---

## Major 5: Confound-completeness gate is too narrow and statistically brittle

The revision specifies:

> “site, scanner, field strength, or processing-software version, drawn from a fixed pre-declared list”

and association with grouping variable by:

> “χ² or one-way ANOVA test at the 0.05 level”

This helps specificity, but creates new problems.

1. The fixed list excludes major neuroimaging confounds: age, sex, motion, medication, illness duration, handedness, intracranial volume, acquisition protocol details, diagnosis/site interactions, QC/failure rates, socioeconomic variables, etc.

2. A variable associated with group is not necessarily a confound unless it is also related to the outcome/measurement. The gate detects imbalance, not confounding.

3. A p=0.05 uncorrected association screen is arbitrary. In small samples it misses important imbalances; in large samples it flags trivial ones. No effect-size threshold is mentioned.

4. Multiple structural confounds are tested, but no correction or hierarchical rule is described.

5. The gate can abstain due to site imbalance even when harmonization or stratified modeling would be more appropriate. That may be fine for conservative governance, but then the paper should not present this as “statistical admissibility” broadly.

---

## Major 6: The runnable-agent comparison remains weak

The paper says:

> “against a runnable agent, CONFIRM matches positive recall while showing no observed false confirmations”

But this is on a very small shared set:

- NeuroClaw: 9/10 TPR, 5/15 FCR
- CONFIRM: 10/10 TPR, 0/15 FCR
- Retrofit: preserves 9/10 and removes 5 false confirms

This is suggestive but not strong evidence. The failures are all:

> “five site-confounded nulls”

That sounds like a narrow case where CONFIRM’s site/confound gate is directly targeted. There are no simpler governance baselines: e.g., “always require replication,” “rule-based site covariate checker,” “preregistration checklist,” “BH correction + site covariate,” etc. The paper admits this in limitations, but the main text still frames the result as “governance improves faithfulness” too generally.

---

## Major 7: Multi-LLM stability claim is not supported

E3 says:

> “cross-model verdict agreement is 7/9 on the nine shared benchmark claims”

Then concludes:

> “Faithfulness is thus a property of the architecture, not of any one model.”

7/9 agreement is not strong stability. Two disagreements out of nine is substantial. The paper says disagreement is confined to “borderline-positive recall rather than false confirmations,” but that still means the LLM affects whether true claims are confirmed. If the contract drafting changes the estimand/covariates/cohorts enough to alter positives, then the architecture is not fully model-invariant.

Also, the “9 shared benchmark claims” denominator remains under-explained in the provided text. Why 9? Out of 39? Out of the 177? Were these selected because all models could run them? Were failures excluded? This needs a clear denominator accounting.

---

## Major 8: The contribution may be more of a statistical checklist/workflow than a WACV-level technical contribution

The gates are conventional: multiplicity correction, confound checking, power, multiverse, replication. The paper’s novelty is packaging these into an LLM-agent governance layer. That is potentially useful for applications, but the technical novelty is modest. Claims like:

> “Two gates are, to our knowledge, new in this setting”

are plausible only in the narrow “agentic neuroimaging” setting, not methodologically. Search-provenance-aware multiplicity and confound audits are not new statistical ideas.

For WACV applications track this may still be acceptable if the evaluation is compelling. Currently, the evaluation is too small, too synthetic, and too gate-targeted.

---

# Remaining minor issues

## Minor 1: “Execution integrity” is somewhat straw-manned

The paper repeatedly implies existing agents validate only execution integrity:

> “they validate execution integrity … rather than whether a finding is statistically admissible”

This may be broadly true for the cited systems, but the claim needs careful support. Some agentic or provenance systems may include statistical sanity checks. The related work should distinguish what each baseline actually does instead of treating them uniformly.

---

## Minor 2: “Statistically admissible” is an overloaded term

The paper uses “statistically admissible” as if passing CONFIRM’s gates is equivalent to admissibility. But admissibility depends on design, scientific context, measurement validity, causal assumptions, model assumptions, preprocessing, and domain-specific norms. CONFIRM’s gates are a conservative checklist, not a universal admissibility criterion.

---

## Minor 3: Numeric guard claim needs precision

The paper says:

> “The LLM never emits numbers; a numeric guard strips any it introduces.”

This is internally awkward. The LLM does emit numbers; the final system strips unauthorized ones. Say “the final report cannot contain unauthorized numbers” or “the LLM is not authorized to emit statistics.”

Also, catching 40 introduced numbers is not meaningful without knowing whether they were hallucinated, legitimate but unrecorded, dates/model names, citations, cohort counts, etc.

---

## Minor 4: Cross-cohort ComBat usage is underspecified

> “After ComBat harmonization of the replication cohort”

ComBat can remove or attenuate biological signal if batch/site and biological variables are entangled. How are biological covariates protected? Is ComBat fit only on replication? Is there discovery/replication leakage? Are batch levels sufficient? This is not a detail; it can change replication outcomes.

---

## Minor 5: Coverage loss needs clearer interpretation

Coverage drops to 0.300 on the main subset. That is a severe abstention rate. This may be intended, but the paper should more explicitly discuss practical usability: if 70% of claims are not confirmed, what workflow burden does this impose? Are abstentions actionable? How often do abstentions reflect fixable design flaws versus conservative false negatives?

---

## Minor 6: Related work is incomplete

Missing or underdeveloped areas:

- Preregistration/statistical analysis plans.
- BIDS/fMRIPrep/Nipype-style reproducible neuroimaging workflows.
- Neuroimaging quality control and confound modeling literature.
- Replication/meta-analysis standards in neuroimaging.
- Prior benchmark/governance/checklist approaches outside LLM agents.
- Statistical auditing/provenance systems.

---

# (2) New problems or inconsistencies introduced by the revisions

## New/increased problem 1: Central 177-claim result is now contradictory

The revisions appear to have inserted the “no observed false confirmations over 177” caveat into abstract/conclusion/limitations, but E4/E5 still report 1/177. This is the most serious new inconsistency.

Quoted contradictions:

> “no observed false confirmations over 177 synthetic, gate-targeted stress claims”

versus:

> “The full ladder yields a combined FCR of 0.006 (1/177)”

and:

> “The single false confirmation in E4 is neg_underpowered_hcp_s3”

This must be fixed.

---

## New/increased problem 2: The exact upper bound is attached to the wrong event count

> “no observed false confirmations … exact 95% upper bound 3.1%”

For no observed false confirmations over 177, the upper bound is not 3.1%. The 3.1% bound corresponds to the 1/177 result. This creates the impression that the authors are mixing the more favorable numerator with the less favorable confidence interval.

---

## New/increased problem 3: The confound gate is now more concrete but exposes statistical arbitrariness

Specifying χ²/ANOVA at p=0.05 improves reproducibility, but also reveals an uncorrected, thresholded imbalance screen. This is brittle and underjustified. The fixed confound list also narrows the claim substantially.

---

## New/increased problem 4: The “9-claim multi-LLM denominator” is still not adequately explained

The paper says:

> “cross-model verdict agreement is 7/9 on the nine shared benchmark claims”

But why nine? Which nine? How selected? Were failed runs excluded? Are these positives, negatives, fragile claims? The text still does not provide enough denominator accounting.

---

## New/increased problem 5: The paper now says “small shared set” in some places but not consistently enough

The conclusion says:

> “on a small shared set”

Good. But the abstract still says:

> “As a modular layer over a runnable agent it converts that agent’s false confirmations to abstentions while preserving its positive recall”

This is technically true but lacks the “small shared set” qualifier in the same sentence. Given the tiny denominator, the qualifier should appear every time the result is summarized.

---

# (3) Honesty check: residual statements that outrun evidence

Below are statements I would require to be softened or fixed.

## False/inconsistent

> “no observed false confirmations over 177 synthetic, gate-targeted stress claims”

False given E4/E5. Should be “one observed false confirmation over 177 intended negatives.”

---

> “The headline figure---no observed false confirmations, with a 3.1% exact upper bound over 177 claims”

Also false/inconsistent. No observed false confirmations over 177 would not give 3.1%.

---

> “showing no observed false confirmations on a 177-claim synthetic, gate-targeted stress suite”

Contradicted by 1/177.

---

## Overstated

> “truth status is known independently of the agent”

Too strong. Labels are externally anchored/adjudicated, not known truth.

---

> “known null (no credible effect...)”

“No credible effect” is not a known null. It is at best a literature-unsupported or low-prior claim.

---

> “Faithfulness is thus a property of the architecture, not of any one model.”

Not supported by 7/9 agreement over nine claims. Better: “false confirmations were stable across this small multi-LLM sweep.”

---

> “its verdicts are stable across six LLMs”

7/9 agreement is not “stable” without qualification. Say “mostly stable on a nine-claim subset, with disagreements on borderline positives.”

---

> “CONFIRM recovers known-positive effects at full recall”

Based on 10/10 or 11/11 mostly canonical effects. Should always be qualified by the small benchmark and wide uncertainty.

---

> “statistically admissible”

The gates encode one conservative admissibility policy. They do not establish statistical admissibility in general.

---

> “strong internal stress-test validation”

Maybe acceptable in limitations, but it still sounds promotional. Given the suite is gate-targeted and synthetic, “targeted internal validation” is more accurate.

---

> “The strongest test of a governor is a large adversarial null set.”

Not necessarily. A large synthetic gate-targeted null set is not stronger than an external preregistered benchmark of real claims. Better: “A useful internal stress test…”

---

# (4) Final score and recommendation

## Score: 4 / 10

## Recommendation: Reject

The idea is useful and the revised framing is more honest than before, but the paper still falls short for WACV acceptance. The main reasons are:

1. The central 177-claim FCR claim is internally contradictory.
2. The real benchmark is very small.
3. The synthetic stress suite is gate-targeted and not independent enough to support broad operating-characteristic claims.
4. Key gates remain under-specified or statistically questionable.
5. The runnable-agent and multi-LLM evaluations are too small.
6. Several statements still overclaim stability, truth labels, and statistical admissibility.

If the authors fix the 0/177 vs 1/177 inconsistency, fully specify the gates, add simple governance baselines, and run a larger external/preregistered real-claim benchmark, this could become a solid applications-track paper. In its current form, it is not yet there.
