# Paper review round 1 (via anthropic:claude-sonnet-4-6)

## Review of CONFIRM: A Claim-Governance Layer for Agentic Neuroimaging

---

### (1) Summary

The paper presents CONFIRM, a modular governance layer for agentic neuroimaging systems that separates LLM-drafted "frozen claim contracts" from code-computed statistics, adjudicating claims through a sequential gate ladder (multiplicity/search-provenance, confound completeness, power, multiverse stability, cross-cohort replication). The system emits `confirmed` only when all gates pass, otherwise returning typed abstentions. Evaluated on 13 cohorts across Alzheimer's, schizophrenia, and aging domains, the authors report zero observed false confirmations on a 177-claim adversarial stress suite (95% CI upper bound 3.1%), full recall on known positives, and stable verdicts across six LLMs. The paper explicitly scopes its contribution to faithfulness (lower false-confirmation rate at fixed recall), not capability, and frames the FCR bound as an internal stress-test result rather than a real-world guarantee.

---

### (2) CRITICAL Issues (blocking acceptance)

**C1. Circular benchmark construction — the fundamental validity threat.**
The adversarial null suite (150/177 claims) was generated to target "exactly the failure modes the gates were designed to catch" (Section 4, Limitations). This is not adversarial evaluation; it is confirmation that the gates catch what they were built to catch. A genuine adversarial suite would include failure modes *not* anticipated by the designers. The 3.1% upper bound is therefore vacuous as a measure of real-world faithfulness — the paper acknowledges this in Limitations but then continues to foreground the bound in the Abstract, Introduction, and Conclusion as the headline result. The circularity is not merely a limitation; it undermines the primary empirical claim.

**C2. Benchmark scale is critically insufficient for the claims made.**
The known-positive set is 10–11 claims; the known-null set (excluding synthetic adversarials) is 27–29 claims. At these counts, the Clopper-Pearson intervals are so wide as to be nearly uninformative (e.g., 0/27 → [0.000, 0.128]). The paper uses exact intervals correctly, but the intervals themselves reveal that the evidence base is too thin to support the framing. TPR = 10/10 with a 95% CI of [0.69, 1.00] means the paper cannot distinguish "perfect recall" from "misses 30% of positives." This is not a minor limitation; it is a fundamental constraint on what can be concluded.

**C3. Single external baseline with no access transparency.**
The head-to-head comparison (E2) uses only NeuroClaw, and the paper acknowledges "only one was runnable on our infrastructure." The retrofit result (FCR 0.333→0.000 on 15 claims) is the paper's strongest practical claim, but it rests on 5 false confirmations converted to abstentions — a count too small to generalize. The paper cannot claim CONFIRM "beats a runnable agent" in any statistically meaningful sense at n=15.

**C4. The confound-completeness gate is underspecified to the point of non-reproducibility.**
The gate checks whether a structural confound "is present in the data with more than one level, is associated with the grouping variable, and is absent from the contracted covariates." The association threshold, the test used, the correction applied, and the definition of "structural confound" (is age a structural confound? IQ?) are not specified. Without these, the gate cannot be independently implemented or its behavior predicted on new data.

---

### (3) MAJOR Issues

**M1. The "numeric guard" is described but not validated as a standalone component.**
The paper reports the guard "strips 40 LLM-introduced numbers across the sweep" (E3) but provides no analysis of false positives (legitimate numbers stripped) or false negatives (hallucinated numbers that passed). Without a precision/recall characterization of the guard itself, the claim that "faithfulness is a property of the architecture" is unsubstantiated — a leaky guard would undermine the entire design.

**M2. Cross-cohort replication gate design choices are unjustified.**
The paper adopts "sign-and-significance" rather than CI overlap for replication, justifying this by saying CI overlap "spuriously rejects strong, same-signed effects that differ only in scale." This is a reasonable position, but sign-and-significance is also known to be liberal (two studies can both reach p<0.05 with opposite-direction point estimates if CIs are wide). The threshold choice directly determines FCR and TPR, and the paper does not report sensitivity to this choice — which is ironic given that multiverse stability is a gate.

**M3. The multiverse gate's "pre-declared set of defensible analytic choices" is not described.**
What choices are included? How many? Who declared them and when? The paper cites Steegen et al. but provides no operationalization. This is a core gate and its implementation is opaque.

**M4. ComBat harmonization applied to replication cohort raises methodological concerns.**
ComBat removes site/scanner variance, but if the effect of interest is correlated with site (e.g., a patient group recruited at one site), ComBat can attenuate or eliminate real effects. The paper does not discuss this risk or report sensitivity analyses.

**M5. LLM contract drafting failure modes are not characterized.**
The paper reports "draft-success" rates in E3 but does not define what constitutes a failed draft, how failures are handled, or what fraction of real-world questions would produce valid contracts. If the LLM systematically mis-specifies estimands for certain question types, the gate ladder operates on a flawed foundation regardless of its internal validity.

**M6. The paper does not compare against simpler baselines.**
A pre-registration checklist, a human statistician review, or even a rule-based confound checker would be natural comparators. The only comparison is against NeuroClaw (execution-integrity only), which is the weakest possible baseline. The paper cannot establish that the complexity of CONFIRM is necessary.

---

### (4) MINOR Issues

**m1.** The architecture figure (TikZ) is functional but cramped; gate labels are truncated and will be illegible in print at column width.

**m2.** "cross-cohort non-replication (3)" in the adversarial suite — only 3 claims in this family makes the 0/3 result essentially uninformative. This should be flagged more prominently than it is.

**m3.** The paper refers to `tab_positives.tex`, `tab_gate_ladder.tex`, and `tab_negatives.tex` as `\input` commands but these tables are not included in the submission text provided for review. Reviewers cannot evaluate the primary quantitative results.

**m4.** "verdict agreement is 7/9 on the shared claims" (E3) — the denominator 9 is never explained. Why 9 claims for the multi-LLM sweep when the benchmark has 10+ positives?

**m5.** The paper uses "faithfulness" in a domain-specific sense (low FCR) that conflicts with the NLP/hallucination literature's use of the same term. A brief disambiguation would help readers from both communities.

**m6.** Section 5.5 (E5) describes the single false confirmation as a "mis-constructed negative" but counts it anyway. This is honest, but the paper should also report whether the power gate threshold was set before or after observing this case.

**m7.** The conclusion states CONFIRM "beats a runnable agent both head-to-head and as a retrofit layer" — "beats" is too strong given the sample sizes involved.

---

### (5) Honesty Check: Residual Overclaims

The paper's calibrated framing is largely maintained and is genuinely better than most systems papers. However, several statements outrun their evidence:

**Overclaim 1 (Abstract):** *"CONFIRM recovers known-positive effects at full recall while abstaining on confounded, p-fished, and non-replicating nulls, with no observed false confirmations over 177 adversarial stress claims."*
The 177 claims are synthetic, gate-targeted nulls. Presenting this as a unified headline result without the circularity caveat in the abstract itself is misleading. The caveat appears only in Limitations.

**Overclaim 2 (Introduction, contributions bullet):** *"Evidence that governance improves faithfulness: against a runnable agent, CONFIRM matches positive recall while showing no observed false confirmations, both head-to-head and as a retrofit layer."*
The head-to-head is on a shared claim set of unspecified size
