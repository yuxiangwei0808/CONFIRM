# Paper review round 3 (via openai:gpt-5.5)

## Bottom line

This is **substantially cleaner and more defensible** than the prior version, especially after the power-gate fix and the explicit downgrade from preregistration to frozen-configuration evaluation. The core claim—**a conservative governance layer reduces false confirmations by abstaining, not by improving discovery capability**—is now mostly aligned with the evidence.

But there are still several important issues. The biggest are:

1. **The 150 vs 177 stress-suite accounting is internally muddled.**
2. **The ds000030 section is not yet clean: the “power gate satisfied at n≈127 for d=0.3” claim looks statistically suspect, and the verdict labels are inconsistent.**
3. **“Statistically admissible,” “robust,” and “trustworthy discovery” still oversell what is actually a heuristic conservative claim filter.**
4. **The external validation remains initial and partly procedural-null-based, not field-level evidence of real-world specificity.**
5. **If the WACV template boilerplate is actually in the submitted PDF, the paper is dead on arrival.**

My likely score as a WACV applications reviewer: **6/10, borderline / weak accept only if the inconsistencies are fixed.** As currently written, I would lean **borderline reject** because the remaining wording and accounting problems are exactly the kind that undermine a paper about claim governance.

---

# 1. Internal consistency audit

## Major inconsistency: 150 synthetic vs 177 stress suite

This is the most obvious numerical/accounting problem.

In the benchmark section:

> “we generate 150 synthetic known-null and fragile claims … Combined with the adjudicated negatives from the main benchmark, this gives 177 negative claims.”

So:  
- **150 = synthetic gate-targeted stress claims**  
- **177 = 150 synthetic + 27 main adjudicated negatives**

But elsewhere the paper repeatedly says:

> “177-claim synthetic, gate-targeted stress suite”

Examples:
- Abstract: “across a 177-claim synthetic, gate-targeted stress suite”
- Intro contribution: “plus a 177-claim synthetic, gate-targeted stress suite”
- E4 table: “Stress suite … procedural nulls … 177 null”
- Conclusion: “across a 177-claim synthetic, gate-targeted stress suite”

This is not just wording. It changes what the bound means.

If 177 includes the 27 main benchmark negatives, then it is **not** a 177-claim synthetic suite. It is a **177-negative pooled set**, consisting of **150 synthetic gate-targeted negatives plus 27 adjudicated benchmark negatives**.

Fix everywhere:

> “150 synthetic gate-targeted stress negatives; pooled with 27 adjudicated benchmark negatives, the 177-negative set yields 0/177 false confirmations, exact 95% upper bound 2.1%.”

Do not call 177 “synthetic” unless all 177 are actually synthetic.

## Abstract currently double-counts or ambiguously stratifies 0/27 and 0/177

Abstract:

> “no false confirmations on the adjudicated benchmark (0/27) or across a 177-claim synthetic, gate-targeted stress suite”

If the 177 includes the 27, this reads as two independent strata when they are not. Say:

> “0/27 on the adjudicated benchmark negatives and 0/177 on the pooled negative set after adding 150 synthetic gate-targeted stress negatives…”

That is precise.

## “Four questions” but six experiment labels

Experiments section:

> “We ask four questions. (E1) … (E2) … (E3) … (E4) … (E6) … We also report … (E5).”

This is sloppy. Either say “six questions” or do not number it this way. For a governance paper, this kind of basic inconsistency is damaging.

## ds000030 verdict labeling is inconsistent

The ds000030 paragraph says:

> “CONFIRM confirmed nothing (0/14 positives) … the claims fail at significance, multiverse, and replication, not power.”

But also:

> “so CONFIRM withholds all 14 as fragile rather than confirm noise”

Those cannot both be true unless all 14 failed specifically at the multiverse gate. But the paragraph says:
- only **1/14** reaches discovery significance,
- **none** replicate,
- the lone significant effect is multiverse-fragile.

So the natural verdicts should be something like:
- 13/14: fail discovery/multiplicity/significance or replication, likely **non_replicated** or some “not_significant”/“not_confirmed” state.
- 1/14: **fragile** if multiverse gate precedes replication and catches it first, or **non_replicated** if replication is assessed after.

Currently the text says “all 14 as fragile,” which is unsupported. This needs correction.

Also, your method taxonomy does not explicitly include a “not significant in discovery” abstention, even though ds000030 says many fail at significance. Multiplicity gate presumably handles this, but the taxonomy lists:

> non_replicated, under_powered, fragile, unverifiable_search_provenance, confound_incomplete, execution-error

There is no clean label for “does not pass discovery significance after multiplicity.” You need either:
- add a verdict, e.g. `not_significant` / `not_admissible`, or
- define failed multiplicity as one of the existing abstentions.

Right now the gate ladder says multiplicity can abstain only for unverifiable search provenance, not for non-significance. That is a method/text inconsistency.

## ds000030 power claim looks mathematically suspicious

This sentence is a red flag:

> “The power gate is satisfied here: the design is adequately powered for the d=0.3 reference effect at n≈127.”

For a standard two-sample comparison, **n≈127 total is not 80% powered for d=0.3** at α=0.05. Even **127 per group** is typically not enough for 80% power for d=0.3; you need on the order of ~175 per group for a simple two-sample test.

If ds000030 has **n=265 total across schizophrenia, bipolar disorder, ADHD, and controls**, then each diagnostic contrast is likely much smaller than n=127 per group. So “power gate satisfied for d=0.3” is not credible without more detail.

This is serious because the prior bug was specifically about the power gate. If I were reviewing, I would immediately check this.

You need to state:
- exact group sizes per contrast,
- whether n≈127 is total, per arm, or effective sample size,
- test type,
- α level after multiplicity,
- one-sided or two-sided,
- target power threshold,
- formula/software used.

As written, I do **not** trust the ds000030 power statement.

## External positives vs “known-positive recall”

The paper says CONFIRM recovers known positives at full recall, but ds000030 has:

> “0/14 positives”

You try to frame this as “correct outcome” because the effects do not replicate in the small cohort. That may be scientifically reasonable, but then these should not be reported as ordinary “known-positive recall” cases.

There are really two different label types:

1. **Sample-detectable known positives**: effects expected to be detectable and replicable in the analyzed cohort/split. NACC AD atrophy fits this.
2. **Literature-positive but sample-insufficient effects**: ENIGMA effects that may be real in huge meta-analyses but not reliably detectable in a modest single cohort. ds000030 fits this.

Calling ds000030 “14 positives” and then saying 0/14 confirmed is “correct” creates metric confusion. It is not positive recall evidence. It is a conservatism / non-replication stress test.

Table should not list ds000030 as “14 pos” in the same conceptual category as NACC’s 9/9 positives unless you very explicitly distinguish “literature-positive but not expected-confirmable in this sample.”

## NACC positive count is under-described

The text lists:

> hippocampal, entorhinal, medial- and lateral-temporal, fusiform, and ventricular enlargement

That sounds like 6 broad effects, not obviously 9. Maybe AD and MCI variants, bilateral regions, or separate volumetric measures bring it to 9. But the paper should list the 9 claims explicitly, either in a table or appendix. For a paper built around claim accounting, every claim count should be auditable from the main text or supplement.

## Thirteen cohorts vs external cohorts

The abstract says:

> “Across Alzheimer's disease, schizophrenia, and aging on thirteen cohorts…”

Then later discusses unseen NACC and ds000030. This is probably fine if “thirteen” refers to the development/internal benchmark only. But because external validation is heavily emphasized, readers may wonder whether NACC and ds000030 are included in the 13. They are not, apparently. Clarify:

> “On a 13-cohort development benchmark…”  
> “On two unseen external cohorts…”

Otherwise the total cohort accounting is muddy.

## E3 “stable across six LLMs” is weakly supported

E3 says:

> “cross-model verdict agreement is 7/9”

That is not strong stability. It is moderate agreement on a tiny set. The more relevant claim is “no model false-confirmed the confounded null,” but that is based on **one null** in the nine shared claims.

So this is internally consistent numerically, but the interpretation is too strong. Use:

> “In a small six-model probe, the no-false-confirm behavior held on the tested null; verdict agreement was 7/9.”

Do not call this robust.

## “Full recall while abstaining…” is too broad

Abstract:

> “CONFIRM recovers known-positive effects at full recall while abstaining on confounded, p-fished, and non-replicating nulls.”

This is true for the internal main/full benchmark and NACC, but not true globally once ds000030 “known positives” are introduced. Qualify:

> “On the internal benchmark and NACC AD/MCI external test, CONFIRM recovers all expected detectable positives…”

## The modular retrofit claim conflicts with search-provenance requirements

Method says multiplicity correction uses:

> “the maximum of the declared comparisons and the comparisons actually searched”

But E2 says CONFIRM can be a modular layer:

> “without otherwise changing the agent”

and discussion says:

> “without access to its internals”

If you do not have access to the existing agent’s internal search log, how do you know “comparisons actually searched”? If you cannot know it, the search-provenance gate should abstain.

This needs to be made precise. Options:
- The retrofit layer requires a claim bundle containing search logs.
- If logs are absent, it uses a conservative declared family supplied by the evaluator.
- If neither is available, it emits `unverifiable_search_provenance`.

Currently, the paper sounds like the retrofit layer can govern arbitrary agent outputs without internals, while also requiring actual search provenance. Those are in tension.

## Catastrophic formatting/content issue: template boilerplate appears included

The pasted “paper” includes the WACV template sections after the actual conclusion:

- `0_abstract.tex`
- `1_intro.tex`
- `2_formatting.tex`
- `3_finalcopy.tex`

If these are actually in the submitted PDF, the paper should be **desk rejected / not reviewed** for being malformed and certainly over length. I assume this is a paste artifact, but if not, fix immediately.

---

# 2. Remaining overclaims

## “Statistically admissible”

This is still too strong.

You can say CONFIRM checks whether a claim satisfies a **predefined admissibility policy**. You cannot generally say it determines whether a finding is “statistically admissible” in the abstract sense. The gates are reasonable but heuristic:
- sign-and-significance replication is one defensible criterion, not the criterion;
- ComBat is a modeling choice, not a guarantee;
- confound-completeness covers only a fixed list;
- power uses d=0.3 fallback, which may be inappropriate for many neuroimaging effects;
- multiverse choices are predeclared but necessarily incomplete.

Suggested wording:

> “whether a finding satisfies a predeclared statistical admissibility policy”

or

> “whether a claim is licensed by a fixed governance protocol”

Avoid naked “statistically admissible” unless you define it as policy-relative.

## “Trustworthy”

If the title is “Trustworthy … Agentic Neuroimaging Discovery,” it still overclaims. The system is not making discovery trustworthy in the broad sense. It is filtering claims.

Better titles:
- **“CONFIRM: Claim Governance for Agentic Neuroimaging”**
- **“CONFIRM: A Faithfulness Layer for Agentic Neuroimaging Claims”**
- **“Governing Confirmed Claims in Agentic Neuroimaging”**
- **“CONFIRM: Reducing False Confirmations in Agentic Neuroimaging”**

If you keep “Trustworthy,” use it only as an aspiration, not as a guaranteed property.

## “Discovery”

Also problematic. The paper repeatedly says the contribution is **not capability** and not better discovery. So a title emphasizing “Discovery” fights your own scoping. “Agentic Neuroimaging Claims” is much safer than “Agentic Neuroimaging Discovery.”

## “Robust”

Do not use “robust” for E3. A 9-claim, 6-model probe with 7/9 agreement is not robustness evidence. It is a small sensitivity probe.

Use:
> “stable in a small six-model probe”

or:
> “not obviously model-dependent in a small probe”

## “Label authority”

The paper still leans too hard on “known null” and “known positive.”

Safer distinctions:
- literature-anchored positives,
- procedural negative controls,
- engineered artifact controls,
- fragile/borderline cases,
- sample-detectable positives,
- literature-positive but sample-insufficient effects.

Especially avoid “known null” for “no literature-supported effect.” Absence of literature support is not a known null.

## Provenance / preregistration-like language

You improved this, but one sentence in related work remains a little strong:

> “the frozen ClaimContract is a machine-checkable, preregistration-like commitment”

That is acceptable if clearly distinguished from formal preregistration, but I would add:

> “not a public preregistration”

The NACC language is now mostly okay:

> “configuration fixed and hashed before execution”

Good. Do not call it preregistered.

---

# 3. ds000030 and ablation framing

## ds000030: scientifically plausible, but current framing invites criticism

The intended argument is defensible:

> “CONFIRM should not confirm ENIGMA-scale psychiatric effects in a modest single-site/split cohort if they do not replicate internally.”

That is a good governance story.

But the current phrasing has three problems.

### Problem 1: “0/14 positives” sounds like failed recall

If these are listed as positives, then 0/14 is bad under your own metric. You cannot simultaneously call them positives and say confirming none is the correct recall-preserving behavior unless you redefine what “positive” means.

Fix by calling them:

> “literature-positive, sample-challenging claims”

not “known-positive recall claims.”

### Problem 2: Power statement is likely wrong or underspecified

As above, the d=0.3/n≈127/80% power claim needs exact detail. If it is wrong, remove it. If power is satisfied only under an optimistic simplification, reviewers will attack it.

### Problem 3: Verdict labels are inconsistent

Do not say all 14 are `fragile` unless all 14 actually fail at the multiverse gate. If most fail significance or replication, report the breakdown:

Example:

> “Of the 14 literature-anchored psychiatric claims, 13 failed discovery/replication criteria and 1 was multiverse-fragile; none reached confirmed.”

That is honest and much better.

## “Power gate satisfied” but significance/replication fail: defensible only with explanation

This is not inherently contradictory. A design can be powered for d=0.3 and still observe no significant effect if:
- the true effect is smaller,
- the cohort differs,
- measurement noise is high,
- split replication is harsh,
- multiple testing reduces power.

But you need to say that. Also, ENIGMA psychiatric effects are often smaller than d=0.3, so the d=0.3 fallback may be too lenient. The correct interpretation may be:

> “The gate says the design is not underpowered for the policy MDE d=0.3, but this does not imply adequate power for the smaller ENIGMA-estimated psychiatric effects.”

That would be honest.

## Ablation: multiverse non-binding is okay, but weak

The ablation framing is mostly honest:

> “multiverse does not bind and is retained only for fragile-claim sign-instability”

This is defensible if you treat multiverse as a safety gate for a failure mode not heavily represented in the current stress suite.

But reviewers will ask: why include a gate that does not change the main ablation? You need one of:
1. a small explicit fragile case where multiverse uniquely matters;
2. an appendix experiment with synthetic sign-instability claims;
3. reframe it as optional / diagnostic rather than load-bearing.

Right now, the paper says the stress suite includes “known-null and fragile claims,” but then says multiverse rejects no null. That makes me wonder whether the suite actually exercises fragile claims. If fragile claims are important, include a fragile family and show the gate does something.

The honest current claim should be:

> “In our present 150-claim synthetic null suite, multiverse is not load-bearing; we retain it as a prespecified diagnostic against analytic sign instability and report fragile cases separately.”

That is fine. Do not pretend it contributes to the 0/177 result unless it does.

---

# 4. Single most important remaining weakness

The most important weakness is **external validity of the evaluation**, not the gate design.

The evidence is still heavily based on:
- development cohorts,
- synthetic gate-targeted negatives,
- small adjudicated claim sets,
- procedural random-label nulls,
- one runnable agent baseline,
- tiny multi-LLM probe,
- no preregistered independent benchmark with clean literature nulls.

The NACC result is useful but mostly validates AD atrophy positives and random-label controls. The ds000030 result is useful as a conservatism stress case, but it does not validate positive recall. The paper still lacks a genuinely independent, preregistered, multi-disorder external benchmark with both:
- sample-detectable positives, and
- clean literature-anchored nulls.

That is the remaining gap between “interesting governance prototype with promising validation” and “field-level trustworthy agentic neuroimaging governor.”

---

# 5. Review-style assessment

## Strengths

- The scoped claim—faithfulness, not capability—is the right one.
- The architecture is sensible: contract freezing, code-owned numbers, numeric guard, and explicit abstention taxonomy.
- The power-gate fix is important and directly addresses a real vulnerability.
- The paper now reports exact binomial intervals and acknowledges small counts.
- The NACC frozen-configuration external run is a meaningful addition.
- The ablation is much more credible now: confound-completeness and replication are clearly load-bearing.
- The paper is more honest than most agentic-AI application papers about abstention and non-confirmation.

## Weaknesses

- The 150/177 stress-suite wording is inconsistent and must be fixed.
- ds000030 has unresolved statistical and labeling problems.
- “Statistically admissible,” “robust,” and “trustworthy discovery” remain too strong.
- The external null evidence is still mostly procedural random-label controls.
- Label authority remains heterogeneous; “known null” is too strong in places.
- The retrofit setting is under-specified with respect to search provenance.
- Multi-LLM stability evidence is very small and should not be sold as robustness.
- If template boilerplate is included, the submission is invalid.

## Specific required edits before submission

1. Replace every “177-claim synthetic stress suite” with:
   > “150 synthetic gate-targeted stress negatives; 177 pooled negatives including 27 adjudicated benchmark negatives.”

2. Fix E section count:
   > “We ask six questions” or remove the numbered-question wording.

3. Rewrite ds000030 with exact verdict counts:
   - how many fail discovery/multiplicity,
   - how many fail multiverse,
   - how many fail replication,
   - how many fail power.

4. Recheck and document ds000030 power. If n≈127 is total, the current claim is likely wrong.

5. Stop calling ds000030 “positive recall” evidence. Frame it as a “literature-positive but sample-challenging conservatism test.”

6. Add a table listing the 9 NACC positive claims.

7. Replace “robustness across LLMs” with “six-model sensitivity probe.”

8. Define what happens when retrofit inputs lack search provenance.

9. Replace “statistically admissible” with “satisfies a predeclared admissibility policy.”

10. Change the title if it currently contains “Trustworthy … Discovery.”

---

# Final score

**Score: 6/10.**

**Recommendation: Borderline.**

If the accounting and ds000030/power issues are fixed, I could support **weak accept** for WACV applications because the problem is timely, the system idea is practical, and the evaluation is unusually honest for this area.

As currently written, I would lean **borderline reject**, because a paper about claim governance cannot afford ambiguous claim accounting, questionable power statements, or overconfident external-validation framing.
