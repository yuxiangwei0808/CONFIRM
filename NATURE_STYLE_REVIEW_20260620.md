# Nature-Style Reviewer Assessment: CONFIRM

Date: 2026-06-20

Paper reviewed: `paper/main.tex` and associated sections, tables, and local result artifacts for CONFIRM.

Scope note: This is a referee-style assessment using the local manuscript and repository artifacts only. It is not an editorial decision, not a citation audit, and not a claim of privileged reviewer expertise. I treat the paper's Nature-relevant bar as: clear conceptual advance, broad scientific importance, technical soundness, strong comparison to alternatives, and readability for an interdisciplinary audience.

## Shared Summary

The paper proposes CONFIRM, a statistical claim-governance layer for agentic neuroimaging discovery. An LLM drafts a frozen `ClaimContract`, but numerical analysis is delegated to deterministic code. Claims are then filtered through gates for multiplicity/search provenance, confound handling and confound completeness, power, multiverse robustness, and independent cross-cohort replication. The system returns governance verdicts such as confirmed, non-replicated, under-powered, or fragile, and strips unsupported numeric claims from generated prose.

The central empirical story is that permissive execution confirms many false or fragile claims, while the full gate ladder sharply reduces false confirmations. The manuscript reports 0/27 false confirmations on the main adjudicated benchmark, 1/177 false confirmations on a synthetic gate-targeted stress suite, improved false-confirmation behavior relative to a NeuroClaw baseline, and an external NACC run with 9/9 known AD/MCI positives and 0/28 random-label negative controls. A local CNP external artifact reports 0/14 psychiatric positives and 0/16 random nulls, which is important negative evidence if it is part of the paper's intended evidence base.

The contribution is best understood as a governance and audit architecture for preventing agentic systems from over-claiming statistical findings. It should not be framed as a new biomarker discovery engine, a new statistical test, or a broad validation of neuroimaging claims.

## Report 1

### Overall Assessment

This is a timely and potentially useful systems paper. The motivating problem is real: LLM-based scientific agents can fabricate, overfit, or selectively report statistical claims unless there is a strict interface between natural-language claims and executable statistical evidence. CONFIRM's strongest idea is the `ClaimContract` abstraction plus a conservative gate ladder that turns free-form scientific assertions into auditable statistical objects.

However, the technical case is not yet strong enough for a high-impact interdisciplinary venue as currently framed. The manuscript's headline evidence depends on small labeled benchmarks, synthetic negatives generated around the system's known gates, and an external benchmark whose provenance is internally inconsistent across repository documents. The method is promising, but the current evidence package does not yet establish robust false-confirmation control in real neuroimaging discovery settings.

### Major Strengths

1. The paper identifies a concrete failure mode for agentic science systems: numerical claims can be linguistically plausible while statistically unsupported. This is a strong and broadly relevant problem framing.

2. The method has a clean division of labor. The LLM proposes claim structure, while code computes all estimates, p-values, intervals, power checks, multiverse summaries, and replication tests. This is a defensible architecture for limiting LLM authority.

3. The gate ladder is interpretable. Multiplicity, confounding, power, multiverse robustness, and replication correspond to recognizable scientific validity concerns.

4. The numeric guard is practically important. Stripping or rejecting unauthorized numbers directly addresses one of the most damaging failure modes of LLM-generated scientific prose.

5. The NeuroClaw comparison is useful because it tests CONFIRM as a post-hoc governance layer over an existing neuroimaging agent, not only as a standalone pipeline.

6. The manuscript is unusually candid about several limitations: conservatism, low coverage, synthetic stress tests, correlated AD evidence, and the fact that governance is not the same as discovery capability.

### Major Concerns

1. The external benchmark provenance is not yet credible enough for the manuscript's wording. `EXTERNAL_BENCHMARK_PREREG.md` still states that the benchmark is a draft preregistration and "not yet frozen", while `EXTERNAL_BENCHMARK_RESULTS.md` describes the NACC run as preregistered, frozen, and run once. `docs/STAGED_RESULTS_INDEX.md` also states that the workspace is not currently a Git repository. This is not a minor documentation issue: the claim of a frozen, run-once external benchmark is central to the paper's credibility. The manuscript should either provide an immutable lockfile/tag/hash trail or downgrade all preregistration language.

2. The false-confirmation estimates are too easily overinterpreted. The reported 1/177 stress-suite result is valuable as a hardening check, but the negatives are synthetic and gate-targeted, and many are on cohorts used during development. It cannot support a general false-confirmation rate for real-world neuroimaging discovery.

3. The external validation is narrow. The NACC result is encouraging for AD/MCI structural effects, but it is one disease family and one data type, with random-label negative controls rather than literature-anchored null claims. The reported 0/28 null result has an upper 95% confidence bound around 12.3%, which does not meet the preregistered target of an upper bound below 10%. If the CNP artifact is included, the 0/14 positive recovery for psychiatric positives is a major transportability failure that must be integrated into the paper.

4. The baselines are not yet sufficient. Significance-only is a necessary baseline, but it is weak. NeuroClaw is relevant, but it is a comparison to another agent rather than to simpler governance alternatives. The paper needs stronger baselines such as FDR plus replication only, replication-only, FDR plus confound adjustment, a no-LLM statistical-template pipeline, and perhaps a human-analyst preregistration baseline on the same claim set.

5. The power gate requires more justification. If the gate relies partly on observed effects, then lucky large effects in small samples can pass, as seen in the underpowered false-confirmation case. The paper should clarify whether power is prospective, observed, MDE-based, or a mixture, and should report how often this gate rejects claims independently of replication.

6. "Confound completeness" is too strong as phrasing. A fixed checklist of measured covariates can detect some obvious missing adjustments, but it cannot certify complete confounding control. The manuscript should call this a measured-confound coverage check or a prespecified covariate-completeness check.

7. Coverage is low and should be foregrounded. In the main benchmark, the full ladder confirms only about 12/40 claims while maintaining high positive-control recall. That is a legitimate design choice, but the paper must make clear that CONFIRM trades discovery throughput for lower false confirmations.

### Technical Revisions Needed

1. Provide a reproducibility manifest with exact claim set hashes, gate configuration hash, code commit or equivalent immutable source snapshot, data versions, and run timestamps for every benchmark table.

2. Replace or qualify "preregistered", "frozen", and "run once" unless the repository contains immutable evidence supporting those claims.

3. Add a complete per-claim table with label source, cohort, discovery cohort, replication cohort, family size, confound set, power value, multiverse fraction, replication statistic, and final verdict.

4. Add stronger non-agent governance baselines and ablations. The most important comparison is not only CONFIRM versus significance-only, but CONFIRM versus simpler combinations of FDR, covariate adjustment, and independent replication.

5. Separate internal development evidence, synthetic stress evidence, and external evidence in all text and figures. Do not pool them rhetorically.

6. If CNP is part of the result package, report it plainly. A method that protects false confirmations but recovers 0/14 external psychiatric positives is still useful, but it changes the paper's claim from "general external validation" to "domain-specific external validation with unresolved transportability".

### Criteria Assessment

Originality: Moderate to strong as a governance architecture for agentic neuroimaging. The individual statistical gates are standard; the novelty is their contractual integration into an LLM-scientist workflow.

Scientific importance: Potentially high, but not yet established at Nature level. The problem is important, but the evidence currently supports a narrower claim about conservative claim adjudication rather than a broad advance in scientific discovery.

Interdisciplinary interest: Moderate. The LLM governance framing is broadly relevant, but the empirical evidence is still specialized and neuroimaging-specific.

Technical soundness: Promising but not fully established. Provenance, external validation, baseline strength, and power-gate interpretation need substantial tightening.

Readability: Good for technical readers, but too dense for a broad interdisciplinary audience. The gate logic is understandable, but the many denominators and benchmark subsets are hard to track.

Recommendation posture: Major revision before the paper can credibly make broad claims.

## Report 2

### Overall Assessment

The manuscript's core novelty is not a new statistical method; it is a workflow constraint on agentic science. That distinction matters. If the authors present CONFIRM as a statistical innovation, the paper is not convincing. If they present it as an enforceable claim-governance layer that makes scientific agents less likely to confirm false claims, the work becomes much more compelling.

The current version sometimes reaches beyond that defensible center. The reported gains are strongest on benchmark designs that favor gate-based conservatism. The manuscript needs a sharper comparison to prior agentic neuroimaging systems and to simple statistical pipelines that do not use LLMs.

### Novelty Assessment

The `ClaimContract` abstraction is the clearest novelty. Freezing an estimand, variable roles, cohort split, covariates, search family, and evidence gates before text generation is a useful mechanism for turning LLM output into something auditable.

The search-provenance and confound-completeness gates may be novel in the local context of neuroimaging agents, but they are not obviously novel statistical concepts. The paper should avoid overclaiming these as new methods. The novelty is in operationalizing them as mandatory checks inside an agentic discovery loop.

The numeric guard is practically valuable and should be emphasized more. Preventing unsupported numeric prose may be as important as the formal verdict system, especially for readers concerned about LLM hallucination.

The NeuroClaw layer result is a strong framing device: CONFIRM can be a wrapper around existing discovery agents rather than a replacement. This is a clearer contribution than trying to claim that CONFIRM itself is a better discovery agent.

### Baseline Assessment

The significance-only baseline is appropriate but too weak to carry the paper. It mainly demonstrates a known fact: uncorrected or lightly governed significance testing is vulnerable to false positives. The paper needs baselines that a careful statistician would actually use.

The NeuroClaw comparison is relevant, but it is not enough. The shared-claim comparison shows that CONFIRM reduces false confirmations relative to NeuroClaw on the benchmark, and the post-hoc layer result is especially useful. Still, a reviewer will ask whether a simpler rule such as "require same-sign independent replication after FDR correction" would achieve most of the benefit.

The gate ladder is a good ablation, but it is not the same as a baseline suite. It shows which CONFIRM gates matter internally; it does not show that the full architecture is necessary relative to simpler non-agent statistical protocols.

Recommended additional baselines:

1. FDR plus replication only.

2. FDR plus covariates plus replication.

3. Replication-only on claims that pass initial execution.

4. A no-LLM deterministic claim template using the same inventory.

5. A conservative "human preregistration" rule set, even if simulated from fixed templates.

6. A baseline that uses the same external claim labels and cohorts, not only internal stress claims.

### Evidence Assessment

The internal benchmark supports the narrow conclusion that the full gate ladder prevents false confirmations on the authors' labeled claim set while preserving known-positive recall. This is meaningful but small-scale.

The 177-negative stress suite supports the conclusion that the gates catch the failure modes they are designed to catch. It does not establish that the system will catch unknown failure modes or genuine literature-null claims on unseen data.

The NACC external result supports the claim that CONFIRM can recover strong, well-known AD/MCI structural effects on a large unseen cohort. It does not support broad neuroimaging generality. The random-label null controls are useful but are not the same as externally sourced null scientific claims.

The CNP result, if included, substantially weakens broad claims. A 0/14 known-positive recovery rate for psychiatric positives suggests either dataset mismatch, measurement mismatch, insufficient power, incompatible labels, or pipeline limitations. Any of these explanations is plausible, but the paper must make the failure visible.

### Major Concerns

1. The manuscript should not describe the result as "trustworthy agentic neuroimaging discovery" without consistently emphasizing abstention and false-confirmation prevention. "Discovery" implies finding true new effects; the evidence mostly supports safe adjudication of proposed claims.

2. The label provenance is underdeveloped. Terms such as known-positive, known-null, fragile, and random-null need a table with explicit source, rationale, and whether the label was assigned before or after any CONFIRM run.

3. The positive-control set appears too AD-heavy in the external evidence. AD atrophy effects are large and well established; recovering them is reassuring but not a demanding demonstration of generality.

4. The paper should not lead with the 3.1% exact upper bound as if it were a general operating characteristic. It is a bound for a particular synthetic stress suite.

5. The multi-LLM experiment is a smoke test, not a robustness study. Six models over nine shared claims, with agreement on seven, is helpful but far too small for broad model-stability claims.

6. The manuscript needs clearer separation between "the system rejects claims" and "the claim is false". Non-confirmed claims include underpowered, non-replicated, fragile, or unassessable findings; these should not be described as negatives.

### Criteria Assessment

Originality: Stronger as workflow design than as statistical novelty. The authors should explicitly position it this way.

Scientific importance: High potential if the paper demonstrates that claim governance generalizes beyond internal benchmarks. Current evidence supports a more limited importance claim.

Interdisciplinary interest: Potentially broad because the same pattern could apply to other agentic scientific domains, but the manuscript needs to make that transfer argument without pretending it has already been empirically proven.

Technical soundness: The architecture is sound in principle; the empirical validation is incomplete.

Readability: The paper is technically organized but would benefit from a single "what each benchmark proves and does not prove" table.

Recommendation posture: Revise to narrow the novelty claim, strengthen baselines, and make negative/transportability results explicit.

## Report 3

### Overall Assessment

The paper has a compelling central message for a broad audience: automated scientific agents should not be trusted because they sound scientific; they should be trusted only when their claims survive executable, auditable statistical governance. This is a strong story. The current manuscript, however, is written more like an internal technical report than a broad scientific paper. It contains many important safeguards and limitations, but the reader has to work too hard to understand which claims are proven, which are stress-tested, and which remain aspirational.

For a high-impact paper, the authors should make the conceptual contribution simpler and the empirical boundaries sharper. The paper should repeatedly distinguish three levels of evidence: internal adjudicated benchmarks, synthetic gate-targeted stress tests, and external unseen-cohort validation.

### Readability and Framing

The introduction should foreground the failure mode in ordinary scientific terms: an agent can convert flexible data analysis into confident prose unless there is a contract that binds every sentence to a prespecified computation. That framing will travel beyond neuroimaging.

The method section should include a compact example of one claim moving through the full pipeline: proposed claim, frozen contract, executed statistics, gate failures or passes, final text allowed by the numeric guard. This would make the architecture much clearer than a purely abstract gate description.

The results section needs a denominator map. The manuscript currently moves across 27, 29, 40, 55, 150, 177, 25, 9, 28, and potentially 37-claim subsets. These numbers may all be correct, but broad readers will lose track. A single table should state each benchmark, number of claims, label types, data source, whether the cohort was used in development, baseline, and what conclusion is licensed.

The limitations are strong, but they should be moved closer to the corresponding claims. For example, the NACC external result should immediately state that it is AD/MCI structural evidence, not broad external validation. The stress-suite result should immediately state that the negatives are synthetic and gate-targeted.

### Major Concerns

1. The title and abstract risk overclaiming. "Trustworthy Agentic Neuroimaging Discovery" sounds like the system discovers trustworthy findings. The evidence more directly supports "statistical claim governance" or "false-confirmation control for agentic neuroimaging claims".

2. The manuscript needs a clearer audience contract. Neuroimaging readers will ask about cohorts, confounds, harmonization, effect-size labels, and replication. AI readers will ask what the LLM actually does, how hallucinated numbers are blocked, and whether the system generalizes across models. Statistics readers will ask whether the gates amount to standard practice and whether the baselines are fair. The current manuscript partially answers all three, but not in a clean order.

3. Some terminology implies more certainty than the evidence warrants. "Known null", "confound completeness", "preregistered", "frozen", "external validation", and "robust across LLMs" should be used only where the paper has direct support.

4. The paper should not hide conservatism. Low coverage is not a defect if the goal is claim safety, but it changes how the paper should be read. The main message should be "CONFIRM says no or not-yet in many cases, and that is the point."

5. The external negative controls need plainer explanation. Random-label controls within cognitively normal subjects are useful for testing whether the pipeline hallucinates signal, but they are not the same thing as real scientific null hypotheses drawn from the literature.

6. The CNP result, if retained in the repository evidence base, should be discussed. A broad reader will not trust a paper that reports only the successful external benchmark while a local artifact shows an external positive-recovery failure.

### Suggested Paper Restructure

1. Lead with the problem: LLM scientific agents need enforceable statistical contracts.

2. Define CONFIRM in one figure: LLM proposal, contract freeze, deterministic execution, gates, allowed text.

3. Present internal benchmark: what it tests and what it cannot test.

4. Present NeuroClaw wrapper experiment: governance over an existing agent.

5. Present stress suite: targeted failure-mode hardening, not real-world FCR.

6. Present external evidence: NACC success and any CNP or other external failure in the same section.

7. End with the honest claim: CONFIRM is a conservative adjudication layer that reduces false confirmations and exposes uncertainty, not a replacement for domain expertise or independent validation.

### Criteria Assessment

Originality: The manuscript has a clear conceptual contribution if framed around enforceable claim contracts and numeric authorization.

Scientific importance: The safety problem is important, but the paper's biological evidence is not the main contribution. The paper should not sell itself as a neuroimaging biomarker paper.

Interdisciplinary interest: Potentially strong if the writing is simplified and the transferability of the governance idea is made explicit.

Technical soundness: The reported results support a conservative claim-governance mechanism, but not broad external generalization.

Readability: Needs substantial restructuring for non-specialists, especially around benchmark subsets and claim boundaries.

Recommendation posture: Potentially publishable after reframing and evidence/provenance tightening; not ready as a broad "trustworthy discovery" claim.

## Cross-Review Synthesis

All three reports converge on the same high-level view: CONFIRM is a promising governance architecture, but the paper must narrow and harden its claims before submission.

Consensus strengths:

1. The `ClaimContract` abstraction is the paper's strongest technical and conceptual contribution.

2. Deterministic execution plus numeric-claim authorization is a serious response to LLM hallucination in scientific writing.

3. The gate ladder is interpretable and maps to real scientific validity concerns.

4. The NeuroClaw wrapper experiment is an effective demonstration of CONFIRM as a governance layer over existing agents.

5. The paper is commendably honest in several limitations, especially around conservatism and synthetic stress tests.

Consensus weaknesses:

1. The external benchmark's frozen/preregistered status is not adequately documented and is internally inconsistent across local files.

2. The empirical support is narrower than the title and abstract imply.

3. The baseline suite is too weak to show that the full architecture is necessary.

4. The false-confirmation statistics are likely to be overread unless the manuscript clearly separates internal, synthetic, and external evidence.

5. External validation is AD-heavy and should not be generalized to neuroimaging broadly without reporting failures such as CNP if those artifacts remain part of the evidence base.

6. The manuscript is too dense for a broad audience and needs a clearer benchmark map.

Most important revision priorities:

1. Fix provenance. Either produce immutable evidence for frozen, run-once external evaluation or remove that language.

2. Reframe the central claim from "trustworthy discovery" to "conservative statistical claim governance".

3. Add stronger baselines and ablations against simple statistical governance pipelines.

4. Add a complete per-claim provenance and verdict table.

5. Report external successes and failures together.

6. Add a one-page conceptual walkthrough of one claim from LLM proposal to final permitted prose.

## Risk and Unsupported-Claim Audit

These are phrases or arguments that should be softened, substantiated, or removed before paper writing.

1. "Preregistered external benchmark" or "frozen, run once" - Unsupported unless the project can show an immutable freeze record. The local prereg file says draft/not frozen.

2. "False-confirmation rate below 3.1%" - Supportable only for the 177-claim synthetic/local stress suite as an exact upper bound, not as a general operating characteristic.

3. "Known null" - Risky unless every null label has external literature provenance. Random-label controls are genuine procedural negatives, but they are not literature-null scientific claims.

4. "External validation" - Too broad if it mainly refers to NACC AD/MCI structural effects. Use "external NACC validation for AD/MCI structural positives and random-label null controls" unless broader external evidence is reported.

5. "Robust across LLMs" - The six-model, nine-claim experiment is a smoke test. Use "stable in a small multi-model probe" unless expanded.

6. "Confound completeness" - Too strong. Prefer "prespecified measured-confound coverage" or "covariate completeness check".

7. "Trustworthy discovery" - Overstates what is shown. "Trustworthy claim adjudication" or "false-confirmation governance" better matches the evidence.

8. "General neuroimaging agent" - Overbroad unless external results cover multiple modalities, disorders, and real nulls.

9. "NeuroClaw comparison proves superiority" - Too broad. It shows reduced false confirmations on a shared benchmark and usefulness as a wrapper; it does not establish general superiority over all agentic or statistical baselines.

10. "The system controls false discoveries" - Use carefully. CONFIRM reduces false confirmations under the benchmark verdict definition; it is not equivalent to formal global FDR control over arbitrary scientific exploration.

## Bottom Line

The paper is worth developing, and the underlying project has a defensible contribution. The most publishable version is not "an AI scientist that discovers reliable neuroimaging findings"; it is "a conservative claim-governance system that prevents agentic neuroimaging workflows from converting weak statistical evidence into confident prose."

For the paper-writing phase, the authors should make the method less grand and the evidence more auditable. If provenance is fixed, baselines are strengthened, and external failures are reported alongside successes, the work can become a credible systems paper about statistical governance for scientific agents.
