# Research Brief

Updated: 2026-07-23

CONFIRM is a claim-governance pipeline for agentic neuroimaging analysis. It
turns literature-derived or LLM-proposed scientific questions into frozen,
Pydantic-validated claim contracts and evaluates them with deterministic
statistical gates.

The current workflow has five target families: normative fMRI, ADHD, ASD,
AD/aging, and psychosis. Stage 0 grounds literature seeds, Stage 1 drafts frozen
contracts, and Stage 2 applies the unchanged CONFIRM gate ladder. The completed
run evaluated 289 contracts: 74 confirmed, 169 fragile, 38 non-replicated, and
8 under-powered, with zero execution errors.

NeuroClaimBench v2.1 is the current claim-level benchmark. It contains 544
canonical source items, 489 executable claims, and 268 score-eligible
references. On scientific literature references, CONFIRM recovers 21/51
confirmable claims and confirms 2/19 abstention references. NACC and ds000030
external literature results are 0/7 and 0/11 positive recovery, respectively;
constructed controls are reported separately and yield 0/14 NACC, 0/8
ds000030, and 0/150 synthetic confirmations. The remaining 221 executable
claims are unresolved and excluded from accuracy denominators.

The optional feedback layer starts from all 215 non-confirmed contracts. GPT-5.5
proposes multiple connected executable claims over bounded rounds. Deterministic
code validates every proposal, rejects unsafe or no-op changes, evaluates valid
unique candidates on source data, and re-adjudicates all provisional passes
under the final realized multiplicity burden. No deterministic template
generator is used in production.

Same-data candidate support is labeled `exploratory_confirmed`. A separate,
outcome-blind frozen audit evaluates internally supported candidates on
compatible holdout and external evidence. Existing holdout, NACC, and CNP data
are marked `previously_queried`, so those results support retrospective
concordance analyses rather than fresh confirmation.

The defensible project claim is that CONFIRM can diagnose failed claims and use
an LLM to generate connected, provenance-audited follow-up hypotheses. It does
not guarantee that a candidate survives the gates, establish a causal benefit
of structured feedback from one model realization, or provide prospective
external validation without untouched data.

In the matched `R=3, C=5` descriptive control, structured diagnosis supports 70
candidates across 24 parents versus 49 across 20 parents for generic retry.
The parent-level cells are 17 both, seven structured-only, three generic-only,
and 188 neither. The completed traces contain 653 and 693 prompt records;
generic retry also incurred 72 superseded transport-failed attempts. This
motivates repeated-model evaluation but is not itself a causal effect estimate.
