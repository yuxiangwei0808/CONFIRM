# EXTERNAL_BENCHMARK_PREREG — CONFIRM external faithfulness benchmark

**Status:** DRAFT preregistration (not yet frozen). Freeze = git tag `prereg-extbench-v1` + config hash, *before* any run.

## Why this exists
The current bound (1 false confirm / 177, exact 95% upper ≈3.1%) comes from **synthetic, gate-targeted** nulls generated on cohorts CONFIRM was **developed on**. A brutal reviewer (correctly) discounts it: the nulls are the failure modes the gates were built to catch, and the cohorts aren't independent. This benchmark estimates CONFIRM's faithfulness on **real claims**, with **externally-sourced labels**, on **unseen cohorts**, **gates frozen**, **run once**.

## Five non-negotiable principles
1. **External labels** — ground truth comes from the literature/consortia, not from CONFIRM's own runs or our post-hoc judgement.
2. **Unseen cohorts** — not used anywhere in CONFIRM development (excludes ADNI, OASIS3, COBRE, FBIRN, ABIDE1, GSP, ChineseSZ, BSNIP2, Olin, JH, and any benchmark_ready cohort already run).
3. **Gates frozen** — the gate ladder + every threshold locked (git tag + hash recorded here) before the run.
4. **Run once** — no iteration on results; failures are reported, not patched.
5. **Blinded scoring** — labels withheld from the pipeline; verdicts scored against labels only afterward.

## Claims & labels  ▸ DECISION 1 (label provenance)
Target **~120–180 claims**, balanced ≈ 40% known-positive / 40% known-null (incl. negative controls) / 20% fragile. Each claim = an estimand (group-diff or association) + a discovery cohort + an independent replication cohort.

- **Option A — literature/consortium-anchored (recommended).** Labels from published meta-analyses, above all **ENIGMA** case–control standardized effect sizes (AD, SZ, MDD, BD, ASD, ADHD, OCD × subcortical/cortical structures) plus large GWAS/BWAS nulls. A claim's label (effect present + sign, or null) is read off the *independent* meta-analysis, never from our data. Most directly rebuts the circularity critique. Cost: compile ENIGMA tables (public) and match to cohorts that carry those structures/disorders.
- **Option B — large-N held-out arbiter.** A huge cohort *not* used in development (e.g., a held-out UK Biobank slice) is the truth arbiter: an effect robustly significant at large N with strict multiplicity = positive; clearly null = null. CONFIRM is then tested on *smaller, independent* cohorts. Pragmatic and data-light, but the arbiter is still "our computation," so less external than A.
- **Option C — hybrid.** Literature/ENIGMA sets the label; the large-N arbiter confirms it; CONFIRM is tested on independent smaller cohorts. Strongest, most work.

Negative controls (real but expected-null associations, e.g., a phenotype with no literature link to a region) are included explicitly so the null set is not all "obvious."

## Cohorts  ▸ DECISION 2 (unseen-data source)
Every claim needs a discovery + an independent replication cohort, both **unseen**. Candidate sources:
- **(i) Remote cluster** `/data/qneuromark`, `/data/neuromark2` via `ssh arcdev` — cohorts we have **not** pulled yet (needs GSU VPN; was flaky before).
- **(ii) Fresh OpenNeuro** datasets downloaded for this benchmark.
- **(iii) Held-out partitions** of large, under-used cohorts (pre-registered split, never touched in dev).

## Frozen gate configuration
Exact ladder as shipped: multiplicity FDR α=0.05 over the effective (search-provenance) family; confound-completeness (χ²/one-way ANOVA @0.05 over the fixed structural list site/scanner/field-strength/fs-version); power ≥0.8; multiverse min-consistent fraction 0.6; replication = same-sign + independent p<0.05 post-ComBat. Locked via tag `prereg-extbench-v1` + a recorded config hash.

## Analysis & success criteria (set *before* running)  ▸ DECISION 3
- **Primary:** false-confirmation rate (exact 95% CI) on external nulls; known-positive recall (TPR).
- **Baseline:** a no-governance "significance-only" decision on the *same* claims; CONFIRM must show materially lower FCR at comparable recall.
- **Pre-set bar (proposed, adjust):** FCR upper 95% CI **< 10%** AND TPR **> 70%** on the frozen external set.
- **Secondary:** per-gate attribution; abstention-reason distribution; transportability sub-types.

## Leakage controls
Claim+label table frozen and hashed before any data touches the pipeline; unseen cohorts ingested through existing adapters with **no** threshold tuning; exactly one run; results frozen on completion.

## Workflow
1. Compile the external claim+label table (DECISION 1 source) → freeze + hash.
2. Inventory + ingest unseen cohorts (DECISION 2 source).
3. Freeze gate config (git tag).
4. Single blinded run → FCR/TPR + exact CIs + baseline comparison.
5. Write it up as the headline external-validation section (replaces "internal stress test" as the lead evidence).

## Decisions (locked 2026-06-19)
- **D1 — label provenance:** **A, literature/ENIGMA-anchored.**
- **D2 — unseen-data source:** OpenNeuro + held-out partitions **now**; **remote cluster (`ssh arcdev`) deferred** until the user signals it is ready, then folded in.
- **D3 — scope & bar:** target ~120–180 claims; pre-set success **FCR upper 95% CI < 10% AND TPR > 70%**, with a no-governance significance-only baseline on the same claims.
