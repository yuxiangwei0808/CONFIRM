# fMRI Benchmark Label Ledger

Date: 2026-06-16

This ledger separates empirical benchmark labels from literature expectations. It is intentionally conservative:
`small_positive_expected` claims should not count as false confirmations merely because their effects are small;
they should instead be scored separately from `known_null` and `stress_test_fragile` claims.

## Evidence Anchors

1. Marek et al. 2022, *Nature*, "Reproducible brain-wide association studies require thousands of individuals."
   Key use here: brain-behavior associations involving RSFC and cognition are expected to be small but real at
   large N. The paper reports median RSFC-cognition effect sizes around `|r| = 0.02-0.03` across HCP/ABCD/UKB
   size-matched analyses, with reliable estimation requiring thousands of participants.
   Source: https://www.nature.com/articles/s41586-022-04492-9

2. Orlichenko et al. 2023, "Somatomotor-Visual Resting State Functional Connectivity Increases After Two Years
   in the UK Biobank Longitudinal Cohort."
   Key use here: age-related resting-state FC effects, especially somatomotor-visual connectivity, are plausible
   positive/stability controls, though the authors caution about possible UKB task/acquisition changes.
   Source: https://arxiv.org/abs/2308.07992

3. Leming and Suckling 2020, "Stochastic encoding of graphs ... UK Biobank."
   Key use here: sex classification from UKB resting-state/task connectomes is feasible after covariate balancing,
   supporting sex-FC as a biological/stability positive control, not as a disease biomarker claim.
   Source: https://arxiv.org/abs/2002.10936

4. Abraham et al. 2016, "Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based
   example."
   Key use here: ABIDE autism resting-state biomarker claims are feasible but challenged by multi-site
   heterogeneity; ASD cross-cohort diagnostic claims should remain `fragile_or_candidate`, not known positives.
   Source: https://arxiv.org/abs/1611.06066

5. Kernbach et al. 2018, *Translational Psychiatry*, "Shared endo-phenotypes of default mode dysfunction in
   attention deficit/hyperactivity disorder and autism spectrum disorder."
   Key use here: ADHD/ASD default-mode dysfunction is biologically plausible, but cross-dataset diagnostic
   effects remain heterogeneous and should be treated as candidate/fragile until benchmark-specific literature
   labels are sharpened.
   Source: https://www.nature.com/articles/s41398-018-0179-6

6. Svaldi et al. 2019, "Optimizing Differential Identifiability Improves Connectome Predictive Modeling of
   Cognitive Deficits in Alzheimer's Disease."
   Key use here: AD functional-connectivity/cognition links are plausible but less canonical than sMRI/PET AD
   anchors; fMRI AD claims should remain `positive_or_fragile_candidate` until a stronger meta-analytic label is
   selected.
   Source: https://arxiv.org/abs/1908.06197

## Current Claim Label Recommendations

| claim_id | current inventory label | recommended scoring label | rationale |
|---|---|---|---|
| `age_fc_ukb_hcpaging` | `positive_candidate` | `positive_stability` | Large cohort age-FC effects are plausible; not a disease biomarker. |
| `age_fc_hcpa_hcp` | `positive_candidate` | `positive_stability` | Lifespan/HCP age-range effects plausible but weaker; keep as stability positive. |
| `sex_fc_ukb_hcp` | `positive_candidate` | `positive_stability` | Sex-FC signal is plausible in large connectomes; interpret as biological covariate/stability, not clinical claim. |
| `sex_fc_abcd_hcp` | `positive_candidate` | `positive_stability_cross_age_stress` | Positive/stability, but ABCD child to HCP adult transfer is a stress test. |
| `cognition_fc_ukb_abcd` | `fragile_or_small_effect` | `small_positive_expected` | Marek et al. supports small RSFC-cognition effects across UKB/ABCD/HCP; should not count as false confirmation by default. |
| `cognition_fc_hcp_hcpa` | `fragile_or_small_effect` | `small_positive_expected_or_underpowered` | Same literature anchor, but smaller cohorts and age-range mismatch make non-confirmation acceptable. |
| `cognition_dyno_ukb_abcd` | `fragile_or_small_effect` | `small_positive_candidate` | Dynamic descriptor link is less established than RSFC edge cognition; separate from known nulls. |
| `adhd_fc_adhd200_abcd` | `fragile_or_positive_candidate` | `fragile_candidate` | ADHD rsFC effects plausible but heterogeneous; cross-dataset phenotype mismatch is substantial. |
| `adhd_region_adhd200_abcd` | `fragile_or_positive_candidate` | `fragile_candidate` | Same as ADHD FC; harmonization-sensitive result should not be a known positive. |
| `asd_fc_abide2_abcd` | `fragile_or_positive_candidate` | `fragile_candidate` | ABIDE literature supports possible biomarkers but emphasizes heterogeneity; ABCD ASD flag is not equivalent. |
| `ad_fc_adni_oasis3` | `positive_or_fragile_candidate` | `positive_or_fragile_candidate` | AD rsFC plausible but less canonical than structural/PET anchors. |
| `ad_region_adni_oasis3` | `positive_or_fragile_candidate` | `positive_or_fragile_candidate` | AD fMRI regional descriptor effects need a sharper literature target. |
| `ad_dyno_adni_oasis3` | `positive_or_fragile_candidate` | `positive_or_fragile_candidate` | AD dynamics plausible but not a strong ground-truth positive yet. |
| injected null claims | `null_expected` | `known_null_synthetic` | Generated random/site/fishing labels are true benchmark nulls. |
| underpowered site claims | `underpowered_or_fragile` | `stress_test_fragile` | Designed to test low-power/site split behavior, not literature truth. |

## Scoring Consequence

For paper-facing FCR, use only:
- `known_null_synthetic`
- `stress_test_fragile`
- externally adjudicated `known_null`

Report `small_positive_expected` separately. Otherwise the current benchmark unfairly treats confirmed
RSFC-cognition effects as false confirmations even though large-sample BWAS literature predicts small
replicable effects.

## Literature Work Still Needed

1. Replace this first-pass ledger with a formal table containing DOI, cohort, modality, phenotype, expected
   direction, effect-size prior, and confidence.
2. Add sMRI/PET AD labels from canonical ADNI/NIA-AA/BrainAGE literature.
3. Add psychosis labels only after imaging features and clinical labels are aligned.
4. Decide whether `small_positive_expected` contributes to TPR, a separate small-effect recovery metric, or an
   uncertainty bucket.
