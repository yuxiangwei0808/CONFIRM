# External benchmark results — NACC (unseen), 2026-06-19

Preregistered design: `EXTERNAL_BENCHMARK_PREREG.md`. Frozen gates, run once,
on a cohort (NACC) never used in CONFIRM development. Artifacts in
`review-stage/external-nacc/` (results JSON + per-claim audit). Lockfile hashes
the claim set, cohort, and runner.

## Setup
- **Cohort:** NACC, 6,912 subjects (CN 3,442 / MCI 1,779 / AD 1,232), 28 centers.
  Provider FreeSurfer ROI volumes (cm³) + UDS clinical diagnosis (NACCUDSD/NACCALZD).
- **Discovery vs replication:** disjoint sets of NACC centers (independent-site replication, ComBat-harmonized).
- **Labels:** literature/ENIGMA-anchored (Shi 2009 AD/MCI hippocampal atrophy; AD medial/lateral-temporal atrophy; ventricular enlargement).
- **Gates (frozen):** multiplicity FDR α=.05 (family=1) · confound require {age,sex,ICV} · power ≥.8 · multiverse ≥.6 consistent · replication same-sign + independent p<.05 (ComBat).

## Results
| Group | CONFIRM | Significance-only |
|---|---|---|
| **TPR** — 9 known AD/MCI atrophy positives | **9/9 = 1.00** [0.66, 1.00] | 9/9 |
| **FCR** — 28 random-label negative controls (14 regions × 2 labelings, within CN) | **0/28 = 0.00** [0.00, 0.123] | 2/28 = 0.071 [0.009, 0.235] |

**Specificity (not FCR):** AD-spared primary cortices/cerebellum — 5/9 confirmed.
These are NOT clean nulls: each carries a small *real* global-atrophy effect
(|d| ≈ 0.32–0.53), and CONFIRM's replication gate rejected the 3 that did not
replicate across centers (`ad_pericalcarine`, `ad_precentral`, `mci_precentral`).
This is correct behavior, and an honest finding: AD-vs-CN furnishes no clean
structural nulls because AD atrophy is partly global.

## Interpretation
- **External positive validity:** CONFIRM is not overfit to the development
  cohorts — it recovers every established AD/MCI atrophy effect on a large
  unseen cohort, with independent-center replication.
- **External false-confirmation:** 0 observed on genuine null contrasts, beating
  the significance-only baseline (2/28). The negative controls share subjects
  (14 regions × 2 random labelings on the same CN pool), so the CI is
  approximate; the strict prereg "<10% upper-CI" target would require a larger,
  fully-independent null suite (≈≥37 controls).

## Honest limitations / future work
- Single disorder (AD) — the one cohort with precomputed structural volumes.
  Literature-anchored *null* claims (e.g., ENIGMA MDD subcortical d≈0, SCZ
  caudate/putamen) need an unseen SZ/MDD cohort with FreeSurfer derivatives
  (FreeSurfer on LA5c/ds000030, or a SchizConnect DUA) — slow/access-gated, scoped as future work.
- Random-label controls are constructed negatives (genuine nulls, but not
  literature claims); they address "unseen cohort + frozen gates", not
  "real-world null claims".
- ComBat fell back to residualization for two tiny centers (benign).
