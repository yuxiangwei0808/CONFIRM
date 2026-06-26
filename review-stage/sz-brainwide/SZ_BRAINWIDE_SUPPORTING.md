# SZ Brain-wide FNC Replication — Supporting Result (NOT a confirmed positive)

Date: 2026-06-17 · Status: **CANDIDATE / supporting evidence** · **NOT counted in TPR/FCR**

## Question
Does the schizophrenia-vs-control functional-connectivity (FNC) pattern replicate across independent cohorts — separately from the 160-IC COBRE↔FBIRN confirmed positive?

## Data
Four independent NeuroMark **100-component** cohorts, **edge-level FNC** (4950 edges), uniform raw-Pearson→Fisher-z, covariates age+sex (no motion covariate available). Local: `data/prepared_data/sz_edges/`.
- ChineseSZ — 738 (483 SZ / 255 HC), China
- BSNIP2 — 586 (248 / 338), US (5 sites)
- BSNIP-1 (a.k.a. Olin_SZ) — 419 (181 / 238), US (6 sites)
- JH — 142 (48 / 94), US

## Per-cohort SZ effect (FDR-significant edges of 4950)
ChineseSZ 1565 · BSNIP2 1548 · BSNIP-1 1836 · JH 1221 — a large *distributed* SZ FNC effect in **every** cohort. (The 4 summary scalars like fc_mean_abs were individually null — the signal is the distributed pattern, not a scalar.)

## Pairwise pattern correlation of SZ−HC edge-effect maps (all positive)
| pair | r |
|---|---|
| ChineseSZ–JH | +0.606 |
| BSNIP2–JH | +0.579 |
| BSNIP-1–JH | +0.568 |
| ChineseSZ–BSNIP2 | +0.467 |
| BSNIP2–BSNIP-1 | +0.452 |
| ChineseSZ–BSNIP-1 | +0.388 |

Mean ≈ 0.51. China↔US: 0.47–0.61.

## CONFIRM brain-wide verdict (discovery ChineseSZ → BSNIP2/JH/BSNIP-1; pre-declared pattern_corr_min=0.40, all-must-pass)
- BSNIP2: r=0.457 — PASS
- JH: r=0.605 — PASS
- BSNIP-1: r=0.394 — FAIL (0.006 under the bar)
- **Overall: `non_replicated`** (all replication cohorts must pass; one sits at threshold). At CONFIRM's default 0.50 bar, more pairs fail.

## Honest interpretation
- The SZ FNC dysconnectivity pattern **reproduces** across 4 independent cohorts (all pairwise r positive, 0.39–0.61, incl. China↔US) — a genuine replication by FNC-literature standards.
- It is **moderate and threshold-sensitive**: CONFIRM's conservative gate returns `non_replicated` because BSNIP-1 lands at the threshold.
- **We do not count it as a confirmed positive** — no tuning of design/threshold/cohort-selection to force a pass (that would be the cherry-picking the integrity audit guards against). Recorded as CANDIDATE / supporting.
- Caveats: no motion covariate (a known SZ-FNC confound); r≈0.5 is moderate; 100-IC template, separate from the 160-IC confirmed pair.

## Role in the project
- **SZ confirmed positive remains COBRE↔FBIRN** (160-IC, within-network FC).
- This 4-cohort 100-IC result is (1) **supporting evidence** that SZ dysconnectivity reproduces broadly, and (2) an **on-thesis demonstration of CONFIRM's conservatism**: an execution-only agent would declare SZ "found" from ChineseSZ's 1565 significant edges; CONFIRM withholds confirmation because the cross-cohort pattern only moderately replicates.
