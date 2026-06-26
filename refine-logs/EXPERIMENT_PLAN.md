# EXPERIMENT PLAN — CONFIRM (claim-driven)

**Compute:** CPU only, precomputed derivatives. **Target:** ~3–6 mo, solo-ish. Each block maps to a claim.

## Data / derivatives — PUBLIC-DATA-FIRST, disease → ≥2 independent cohorts
Lead with **open** datasets so the benchmark is reproducible by any lab; use access-gated cohorts
(UKB/ADNI/ABCD) for **scale + extra replication**. Primary substrate = **sMRI FreeSurfer IDPs**
(volumes/thickness) — the one derivative available across nearly all cohorts and the cleanest for
cross-cohort replication; add **PET** where available; **FC/fMRI** as a secondary track.

Each claim needs **≥2 independent cohorts of the same condition** (the replication split is across
cohorts, never just within-site):
| Disease family | Cohorts (open unless noted) | Example claim | GT status |
|---|---|---|---|
| Alzheimer's / MCI | ADNI* ↔ OASIS-3 ↔ AIBL | hippocampal/entorhinal atrophy; FDG/amyloid/tau PET | **positive** (d≈1.5–2.0) |
| Aging | UKB* ↔ CamCAN ↔ IXI ↔ OASIS-3 | brain-age; global/regional atrophy; sex/ICV | **positive** (huge, universal) |
| Autism | ABIDE-I ↔ ABIDE-II | cortical/subcortical + FC differences | mixed / **fragile** |
| ADHD | ADHD-200 (multi-site) | structural/FC; split-site replication | mostly **fragile** |
| Schizophrenia | COBRE / SchizConnect / OpenNeuro | ventricular enlargement, GM reduction | **positive** (moderate) |
| Parkinson's | PPMI | subcortical / DAT measures | **positive** |
| Dev. / psychiatric dims | ABCD* | brain–behavior associations | largely **fragile** (Marek) |
| Analytic variability | NARPS (OpenNeuro ds001734) | the 9-hypothesis set | **fragile/variable** |
*(* = access-gated but on hand; all others are openly downloadable.)*

- Harmonization: **ComBat / CovBat** across site/scanner, *predeclared* in the contract; separate
  "non-replicated because false" from "non-replicated because cohort mismatch" via predeclared
  phenotype/feature alignment.
- **First paper:** cover ~3–4 disease families deeply (AD + aging + one of ASD/SCZ/PD) × the three task
  classes below; grow into a standing leaderboard later.

## Task suite — NeuroDecide-Bench-lite (~12–15 tasks)
- **Adversarial injected-nulls (ground-truth null):** head-motion leakage, site imbalance, age/diagnosis
  collider bias, label leakage, bad multiplicity. *Trap structure masked in realistic derived variables.*
- **Known-positives (ground-truth signal):** AD hippocampal atrophy + FDG (ADNI/OASIS-3/AIBL); aging→volume
  + sex/ICV (UKB/CamCAN/IXI); schizophrenia GM/ventricles (COBRE); Parkinson's subcortical (PPMI). Each has
  a published meta-analytic effect size as the reference.
- **Real fragile:** NARPS hypotheses; small-N brain–behavior associations (ABCD/UKB); ASD/ADHD effects
  (ABIDE/ADHD-200) known to be unstable across sites.

## Metrics
- **Primary:** false-confirmed rate **@ fixed known-positive recall**; **area under risk-coverage curve.**
- Effect-size **calibration error** (vs meta-analytic / large-N reference).
- **Abstention quality** (correct abstain on nulls/fragile; not on known-positives).
- **Bundle reproducibility** (re-run determinism across machine/seed).

## Baselines (all mapped to {confirmed/qualified/abstained} by a blinded claim extractor)
1. Single-cohort agent (no replication). 2. Generic DS agent (Data-Interpreter/CodeAct-style).
3. Execution-valid runner (NeuroClaw/NIAgent-style: runs + validates artifacts, no admissibility gates).
4. Trivial "always adjust motion/site/age/sex" heuristic (benchmark sanity floor).

## Blocks & run order (with decision gates)
- **B0 — Infra (enabling):** claim-contract schema + executor + provenance bundle + ComBat harness +
  replication harness. *Gate: schema validates; one end-to-end task runs deterministically.*
- **B1 — ANCHOR / must-win (C1):** curated traps where baseline-3 reports significant but effect fails
  replication / collapses under confound+multiverse; CONFIRM abstains. *Gate: clear false-confirmed-rate
  separation; if none → RETHINK gates before scaling.*
- **B2 — Known-positive recall (C3):** AD atrophy / FDG / age-sex must be **confirmed**. *Gate: recall high;
  if CONFIRM abstains on these → fix calibration (abstain-all failure).*
- **B3 — Gate-ablation ladder (C2):** exec-only→+confound→+power→+multiverse→+replication. *Gate:
  replication gate is the dominant driver of false-confirmed reduction.*
- **B4 — Real fragile / NARPS robustness:** verify CONFIRM labels fragile, matches NARPS consensus direction.
- **B5 — Reproducibility (C4):** re-run bundles on a 2nd machine/seed → determinism within numerical noise.
- **B6 — Construct-validity controls:** trivial baselines fail the traps; masked traps not solvable by
  column-name heuristics; "non-replicated-false" vs "cohort-mismatch" separated via predeclared harmonization.

## Budget / sequencing
B0 (≈2–3 wk) → B1 (≈2 wk, decision gate) → B2 (≈1 wk) → B3 (≈2 wk) → B4/B5/B6 (≈2–3 wk). All CPU.
Pilot subset first (see pilots P1–P3) before full suite.

## Pilots (cheap, run once data mounted)
- **P1:** ADNI→HCP/UKB replication gate on AD hippocampal atrophy — **must CONFIRM** (sanity).
- **P2:** one injected-null task — **must ABSTAIN** (winner's-curse trap).
- **P3:** one NARPS-fragile effect through multiverse — **must label FRAGILE**.
