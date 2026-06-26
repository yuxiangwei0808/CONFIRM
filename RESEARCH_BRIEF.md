# Research Brief — Agentic Framework for Reproducible Neuroimaging Analysis

**Created:** 2026-06-01
**Owner:** yuxiang.wei@joinhandshake.com

## Problem statement / direction
Build a **general agentic (LLM-driven) framework for reproducible neuroimaging analysis**.
Core contribution = a **reproducible-analysis agent**: given a scientific/clinical question
+ a neuroimaging cohort, the agent autonomously **plans → executes → QCs → interprets** a
reproducible statistical analysis, operating mostly on **precomputed derivatives** (so the
core loop runs on CPU/laptop, no heavy preprocessing).

This is deliberately positioned on the **analysis / interpretation / discovery** part of the
lifecycle (post-preprocessing), NOT radiology, segmentation, or presurgical mapping. Rationale:
the user's data are large *research cohorts* (already curated), so the value and novelty live
in the analysis/decision layer, not preprocessing cleanup.

## What we are NOT doing (non-goals)
- Not a clinical/diagnostic decision system, not presurgical eloquent-cortex mapping.
- Not "just an LLM wrapper that runs fMRIPrep." Preprocessing is largely solved for these cohorts.
- Not a broad shallow "general framework" with no evaluable core claim.

## Data available (the moat)
- **UK Biobank (UKB)**, **HCP**, **ADNI**, **ABCD**, **OpenNeuro** — on hand.
- Modalities: **fMRI, sMRI, PET**. Multi-disease (AD/MCI via ADNI; development/psychiatric via
  ABCD; aging/population via UKB; healthy via HCP; heterogeneous via OpenNeuro).
- Mostly available as **precomputed derivatives** + rich phenotypic/clinical metadata; longitudinal.
- **Public-data-first preference (2026-06-01):** lean on openly downloadable cohorts so the benchmark/tool
  is reproducible by any lab — OASIS-3, AIBL, IXI, CamCAN, ABIDE-I/II, ADHD-200, PPMI, COBRE/SchizConnect,
  OpenNeuro (+HCP open). Use access-gated UKB/ADNI/ABCD for scale + extra replication.
- **Breadth goal:** test many biomarkers across many diseases (AD, aging, autism, ADHD, schizophrenia,
  Parkinson's, development), each with ≥2 independent cohorts to enable cross-cohort replication. Primary
  substrate = sMRI FreeSurfer IDPs (universally available); PET/FC secondary.

## Evaluation strategy (the differentiator)
The agent's output is scored on objective, computable criteria:
1. **Known-effect recovery** — does it recover well-established effects (aging atrophy, AD
   hypometabolism/atrophy, sex differences, heritability)?
2. **Cross-cohort / cross-modal replication** — do findings replicate across UKB↔HCP↔ADNI↔ABCD↔OpenNeuro?
3. **Reproducibility** — deterministic, provenance-tracked, re-runnable; specification-curve stability.

## Constraints
- Compute: assume **no/limited GPU** for the core loop (use precomputed derivatives). Pilots
  are paper-only / lightweight unless the user confirms GPU access.
- Timeline/venue: TBD. End goal = **research-backed open-source tool** (a paper AND an adopted tool).
  Weight **novelty + usefulness jointly.**

## Decisions locked (2026-06-01)
- Core framing: **Reproducible-analysis agent** (chosen over biomarker-discovery-only and
  general-framework-anchored variants; those remain backup framings).
- End goal: **Both** — research paper/system + open-source tool.

## Pipeline
research-lit (survey) → idea-creator (brainstorm) → novelty-check → research-review (critique)
→ research-refine-pipeline (proposal + experiment plan). Orchestrated directly (parallel search
agents + Codex/gpt-5.5 for cross-model novelty/review) rather than via deep nested skill calls.
Canonical deliverable: `idea-stage/IDEA_REPORT.md`.
