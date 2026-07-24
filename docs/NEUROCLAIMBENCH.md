# NeuroClaimBench v2.1 Construction

NeuroClaimBench v2.1 is the only active benchmark version. Its three tracks
share schemas but retain separate scoring denominators:

- scientific literature claims;
- synthetic stress controls;
- external-transfer literature claims and matched constructed controls.

The construction workflow uses two local package directories:

- `data/neuroclaimbench/v2.1-source/` is the frozen, pre-alignment source
  snapshot. It is a build input, not a benchmark release.
- `data/neuroclaimbench/v2.1/` is the canonical repaired package.

The publishable metadata release is `benchmark/neuroclaimbench-v2.1/`.
Raw cohort data, local PubMed abstracts, LLM checkpoints, and development
archives are not part of that release.

## Pipeline

Three public launchers expose the actual method boundaries:

1. `launch_neuroclaimbench_build.sh` assembles/resumes the frozen source,
   alignment, PubMed-cache, adjudication, package, and reference phases.
   Network or model phases fail unless `ALLOW_API=1` is explicit.
2. `launch_neuroclaimbench_evaluate.sh` runs local CONFIRM evaluation over the
   frozen tasks.
3. `launch_neuroclaimbench_analyze.sh` creates the compact public schema,
   deterministic analyses, feedback crosswalk, release, and audit archive.

The outcome-blind alignment step used Gemini 3.5 Flash as an **eligibility
adjudicator**, not merely as an advisory annotation. Deterministic rules alone
authorized seven contract repairs, but Gemini could classify an otherwise
deterministically aligned literature item as ambiguous. It did so for 40
items; those items remain in v2.1 as unresolved and are not restored or
rescored.

## Commands

The current frozen package can be rebuilt and analyzed locally with:

```bash
PHASE=package scripts/launch_neuroclaimbench_build.sh
PHASE=reference scripts/launch_neuroclaimbench_build.sh
PHASE=all MAX_WORKERS=8 scripts/launch_neuroclaimbench_evaluate.sh
PHASE=all scripts/launch_neuroclaimbench_analyze.sh
```

Historical API phases remain reproducible through the build launcher, for
example `PHASE=alignment ALLOW_API=1` or
`PHASE=adjudication-full ALLOW_API=1`. Local analysis and release phases never
initialize an LLM or PubMed client.

## Release Contract

Release schema 2 contains `cases`, `references`, `tasks`, and compact
`outcomes`, plus summaries, crosswalks, hashes, and interpretation
restrictions. Detailed alignment records, votes, prompts, and gate bundles are
kept in a checksummed audit archive rather than the default public model. The
release intentionally excludes raw cohort data and PubMed abstracts.
`SHA256SUMS` must verify before any paper table is generated from the release.

NeuroClaimBench v2.1 is a retrospective benchmark revision. Constructed
controls are not literature references, underpowered but identifiable claims
remain executable, and unresolved literature claims are excluded from scored
accuracy denominators.
