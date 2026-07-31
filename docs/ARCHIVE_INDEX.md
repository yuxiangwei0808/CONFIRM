# Archive index

Archived material is retained for lab provenance, not treated as current
evidence. Each archive must contain an `ARCHIVE_MANIFEST.md` with the original
path, archive date, reason, replacement (if any), and SHA-256 checksum list.

## Existing archives

| Archive | Scope | Current replacement |
|---|---|---|
| `review-stage/_archive_20260719_pre_v7/` | Pre-v7 search outputs | v7 sweep and control runs in `RESULTS_MANIFEST.md` |
| `review-stage/_archive_20260723_claim_search_v7_original/` | Original full v7 arm artifacts | normalized v7 representation |
| `review-stage/_archive_20260723_neuroclaimbench_release_v1/` | Release-schema-1 benchmark payload | NeuroClaimBench v2.1 |
| `review-stage/_archive_20260723_simplification_source/` | Inputs retained for simplification audit | compact v2.1 release and audit |
| `docs/archive/data_discovery_20260730/` | Historical cluster and data-discovery notes | active data layout and configuration docs |
| `docs/archive/idea_stage_20260730/` | Early project ideation report | `RESEARCH_BRIEF.md` and current paper |
| `docs/archive/legacy_inputs_20260730/` | Unreferenced NeuroMark spreadsheet | Current data manifests and prepared data |
| `review-stage/_archive_20260730_inactive_probes/` | Multi-LLM drafting probe | None; restore only for model-comparison work |
| `review-stage/_archive_20260730_superseded_runs/` | v6 control and simplification audit | v7 control and v2.1 compact audit |

## Archive procedure

1. Verify checksums of the source and record them in the archive manifest.
2. Move, rather than copy, the obsolete material once the manifest is written.
3. Update `MANIFEST.md` if the moved material was previously described there.
4. Keep archives out of active launcher defaults and paper inputs.
