# NeuroClaimBench v2.1

This is the compact, metadata-only NeuroClaimBench v2.1 release using
`release_schema_version=2`.

- `cases.jsonl` stores canonical questions and frozen contracts.
- `references.jsonl` stores literature or constructed-control dispositions.
- `tasks.jsonl` stores exact evidence identities and executable contracts.
- `outcomes.jsonl` stores compact CONFIRM verdicts. Detailed gate bundles are
  retained in the checksummed audit archive.
- Constructed controls are never described as literature references.
- Low-powered but identifiable claims remain executable and are handled by the
  unchanged CONFIRM power gate.
- v2.1 is a retrospective benchmark revision. Its repair and eligibility
  policy was frozen before the v2.1 rerun.

Raw cohort data and the local PubMed abstract cache are not redistributed.
