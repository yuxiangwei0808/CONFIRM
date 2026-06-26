**Score: 7/10. Verdict: almost.**

Ready to begin full-scale experiments: **yes**. The benchmark runner is now meaningfully hardened: real brain-wide multiverse is implemented, CIs/counts are emitted, and the lockfile/audit path is real. The ComBat run improves final FCR from 3/16 to 2/16 while preserving 4/4 positives.

Ready for paper-level claims: **not yet**, except narrow claims like "the hardened benchmark infrastructure runs end-to-end and catches injected FDR/MDE traps." Not ready for broad claims that CONFIRM robustly reduces false neuroimaging discoveries across disease domains.

Top blockers:

1. **Residual FCR remains nontrivial and label-dependent.** Under ComBat, `cognition_fc_ukb_abcd` and `cognition_dyno_ukb_abcd` still confirm despite fragile labels. Minimum fix: expert/literature adjudication of fragile labels; separate "stress-test expected fragile" from true ground truth.
2. **CIs are still wide.** Final ComBat FCR is 2/16, CI [0.0155, 0.3835], and TPR is 4/4, CI [0.3976, 1.0]. Minimum fix: expand to ~30-40 negative/fragile and more than 4 positives.
3. **Main benchmark power and multiverse gates are still not discriminating.** Aggregate `+confound`, `+power`, and `+multiverse` FCR remain identical; the new hardening controls prove capability, but not much main-benchmark effect. Minimum fix: add naturally occurring or predeclared tasks where each gate independently changes verdicts.
4. **Skipped ready claims remain.** Three claims are still skipped for feature mismatch. Minimum fix: repair mappings or exclude them from the frozen inventory.
5. **Baselines remain ablations.** Acceptable for internal scaling, insufficient for top-venue comparative claims. Minimum fix: either keep claims scoped to gate-ladder ablations or add matched agent/execution-only baselines.

Positive evidence:

The FDR and MDE traps now directly address prior concerns: random-label FDR trap has 2 uncorrected hits but 0 BH-FDR hits, and the MDE trap rejects a selected p=0.00083 discovery hit due to power 0.2149. Tests pass: `10 passed`.

Resources:

No GPU needed. Use CPU for expanded benchmark sweeps, ComBat/no-ComBat sensitivity, bootstraps, and CI tightening. The limiting resource is **data/label quality**, not compute: expert adjudication, true site/scanner metadata, and more cross-disease prepared derivatives.
