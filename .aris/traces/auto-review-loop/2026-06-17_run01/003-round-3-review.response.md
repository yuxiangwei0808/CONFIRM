# Round 3 Review — Codex gpt-5.5 (xhigh)

threadId: 019ed7dd-905b-7fd1-8dba-3fb5c2cfb153 (fresh thread; prior threads lost to MCP restarts, context re-supplied)
Date: 2026-06-17

**Score: 6.4/10. Verdict: Almost (not ready yet). STOP CONDITION MET (>=6 AND almost).**

## Raw response (verbatim)

Score: 6.4/10. Verdict: Almost, not ready yet.

B1 is cleared as a hard blocker if implemented as described. The search-provenance gate is the right fix: it attacks the search process, not just the threshold. The subject-level split fix also matters; row-level leakage would have been fatal if left in.

B2 is mostly cleared, with a narrower interpretation. The benchmark no longer grades itself if MAIN excludes provisional/author-assigned labels and metrics are foregrounded there. However MAIN is now small: 4/4 known-positive recall and 0/21 false-confirmed is directionally good but not yet a convincing benchmark.

Conditional on Round 4 delivering the three items, could reach ready / weak accept territory, ~7.0-7.4/10, but only if: (1) new positives are genuinely externally labeled and not selected because CONFIRM can pass them; (2) NeuroClaw is run fairly, same inputs/resources, no hidden advantage for CONFIRM; (3) psychosis result has real cross-cohort replication, not pooled significance; (4) the paper leads with calibrated claims ("reduces false confirmations by abstention under governance gates," not "discovers neuroscience reliably"); (5) coverage/abstention analyzed as a first-class tradeoff, not buried.

Critical weakness you may still underestimate: coverage. MAIN coverage 6/28 means safety is partly achieved by refusing most claims. Acceptable only if the paper frames CONFIRM as a claim-governance layer, not a high-throughput discovery engine. Reviewers will ask: "Does this just confirm easy canonical facts and abstain elsewhere?" Round 4 must answer directly.

Second: positive recall too narrow and AD-heavy. Psychosis case/control with cross-cohort replication would help a lot.

Third: FCR upper bounds still not tight (0/21 and 0/23, CI upper ~0.15-0.16) — claim "no observed false confirmations," not "low false confirmation rate" without qualification.

Round 4 bar for "ready": MAIN externally labeled positives ~10-15 across >1 disease/domain; real NeuroClaw baseline; fishing/null stress tests remain zero confirmed after new data; cross-cohort psychosis replication succeeds or is honestly presented as conservative governance; a coverage-vs-FCR curve showing what each gate buys and costs.

Bottom line: you fixed the two real blockers. The work is now credible. Not yet broadly persuasive enough for NeurIPS/ICML unless Round 4 proves this generalizes beyond a small, AD-heavy, high-abstention setting.
