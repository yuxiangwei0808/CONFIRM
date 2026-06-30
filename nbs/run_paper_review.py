#!/usr/bin/env python
"""Round-3 fresh-context review/audit via DIRECT API (bypasses flaky Codex MCP)."""
from __future__ import annotations
import sys, glob
from pathlib import Path

sys.path.insert(0, "src")
from confirm.env import load_env
from confirm.llm import make_llm

load_env()

SYSTEM = (
    "You are a senior, brutally honest WACV/CVPR area chair and reviewer. "
    "Do not reward effort. Be specific and concrete. Focus on whether every "
    "claim is matched by the evidence, internal consistency, and residual overclaims."
)

secs = sorted(glob.glob("paper/sec/*.tex"))
paper = "\n\n".join(f"% ==== {Path(s).name} ====\n{Path(s).read_text()}" for s in secs)
bib = Path("paper/references.bib").read_text()

USER = f"""Heavily REVISED WACV 2027 (applications track) submission: CONFIRM, a
claim-governance layer for agentic neuroimaging. Scope = FAITHFULNESS (lower
false-confirmation rate at fixed known-positive recall), NOT capability.

Since prior reviews the authors: (a) fixed a power-gate bug (it had fallen back to
the OBSERVED effect; it now judges power against a pre-declared MDE d=0.3,
fail-closed) and re-ran everything -> stress-suite false confirmations 1/177 ->
0/177 (exact upper bound 2.1%); (b) added external validation on UNSEEN cohorts
(NACC: 9/9 AD/MCI positives, 0/28 random-label controls, baseline 2/28;
ds000030: 0/14 positives, correctly withheld as NON-REPLICATED with the power
gate SATISFIED, plus 0/16 controls; combined 0/44 external nulls); (c) added a
baselines + leave-one-out ablation (only the full ladder reaches 0 on the
150-claim stress set; confound-coverage and replication load-bearing; multiverse
does not bind and is retained only for fragile-claim sign-instability); (d)
downgraded provenance to "frozen-configuration" (SHA-256 hashes + a results
manifest, NOT formal preregistration); (e) reworded overclaims.

Brutally honest audit + review:
(1) INTERNAL CONSISTENCY: does any number look inconsistent or unsupported across
    abstract/intro/experiments/conclusion? (Every headline number has been
    re-derived from raw artifacts and matches; flag anything that still looks off.)
(2) REMAINING OVERCLAIMS: provenance, "statistically admissible", "robust",
    label authority, and the title "Trustworthy ... Agentic Neuroimaging Discovery".
(3) Are the ds000030 framing (withheld by significance/replication/multiverse,
    power gate satisfied) and the ablation framing (multiverse non-binding,
    retained for fragile claims) honest and defensible, or do they invite "then
    why keep those gates"?
(4) The single most important remaining weakness.
(5) Final score 1-10 and accept / borderline / reject for WACV applications.

=== PAPER ===
{paper}

=== REFERENCES ===
{bib}
"""


def main() -> None:
    for spec in ["openai:gpt-5.5", "anthropic:claude-sonnet-4-6", "openai:gpt-4o",
                 "openrouter:deepseek/deepseek-chat"]:
        try:
            llm = make_llm(spec)
            if hasattr(llm, "max_tokens"):
                llm.max_tokens = 8000
            text = llm.complete(SYSTEM, USER)
            if text and text.strip():
                out = Path("review-stage/paper-review-round3.md")
                out.write_text(f"# Paper review round 3 (via {spec})\n\n{text}\n", encoding="utf-8")
                print(f"=== REVIEW via {spec} ===\n\n{text}")
                return
        except Exception as exc:  # noqa: BLE001
            print(f"[{spec} failed: {type(exc).__name__}: {exc}]")
    print("ALL REVIEWER MODELS FAILED")


if __name__ == "__main__":
    main()
