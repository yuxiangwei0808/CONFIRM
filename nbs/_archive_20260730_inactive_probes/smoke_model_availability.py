"""Probe whether each candidate drafting model is reachable before a sweep.

The multi-model drafting probe costs one full drafting pass per model, so an
unreachable model id or a missing key should surface here rather than halfway
through the sweep.
"""

from __future__ import annotations

import argparse
import sys

from confirm.llm import make_llm

# Claude is routed through OpenRouter: the direct Anthropic key returns 401.
CANDIDATES = [
    "openai:gpt-5.5",
    "openai:gpt-5.6-luna",
    "openai:gpt-5.4",
    "google:gemini-3.5-flash",
    "google:gemini-3.5-flash-lite",
    "openrouter:anthropic/claude-sonnet-5",
]

SYSTEM = "You are a terse assistant."
USER = "Reply with the single word: ok"


def probe(spec: str) -> tuple[str, str]:
    try:
        client = make_llm(spec)
    except Exception as exc:  # noqa: BLE001 - report any construction failure
        return "CONSTRUCT_FAIL", f"{type(exc).__name__}: {exc}"[:160]
    try:
        reply = client.complete(SYSTEM, USER)
    except Exception as exc:  # noqa: BLE001
        return "CALL_FAIL", f"{type(exc).__name__}: {exc}"[:160]
    return "OK", " ".join(str(reply).split())[:60]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None)
    args = parser.parse_args(argv)

    specs = args.model or CANDIDATES
    failures = 0
    for spec in specs:
        status, detail = probe(spec)
        if status != "OK":
            failures += 1
        print(f"{spec:34s} {status:14s} {detail}", flush=True)
    print(f"\n{len(specs) - failures}/{len(specs)} reachable")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
