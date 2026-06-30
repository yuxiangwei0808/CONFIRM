"""Compare retry, feedback, single-shot proposals, and iterative claim search."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.run_agentic_proposal_benchmark import run as run_single_shot_benchmark
from bench.run_iterative_claim_search_replay import run as run_search_replay

DEFAULT_INPUT = "review-stage/claim-search-llm-20260626/source/multi_model_claim_source.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    single_dir = out_dir / "single-shot-proposal"
    search_dir = out_dir / "iterative-search"
    single = run_single_shot_benchmark(argparse.Namespace(input=args.input, out_dir=str(single_dir)))
    search = run_search_replay(
        argparse.Namespace(
            input=args.input,
            out_dir=str(search_dir),
            max_rounds=args.max_rounds,
            max_candidates=args.max_candidates,
            schema_retries=args.schema_retries,
            llm=args.llm,
            data_root=args.data_root,
            external_data_root=args.external_data_root,
        )
    )
    iterative_summary = search["summary"]
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "E16 one-step versus LLM iterative claim-generation benchmark.",
        "input": args.input,
        "llm": args.llm,
        "config": {
            "max_rounds": args.max_rounds,
            "max_candidates": args.max_candidates,
            "schema_retries": args.schema_retries,
        },
        "summary": {
            "generic_retry": single["summary"]["generic_retry"],
            "structured_feedback": single["summary"]["structured_feedback"],
            "single_shot_proposal_layer": single["summary"]["claim_proposal_layer"],
            "iterative_candidate_generation": {
                "candidate_count": iterative_summary["candidate_count"],
                "valid_connected_candidate_count": iterative_summary["valid_connected_candidate_count"],
                "valid_connected_candidate_rate": iterative_summary["valid_connected_candidate_rate"],
                "admissible_evaluation_count": iterative_summary["admissible_evaluation_count"],
                "confirmed_on_excluded_evidence_count": iterative_summary["confirmed_on_excluded_evidence_count"],
                "false_current_data_confirmation_count": iterative_summary["false_current_data_confirmation_count"],
                "hacking_block_count": iterative_summary["hacking_block_count"],
                "stopped_without_confirmation_count": sum(
                    count
                    for reason, count in iterative_summary["stopped_reason_counts"].items()
                    if reason != "confirmed"
                ),
                "stopped_reason_counts": iterative_summary["stopped_reason_counts"],
            },
        },
        "single_shot_artifact": str(single_dir / "agentic_proposal_benchmark.json"),
        "iterative_search_artifact": str(search_dir / "iterative_candidate_replay.json"),
    }
    path = out_dir / "iterative_claim_search_benchmark.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="review-stage/claim-search-llm-20260626/llm-iterative-benchmark")
    parser.add_argument("--llm", default=None, help="LLM spec such as openai:gpt-4o; defaults to CONFIRM_LLM")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--data-root", action="append", default=None)
    parser.add_argument("--external-data-root", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
