"""Run phased retrospective holdout/external audits for frozen initial claims."""

from __future__ import annotations

import argparse
import json

from confirm.frozen_evidence import (
    build_evidence_preflight,
    evaluate_query_plan,
    freeze_initial_claims,
    summarize_initial_claim_evidence,
)


def run(args: argparse.Namespace) -> dict:
    if args.phase == "freeze":
        return freeze_initial_claims(args.initial_results, args.out_dir)
    if args.phase == "preflight":
        return build_evidence_preflight(
            args.out_dir,
            args.evidence_manifest,
            evidence_roots=args.evidence_root,
            source_roots=args.source_root,
            schedule_all_parents=True,
        )
    if args.phase == "evaluate":
        return evaluate_query_plan(
            args.out_dir,
            max_workers=args.max_workers,
            parallel_backend=args.parallel_backend,
            progress=not args.no_progress,
        )
    return summarize_initial_claim_evidence(args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=["freeze", "preflight", "evaluate", "summarize"],
    )
    parser.add_argument(
        "--initial-results",
        default="review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--evidence-manifest",
        default="data/prepared_data/evidence_partitions/manifest.json",
    )
    parser.add_argument(
        "--evidence-root",
        action="append",
        default=None,
        help="Repeatable root containing exact HOLDOUT/EXTERNAL partition parquets.",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=None,
        help="Repeatable root containing exact source DISC/REP partition parquets.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.evidence_root is None:
        args.evidence_root = ["data/prepared_data/evidence_partitions/cohorts"]
    if args.source_root is None:
        args.source_root = [
            "data/prepared_data/evidence_partitions/benchmark_ready/cohorts",
            "data/prepared_data/evidence_partitions/cohorts",
        ]
    result = run(args)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "out_dir": args.out_dir,
                "status": "completed",
                "summary": {
                    key: value
                    for key, value in result.items()
                    if key
                    in {
                        "observed_counts",
                        "preflight_status_counts",
                        "deduplicated_query_task_count",
                        "status_counts",
                        "interpretation_label_counts",
                        "overall",
                    }
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
