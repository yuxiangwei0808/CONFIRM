"""Command-line interface for CONFIRM."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from confirm.agent import run_claim, run_question
from confirm.ingest.abide import AbideAdapter
from confirm.ingest.adhd200 import Adhd200Adapter
from confirm.ingest.generic_csv import GenericCsvAdapter
from confirm.ingest.oasis1 import Oasis1Adapter

LOGGER = logging.getLogger(__name__)


def _cmd_run(args: argparse.Namespace) -> int:
    verdict = run_claim(args.contract, args.data_dir, args.out, command=sys.argv)
    print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    return 0 if verdict.label in args.accept_label else 1


def _cmd_ask(args: argparse.Namespace) -> int:
    verdict = run_question(args.question, args.data_dir, args.out, approve=not args.auto)
    print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
    return 0 if verdict.label in args.accept_label else 1


def _cmd_ingest(args: argparse.Namespace) -> int:
    if args.cohort == "abide":
        adapter = AbideAdapter(data_dir=args.source, n_subjects=args.n_subjects)
    elif args.cohort == "adhd200":
        adapter = Adhd200Adapter(data_dir=args.source, n_subjects=args.n_subjects)
    elif args.cohort == "oasis1":
        adapter = Oasis1Adapter(data_dir=args.source)
    elif args.cohort == "generic_csv":
        if not args.raw_csv or not args.mapping_yaml:
            raise SystemExit("generic_csv requires --raw-csv and --mapping-yaml")
        adapter = GenericCsvAdapter(args.raw_csv, args.mapping_yaml)
    else:
        raise SystemExit(f"Unknown cohort: {args.cohort}")
    parquet_path, dict_path = adapter.write(args.out)
    LOGGER.info("Wrote %s and %s", parquet_path, dict_path)
    print(json.dumps({"parquet": str(parquet_path), "dictionary": str(dict_path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="confirm", description="CONFIRM statistical claim governance")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a claim contract")
    run.add_argument("--contract", required=True)
    run.add_argument("--data-dir", default="data/canonical")
    run.add_argument("--out", required=True)
    run.add_argument("--accept-label", action="append", default=["confirmed"])
    run.set_defaults(func=_cmd_run)

    ask = sub.add_parser("ask", help="Draft and run a claim from a natural-language question")
    ask.add_argument("question")
    ask.add_argument("--data-dir", default="data/canonical")
    ask.add_argument("--out", required=True)
    ask.add_argument("--auto", action="store_true", help="Skip the contract approval prompt")
    ask.add_argument("--accept-label", action="append", default=["confirmed"])
    ask.set_defaults(func=_cmd_ask)

    ingest = sub.add_parser("ingest", help="Ingest a cohort to canonical parquet")
    ingest.add_argument("--cohort", required=True, choices=["abide", "adhd200", "oasis1", "generic_csv"])
    ingest.add_argument("--out", default="data/canonical")
    ingest.add_argument("--source", default=None, help="Dataset cache directory for nilearn fetchers")
    ingest.add_argument("--n-subjects", type=int, default=None)
    ingest.add_argument("--raw-csv", default=None)
    ingest.add_argument("--mapping-yaml", default=None)
    ingest.set_defaults(func=_cmd_ingest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
