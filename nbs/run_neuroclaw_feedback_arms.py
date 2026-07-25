"""Run matched feedback arms in which an existing agent revises its own claims.

Stage 1 (`parents`) re-executes NeuroClaw-adapted's drafted contracts and keeps
the ones CONFIRM did not support. Those are the claims the agent must improve.

Stage 2 (`run`) replays a bounded search over those parents under one of two
arms. Both share the model, budget, validator, and cumulative multiplicity
policy, and differ only in the feedback the agent receives:

  self_critique      the agent critiques its own unsupported claim, with the
                     gate-specific diagnosis withheld
  confirm_diagnosis  the agent receives CONFIRM's typed gate localization

Stage 3 (`analyze`) compares the arms against the no-feedback baseline, which is
simply the parents' own gate outcome before any revision.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from bench.claim_generation_integration import draft_and_gate
from bench.neuroclaw_feedback import (
    NeuroClawDiagnosisGenerator,
    NeuroClawSelfCritiqueGenerator,
)
from bench.run_initial_claim_drafting import (
    DEFAULT_DATA_ROOTS,
    ClaimQuestion,
    _merge_catalogs,
    _preflight_context_from_catalog,
)
from confirm.candidate_preflight import CandidatePreflightContext
from confirm.claim_search import CandidateClaimProposal, ClaimSearchConfig, run_claim_search
from confirm.contract import ClaimContract
from confirm.execution import evaluate_contract, jsonable, resolve_execution_root
from confirm.llm import make_llm

PROTOCOL_VERSION = "neuroclaimbench-neuroclaw-feedback-arms-v1"
DRAFTER = "neuroclaw_adapted_drafter"
FAMILIES = ("ad_aging", "adhd", "asd", "normative_fmri", "psychosis")

ARMS = {
    "self_critique": NeuroClawSelfCritiqueGenerator,
    "confirm_diagnosis": NeuroClawDiagnosisGenerator,
}


@contextmanager
def exclusive_stage(out_dir: Path, stage: str) -> Iterator[None]:
    """Refuse to start a stage that another process is already running.

    Two concurrent writers over the same per-item cache silently corrupt a run,
    because each writes its own realization and the last writer wins.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / f".lock_{stage}"
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = lock_path.read_text() if lock_path.exists() else "unknown"
        raise SystemExit(
            f"Stage '{stage}' is already locked by: {holder.strip()}\n"
            f"If that process is definitely dead, remove {lock_path} and retry."
        )
    os.write(handle, f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
    os.close(handle)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _evaluator(data_roots: list[Path]):
    def evaluate(candidate: CandidateClaimProposal) -> dict[str, Any]:
        contract = candidate.proposed_contract
        if contract is None:
            raise ValueError("candidate has no executable proposed_contract")
        root = resolve_execution_root(contract, data_roots)
        verdict, results, _paths = evaluate_contract(
            contract,
            root,
            ref_effect=contract.gates.power.ref_effect,
        )
        return {"final_label": verdict.label, "gate_results": jsonable(results)}

    return evaluate


def _draft_one(
    *,
    question_payload: dict[str, Any],
    catalog: dict[str, Any],
    model: str,
    schema_retries: int,
    work_path: str,
) -> dict[str, Any]:
    destination = Path(work_path)
    if destination.exists():
        return json.loads(destination.read_text())

    question = ClaimQuestion.model_validate(question_payload)
    outcome, _prompts, _responses = draft_and_gate(
        question,
        "positive",
        catalog,
        make_llm(model),
        DRAFTER,
        schema_retries=schema_retries,
        preflight_context=_preflight_context_from_catalog(catalog),
        data_roots=list(DEFAULT_DATA_ROOTS),
    )
    record = outcome.model_dump(mode="json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, sort_keys=True, default=str))
    return record


def run_draft(args: argparse.Namespace) -> None:
    """Draft contracts with the agent's own persona across the question corpus."""

    out_dir = Path(args.out_dir)
    with exclusive_stage(out_dir, "draft"):
        work_dir = out_dir / ".work" / "draft"
        work_dir.mkdir(parents=True, exist_ok=True)

        rows = [
            json.loads(line)
            for line in Path(args.questions).read_text().splitlines()
            if line.strip()
        ]
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_family[row["target_family"]].append(row)
        selected: list[dict[str, Any]] = []
        for family in FAMILIES:
            family_rows = sorted(by_family.get(family, []), key=lambda r: r["claim_id"])
            selected.extend(family_rows[: args.per_family])

        catalog = _merge_catalogs([Path(item) for item in DEFAULT_DATA_ROOTS])
        payloads = [
            {
                "question_payload": row,
                "catalog": catalog,
                "model": args.model,
                "schema_retries": args.schema_retries,
                "work_path": str(work_dir / f"{row['claim_id']}.json"),
            }
            for row in selected
        ]

        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_draft_one, **payload) for payload in payloads]
            for future in as_completed(futures):
                records.append(future.result())

        records.sort(key=lambda row: row["claim_id"])
        (out_dir / "drafted_outcomes.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True, default=str) for row in records) + "\n"
        )
        confirmed = sum(1 for row in records if row.get("confirm_support"))
        print(
            f"drafted {len(records)} questions; confirmed {confirmed}; "
            f"abstained {len(records) - confirmed}"
        )


def run_parents(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    with exclusive_stage(out_dir, "parents"):
        _run_parents(args)


def _run_parents(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    roots = [Path(item) for item in args.data_roots]

    if args.parent_source == "synthetic_controls":
        _run_control_parents(args)
        return

    scaled = out_dir / "drafted_outcomes.jsonl"
    source = scaled if scaled.exists() else Path(args.generation_outcomes)
    records = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    drafted = sorted(
        (
            row
            for row in records
            if row.get("drafter", DRAFTER) == args.drafter
            and row["question_class"] == "positive"
        ),
        key=lambda row: row["claim_id"],
    )
    print(f"parent source: {source}")

    parents: list[dict[str, Any]] = []
    confirmed = 0
    not_executable = 0
    for row in drafted:
        if not row.get("drafted_contract"):
            # Schema-valid draft that preflight could not execute; not an abstention.
            not_executable += 1
            continue
        contract = ClaimContract.model_validate(row["drafted_contract"])
        root = resolve_execution_root(contract, roots)
        verdict, results, _paths = evaluate_contract(
            contract,
            root,
            ref_effect=contract.gates.power.ref_effect,
        )
        if verdict.label == "confirmed":
            confirmed += 1
            continue
        parents.append(
            {
                "claim_id": row["claim_id"],
                "target_family": row["target_family"],
                "contract": contract.model_dump(mode="json"),
                "gate_verdict": jsonable(verdict.gates),
                "gate_results": jsonable(results),
                "parent_label": verdict.label,
            }
        )

    (out_dir / "parents.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in parents) + "\n"
    )
    (out_dir / "parents_summary.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "drafter": args.drafter,
                "drafted_count": len(drafted),
                "not_executable_count": not_executable,
                "executed_count": len(drafted) - not_executable,
                "confirmed_without_feedback": confirmed,
                "abstained_parent_count": len(parents),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"drafted {len(drafted)}; not executable {not_executable}; "
        f"executed {len(drafted) - not_executable}; "
        f"confirmed without feedback {confirmed}; abstained parents {len(parents)}"
    )


def _run_control_parents(args: argparse.Namespace) -> None:
    """Use frozen site-confounded benchmark controls as safety parents.

    The controls are already executable contracts, so the agent never drafts
    them. That matters: when the agent fills in a ClaimContract it declares
    ``site`` as a covariate, CONFIRM adjusts for it, and the planted confound
    disarms before any gate runs.
    """

    out_dir = Path(args.out_dir)
    roots = [Path(item) for item in args.data_roots]
    rows = [
        json.loads(line)
        for line in Path(args.benchmark_tasks).read_text().splitlines()
        if line.strip()
    ]
    controls = sorted(
        (
            row
            for row in rows
            if str(row["contract"].get("claim_id", "")).startswith(args.control_prefix)
        ),
        key=lambda row: row["contract"]["claim_id"],
    )

    parents: list[dict[str, Any]] = []
    confirmed = 0
    for row in controls:
        contract = ClaimContract.model_validate(row["contract"])
        root = resolve_execution_root(contract, roots)
        verdict, results, _paths = evaluate_contract(
            contract,
            root,
            ref_effect=contract.gates.power.ref_effect,
        )
        if verdict.label == "confirmed":
            confirmed += 1
            continue
        parents.append(
            {
                "claim_id": contract.claim_id,
                "target_family": "synthetic_control",
                "contract": contract.model_dump(mode="json"),
                "gate_verdict": jsonable(verdict.gates),
                "gate_results": jsonable(results),
                "parent_label": verdict.label,
            }
        )

    (out_dir / "parents.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in parents) + "\n"
    )
    (out_dir / "parents_summary.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "parent_source": "synthetic_controls",
                "control_prefix": args.control_prefix,
                "control_count": len(controls),
                "confirmed_without_feedback": confirmed,
                "abstained_parent_count": len(parents),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"controls {len(controls)}; already confirmed {confirmed}; "
        f"safety parents {len(parents)}"
    )


def _run_one(
    *,
    row: dict[str, Any],
    arm: str,
    model: str,
    rounds: int,
    candidates: int,
    data_roots: list[str],
    work_path: str,
) -> dict[str, Any]:
    destination = Path(work_path)
    if destination.exists():
        return json.loads(destination.read_text())

    roots = [Path(item) for item in data_roots]
    contract = ClaimContract.model_validate(row["contract"])
    config = ClaimSearchConfig(max_rounds=rounds, max_candidates_per_round=candidates)
    preflight = CandidatePreflightContext.from_roots(roots)
    generator = ARMS[arm](make_llm(model), preflight_context=preflight)

    error: str | None = None
    try:
        state = run_claim_search(
            contract,
            row["gate_verdict"],
            row["gate_results"],
            config=config,
            candidate_generator=generator,
            evaluator=_evaluator(roots),
            preflight_context=preflight,
        )
        payload = jsonable(state.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001 - recorded per parent
        error = str(exc)
        payload = {}

    record = {
        "claim_id": row["claim_id"],
        "target_family": row["target_family"],
        "arm": arm,
        "parent_label": row["parent_label"],
        "llm_calls": len(generator.prompt_records),
        "state": payload,
        "error": error,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, sort_keys=True, default=str))
    return record


def run_arm(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    with exclusive_stage(out_dir, f"arm_{args.arm}"):
        _run_arm(args)


def _run_arm(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    work_dir = out_dir / ".work" / args.arm
    work_dir.mkdir(parents=True, exist_ok=True)
    parents = [
        json.loads(line)
        for line in (out_dir / "parents.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        parents = parents[: args.limit]

    payloads = [
        {
            "row": row,
            "arm": args.arm,
            "model": args.model,
            "rounds": args.rounds,
            "candidates": args.candidates,
            "data_roots": list(args.data_roots),
            "work_path": str(work_dir / f"{row['claim_id']}.json"),
        }
        for row in parents
    ]

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_run_one, **payload) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda row: row["claim_id"])
    (out_dir / f"arm_{args.arm}.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True, default=str) for row in records) + "\n"
    )
    errors = sum(1 for row in records if row["error"])
    calls = sum(row["llm_calls"] for row in records)
    print(f"arm={args.arm} parents={len(records)} llm_calls={calls} errors={errors}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["draft", "parents", "run"])
    parser.add_argument("--arm", choices=sorted(ARMS), default="confirm_diagnosis")
    parser.add_argument(
        "--generation-outcomes",
        default="review-stage/neuroclaimbench-v2.1/claim-generation-integration-v1/generation_outcomes.jsonl",
    )
    parser.add_argument("--drafter", default="neuroclaw_adapted_drafter")
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/neuroclaw-feedback-v1",
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=["data/prepared_data/evidence_partitions/benchmark_ready/cohorts"],
    )
    parser.add_argument("--model", default="openai:gpt-5.5")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--questions",
        default="review-stage/initial-claims-all-gpt55/claim_questions.jsonl",
    )
    parser.add_argument("--per-family", type=int, default=62)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument(
        "--parent-source",
        choices=["drafts", "synthetic_controls"],
        default="drafts",
    )
    parser.add_argument(
        "--benchmark-tasks",
        default="benchmark/neuroclaimbench-v2.1/tasks.jsonl",
    )
    parser.add_argument("--control-prefix", default="neg_site_confound")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.phase == "draft":
        run_draft(parsed)
    elif parsed.phase == "parents":
        run_parents(parsed)
    else:
        run_arm(parsed)
