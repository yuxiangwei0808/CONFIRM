"""Replay existing CONFIRM failures through iterative candidate generation."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bench.run_claim_proposal_replay import _contract_from_row, _iter_initial_rows
from confirm.candidate_preflight import CandidatePreflightContext
from confirm.claim_search import ClaimSearchConfig, ClaimSearchState, run_claim_search, summarize_claim_search
from confirm.claim_search import CandidateClaimProposal
from confirm.agent import _jsonable, _load_canonical, _run_brainwide_contract, _run_scalar_contract
from confirm.contract import ClaimContract
from confirm.llm import get_llm, make_llm

DEFAULT_INPUT = "review-stage/claim-search-llm-20260626/source/multi_model_claim_source.json"
DEFAULT_DATA_ROOTS = (
    "data/prepared_data/benchmark_ready/cohorts",
    "data/prepared_data/smri_disease",
    "data/prepared_data/cluster_recovered",
    "data/prepared_data/fmri_descriptors",
    "data/canonical",
)


def _needs_search(row: dict[str, Any]) -> bool:
    return bool(
        not row.get("draft_success")
        or not row.get("gate_success")
        or not row.get("estimand_match", True)
        or str(row.get("gate_verdict_label")) != "confirmed"
    )


def _cohort_path(data_roots: list[Path], cohort: str) -> Path:
    for candidate in _cohort_aliases(cohort):
        for root in data_roots:
            path = root / f"{candidate}.parquet"
            if path.exists():
                return path
    roots = ", ".join(str(root) for root in data_roots)
    raise FileNotFoundError(f"Cohort {cohort!r} was not found in data roots: {roots}")


def _cohort_aliases(cohort: str) -> list[str]:
    aliases = [cohort]
    suffixes = (
        "_DISC_SITES",
        "_REP_SITES",
        "_DISC",
        "_REP",
        "_CN",
    )
    for suffix in suffixes:
        if cohort.endswith(suffix):
            aliases.append(cohort[: -len(suffix)])
    for split in ("_DISC_s", "_REP_s"):
        if split in cohort:
            aliases.append(cohort.split(split, 1)[0])
    return list(dict.fromkeys(item for item in aliases if item))


def _execute_candidate_contract(contract: ClaimContract, data_roots: list[Path]) -> dict[str, Any]:
    discovery_path = _cohort_path(data_roots, contract.discovery_cohort)
    replication_paths = [_cohort_path(data_roots, cohort) for cohort in contract.replication_cohorts]
    discovery_df = _load_canonical(discovery_path)
    replication_dfs = [_load_canonical(path) for path in replication_paths]
    if contract.estimand.unit == "brainwide":
        verdict, results = _run_brainwide_contract(contract, discovery_df, replication_dfs)
    else:
        verdict, results = _run_scalar_contract(
            contract,
            discovery_df,
            replication_dfs,
            ref_effect=contract.gates.power.ref_effect,
        )
    return {
        "final_label": verdict.label,
        "gate_results": _jsonable(
            {
                "contract": contract.model_dump(mode="json"),
                "data_paths": {
                    "discovery": str(discovery_path),
                    "replication": [str(path) for path in replication_paths],
                },
                **results,
            }
        ),
    }


def _candidate_evaluator(data_roots: list[Path]):
    def evaluator(candidate: CandidateClaimProposal) -> dict[str, Any]:
        return _execute_candidate_contract(candidate.proposed_contract, data_roots)

    return evaluator


def _configured_roots(values: list[str] | None, defaults: tuple[str, ...] = ()) -> list[Path]:
    return [Path(item) for item in (values if values is not None else list(defaults))]


def _log(message: str) -> None:
    print(message, flush=True)


def _row_for_state(source_row: dict[str, Any], state: Any, candidate_generator_model_spec: str) -> dict[str, Any]:
    evaluations = state.evaluations
    valid = [item for item in evaluations if item.validation.ok]
    blocked = [item for item in evaluations if item.blocked_reason]
    execution_errors = [item for item in evaluations if item.execution_error]
    final_labels = [str(item.final_label) for item in evaluations if item.final_label]
    source_model_spec = source_row.get("model_spec")
    return {
        "model_spec": source_model_spec,
        "source_model_spec": source_model_spec,
        "candidate_generator_model_spec": candidate_generator_model_spec,
        "claim_id": source_row.get("claim_id"),
        "source_verdict": source_row.get("gate_verdict_label"),
        "source_scoring_label": source_row.get("source_scoring_label") or source_row.get("scoring_label") or source_row.get("label_class"),
        "source_label_authority": source_row.get("source_label_authority") or source_row.get("label_authority"),
        "source_result_path": source_row.get("source_result_path"),
        "known_negative_or_fragile_source": _known_negative_or_fragile_source(source_row),
        "candidate_count": len(state.candidate_history),
        "valid_candidate_count": len(valid),
        "blocked_candidate_count": len(blocked),
        "confirmed_candidate_count": len(state.confirmed_candidates),
        "exploratory_confirmed_count": sum(1 for item in evaluations if item.final_label == "exploratory_confirmed"),
        "external_confirmed_count": sum(1 for item in evaluations if item.external_confirmed),
        "execution_error_count": len(execution_errors),
        "stopped_reason": state.stopped_reason,
        "transform_types": ";".join(candidate.transform_type for candidate in state.candidate_history),
        "blocked_reasons": ";".join(str(item.blocked_reason) for item in blocked),
        "execution_errors": " | ".join(str(item.execution_error) for item in execution_errors),
        "final_labels": ";".join(final_labels),
    }


def _known_negative_or_fragile_source(source_row: dict[str, Any]) -> bool:
    claim_id = str(source_row.get("claim_id") or "")
    labels = {
        str(source_row.get("source_scoring_label") or "").lower(),
        str(source_row.get("scoring_label") or "").lower(),
        str(source_row.get("label_class") or "").lower(),
        str(source_row.get("source_verdict") or "").lower(),
        str(source_row.get("gate_verdict_label") or "").lower(),
    }
    negative_terms = {
        "known_null",
        "null_expected",
        "random_null",
        "fragile",
        "underpowered",
        "under_powered",
        "non_replicated",
    }
    return claim_id.startswith("neg_") or bool(labels & negative_terms)


def _replay_specific_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    negative_rows = [row for row in rows if row.get("known_negative_or_fragile_source")]
    negative_exploratory = sum(int(row.get("exploratory_confirmed_count") or 0) for row in negative_rows)
    return {
        "known_negative_or_fragile_search_count": len(negative_rows),
        "known_negative_or_fragile_exploratory_confirmed_count": negative_exploratory,
    }


def _run_single_claim(
    *,
    index: int,
    total: int,
    row: dict[str, Any],
    config: ClaimSearchConfig,
    data_roots: list[Path],
    external_data_roots: list[Path],
    candidate_evaluation: str,
    llm_model_spec: str,
    llm: Any,
    preflight_context: CandidatePreflightContext | None,
) -> dict[str, Any]:
    claim_id = str(row.get("claim_id"))
    try:
        contract = _contract_from_row(row)
        if contract is None or not isinstance(row.get("gate_verdict"), dict):
            raise ValueError("missing executable contract or gate verdict")
        results = row.get("gate_results") if isinstance(row.get("gate_results"), dict) else None
        evaluator = _candidate_evaluator(data_roots) if candidate_evaluation == "on" else None
        external_evaluator = (
            _candidate_evaluator(external_data_roots)
            if candidate_evaluation == "on" and external_data_roots
            else None
        )
        state = run_claim_search(
            contract,
            row["gate_verdict"],
            results,
            config=config,
            llm=llm,
            evaluator=evaluator,
            external_evaluator=external_evaluator,
            excluded_validation_available=candidate_evaluation == "on" and external_evaluator is not None,
            preflight_context=preflight_context,
        )
    except Exception as exc:
        return {
            "index": index,
            "total": total,
            "claim_id": claim_id,
            "status": "skipped",
            "skip": {"claim_id": claim_id, "reason": str(exc)},
            "message": f"[claim {index}/{total}] skipped claim_id={claim_id} reason={exc}",
        }

    output_row = _row_for_state(row, state, llm_model_spec)
    return {
        "index": index,
        "total": total,
        "claim_id": claim_id,
        "status": "completed",
        "row": output_row,
        "state": state.model_dump(mode="json"),
        "message": (
            f"[claim {index}/{total}] done claim_id={claim_id} "
            f"candidates={output_row['candidate_count']} valid={output_row['valid_candidate_count']} "
            f"exploratory_confirmed={output_row['exploratory_confirmed_count']} "
            f"stopped={output_row['stopped_reason']}"
        ),
    }


def _run_single_claim_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config = ClaimSearchConfig.model_validate(payload["config"])
    llm_spec = payload.get("llm_spec")
    llm = make_llm(llm_spec) if llm_spec else get_llm()
    llm_model_spec = llm_spec or getattr(llm, "model", type(llm).__name__)
    data_roots = [Path(item) for item in payload["data_roots"]]
    preflight_context = CandidatePreflightContext.from_roots(data_roots)
    return _run_single_claim(
        index=int(payload["index"]),
        total=int(payload["total"]),
        row=payload["row"],
        config=config,
        data_roots=data_roots,
        external_data_roots=[Path(item) for item in payload["external_data_roots"]],
        candidate_evaluation=str(payload["candidate_evaluation"]),
        llm_model_spec=llm_model_spec,
        llm=llm,
        preflight_context=preflight_context,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    config = ClaimSearchConfig(
        max_rounds=args.max_rounds,
        max_candidates_per_round=args.max_candidates,
        llm_schema_retries=args.schema_retries,
    )
    data_roots = _configured_roots(getattr(args, "data_root", None), DEFAULT_DATA_ROOTS)
    external_data_roots = _configured_roots(getattr(args, "external_data_root", None))
    candidate_evaluation = str(getattr(args, "candidate_evaluation", "on"))
    preflight_context = CandidatePreflightContext.from_roots(data_roots)
    llm = make_llm(args.llm) if args.llm else get_llm()
    llm_model_spec = args.llm or getattr(llm, "model", type(llm).__name__)
    max_workers = max(1, int(getattr(args, "max_workers", 1) or 1))
    parallel_backend = str(getattr(args, "parallel_backend", "process"))
    active_parallel_backend = parallel_backend
    rows: list[dict[str, Any]] = []
    states: list[ClaimSearchState] = []
    skipped: list[dict[str, str]] = []
    rows_by_index: dict[int, dict[str, Any]] = {}
    states_by_index: dict[int, ClaimSearchState] = {}
    skipped_by_index: dict[int, dict[str, str]] = {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def refresh_lists() -> None:
        nonlocal rows, states, skipped
        rows = [rows_by_index[index] for index in sorted(rows_by_index)]
        states = [states_by_index[index] for index in sorted(states_by_index)]
        skipped = [skipped_by_index[index] for index in sorted(skipped_by_index)]

    def write_current(status: str) -> dict[str, Any]:
        refresh_lists()
        result = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": "E15 LLM-driven iterative candidate replay.",
            "status": status,
            "input": str(source),
            "llm_model": llm_model_spec,
            "max_workers": max_workers,
            "parallel_backend": active_parallel_backend,
            "config": config.model_dump(mode="json"),
            "candidate_evaluation": candidate_evaluation,
            "data_roots": [str(root) for root in data_roots],
            "external_data_roots": [str(root) for root in external_data_roots],
            "summary": {**summarize_claim_search(states), **_replay_specific_summary(rows)},
            "skipped": skipped,
            "rows": rows,
            "states": [state.model_dump(mode="json") for state in states],
        }
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        _log(f"[checkpoint] status={status} completed={len(rows)} skipped={len(skipped)} -> {json_path}")
        return result

    json_path = out_dir / "iterative_candidate_replay.json"
    csv_path = out_dir / "iterative_candidate_replay.csv"
    checkpoint_every = max(0, int(getattr(args, "checkpoint_every", 0) or 0))
    initial_rows = [row for row in _iter_initial_rows(payload) if _needs_search(row)]

    _log(
        "[start] iterative claim search "
        f"input={source} out_dir={out_dir} llm={llm_model_spec} "
        f"searchable_claims={len(initial_rows)} max_rounds={config.max_rounds} "
        f"max_candidates={config.max_candidates_per_round} candidate_evaluation={candidate_evaluation} "
        f"max_workers={max_workers} parallel_backend={parallel_backend}"
    )

    def record_result(result: dict[str, Any]) -> None:
        index = int(result["index"])
        if result["status"] == "completed":
            rows_by_index[index] = result["row"]
            states_by_index[index] = ClaimSearchState.model_validate(result["state"])
        else:
            skipped_by_index[index] = result["skip"]
        _log(str(result.get("message") or ""))
        if checkpoint_every and (len(rows_by_index) + len(skipped_by_index)) % checkpoint_every == 0:
            write_current("running")

    if max_workers == 1:
        for index, row in enumerate(initial_rows, start=1):
            claim_id = str(row.get("claim_id"))
            _log(f"[claim {index}/{len(initial_rows)}] start claim_id={claim_id} source_label={row.get('gate_verdict_label')}")
            record_result(
                _run_single_claim(
                    index=index,
                    total=len(initial_rows),
                    row=row,
                    config=config,
                    data_roots=data_roots,
                    external_data_roots=external_data_roots,
                    candidate_evaluation=candidate_evaluation,
                    llm_model_spec=llm_model_spec,
                    llm=llm,
                    preflight_context=preflight_context,
                )
            )
    else:
        worker_count = min(max_workers, len(initial_rows)) if initial_rows else 1
        tasks = [
            {
                "index": index,
                "total": len(initial_rows),
                "row": row,
                "config": config.model_dump(mode="json"),
                "data_roots": [str(root) for root in data_roots],
                "external_data_roots": [str(root) for root in external_data_roots],
                "candidate_evaluation": candidate_evaluation,
                "llm_spec": args.llm,
            }
            for index, row in enumerate(initial_rows, start=1)
        ]
        for task in tasks:
            _log(
                f"[claim {task['index']}/{task['total']}] queued "
                f"claim_id={task['row'].get('claim_id')} source_label={task['row'].get('gate_verdict_label')}"
            )

        def run_parallel(backend: str) -> None:
            nonlocal active_parallel_backend
            active_parallel_backend = backend
            executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
            _log(f"[workers] launching {backend} pool workers={worker_count}")
            with executor_cls(max_workers=worker_count) as pool:
                future_to_task = {pool.submit(_run_single_claim_worker, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        index = int(task["index"])
                        claim_id = str(task["row"].get("claim_id"))
                        result = {
                            "index": index,
                            "total": int(task["total"]),
                            "claim_id": claim_id,
                            "status": "skipped",
                            "skip": {"claim_id": claim_id, "reason": f"worker failed: {exc}"},
                            "message": f"[claim {index}/{task['total']}] skipped claim_id={claim_id} reason=worker failed: {exc}",
                        }
                    record_result(result)

        try:
            run_parallel(parallel_backend)
        except (OSError, PermissionError) as exc:
            if parallel_backend != "process":
                raise
            _log(f"[workers] process pool unavailable ({exc}); falling back to thread pool")
            run_parallel("thread")

    result = write_current("completed")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="review-stage/claim-search-llm-20260626/llm-candidate-replay")
    parser.add_argument("--llm", default=None, help="LLM spec such as openai:gpt-4o; defaults to CONFIRM_LLM")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker processes for claim-level replay parallelism.")
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--candidate-evaluation", choices=["on", "off"], default="on")
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="Repeatable root containing canonical cohort parquet files. Defaults cover prepared sMRI/fMRI roots.",
    )
    parser.add_argument(
        "--external-data-root",
        action="append",
        default=None,
        help="Repeatable optional external/holdout cohort root. Passing this enables external confirmation attempts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
