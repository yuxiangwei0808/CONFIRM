"""Run and summarize direct-LLM and significance claim-evaluation baselines."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bench.benchmark import BenchmarkDataset
from bench.claim_evaluation_baselines import (
    DIRECT_JUDGE_SYSTEM_PROMPT,
    NEUROCLAW_PERSONA_ORDER,
    ClaimEvaluationDecision,
    DirectJudgeOutput,
    NeuroClawPersonaOutput,
    conventional_significance_decision,
    direct_judge_prompt,
    direct_llm_decision,
    neuroclaw_adapted_decision,
    neuroclaw_persona_prompt,
    neuroclaw_persona_system,
    veritas_adapted_decision,
)
from bench.progress import iter_progress
from confirm.llm import complete_structured_with_retries, make_llm
from nbs.claim_search_analysis_common import (
    iter_jsonl,
    sha256_file,
    sha256_json,
    write_csv_atomic,
    write_json_atomic,
)

PROTOCOL_VERSION = "neuroclaimbench-claim-evaluation-baselines-v1"
METHOD_LABELS = {
    "direct_llm_judge": "Direct LLM judge",
    "neuroclaw_adapted_judge": "NeuroClaw-adapted",
    "veritas_adapted": "VERITAS-adapted",
    "conventional_significance": "Discovery + replication significance",
    "confirm": "CONFIRM",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _task_scope(
    package_dir: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    dataset = BenchmarkDataset(package_dir)
    references = {
        reference.benchmark_case_id: reference for reference in dataset.references
    }
    cases = {case.benchmark_case_id: case for case in dataset.cases}
    rows = []
    for task in dataset.tasks:
        reference = references[task.benchmark_case_id]
        if not reference.score_eligible:
            continue
        case = cases[task.benchmark_case_id]
        rows.append(
            {
                "task_id": task.task_id,
                "benchmark_case_id": task.benchmark_case_id,
                "benchmark_track": case.benchmark_track,
                "target_family": case.target_family,
                "dataset_id": task.dataset_id,
                "unit": task.contract.estimand.unit,
            }
        )
    rows.sort(key=lambda row: row["task_id"])
    if len(rows) != 268:
        raise ValueError(f"Expected 268 score-eligible tasks, found {len(rows)}")
    return rows[:limit] if limit is not None else rows


def _protocol_payload(
    *,
    package_dir: Path,
    scope_path: Path,
    model: str,
    retries: int,
    limit: int | None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    implementations = [
        Path(__file__).resolve(),
        root / "src/bench/claim_evaluation_baselines.py",
    ]
    return {
        "version": PROTOCOL_VERSION,
        "implementation": [
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for path in implementations
        ],
        "package_manifest_sha256": sha256_file(
            package_dir / "benchmark_manifest.json"
        ),
        "task_scope_sha256": sha256_file(scope_path),
        "task_scope_count": sum(1 for _ in iter_jsonl(scope_path)),
        "task_scope_limit": limit,
        "methods": {
            "conventional_significance": {
                "alpha": 0.05,
                "rule": (
                    "unadjusted p < 0.05 in discovery and every executable "
                    "replication cohort with matching effect direction"
                ),
                "applicable_unit": "scalar",
                "multiplicity_adjustment": "none",
            },
            "direct_llm_judge": {
                "model": model,
                "schema_retries": retries,
                "system_prompt_sha256": sha256_json(
                    {"prompt": DIRECT_JUDGE_SYSTEM_PROMPT}
                ),
                "response_schema_sha256": sha256_json(
                    DirectJudgeOutput.model_json_schema()
                ),
                "applicable_units": ["scalar", "brainwide"],
            },
            "neuroclaw_adapted_judge": {
                "classification": "adapted",
                "repo": "https://github.com/CUHK-AIM-Group/NeuroClaw",
                "commit": "b9e3833a795b0f3a5d6348ffab814b0b4c904c3e",
                "component": (
                    "three-perspective critic personas with majority + "
                    "methodology-weighted aggregation"
                ),
                "model": model,
                "schema_retries": retries,
                "persona_order": list(NEUROCLAW_PERSONA_ORDER),
                "persona_system_sha256": {
                    persona: sha256_json(
                        {"system": neuroclaw_persona_system(persona)}
                    )
                    for persona in NEUROCLAW_PERSONA_ORDER
                },
                "response_schema_sha256": sha256_json(
                    NeuroClawPersonaOutput.model_json_schema()
                ),
                "applicable_unit": "scalar",
            },
            "veritas_adapted": {
                "classification": "adapted",
                "repo": "https://github.com/LucZot/veritas",
                "commit": "17dbdc96cef23c29a4efbf0291de6d6295908a17",
                "component": (
                    "compute_evidence_label deterministic epistemic label "
                    "(SUPPORTED->confirm; REFUTED/UNDERPOWERED/INVALID->abstain)"
                ),
                "sesoi_profile": "standard",
                "alpha": 0.05,
                "applicable_unit": "scalar",
            },
            "confirm": {
                "source": "frozen NeuroClaimBench v2.1 task outcomes",
            },
        },
        "decision_policy": {
            "references_joined_after_decisions": True,
            "direct_judge_identifiers_anonymized": True,
            "direct_judge_forbidden_inputs": [
                "benchmark references",
                "CONFIRM gate configuration or thresholds",
                "CONFIRM gate pass/fail fields",
                "CONFIRM verdict",
                "source mode",
                "holdout or feedback outcomes",
            ],
        },
    }


def _protocol_hash(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def freeze_protocol(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scope_path = out_dir / "task_scope.jsonl"
    _atomic_jsonl(
        scope_path,
        _task_scope(Path(args.package_dir), limit=args.limit),
    )
    payload = _protocol_payload(
        package_dir=Path(args.package_dir),
        scope_path=scope_path,
        model=args.model,
        retries=args.schema_retries,
        limit=args.limit,
    )
    document = {**payload, "protocol_sha256": _protocol_hash(payload)}
    write_json_atomic(out_dir / "protocol.json", document)
    return document


def _load_protocol(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze the protocol first: {path}")
    document = _read_json(path)
    stored_hash = str(document.pop("protocol_sha256"))
    if _protocol_hash(document) != stored_hash:
        raise ValueError("Stored protocol hash is invalid")
    current = _protocol_payload(
        package_dir=Path(args.package_dir),
        scope_path=Path(args.out_dir) / "task_scope.jsonl",
        model=args.model,
        retries=args.schema_retries,
        limit=args.limit,
    )
    if current != document:
        raise ValueError(
            "Current code, model, schema, or benchmark inputs do not match "
            "the frozen protocol"
        )
    return {**document, "protocol_sha256": stored_hash}


def _checkpoint(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    path = Path(args.checkpoint_dir) / f"{task_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = _read_json(path)
    if str(checkpoint.get("task_id")) != task_id:
        raise ValueError(f"Checkpoint task mismatch: {path}")
    return checkpoint


def run_significance(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    scope = list(iter_jsonl(Path(args.out_dir) / "task_scope.jsonl"))
    scalar = [row for row in scope if row["unit"] == "scalar"]
    for row in iter_progress(
        scalar,
        total=len(scalar),
        desc="conventional significance",
        enabled=not args.no_progress,
        unit="claim",
    ):
        decision = conventional_significance_decision(
            _checkpoint(args, str(row["task_id"])),
            str(protocol["protocol_sha256"]),
        )
        rows.append(decision.model_dump(mode="json"))
    rows.sort(key=lambda row: row["task_id"])
    path = Path(args.out_dir) / "conventional_significance_decisions.jsonl"
    _atomic_jsonl(path, rows)
    summary = {
        "task_count": len(rows),
        "supported_count": sum(bool(row["supported"]) for row in rows),
        "available_count": sum(bool(row["available"]) for row in rows),
        "protocol_sha256": protocol["protocol_sha256"],
        "decision_sha256": sha256_file(path),
    }
    write_json_atomic(
        Path(args.out_dir) / "conventional_significance_summary.json",
        summary,
    )
    return summary


def _judge_one(
    *,
    checkpoint_path: str,
    work_path: str,
    model: str,
    retries: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    checkpoint = _read_json(Path(checkpoint_path))
    prompt = direct_judge_prompt(checkpoint)
    prompt_hash = sha256_json(
        {"system": DIRECT_JUDGE_SYSTEM_PROMPT, "user": prompt}
    )
    destination = Path(work_path)
    if destination.exists():
        existing = _read_json(destination)
        if (
            existing.get("protocol_sha256") == protocol_sha256
            and existing.get("model_spec") == model
            and existing.get("prompt_sha256") == prompt_hash
        ):
            return existing
        raise ValueError(f"Incompatible direct-judge checkpoint: {destination}")

    llm = make_llm(model)
    try:
        parsed, _, _, attempts = complete_structured_with_retries(
            llm,
            system=DIRECT_JUDGE_SYSTEM_PROMPT,
            prompt=prompt,
            response_model=DirectJudgeOutput,
            retries=retries,
        )
        decision = direct_llm_decision(
            checkpoint,
            protocol_sha256,
            parsed,
        )
        error = None
    except Exception as exc:
        attempts = []
        decision = ClaimEvaluationDecision(
            task_id=str(checkpoint["task_id"]),
            benchmark_case_id=str(
                checkpoint.get("benchmark_item_id")
                or checkpoint.get("benchmark_case_id")
            ),
            method="direct_llm_judge",
            available=False,
            supported=False,
            reason="llm_judge_failed",
            protocol_sha256=protocol_sha256,
            direction=str(
                checkpoint["gate_results"]["contract"]["estimand"]["direction"]
            ),
            unit=str(checkpoint["gate_results"]["contract"]["estimand"]["unit"]),
            details={"error": str(exc)},
        )
        error = str(exc)
    record = {
        "task_id": str(checkpoint["task_id"]),
        "benchmark_case_id": decision.benchmark_case_id,
        "model_spec": model,
        "protocol_sha256": protocol_sha256,
        "system_prompt": DIRECT_JUDGE_SYSTEM_PROMPT,
        "user_prompt": prompt,
        "prompt_sha256": prompt_hash,
        "attempts": attempts,
        "decision": decision.model_dump(mode="json"),
        "error": error,
    }
    write_json_atomic(destination, record)
    return record


def _usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    usage_complete = True
    for record in records:
        for attempt in record.get("attempts") or []:
            calls += 1
            usage = (attempt.get("call_metadata") or {}).get("usage") or {}
            if not usage:
                usage_complete = False
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
    return {
        "llm_call_count": calls,
        "usage_complete": usage_complete,
        "prompt_tokens": prompt_tokens if usage_complete else None,
        "completion_tokens": completion_tokens if usage_complete else None,
        "total_tokens": total_tokens if usage_complete else None,
    }


def run_llm_judge(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    work_dir = out_dir / ".work" / "direct_llm_judge"
    work_dir.mkdir(parents=True, exist_ok=True)
    scope = list(iter_jsonl(out_dir / "task_scope.jsonl"))
    payloads = [
        {
            "checkpoint_path": str(
                Path(args.checkpoint_dir) / f"{row['task_id']}.json"
            ),
            "work_path": str(work_dir / f"{row['task_id']}.json"),
            "model": args.model,
            "retries": args.schema_retries,
            "protocol_sha256": str(protocol["protocol_sha256"]),
        }
        for row in scope
    ]
    records: list[dict[str, Any]] = []
    if args.max_workers == 1:
        for payload in iter_progress(
            payloads,
            total=len(payloads),
            desc="direct LLM judge",
            enabled=not args.no_progress,
            unit="claim",
        ):
            records.append(_judge_one(**payload))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_judge_one, **payload) for payload in payloads]
            completed = iter_progress(
                as_completed(futures),
                total=len(futures),
                desc="direct LLM judge",
                enabled=not args.no_progress,
                unit="claim",
            )
            for future in completed:
                records.append(future.result())

    records.sort(key=lambda row: row["task_id"])
    prompts = [
        {
            "task_id": row["task_id"],
            "model_spec": row["model_spec"],
            "system": row["system_prompt"],
            "user": row["user_prompt"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in records
    ]
    responses = [
        {"task_id": row["task_id"], "attempt_index": index, **attempt}
        for row in records
        for index, attempt in enumerate(row.get("attempts") or [], start=1)
    ]
    decisions = [row["decision"] for row in records]
    _atomic_jsonl(out_dir / "direct_llm_judge_prompts.jsonl", prompts)
    _atomic_jsonl(out_dir / "direct_llm_judge_responses.jsonl", responses)
    _atomic_jsonl(out_dir / "direct_llm_judge_decisions.jsonl", decisions)
    summary = {
        "task_count": len(records),
        "available_count": sum(row["decision"]["available"] for row in records),
        "supported_count": sum(row["decision"]["supported"] for row in records),
        "error_count": sum(row["error"] is not None for row in records),
        "model": args.model,
        "protocol_sha256": protocol["protocol_sha256"],
        **_usage(records),
    }
    write_json_atomic(out_dir / "direct_llm_judge_summary.json", summary)
    return summary


def run_veritas(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    scope = list(iter_jsonl(Path(args.out_dir) / "task_scope.jsonl"))
    scalar = [row for row in scope if row["unit"] == "scalar"]
    for row in iter_progress(
        scalar,
        total=len(scalar),
        desc="veritas-adapted",
        enabled=not args.no_progress,
        unit="claim",
    ):
        decision = veritas_adapted_decision(
            _checkpoint(args, str(row["task_id"])),
            str(protocol["protocol_sha256"]),
        )
        rows.append(decision.model_dump(mode="json"))
    rows.sort(key=lambda row: row["task_id"])
    path = Path(args.out_dir) / "veritas_adapted_decisions.jsonl"
    _atomic_jsonl(path, rows)
    summary = {
        "task_count": len(rows),
        "supported_count": sum(bool(row["supported"]) for row in rows),
        "available_count": sum(bool(row["available"]) for row in rows),
        "protocol_sha256": protocol["protocol_sha256"],
        "decision_sha256": sha256_file(path),
    }
    write_json_atomic(
        Path(args.out_dir) / "veritas_adapted_summary.json",
        summary,
    )
    return summary


def _neuroclaw_one(
    *,
    checkpoint_path: str,
    work_path: str,
    model: str,
    retries: int,
    protocol_sha256: str,
) -> dict[str, Any]:
    checkpoint = _read_json(Path(checkpoint_path))
    prompt = neuroclaw_persona_prompt(checkpoint)
    systems = {
        persona: neuroclaw_persona_system(persona)
        for persona in NEUROCLAW_PERSONA_ORDER
    }
    prompt_hash = sha256_json({"systems": systems, "user": prompt})
    destination = Path(work_path)
    if destination.exists():
        existing = _read_json(destination)
        if (
            existing.get("protocol_sha256") == protocol_sha256
            and existing.get("model_spec") == model
            and existing.get("prompt_sha256") == prompt_hash
        ):
            return existing
        raise ValueError(f"Incompatible neuroclaw-judge checkpoint: {destination}")

    llm = make_llm(model)
    votes: dict[str, NeuroClawPersonaOutput] = {}
    attempts_by_persona: dict[str, list[dict[str, Any]]] = {}
    error = None
    try:
        for persona in NEUROCLAW_PERSONA_ORDER:
            parsed, _, _, attempts = complete_structured_with_retries(
                llm,
                system=systems[persona],
                prompt=prompt,
                response_model=NeuroClawPersonaOutput,
                retries=retries,
            )
            votes[persona] = parsed
            attempts_by_persona[persona] = attempts
        decision = neuroclaw_adapted_decision(checkpoint, protocol_sha256, votes)
    except Exception as exc:
        decision = ClaimEvaluationDecision(
            task_id=str(checkpoint["task_id"]),
            benchmark_case_id=str(
                checkpoint.get("benchmark_item_id")
                or checkpoint.get("benchmark_case_id")
            ),
            method="neuroclaw_adapted_judge",
            available=False,
            supported=False,
            reason="neuroclaw_judge_failed",
            protocol_sha256=protocol_sha256,
            direction=str(
                checkpoint["gate_results"]["contract"]["estimand"]["direction"]
            ),
            unit=str(checkpoint["gate_results"]["contract"]["estimand"]["unit"]),
            details={"error": str(exc)},
        )
        error = str(exc)
    flat_attempts = [
        {**attempt, "persona": persona}
        for persona in NEUROCLAW_PERSONA_ORDER
        for attempt in attempts_by_persona.get(persona, [])
    ]
    record = {
        "task_id": str(checkpoint["task_id"]),
        "benchmark_case_id": decision.benchmark_case_id,
        "model_spec": model,
        "protocol_sha256": protocol_sha256,
        "persona_systems": systems,
        "user_prompt": prompt,
        "prompt_sha256": prompt_hash,
        "votes": {persona: vote.model_dump(mode="json") for persona, vote in votes.items()},
        "attempts": flat_attempts,
        "decision": decision.model_dump(mode="json"),
        "error": error,
    }
    write_json_atomic(destination, record)
    return record


def run_neuroclaw_judge(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    work_dir = out_dir / ".work" / "neuroclaw_adapted_judge"
    work_dir.mkdir(parents=True, exist_ok=True)
    scope = [
        row
        for row in iter_jsonl(out_dir / "task_scope.jsonl")
        if row["unit"] == "scalar"
    ]
    payloads = [
        {
            "checkpoint_path": str(
                Path(args.checkpoint_dir) / f"{row['task_id']}.json"
            ),
            "work_path": str(work_dir / f"{row['task_id']}.json"),
            "model": args.model,
            "retries": args.schema_retries,
            "protocol_sha256": str(protocol["protocol_sha256"]),
        }
        for row in scope
    ]
    records: list[dict[str, Any]] = []
    if args.max_workers == 1:
        for payload in iter_progress(
            payloads,
            total=len(payloads),
            desc="neuroclaw judge",
            enabled=not args.no_progress,
            unit="claim",
        ):
            records.append(_neuroclaw_one(**payload))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(_neuroclaw_one, **payload) for payload in payloads
            ]
            completed = iter_progress(
                as_completed(futures),
                total=len(futures),
                desc="neuroclaw judge",
                enabled=not args.no_progress,
                unit="claim",
            )
            for future in completed:
                records.append(future.result())

    records.sort(key=lambda row: row["task_id"])
    prompts = [
        {
            "task_id": row["task_id"],
            "model_spec": row["model_spec"],
            "persona_systems": row["persona_systems"],
            "user": row["user_prompt"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for row in records
    ]
    responses = [
        {"task_id": row["task_id"], "attempt_index": index, **attempt}
        for row in records
        for index, attempt in enumerate(row.get("attempts") or [], start=1)
    ]
    decisions = [row["decision"] for row in records]
    _atomic_jsonl(out_dir / "neuroclaw_adapted_judge_prompts.jsonl", prompts)
    _atomic_jsonl(out_dir / "neuroclaw_adapted_judge_responses.jsonl", responses)
    _atomic_jsonl(out_dir / "neuroclaw_adapted_judge_decisions.jsonl", decisions)
    summary = {
        "task_count": len(records),
        "available_count": sum(row["decision"]["available"] for row in records),
        "supported_count": sum(row["decision"]["supported"] for row in records),
        "error_count": sum(row["error"] is not None for row in records),
        "model": args.model,
        "protocol_sha256": protocol["protocol_sha256"],
        **_usage(records),
    }
    write_json_atomic(out_dir / "neuroclaw_adapted_judge_summary.json", summary)
    return summary


def finalize_decisions(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    scope = list(iter_jsonl(out_dir / "task_scope.jsonl"))
    scalar_ids = {row["task_id"] for row in scope if row["unit"] == "scalar"}
    all_ids = {row["task_id"] for row in scope}
    significance = [
        ClaimEvaluationDecision.model_validate(row)
        for row in iter_jsonl(out_dir / "conventional_significance_decisions.jsonl")
    ]
    judge = [
        ClaimEvaluationDecision.model_validate(row)
        for row in iter_jsonl(out_dir / "direct_llm_judge_decisions.jsonl")
    ]
    veritas = [
        ClaimEvaluationDecision.model_validate(row)
        for row in iter_jsonl(out_dir / "veritas_adapted_decisions.jsonl")
    ]
    neuroclaw = [
        ClaimEvaluationDecision.model_validate(row)
        for row in iter_jsonl(out_dir / "neuroclaw_adapted_judge_decisions.jsonl")
    ]
    if {row.task_id for row in significance} != scalar_ids:
        raise ValueError("Conventional-significance task coverage mismatch")
    if {row.task_id for row in judge} != all_ids:
        raise ValueError("Direct-judge task coverage mismatch")
    if {row.task_id for row in veritas} != scalar_ids:
        raise ValueError("VERITAS-adapted task coverage mismatch")
    if {row.task_id for row in neuroclaw} != scalar_ids:
        raise ValueError("NeuroClaw-adapted task coverage mismatch")
    decisions = sorted(
        [*significance, *judge, *veritas, *neuroclaw],
        key=lambda row: (row.task_id, row.method),
    )
    for decision in decisions:
        if decision.protocol_sha256 != protocol["protocol_sha256"]:
            raise ValueError(f"Protocol mismatch for {decision.task_id}")
    decision_path = out_dir / "baseline_decisions.jsonl"
    _atomic_jsonl(
        decision_path,
        [row.model_dump(mode="json") for row in decisions],
    )
    manifest = {
        "version": PROTOCOL_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "reference_labels_joined": False,
        "score_eligible_task_count": len(scope),
        "scalar_task_count": len(scalar_ids),
        "decision_count": len(decisions),
        "direct_judge_error_count": sum(
            row.method == "direct_llm_judge" and not row.available
            for row in decisions
        ),
        "neuroclaw_judge_error_count": sum(
            row.method == "neuroclaw_adapted_judge" and not row.available
            for row in decisions
        ),
        "files": {
            "task_scope": sha256_file(out_dir / "task_scope.jsonl"),
            "baseline_decisions": sha256_file(decision_path),
            "direct_llm_judge_prompts": sha256_file(
                out_dir / "direct_llm_judge_prompts.jsonl"
            ),
            "direct_llm_judge_responses": sha256_file(
                out_dir / "direct_llm_judge_responses.jsonl"
            ),
            "neuroclaw_adapted_judge_prompts": sha256_file(
                out_dir / "neuroclaw_adapted_judge_prompts.jsonl"
            ),
            "neuroclaw_adapted_judge_responses": sha256_file(
                out_dir / "neuroclaw_adapted_judge_responses.jsonl"
            ),
            "veritas_adapted_decisions": sha256_file(
                out_dir / "veritas_adapted_decisions.jsonl"
            ),
        },
    }
    write_json_atomic(out_dir / "decision_manifest.json", manifest)
    return manifest


def _stratum(case: Any, reference: Any, task: Any) -> str:
    if case.benchmark_track == "scientific":
        return "internal_scientific"
    if case.benchmark_track == "synthetic_stress":
        return "synthetic_control"
    if case.benchmark_track == "external_transfer":
        basis = (
            "literature"
            if reference.basis == "literature"
            else "constructed_control"
        )
        return f"external_{basis}_{task.dataset_id}"
    return case.benchmark_track


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHOD_LABELS:
        for stratum in sorted({row["stratum"] for row in records}):
            for disposition in ("confirm", "abstain"):
                selected = [
                    row
                    for row in records
                    if row["method"] == method
                    and row["stratum"] == stratum
                    and row["reference_disposition"] == disposition
                ]
                if not selected:
                    continue
                available = [row for row in selected if row["available"]]
                supported_count = sum(row["supported"] for row in available)
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "stratum": stratum,
                        "reference_disposition": disposition,
                        "eligible_count": len(selected),
                        "available_count": len(available),
                        "supported_count": supported_count,
                        "supported_rate": (
                            supported_count / len(available)
                            if available
                            else None
                        ),
                    }
                )
    return rows


def _primary_table(summary: list[dict[str, Any]], path: Path) -> None:
    index = {
        (row["method"], row["stratum"], row["reference_disposition"]): row
        for row in summary
    }

    def cell(method: str, stratum: str, disposition: str) -> str:
        row = index[(method, stratum, disposition)]
        return f"{row['supported_count']}/{row['available_count']}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Claim-evaluation baselines on common scalar coverage.} Higher recall is better; lower unsafe confirmation and synthetic-control support are better.}",
        r"\label{tab:claim_evaluation_baselines}",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Method & Recovery & Unsafe support & Synthetic support \\",
        r"\midrule",
    ]
    for method in (
        "direct_llm_judge",
        "neuroclaw_adapted_judge",
        "veritas_adapted",
        "conventional_significance",
        "confirm",
    ):
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{cell(method, 'internal_scientific', 'confirm')} & "
            f"{cell(method, 'internal_scientific', 'abstain')} & "
            f"{cell(method, 'synthetic_control', 'abstain')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    dataset = BenchmarkDataset(args.package_dir)
    cases = {row.benchmark_case_id: row for row in dataset.cases}
    references = {row.benchmark_case_id: row for row in dataset.references}
    outcomes = {row.task_id: row for row in dataset.outcomes}
    baseline = {
        (row.task_id, row.method): row
        for row in (
            ClaimEvaluationDecision.model_validate(value)
            for value in iter_jsonl(out_dir / "baseline_decisions.jsonl")
        )
    }
    records = []
    for task in dataset.tasks:
        reference = references[task.benchmark_case_id]
        if not reference.score_eligible:
            continue
        case = cases[task.benchmark_case_id]
        for method in METHOD_LABELS:
            if method == "confirm":
                outcome = outcomes[task.task_id]
                available = outcome.status == "completed"
                supported = outcome.confirm_outcome == "confirmed"
                reason = "evaluated" if available else str(outcome.error)
            else:
                decision = baseline.get((task.task_id, method))
                available = decision is not None and decision.available
                supported = bool(decision and decision.supported)
                reason = (
                    decision.reason
                    if decision is not None
                    else "method_not_applicable"
                )
            records.append(
                {
                    "task_id": task.task_id,
                    "benchmark_case_id": task.benchmark_case_id,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "stratum": _stratum(case, reference, task),
                    "benchmark_track": case.benchmark_track,
                    "target_family": case.target_family,
                    "dataset_id": task.dataset_id,
                    "unit": task.contract.estimand.unit,
                    "reference_disposition": reference.disposition,
                    "available": available,
                    "supported": supported,
                    "reason": reason,
                }
            )
    summary = _summary_rows(records)
    write_csv_atomic(out_dir / "joined_decisions.csv", records)
    write_csv_atomic(out_dir / "method_summary.csv", summary)
    common = [
        row
        for row in records
        if row["unit"] == "scalar"
        and row["stratum"] in {"internal_scientific", "synthetic_control"}
    ]
    write_csv_atomic(out_dir / "primary_common_coverage.csv", common)
    _primary_table(
        _summary_rows(common),
        out_dir / "tab_claim_evaluation_baselines.tex",
    )
    counts = {
        "internal_scientific_scalar": len(
            {
                row["task_id"]
                for row in records
                if row["method"] == "confirm"
                and row["stratum"] == "internal_scientific"
                and row["unit"] == "scalar"
            }
        ),
        "internal_scientific_brainwide": len(
            {
                row["task_id"]
                for row in records
                if row["method"] == "confirm"
                and row["stratum"] == "internal_scientific"
                and row["unit"] == "brainwide"
            }
        ),
        "synthetic_scalar": len(
            {
                row["task_id"]
                for row in records
                if row["method"] == "confirm"
                and row["stratum"] == "synthetic_control"
            }
        ),
    }
    expected = {
        "internal_scientific_scalar": 46,
        "internal_scientific_brainwide": 24,
        "synthetic_scalar": 150,
    }
    if counts != expected:
        raise ValueError(f"Coverage mismatch: observed={counts} expected={expected}")
    manifest = {
        "version": PROTOCOL_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "interpretation": [
            "retrospective_matched_claim_evaluation",
            "primary_three_method_comparison_uses_common_scalar_coverage",
            "external_datasets_are_reported_separately",
            "unresolved_references_are_excluded",
        ],
        "counts": counts,
        "files": {
            name: sha256_file(out_dir / name)
            for name in (
                "baseline_decisions.jsonl",
                "joined_decisions.csv",
                "method_summary.csv",
                "primary_common_coverage.csv",
                "tab_claim_evaluation_baselines.tex",
            )
        },
    }
    write_json_atomic(out_dir / "analysis_manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.phase == "protocol":
        return freeze_protocol(args)
    protocol = _load_protocol(args)
    if args.phase == "significance":
        return run_significance(args, protocol)
    if args.phase == "veritas":
        return run_veritas(args, protocol)
    if args.phase == "llm_judge":
        return run_llm_judge(args, protocol)
    if args.phase == "neuroclaw_judge":
        return run_neuroclaw_judge(args, protocol)
    if args.phase == "finalize":
        return finalize_decisions(args, protocol)
    if args.phase == "analyze":
        return analyze(args, protocol)
    if args.phase == "all":
        run_significance(args, protocol)
        run_veritas(args, protocol)
        run_llm_judge(args, protocol)
        run_neuroclaw_judge(args, protocol)
        finalize_decisions(args, protocol)
        return analyze(args, protocol)
    raise ValueError(args.phase)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=(
            "protocol",
            "significance",
            "veritas",
            "llm_judge",
            "neuroclaw_judge",
            "finalize",
            "analyze",
            "all",
        ),
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "review-stage/neuroclaimbench-v2.1/"
            "claim-evaluation-baselines-v1"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="review-stage/neuroclaimbench-v2.1/results/checkpoints",
    )
    parser.add_argument(
        "--package-dir",
        default="benchmark/neuroclaimbench-v2.1",
    )
    parser.add_argument("--model", default="openai:gpt-5.5")
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.schema_retries < 0:
        raise ValueError("--schema-retries must be nonnegative")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
