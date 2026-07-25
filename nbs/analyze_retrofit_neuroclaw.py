"""Measure what CONFIRM adds when retrofitted onto an existing neuroimaging agent.

The claim-generation integration study already routes NeuroClaw-adapted drafts
through CONFIRM's gates. It cannot say what the agent would have reported on its
own, so it shows the composite is safe without showing that CONFIRM changed
anything.

This script adds the missing arm. The same NeuroClaw-adapted contracts are
re-executed, and the agent's own statistical-critic panel decides which claims it
would report from the identical evidence. Comparing that arm against CONFIRM's
gate verdicts on the same claims isolates the retrofit effect.

Phases:
  checkpoints  re-execute drafted contracts (deterministic, no model call)
  judge        run the NeuroClaw-adapted panel over those checkpoints
  analyze      build the two-arm comparison
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bench.claim_evaluation_baselines import (
    NEUROCLAW_PERSONA_ORDER,
    NeuroClawPersonaOutput,
    label_blind_evidence,
    neuroclaw_adapted_decision,
    neuroclaw_persona_prompt,
    neuroclaw_persona_system,
)
from bench.progress import iter_progress
from confirm.contract import ClaimContract
from confirm.execution import evaluate_contract, jsonable, resolve_execution_root
from confirm.frozen_evidence import sha256_json
from confirm.llm import complete_structured_with_retries, make_llm

PROTOCOL_VERSION = "neuroclaimbench-retrofit-neuroclaw-v1"
DRAFTER = "neuroclaw_adapted_drafter"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _drafted_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(
        (row for row in records if row["drafter"] == DRAFTER),
        key=lambda row: row["claim_id"],
    )


def run_checkpoints(args: argparse.Namespace) -> None:
    """Re-execute each drafted contract and store a judge-ready checkpoint."""

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(item) for item in args.data_roots]

    records = _drafted_records(Path(args.generation_outcomes))
    mismatches: list[dict[str, str]] = []
    scope: list[dict[str, Any]] = []

    for row in iter_progress(
        records,
        total=len(records),
        desc="re-execute",
        enabled=not args.no_progress,
        unit="claim",
    ):
        contract = ClaimContract.model_validate(row["drafted_contract"])
        root = resolve_execution_root(contract, roots)
        verdict, results, _paths = evaluate_contract(
            contract,
            root,
            ref_effect=contract.gates.power.ref_effect,
        )
        if verdict.label != row["gate_label"]:
            mismatches.append(
                {
                    "claim_id": row["claim_id"],
                    "stored": str(row["gate_label"]),
                    "recomputed": verdict.label,
                }
            )
        checkpoint = {
            "task_id": row["claim_id"],
            "benchmark_case_id": row["claim_id"],
            "question_class": row["question_class"],
            "target_family": row["target_family"],
            "confirm_gate_label": verdict.label,
            "confirm_support": bool(row["confirm_support"]),
            "gate_results": jsonable(results),
        }
        (checkpoint_dir / f"{row['claim_id']}.json").write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True)
        )
        scope.append(
            {
                "claim_id": row["claim_id"],
                "question_class": row["question_class"],
                "target_family": row["target_family"],
                "unit": contract.estimand.unit,
                "confirm_gate_label": verdict.label,
                "confirm_support": bool(row["confirm_support"]),
            }
        )

    (out_dir / "task_scope.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in scope) + "\n"
    )
    (out_dir / "reexecution_audit.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "claim_count": len(scope),
                "label_mismatch_count": len(mismatches),
                "label_mismatches": mismatches,
                "deterministic_replay": not mismatches,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"re-executed {len(scope)} contracts; label mismatches: {len(mismatches)}")
    if mismatches:
        raise SystemExit("Re-execution did not reproduce stored gate labels")


def _judge_one(
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
        raise ValueError(f"Incompatible judge checkpoint: {destination}")

    llm = make_llm(model)
    votes: dict[str, NeuroClawPersonaOutput] = {}
    error: str | None = None
    try:
        for persona in NEUROCLAW_PERSONA_ORDER:
            parsed, _, _, _attempts = complete_structured_with_retries(
                llm,
                system=systems[persona],
                prompt=prompt,
                response_model=NeuroClawPersonaOutput,
                retries=retries,
            )
            votes[persona] = parsed
        decision = neuroclaw_adapted_decision(checkpoint, protocol_sha256, votes)
        supported = bool(decision.supported)
        reason = decision.reason
        details = decision.details
    except Exception as exc:  # noqa: BLE001 - recorded per claim
        error = str(exc)
        supported = False
        reason = "neuroclaw_error"
        details = {}

    record = {
        "task_id": checkpoint["task_id"],
        "question_class": checkpoint["question_class"],
        "target_family": checkpoint["target_family"],
        "confirm_gate_label": checkpoint["confirm_gate_label"],
        "confirm_support": checkpoint["confirm_support"],
        "neuroclaw_supported": supported,
        "neuroclaw_reason": reason,
        "neuroclaw_details": details,
        "model_spec": model,
        "prompt_sha256": prompt_hash,
        "protocol_sha256": protocol_sha256,
        "error": error,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    return record


def run_judge(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    work_dir = out_dir / ".work" / "neuroclaw_self_judge"
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"

    scope = [
        json.loads(line)
        for line in (out_dir / "task_scope.jsonl").read_text().splitlines()
        if line.strip()
    ]
    protocol_sha256 = sha256_json(
        {
            "version": PROTOCOL_VERSION,
            "personas": list(NEUROCLAW_PERSONA_ORDER),
            "model": args.model,
        }
    )
    payloads = [
        {
            "checkpoint_path": str(checkpoint_dir / f"{row['claim_id']}.json"),
            "work_path": str(work_dir / f"{row['claim_id']}.json"),
            "model": args.model,
            "retries": args.schema_retries,
            "protocol_sha256": protocol_sha256,
        }
        for row in scope
    ]

    records: list[dict[str, Any]] = []
    if args.max_workers == 1:
        for payload in iter_progress(
            payloads,
            total=len(payloads),
            desc="neuroclaw self-judge",
            enabled=not args.no_progress,
            unit="claim",
        ):
            records.append(_judge_one(**payload))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_judge_one, **payload) for payload in payloads]
            for future in iter_progress(
                as_completed(futures),
                total=len(futures),
                desc="neuroclaw self-judge",
                enabled=not args.no_progress,
                unit="claim",
            ):
                records.append(future.result())

    records.sort(key=lambda row: row["task_id"])
    (out_dir / "neuroclaw_self_decisions.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True, default=str) for row in records) + "\n"
    )
    errors = sum(1 for row in records if row["error"])
    print(f"judged {len(records)} claims; errors: {errors}")


def _latex_table(summary: dict[str, dict[str, int]]) -> str:
    pos = summary["positive"]
    neg = summary["negative_control"]
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{\textbf{CONFIRM retrofitted onto an existing neuroimaging",
            r"agent.} NeuroClaw-adapted drafts every contract. The agent arm lets its",
            r"own statistical-critic panel decide what to report from the same frozen",
            r"evidence; the retrofit arm replaces that decision with CONFIRM's gates.",
            r"Negative controls are site-confounded by construction. The positive",
            r"questions carry no reference labels, so that column measures",
            r"selectivity rather than accuracy.}",
            r"\label{tab:retrofit_neuroclaw}",
            r"\small",
            r"\setlength{\tabcolsep}{6pt}",
            r"\begin{tabular}{@{}lrr@{}}",
            r"\toprule",
            r"Reporting decision & Positives reported & Unsafe (neg.) \\",
            r"\midrule",
            f"NeuroClaw-adapted alone & {pos['neuroclaw']}/{pos['n']} & "
            f"{neg['neuroclaw']}/{neg['n']} \\\\",
            f"NeuroClaw-adapted + CONFIRM & {pos['confirm']}/{pos['n']} & "
            f"{neg['confirm']}/{neg['n']} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def run_analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    records = [
        json.loads(line)
        for line in (out_dir / "neuroclaw_self_decisions.jsonl").read_text().splitlines()
        if line.strip()
    ]

    summary: dict[str, dict[str, int]] = {}
    for question_class in ("positive", "negative_control"):
        rows = [row for row in records if row["question_class"] == question_class]
        summary[question_class] = {
            "n": len(rows),
            "neuroclaw": sum(1 for row in rows if row["neuroclaw_supported"]),
            "confirm": sum(1 for row in rows if row["confirm_support"]),
            "caught_by_confirm": sum(
                1 for row in rows if row["neuroclaw_supported"] and not row["confirm_support"]
            ),
            "added_by_confirm": sum(
                1 for row in rows if row["confirm_support"] and not row["neuroclaw_supported"]
            ),
            "agree": sum(
                1 for row in rows if row["neuroclaw_supported"] == row["confirm_support"]
            ),
            "errors": sum(1 for row in rows if row["error"]),
        }

    (out_dir / "retrofit_comparison.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "drafter": DRAFTER,
                "summary": summary,
                "interpretation": [
                    "Both arms score identical drafted contracts and identical frozen evidence.",
                    "Only the negative controls carry construction ground truth.",
                    "Positive questions have no reference labels; the column is selectivity.",
                    "NeuroClaw-adapted is a persona adaptation, not the released system.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    table = _latex_table(summary)
    (out_dir / "tab_retrofit_neuroclaw.tex").write_text(table)
    if args.paper_table:
        Path(args.paper_table).write_text(table)

    for question_class, row in summary.items():
        print(
            f"{question_class:17} n={row['n']:3}  neuroclaw={row['neuroclaw']:3}  "
            f"confirm={row['confirm']:3}  caught_by_confirm={row['caught_by_confirm']:3}  "
            f"added_by_confirm={row['added_by_confirm']:3}  errors={row['errors']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["checkpoints", "judge", "analyze"])
    parser.add_argument(
        "--generation-outcomes",
        default="review-stage/neuroclaimbench-v2.1/claim-generation-integration-v1/generation_outcomes.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/retrofit-neuroclaw-v1",
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=["data/prepared_data/evidence_partitions/benchmark_ready/cohorts"],
    )
    parser.add_argument("--model", default="openai:gpt-5.5")
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--paper-table", default="")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.phase == "checkpoints":
        run_checkpoints(parsed)
    elif parsed.phase == "judge":
        run_judge(parsed)
    else:
        run_analyze(parsed)
