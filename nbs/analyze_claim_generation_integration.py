"""Experiment 2: question->claim generation integration analysis.

Compares Direct GPT drafting with NeuroClaw-adapted drafting on the same fixed
questions, routing every drafted contract through the unchanged CONFIRM gates.

Phases:
  protocol  freeze implementation + drafter-prompt hashes and the fixed selection
  select    sample 10 positive questions/family + derive constructed negatives
  draft     draft + gate each question with each drafter (resumable, live LLM)
  analyze   aggregate the six integration metrics per drafter

This is an integration study, not benchmark accuracy: a newly drafted claim may
differ from any original labeled claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from bench.claim_generation_integration import (
    NEUROCLAW_DRAFTER_SYSTEM,
    GenerationOutcome,
    build_negative_controls,
    draft_and_gate,
)
from bench.progress import iter_progress
from bench.run_initial_claim_drafting import (
    DEFAULT_DATA_ROOTS,
    ClaimQuestion,
    _merge_catalogs,
    _preflight_context_from_catalog,
)
from confirm.agent import DOMAIN_PRIOR_SYSTEM_PROMPT
from confirm.llm import make_llm
from nbs.claim_search_analysis_common import (
    iter_jsonl,
    sha256_file,
    sha256_json,
    write_csv_atomic,
    write_json_atomic,
)

PROTOCOL_VERSION = "neuroclaimbench-claim-generation-integration-v1"
DRAFTERS = ("direct_gpt_drafter", "neuroclaw_adapted_drafter")
DRAFTER_LABELS = {
    "direct_gpt_drafter": "Direct GPT drafter",
    "neuroclaw_adapted_drafter": "NeuroClaw-adapted drafter",
}
FAMILIES = ("normative_fmri", "adhd", "asd", "ad_aging", "psychosis")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def _select(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(Path(args.questions)))
    questions = [ClaimQuestion.model_validate(row) for row in rows]
    by_family: dict[str, list[ClaimQuestion]] = defaultdict(list)
    for question in questions:
        by_family[question.target_family].append(question)
    positives: list[ClaimQuestion] = []
    for family in FAMILIES:
        family_qs = sorted(by_family.get(family, []), key=lambda q: q.claim_id)
        positives.extend(family_qs[: args.per_family])
    negatives = build_negative_controls(positives, per_family=args.negatives_per_family)
    selection = [
        {"question_class": "positive", "question": q.model_dump(mode="json")}
        for q in positives
    ] + [
        {"question_class": "negative_control", "question": q.model_dump(mode="json")}
        for q in negatives
    ]
    return selection


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = _select(args)
    _atomic_jsonl(out_dir / "selection.jsonl", selection)
    counts: dict[str, int] = defaultdict(int)
    for row in selection:
        counts[row["question_class"]] += 1
    return {"selected": len(selection), "counts": dict(counts)}


def _protocol_payload(args: argparse.Namespace, scope_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    implementations = [
        Path(__file__).resolve(),
        root / "src/bench/claim_generation_integration.py",
    ]
    return {
        "version": PROTOCOL_VERSION,
        "implementation": [
            {"path": str(p.relative_to(root)), "sha256": sha256_file(p)}
            for p in implementations
        ],
        "model": args.model,
        "schema_retries": args.schema_retries,
        "selection_sha256": sha256_file(scope_path),
        "drafters": {
            "direct_gpt_drafter": {
                "classification": "native",
                "system_prompt_sha256": sha256_json(
                    {"system": DOMAIN_PRIOR_SYSTEM_PROMPT}
                ),
            },
            "neuroclaw_adapted_drafter": {
                "classification": "adapted",
                "repo": "https://github.com/CUHK-AIM-Group/NeuroClaw",
                "commit": "b9e3833a795b0f3a5d6348ffab814b0b4c904c3e",
                "system_prompt_sha256": sha256_json(
                    {"system": NEUROCLAW_DRAFTER_SYSTEM}
                ),
            },
        },
        "interpretation": [
            "integration_study_not_benchmark_accuracy",
            "identical_schema_catalog_preflight_and_gates_across_drafters",
            "negative_controls_are_site_confounded_by_construction",
        ],
    }


def _protocol_hash(payload: dict[str, Any]) -> str:
    return sha256_json(payload)


def freeze_protocol(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    scope_path = out_dir / "selection.jsonl"
    if not scope_path.exists():
        raise FileNotFoundError("Run PHASE=select before protocol.")
    payload = _protocol_payload(args, scope_path)
    document = {**payload, "protocol_sha256": _protocol_hash(payload)}
    write_json_atomic(out_dir / "protocol.json", document)
    return document


def _load_protocol(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.out_dir) / "protocol.json"
    if not path.exists():
        raise FileNotFoundError("Freeze the protocol first.")
    document = _read_json(path)
    stored = str(document.pop("protocol_sha256"))
    current = _protocol_payload(args, Path(args.out_dir) / "selection.jsonl")
    if _protocol_hash(current) != stored or current != document:
        raise ValueError("Protocol mismatch; re-freeze after code/model/selection changes.")
    return {**document, "protocol_sha256": stored}


def _draft_one(
    *,
    item: dict[str, Any],
    drafter: str,
    catalog: dict[str, Any],
    preflight_context: Any,
    model: str,
    schema_retries: int,
    work_path: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    destination = Path(work_path)
    if destination.exists():
        existing = _read_json(destination)
        if existing.get("protocol_sha256") == protocol_sha256 and existing.get(
            "model"
        ) == model:
            return existing
    question = ClaimQuestion.model_validate(item["question"])
    llm = make_llm(model)
    outcome, prompts, responses = draft_and_gate(
        question,
        item["question_class"],
        catalog,
        llm,
        drafter,  # type: ignore[arg-type]
        schema_retries=schema_retries,
        preflight_context=preflight_context,
    )
    record = {
        "protocol_sha256": protocol_sha256,
        "model": model,
        "outcome": outcome.model_dump(mode="json"),
        "prompts": prompts,
        "responses": responses,
    }
    write_json_atomic(destination, record)
    return record


def run_draft(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    selection = list(iter_jsonl(out_dir / "selection.jsonl"))
    catalog = _merge_catalogs([Path(item) for item in DEFAULT_DATA_ROOTS])
    if not catalog["cohorts"]:
        raise ValueError("No readable cohort parquet files in default data roots.")
    preflight_context = _preflight_context_from_catalog(catalog)

    jobs: list[dict[str, Any]] = []
    for drafter in DRAFTERS:
        work_dir = out_dir / ".work" / drafter
        work_dir.mkdir(parents=True, exist_ok=True)
        for item in selection:
            claim_id = item["question"]["claim_id"]
            jobs.append(
                {
                    "item": item,
                    "drafter": drafter,
                    "catalog": catalog,
                    "preflight_context": preflight_context,
                    "model": args.model,
                    "schema_retries": args.schema_retries,
                    "work_path": str(work_dir / f"{claim_id}.json"),
                    "protocol_sha256": str(protocol["protocol_sha256"]),
                }
            )

    records: list[dict[str, Any]] = []
    if args.max_workers == 1:
        for job in iter_progress(
            jobs, total=len(jobs), desc="draft+gate", enabled=not args.no_progress, unit="job"
        ):
            records.append(_draft_one(**job))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_draft_one, **job) for job in jobs]
            for future in iter_progress(
                as_completed(futures), total=len(futures), desc="draft+gate",
                enabled=not args.no_progress, unit="job",
            ):
                records.append(future.result())

    outcomes = [record["outcome"] for record in records]
    _atomic_jsonl(out_dir / "generation_outcomes.jsonl", outcomes)
    return {
        "jobs": len(jobs),
        "schema_valid": sum(o["schema_valid"] for o in outcomes),
        "executable": sum(o["executable"] for o in outcomes),
        "confirm_support": sum(o["confirm_support"] for o in outcomes),
        "unsafe_support": sum(o["unsafe_support"] for o in outcomes),
    }


def analyze(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    outcomes = [
        GenerationOutcome.model_validate(row)
        for row in iter_jsonl(out_dir / "generation_outcomes.jsonl")
    ]
    rows: list[dict[str, Any]] = []
    for drafter in DRAFTERS:
        for question_class in ("positive", "negative_control"):
            group = [
                o
                for o in outcomes
                if o.drafter == drafter and o.question_class == question_class
            ]
            if not group:
                continue
            n = len(group)
            executable = [o for o in group if o.executable]
            rows.append(
                {
                    "drafter": drafter,
                    "drafter_label": DRAFTER_LABELS[drafter],
                    "question_class": question_class,
                    "n_questions": n,
                    "schema_valid": sum(o.schema_valid for o in group),
                    "schema_valid_rate": sum(o.schema_valid for o in group) / n,
                    "executable": len(executable),
                    "executable_rate": len(executable) / n,
                    "aligned": sum(o.aligned for o in group),
                    "unsupported_variable_or_preflight": sum(
                        o.error_disposition == "unsupported_variable_or_preflight"
                        for o in group
                    ),
                    "unsupported_cohort_or_predictor": sum(
                        o.error_disposition == "unsupported_cohort_or_predictor"
                        for o in group
                    ),
                    "gate_available": sum(o.gate_available for o in group),
                    "confirm_support": sum(o.confirm_support for o in group),
                    "unsafe_support": sum(o.unsafe_support for o in group),
                }
            )
    write_csv_atomic(out_dir / "integration_metrics.csv", rows)
    _write_table(rows, out_dir / "tab_claim_generation_integration.tex")
    manifest = {
        "version": PROTOCOL_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "outcome_count": len(outcomes),
        "interpretation": [
            "integration_study_not_benchmark_accuracy",
            "same_gates_govern_claims_from_both_drafters",
            "negative_controls_are_site_confounded_by_construction",
        ],
        "files": {
            name: sha256_file(out_dir / name)
            for name in (
                "selection.jsonl",
                "generation_outcomes.jsonl",
                "integration_metrics.csv",
                "tab_claim_generation_integration.tex",
            )
        },
    }
    write_json_atomic(out_dir / "analysis_manifest.json", manifest)
    return manifest


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    index = {(row["drafter"], row["question_class"]): row for row in rows}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Question-to-claim integration.} Both drafters use the "
        r"same contract schema, catalog, preflight, and CONFIRM gates. Rates on "
        r"positive questions; unsafe support on site-confounded negative controls.}",
        r"\label{tab:claim_generation_integration}",
        r"\small",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Drafter & Schema-valid & Executable & CONFIRM support & Unsafe (neg.) \\",
        r"\midrule",
    ]
    for drafter in DRAFTERS:
        pos = index.get((drafter, "positive"))
        neg = index.get((drafter, "negative_control"))
        if pos is None:
            continue
        unsafe = (
            f"{neg['unsafe_support']}/{neg['n_questions']}" if neg else "n/a"
        )
        lines.append(
            f"{DRAFTER_LABELS[drafter]} & "
            f"{pos['schema_valid']}/{pos['n_questions']} & "
            f"{pos['executable']}/{pos['n_questions']} & "
            f"{pos['confirm_support']}/{pos['executable']} & "
            f"{unsafe} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.phase == "select":
        return run_select(args)
    if args.phase == "protocol":
        return freeze_protocol(args)
    protocol = _load_protocol(args)
    if args.phase == "draft":
        return run_draft(args, protocol)
    if args.phase == "analyze":
        return analyze(args, protocol)
    if args.phase == "all":
        run_draft(args, protocol)
        return analyze(args, protocol)
    raise ValueError(args.phase)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("select", "protocol", "draft", "analyze", "all"),
        required=True,
    )
    parser.add_argument(
        "--questions",
        default="review-stage/initial-claims-all-gpt55/claim_questions.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/claim-generation-integration-v1",
    )
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--negatives-per-family", type=int, default=2)
    parser.add_argument("--model", default="openai:gpt-5.5")
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
