"""Portable validation, execution, and scoring for the frozen public benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bench.benchmark import BenchmarkDataset
from confirm.cli import REPORTING_STANDARDS
from confirm.execution import evaluate_contract, jsonable
from confirm.verdict import EVIDENCE_TIER_REQUIRED_GATES, classify_support


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def indexed(rows, field):
    result = {}
    for row in rows:
        key = getattr(row, field)
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def verify(root: Path) -> tuple[BenchmarkDataset, dict]:
    root = root.resolve()
    listed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name in listed:
            raise ValueError(f"Invalid or repeated checksum path: {name}")
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        listed.add(name)
    required = {"cases.jsonl", "references.jsonl", "tasks.jsonl", "outcomes.jsonl", "benchmark_manifest.json"}
    if not required <= listed:
        raise ValueError(f"Missing checksums: {sorted(required - listed)}")
    data = BenchmarkDataset(root)
    cases = indexed(data.cases, "benchmark_case_id")
    refs = indexed(data.references, "reference_id")
    tasks = indexed(data.tasks, "task_id")
    outcomes = indexed(data.outcomes, "task_id")
    if set(outcomes) != set(tasks):
        raise ValueError("Frozen outcomes and tasks have different IDs")
    for case in data.cases:
        if case.reference_id not in refs or refs[case.reference_id].benchmark_case_id != case.benchmark_case_id:
            raise ValueError(f"Reference link mismatch: {case.benchmark_case_id}")
        if len(case.task_ids) != len(set(case.task_ids)):
            raise ValueError(f"Repeated task link: {case.benchmark_case_id}")
        for task_id in case.task_ids:
            if task_id not in tasks or tasks[task_id].benchmark_case_id != case.benchmark_case_id:
                raise ValueError(f"Task link mismatch: {task_id}")
    for ref in data.references:
        if ref.benchmark_case_id not in cases or cases[ref.benchmark_case_id].reference_id != ref.reference_id:
            raise ValueError(f"Case link mismatch: {ref.reference_id}")
        if ref.score_eligible and ref.disposition == "unresolved":
            raise ValueError(f"Unresolved reference marked score eligible: {ref.reference_id}")
    for task in data.tasks:
        case = cases.get(task.benchmark_case_id)
        if case is None or task.task_id not in case.task_ids:
            raise ValueError(f"Missing case link: {task.task_id}")
        if outcomes[task.task_id].benchmark_case_id != task.benchmark_case_id:
            raise ValueError(f"Outcome link mismatch: {task.task_id}")
        if (task.discovery_cohort != task.contract.discovery_cohort
                or task.replication_cohorts != task.contract.replication_cohorts):
            raise ValueError(f"Task and contract cohort mismatch: {task.task_id}")
        if task.contract_sha256 != case.contract_sha256:
            raise ValueError(f"Task and case contract hash mismatch: {task.task_id}")
    counts = {name: len(getattr(data, name)) for name in ("cases", "references", "tasks", "outcomes")}
    manifest = json.loads((root / "benchmark_manifest.json").read_text())
    for name in ("cases", "references", "tasks", "outcomes"):
        if manifest["files"][name]["sha256"] != digest(root / f"{name}.jsonl"):
            raise ValueError(f"Frozen record hash mismatch: {name}")
    if any(manifest["counts"][k] != v for k, v in counts.items()):
        raise ValueError("Manifest counts differ from the files")
    return data, {"verified_files": len(listed), **counts,
                  "score_eligible_references": sum(r.score_eligible for r in data.references),
                  "case_tracks": dict(Counter(c.benchmark_track for c in data.cases)),
                  "executable_case_tracks": dict(Counter(cases[t.benchmark_case_id].benchmark_track for t in data.tasks))}


def frozen_predictions(data: BenchmarkDataset, standard: str) -> list[dict]:
    rows = []
    for outcome in data.outcomes:
        row = {"task_id": outcome.task_id, "standard": standard,
               "source": "frozen_confirm_outcomes"}
        gates = (outcome.gate_verdict or {}).get("gates")
        complete = isinstance(gates, dict) and all(
            type(gates.get(name)) is bool
            for name in EVIDENCE_TIER_REQUIRED_GATES["confirmed"]
        )
        if outcome.status == "completed" and complete:
            row.update(status="completed", supported=classify_support(gates, REPORTING_STANDARDS[standard]).supported)
        else:
            row.update(status="error", error=outcome.error or "Missing or invalid completed gate vector")
        rows.append(row)
    return rows


def score(data: BenchmarkDataset, rows: list[dict], allow_partial: bool = False,
          literature_scope: str = "all") -> dict:
    if literature_scope not in {"all", "scalar"}:
        raise ValueError(f"Invalid literature scope: {literature_scope}")
    tasks = indexed(data.tasks, "task_id")
    cases = indexed(data.cases, "benchmark_case_id")
    refs = indexed(data.references, "reference_id")
    predictions = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id not in tasks or task_id in predictions:
            raise ValueError(f"Unknown or duplicate task_id: {task_id}")
        status = row.get("status", "completed")
        if not isinstance(status, str) or status not in {"completed", "error"}:
            raise ValueError(f"Invalid prediction status: {task_id}")
        if status == "completed" and type(row.get("supported")) is not bool:
            raise ValueError(f"supported must be a JSON boolean: {task_id}")
        if status == "error" and "supported" in row:
            raise ValueError(f"An error cannot also be a reporting decision: {task_id}")
        if status == "error" and (not isinstance(row.get("error"), str) or not row["error"].strip()):
            raise ValueError(f"An error row requires an explanatory error string: {task_id}")
        predictions[task_id] = {**row, "status": status}
    missing = sorted(set(tasks) - predictions.keys())
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} task predictions; use --allow-partial for an explicitly partial evaluation")
    groups = {}
    for track in sorted({c.benchmark_track for c in data.cases}):
        group_tasks = [t for t in data.tasks if cases[t.benchmark_case_id].benchmark_track == track]
        summaries = {}
        for name, basis, disposition in (
            ("literature_recall", "literature", "confirm"),
            ("literature_false_support", "literature", "abstain"),
            ("control_false_support", "constructed_control", "abstain"),
        ):
            for strength in ("combined", "strict", "provisional") if basis == "literature" else ("combined",):
                eligible = [t for t in group_tasks
                            if (r := refs[cases[t.benchmark_case_id].reference_id]).score_eligible
                            and r.basis == basis and r.disposition == disposition
                            and (basis != "literature" or literature_scope == "all" or t.contract.estimand.unit == "scalar")
                            and (strength == "combined" or r.strength == strength)]
                submitted = [t for t in eligible if t.task_id in predictions]
                completed = [t for t in eligible if predictions.get(t.task_id, {}).get("status") == "completed"]
                supported = sum(predictions[t.task_id]["supported"] for t in completed)
                summaries[f"{name}_{strength}"] = {
                    "supported": supported, "evaluated": len(submitted), "completed": len(completed),
                    "eligible": len(eligible),
                    "errors": sum(predictions.get(t.task_id, {}).get("status") == "error" for t in eligible),
                    "missing": sum(t.task_id not in predictions for t in eligible),
                    "rate": supported / len(submitted) if submitted else None,
                }
        groups[track] = {"tasks": len(group_tasks), "metrics": summaries}
    return {"task_count": len(tasks), "submitted_predictions": len(predictions),
            "literature_scope": literature_scope, "error_scoring_policy": "count_as_withheld",
            "completed": sum(r["status"] == "completed" for r in predictions.values()),
            "errors": sum(r["status"] == "error" for r in predictions.values()),
            "missing": len(missing),
            "partial": bool(missing) or any(r["status"] == "error" for r in predictions.values()),
            "by_track": groups}


def run_tasks(data: BenchmarkDataset, roots: list[Path], out: Path, standard: str,
              task_ids: list[str] | None = None) -> dict:
    if standard not in REPORTING_STANDARDS:
        raise ValueError(f"Unknown reporting standard: {standard}")
    tasks = indexed(data.tasks, "task_id")
    selected = list(tasks) if task_ids is None else task_ids
    if len(selected) != len(set(selected)) or set(selected) - tasks.keys():
        raise ValueError("Requested task IDs must be unique IDs from tasks.jsonl")
    if any(not name or name in {".", ".."} or "/" in name or "\\" in name for name in selected):
        raise ValueError("Task IDs must be single filenames")
    outcomes = indexed(data.outcomes, "task_id")
    out.mkdir(parents=True, exist_ok=False)
    cache = {}
    predictions = []
    for task_id in selected:
        task = tasks[task_id]
        names = [task.discovery_cohort, *task.replication_cohorts]
        row = {"task_id": task_id, "standard": standard}
        try:
            # Random controls require the frozen, already-generated partitions.
            hashes = outcomes[task_id].generated_cohort_hashes if task.generator_spec else task.partition_hashes
            matching = None
            for root in roots:
                if not all((root / f"{n}.parquet").is_file() for n in names):
                    continue
                valid = True
                for name in names:
                    path = (root / f"{name}.parquet").resolve()
                    if path not in cache:
                        cache[path] = digest(path)
                    valid = valid and bool(hashes.get(name)) and cache[path] == hashes.get(name)
                if valid:
                    matching = root
                    break
            if matching is None:
                raise ValueError("Frozen cohort files are missing or their SHA-256 hashes differ from the benchmark")
            verdict, results, paths = evaluate_contract(task.contract, matching,
                ref_effect=task.contract.gates.power.ref_effect,
                minimum_evidence_tier=REPORTING_STANDARDS[standard])
            detail = {"task_id": task_id, "results": jsonable(results),
                      "input_hashes": {str(p): cache[p.resolve()] for p in paths}}
            (out / f"{task_id}.json").write_text(json.dumps(detail, indent=2, allow_nan=False) + "\n")
            row.update(status="completed", supported=results["support_decision"].supported,
                       raw_label=verdict.label)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            row.update(status="error", error=str(exc))
        predictions.append(row)
    write_rows(out / "predictions.jsonl", predictions)
    return {"tasks_requested": len(selected), "completed": sum(x["status"] == "completed" for x in predictions),
            "errors": sum(x["status"] == "error" for x in predictions), "predictions": str(out / "predictions.jsonl")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="neuroclaimbench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "summary", "predictions", "score", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--benchmark", type=Path, required=True)
        if command in {"predictions", "run"}:
            sub.add_argument("--standard", choices=REPORTING_STANDARDS, default="full")
            sub.add_argument("--out", type=Path, required=True)
        if command == "score":
            sub.add_argument("--predictions", type=Path, required=True)
            sub.add_argument("--allow-partial", action="store_true")
            sub.add_argument("--literature-scope", choices=["all", "scalar"], default="all",
                             help="Use scalar for the paper's common scalar comparison; controls remain included")
        if command == "run":
            sub.add_argument("--data-root", type=Path, action="append", required=True)
            sub.add_argument("--task-id", action="append")
    args = parser.parse_args(argv)
    try:
        data, report = verify(args.benchmark)
        exit_code = 0
        if args.command == "predictions":
            rows = frozen_predictions(data, args.standard)
            write_rows(args.out, rows)
            report = {"written": len(rows), "standard": args.standard, "out": str(args.out)}
        elif args.command == "score":
            report = score(data, read_rows(args.predictions), args.allow_partial, args.literature_scope)
        elif args.command == "run":
            report = run_tasks(data, args.data_root, args.out, args.standard, args.task_id)
            exit_code = 1 if report["errors"] else 0
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return exit_code
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
