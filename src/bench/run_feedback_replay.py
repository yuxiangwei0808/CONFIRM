"""Replay CONFIRM audit artifacts through the deterministic feedback layer."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from confirm.feedback import ClaimFeedback, feedback_from_row, summarize_feedback

DEFAULT_INPUTS = (
    "review-stage/round5-combat/combined_benchmark_audit.csv",
    "review-stage/negatives-expansion/negatives_expansion_audit_20260620_102803.csv",
    "review-stage/external-nacc/nacc_external_audit.csv",
    "review-stage/external-cnp/CNP_external_audit.csv",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["source_artifact"] = str(path)
    return rows


def _feedback_row(feedback: ClaimFeedback) -> dict[str, Any]:
    payload = feedback.model_dump(mode="json")
    return {
        "claim_id": payload["claim_id"],
        "source_verdict": payload["source_verdict"],
        "primary_failure": payload["primary_failure"],
        "repairability": payload["repairability"],
        "failed_gates": ";".join(payload["failed_gates"]),
        "diagnosis": payload["diagnosis"],
        "evidence": " | ".join(payload["evidence"]),
        "allowed_revisions": " | ".join(payload["allowed_revisions"]),
        "forbidden_revisions": " | ".join(payload["forbidden_revisions"]),
        "refinement_actions": " | ".join(payload["refinement_actions"]),
        "next_agent_instruction": payload["next_agent_instruction"],
        "must_preserve": ";".join(payload["must_preserve"]),
        "allowed_contract_changes": ";".join(payload["allowed_contract_changes"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in args.input:
        path = Path(source)
        if path.exists():
            rows.extend(_read_rows(path))
        else:
            missing.append(str(path))
    if not rows:
        raise FileNotFoundError(f"No feedback replay inputs found: {args.input}")

    feedbacks = [feedback_from_row(row) for row in rows]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for row, feedback in zip(rows, feedbacks):
        item = _feedback_row(feedback)
        item["source_artifact"] = row["source_artifact"]
        table_rows.append(item)

    csv_path = out_dir / "feedback_replay.csv"
    json_path = out_dir / "feedback_replay.json"
    examples_path = out_dir / "feedback_examples.csv"

    pd.DataFrame(table_rows).to_csv(csv_path, index=False)
    examples = _select_examples(table_rows)
    pd.DataFrame(examples).to_csv(examples_path, index=False)
    latex_out = getattr(args, "latex_out", None)
    if latex_out:
        _write_latex_examples(examples, Path(latex_out))

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Deterministic feedback replay over existing CONFIRM audit artifacts.",
        "inputs": list(args.input),
        "missing_inputs": missing,
        "summary": summarize_feedback(feedbacks),
        "feedback": [item.model_dump(mode="json") for item in feedbacks],
        "example_table": examples,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {examples_path}")
    if latex_out:
        print(f"wrote {latex_out}")
    return payload


def _select_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = ["confound", "search_provenance", "power", "multiverse", "replication", "multiplicity"]
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for failure in wanted:
        for row in rows:
            if row["primary_failure"] == failure and row["source_verdict"] != "confirmed":
                examples.append(
                    {
                        "failed_gate": row["primary_failure"],
                        "repairability": row["repairability"],
                        "example_claim": row["claim_id"],
                        "agent_instruction": row["next_agent_instruction"],
                    }
                )
                seen.add(failure)
                break
    for row in rows:
        if len(examples) >= 6:
            break
        failure = row["primary_failure"]
        if failure not in seen and row["source_verdict"] != "confirmed":
            examples.append(
                {
                    "failed_gate": failure,
                    "repairability": row["repairability"],
                    "example_claim": row["claim_id"],
                    "agent_instruction": row["next_agent_instruction"],
                }
            )
            seen.add(failure)
    return examples


def _write_latex_examples(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{@{}llll@{}}",
        "\\toprule",
        "Failed gate & Repairability & Example claim & Agent instruction \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex(row['failed_gate'])} & {_tex(row['repairability'])} & "
            f"\\texttt{{{_tex(row['example_claim'])}}} & {_tex(row['agent_instruction'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=list(DEFAULT_INPUTS))
    parser.add_argument("--out-dir", default="review-stage/feedback-replay")
    parser.add_argument("--latex-out", default="paper/figures/tab_feedback_examples.tex")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
