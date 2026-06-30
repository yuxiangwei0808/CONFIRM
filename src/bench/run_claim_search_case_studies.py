"""Build compact case-study traces for iterative claim search."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT = "review-stage/claim-search-llm-20260626/llm-candidate-replay/iterative_candidate_replay.json"


def _case_for_state(state: dict[str, Any]) -> dict[str, Any]:
    claim = state.get("original_claim", {})
    localization = state.get("failure_localization") or {}
    candidates = state.get("candidate_history", [])
    evaluations = state.get("evaluations", [])
    first_eval = evaluations[0] if evaluations else {}
    prompts = state.get("llm_candidate_prompts") or []
    first_prompt = prompts[0] if prompts else {}
    candidate_summaries = [_candidate_summary(candidate, evaluation) for candidate, evaluation in zip(candidates, evaluations)]
    return {
        "claim_id": claim.get("claim_id"),
        "original_claim": claim.get("question"),
        "failed_gates": ";".join(str(item) for item in localization.get("failed_gates") or []),
        "localized_failure": localization.get("failure_kind"),
        "failure_diagnosis": localization.get("diagnosis"),
        "used_generation_evidence": " | ".join(state.get("used_evidence") or []),
        "llm_prompt_excerpt": str(first_prompt.get("user") or "")[:1000],
        "validation_evidence_policy": "excluded validation evidence required unless candidate is a true contract correction",
        "candidate_count": len(candidates),
        "candidate_transforms": ";".join(str(candidate.get("transform_type")) for candidate in candidates),
        "first_candidate_question": (candidates[0] or {}).get("proposed_question") if candidates else None,
        "first_candidate_validation": (first_eval.get("validation") or {}).get("ok"),
        "first_candidate_blocked_reason": first_eval.get("blocked_reason"),
        "candidate_validation_summary": " | ".join(
            "{candidate_id}:{validation_ok}:{blocked_reason}:{violations}".format(**item) for item in candidate_summaries
        ),
        "candidate_proposals": candidate_summaries,
        "confirmed_candidates": ";".join(state.get("confirmed_candidates") or []),
        "stopped_reason": state.get("stopped_reason"),
    }


def _candidate_summary(candidate: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    validation = evaluation.get("validation") or {}
    violations = validation.get("violations") or []
    return {
        "candidate_id": str(candidate.get("candidate_id")),
        "round_index": str(candidate.get("round_index")),
        "transform_type": str(candidate.get("transform_type")),
        "proposal_type": str(candidate.get("proposal_type")),
        "proposed_question": str(candidate.get("proposed_question") or ""),
        "provenance": str(candidate.get("provenance")),
        "validation_split": str(candidate.get("validation_split")),
        "validation_ok": str(validation.get("ok")),
        "blocked_reason": str(evaluation.get("blocked_reason") or ""),
        "violations": ";".join(str(item) for item in violations),
    }


def _write_markdown(cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Iterative Claim Search Case Studies",
        "",
    ]
    for row in cases:
        lines.extend(
            [
                f"## {_md(row.get('claim_id'))}",
                "",
                f"- Original claim: {_md(row.get('original_claim'))}",
                f"- Failed gates: {_md(row.get('failed_gates'))}",
                f"- Localized failure: {_md(row.get('localized_failure'))}",
                f"- Diagnosis: {_md(row.get('failure_diagnosis'))}",
                f"- Generation evidence: {_md(row.get('used_generation_evidence'))}",
                f"- LLM prompt excerpt: {_md(row.get('llm_prompt_excerpt'))}",
                f"- Final status: {_md(row.get('stopped_reason'))}",
                "",
                "| Round | Transform | Proposal type | Validation | Blocked reason | Violations | Proposed question |",
                "| ---: | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for candidate in row.get("candidate_proposals") or []:
            lines.append(
                "| {round_index} | {transform_type} | {proposal_type} | {validation_ok} | {blocked_reason} | {violations} | {proposed_question} |".format(
                    **{key: _md(value) for key, value in candidate.items()}
                )
            )
        lines.append("")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _md(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    states = payload.get("states", [])
    if not isinstance(states, list) or not states:
        raise ValueError(f"No states found in {source}")
    cases = [_case_for_state(state) for state in states[: args.limit]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "claim_search_case_studies.json"
    csv_path = out_dir / "claim_search_case_studies.csv"
    md_path = out_dir / "claim_search_case_studies.md"
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "E17 LLM iterative claim-search case-study traces.",
        "input": str(source),
        "cases": cases,
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(cases).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    _write_markdown(cases, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="review-stage/claim-search-llm-20260626/case-studies")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
