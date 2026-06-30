"""Benchmark typed claim proposals against existing agentic feedback artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.run_claim_proposal_replay import _contract_from_row, _localization_for_row
from confirm.proposals import default_new_claim_proposal, validate_new_claim_proposal

DEFAULT_INPUT = "review-stage/claim-search-llm-20260626/source/multi_model_claim_source.json"


def _needs_attempt(row: dict[str, Any]) -> bool:
    if not row.get("draft_success") or not row.get("gate_success"):
        return True
    if not row.get("estimand_match"):
        return True
    return str(row.get("gate_verdict_label")) != "confirmed"


def _is_negative_label(label_class: str) -> bool:
    return label_class in {"known_null", "fragile"}


def _existing_arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    arm_rows = [row for row in rows if row.get("arm") == arm]
    attempted = [row for row in arm_rows if row.get("attempted")]
    valid = [row for row in attempted if row.get("revision_valid")]
    policy_violations = [row for row in attempted if row.get("policy_ok") is False and row.get("policy_violations")]
    false_confirmed = [
        row
        for row in attempted
        if _is_negative_label(str(row.get("label_class"))) and str(row.get("final_label")) == "confirmed"
    ]
    return {
        "arm": arm,
        "attempted_count": len(attempted),
        "valid_proposal_count": len(valid),
        "valid_proposal_rate": len(valid) / len(attempted) if attempted else math.nan,
        "provenance_compliant_count": len(attempted) - len(policy_violations),
        "provenance_compliant_rate": (len(attempted) - len(policy_violations)) / len(attempted) if attempted else math.nan,
        "useful_proposal_count": len(valid),
        "useful_proposal_rate": len(valid) / len(attempted) if attempted else math.nan,
        "current_data_gaming_violation_count": len(policy_violations),
        "false_current_data_confirmation_count": len(false_confirmed),
        "corrected_contract_success_count": sum(1 for row in valid if row.get("accepted_disposition") is False),
        "corrected_contract_success_denominator": len(valid),
    }


def _proposal_rows_for_model(model: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for initial in model.get("initial_claims", []):
        attempted = _needs_attempt(initial)
        base = {
            "model_spec": model.get("model_spec"),
            "claim_id": initial.get("claim_id"),
            "label_class": initial.get("label_class"),
            "arm": "claim_proposal_layer",
            "attempted": attempted,
            "initial_gate_verdict_label": initial.get("gate_verdict_label"),
            "initial_estimand_match": bool(initial.get("estimand_match")),
        }
        if not attempted:
            rows.append(
                {
                    **base,
                    "proposal_valid": True,
                    "provenance_compliant": True,
                    "useful": True,
                    "current_data_gaming_violation": False,
                    "false_current_data_confirmation": False,
                    "proposal_type": "none",
                    "failure_kind": "none",
                }
            )
            continue
        try:
            contract = _contract_from_row(initial)
            if contract is None:
                raise ValueError("missing contract")
            localization = _localization_for_row(initial, contract)
            proposal = default_new_claim_proposal(localization, contract)
            validation = validate_new_claim_proposal(contract, proposal, localization)
            current_data_gaming = any("current data" in item for item in validation.violations)
            false_confirm = (
                _is_negative_label(str(initial.get("label_class")))
                and proposal.can_confirm_on_current_data
                and validation.ok
            )
            rows.append(
                {
                    **base,
                    "proposal_valid": validation.ok,
                    "provenance_compliant": validation.provenance_compliant,
                    "useful": validation.useful,
                    "current_data_gaming_violation": current_data_gaming,
                    "false_current_data_confirmation": false_confirm,
                    "proposal_type": proposal.proposal_type,
                    "proposal_provenance": proposal.provenance,
                    "failure_kind": localization.failure_kind,
                    "primary_failure": localization.primary_failure,
                    "validation_violations": validation.violations,
                    "localization": localization.model_dump(mode="json"),
                    "proposal": proposal.model_dump(mode="json"),
                    "validation": validation.model_dump(mode="json"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "proposal_valid": False,
                    "provenance_compliant": False,
                    "useful": False,
                    "current_data_gaming_violation": False,
                    "false_current_data_confirmation": False,
                    "proposal_type": "error",
                    "failure_kind": "error",
                    "error": str(exc),
                }
            )
    return rows


def _proposal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [row for row in rows if row.get("attempted")]
    valid = [row for row in attempted if row.get("proposal_valid")]
    provenance = [row for row in attempted if row.get("provenance_compliant")]
    useful = [row for row in attempted if row.get("useful")]
    gaming = [row for row in attempted if row.get("current_data_gaming_violation")]
    false_confirm = [row for row in attempted if row.get("false_current_data_confirmation")]
    corrected_contracts = [row for row in attempted if row.get("proposal_type") == "corrected_contract"]
    corrected_success = [row for row in corrected_contracts if row.get("proposal_valid")]
    return {
        "arm": "claim_proposal_layer",
        "attempted_count": len(attempted),
        "valid_proposal_count": len(valid),
        "valid_proposal_rate": len(valid) / len(attempted) if attempted else math.nan,
        "provenance_compliant_count": len(provenance),
        "provenance_compliant_rate": len(provenance) / len(attempted) if attempted else math.nan,
        "useful_proposal_count": len(useful),
        "useful_proposal_rate": len(useful) / len(attempted) if attempted else math.nan,
        "current_data_gaming_violation_count": len(gaming),
        "false_current_data_confirmation_count": len(false_confirm),
        "corrected_contract_success_count": len(corrected_success),
        "corrected_contract_success_denominator": len(corrected_contracts),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_payloads = []

    for model in payload.get("models", []):
        proposal_rows = _proposal_rows_for_model(model)
        arm_rows = list(model.get("arm_rows", []))
        for row in arm_rows:
            if row.get("arm") in {"generic_retry", "structured_feedback"}:
                by_arm[str(row["arm"])].append(row)
        by_arm["claim_proposal_layer"].extend(proposal_rows)
        model_payloads.append(
            {
                "model_spec": model.get("model_spec"),
                "proposal_rows": proposal_rows,
                "summary": _proposal_summary(proposal_rows),
            }
        )

    summary = {
        "generic_retry": _existing_arm_summary(by_arm["generic_retry"], "generic_retry"),
        "structured_feedback": _existing_arm_summary(by_arm["structured_feedback"], "structured_feedback"),
        "claim_proposal_layer": _proposal_summary(by_arm["claim_proposal_layer"]),
    }
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Agentic benchmark comparing retry/feedback arms with typed claim proposals.",
        "input": str(source),
        "summary": summary,
        "models": model_payloads,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "agentic_proposal_benchmark.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="review-stage/claim-search-llm-20260626/single-shot-proposal-benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
