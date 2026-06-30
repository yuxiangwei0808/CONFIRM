"""Replay existing agentic CONFIRM artifacts through the claim-proposal layer."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from confirm.contract import ClaimContract
from confirm.proposals import (
    FailureLocalization,
    NewClaimProposal,
    ProposalValidation,
    default_new_claim_proposal,
    localization_for_estimand_mismatch,
    localize_failure,
    summarize_proposals,
    validate_new_claim_proposal,
)

DEFAULT_INPUT = "review-stage/claim-search-llm-20260626/source/multi_model_claim_source.json"


def _iter_initial_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in payload.get("models", []):
        model_spec = str(model.get("model_spec") or "")
        for row in model.get("initial_claims", []):
            if isinstance(row, dict):
                rows.append({"model_spec": model_spec, **row})
    return rows


def _contract_from_row(row: dict[str, Any]) -> ClaimContract | None:
    payload = row.get("drafted_contract")
    if not isinstance(payload, dict):
        gate_results = row.get("gate_results")
        if isinstance(gate_results, dict):
            payload = gate_results.get("contract")
    if not isinstance(payload, dict):
        return None
    return ClaimContract.model_validate(payload)


def _localization_for_row(row: dict[str, Any], contract: ClaimContract) -> FailureLocalization:
    if row.get("draft_success") and not row.get("estimand_match", True):
        mismatches = row.get("estimand_mismatches")
        return localization_for_estimand_mismatch(
            str(row.get("claim_id") or contract.claim_id),
            mismatches=mismatches if isinstance(mismatches, dict) else None,
        )
    verdict = row.get("gate_verdict")
    if isinstance(verdict, dict):
        results = row.get("gate_results") if isinstance(row.get("gate_results"), dict) else None
        return localize_failure(contract, verdict, results)
    return localization_for_estimand_mismatch(
        str(row.get("claim_id") or contract.claim_id),
        mismatches={"draft_or_gate": "missing executable gate verdict"},
    )


def _unsafe_probe(localization: FailureLocalization) -> NewClaimProposal:
    return NewClaimProposal(
        source_claim_id=localization.claim_id,
        proposal_type="exploratory_followup_claim",
        rationale="Unsafe probe asks to confirm a post-hoc claim on the current data.",
        proposed_question="Post-hoc current-data follow-up claim.",
        proposed_contract=None,
        disposition_label=None,
        provenance="post_hoc",
        requires_new_evidence=False,
        can_confirm_on_current_data=True,
        supported_by_evidence=localization.evidence,
    )


def _table_row(
    source_row: dict[str, Any],
    localization: FailureLocalization,
    proposal: NewClaimProposal,
    validation: ProposalValidation,
    unsafe_validation: ProposalValidation,
) -> dict[str, Any]:
    return {
        "model_spec": source_row.get("model_spec"),
        "claim_id": localization.claim_id,
        "question": source_row.get("question"),
        "source_verdict": localization.source_verdict,
        "failed_gates": ";".join(localization.failed_gates),
        "primary_failure": localization.primary_failure,
        "failure_kind": localization.failure_kind,
        "diagnosis": localization.diagnosis,
        "evidence": " | ".join(localization.evidence),
        "allowed_proposal_types": ";".join(localization.allowed_proposal_types),
        "current_data_repair_allowed": localization.current_data_repair_allowed,
        "proposal_type": proposal.proposal_type,
        "proposal_provenance": proposal.provenance,
        "proposal_requires_new_evidence": proposal.requires_new_evidence,
        "proposal_can_confirm_on_current_data": proposal.can_confirm_on_current_data,
        "proposal_validation_ok": validation.ok,
        "proposal_violations": " | ".join(validation.violations),
        "unsafe_probe_blocked": not unsafe_validation.ok,
        "unsafe_probe_violations": " | ".join(unsafe_validation.violations),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_rows = _iter_initial_rows(payload)
    if not source_rows:
        raise ValueError(f"No initial claims found in {source}")

    rows: list[dict[str, Any]] = []
    localizations: list[FailureLocalization] = []
    validations: list[ProposalValidation] = []
    unsafe_validations: list[ProposalValidation] = []
    skipped: list[dict[str, str]] = []

    for row in source_rows:
        try:
            contract = _contract_from_row(row)
            if contract is None:
                skipped.append({"claim_id": str(row.get("claim_id")), "reason": "missing contract"})
                continue
            localization = _localization_for_row(row, contract)
            proposal = default_new_claim_proposal(localization, contract)
            validation = validate_new_claim_proposal(contract, proposal, localization)
            unsafe_validation = validate_new_claim_proposal(contract, _unsafe_probe(localization), localization)
        except Exception as exc:
            skipped.append({"claim_id": str(row.get("claim_id")), "reason": str(exc)})
            continue
        localizations.append(localization)
        validations.append(validation)
        unsafe_validations.append(unsafe_validation)
        rows.append(_table_row(row, localization, proposal, validation, unsafe_validation))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "diagnosis_replay.json"
    csv_path = out_dir / "diagnosis_replay.csv"
    summary = summarize_proposals(localizations, unsafe_validations)
    summary["valid_default_proposal_count"] = sum(1 for item in validations if item.ok)
    summary["valid_default_proposal_rate"] = sum(1 for item in validations if item.ok) / len(validations) if validations else 0.0
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Diagnosis replay for the CONFIRM claim-proposal layer.",
        "input": str(source),
        "summary": summary,
        "skipped": skipped,
        "rows": rows,
        "localizations": [item.model_dump(mode="json") for item in localizations],
        "default_proposal_validation": [item.model_dump(mode="json") for item in validations],
        "unsafe_probe_validation": [item.model_dump(mode="json") for item in unsafe_validations],
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="review-stage/claim-search-llm-20260626/single-shot-diagnosis-replay")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
