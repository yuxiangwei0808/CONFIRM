"""Run a one-step agent repair benchmark with CONFIRM feedback."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from bench.run_agentic_benchmark import (
    CLAIMS,
    DEFAULT_MODELS,
    AgenticClaim,
    _estimand_comparison,
    _json_safe,
    _merge_catalogs,
    _run_claim_for_model,
    _safe_model_name,
)
from confirm.agent import _execute_contract, _load_canonical, _parse_contract_text, draft_contract
from confirm.contract import ClaimContract
from confirm.feedback import (
    ClaimFeedback,
    RevisionResponse,
    feedback_for_estimand_mismatch,
    feedback_from_verdict,
    validate_revision_response,
)
from confirm.llm import make_llm

CONTROLLED_FEEDBACK_MODELS = {"feedback-following", "feedback-oracle", "controlled-feedback"}


def _is_controlled_feedback_model(model_spec: str) -> bool:
    return model_spec.strip().lower() in CONTROLLED_FEEDBACK_MODELS


def _runtime_model_spec(model_spec: str) -> str:
    return "standin" if _is_controlled_feedback_model(model_spec) else model_spec


def _draft_failure_feedback(claim: AgenticClaim, row: dict[str, Any]) -> ClaimFeedback:
    error = str(row.get("error") or row.get("error_stage") or "draft failed")
    return ClaimFeedback(
        claim_id=claim.claim_id,
        source_verdict="draft_failed",
        failed_gates=["draft_contract"],
        primary_failure="execution_error",
        repairability="contract_repairable",
        diagnosis=f"The agent did not produce a valid executable ClaimContract: {error}.",
        allowed_revisions=[
            "Return a schema-valid ClaimContract for the original scientific question.",
            "Return a claim disposition if the requested claim cannot be represented.",
        ],
        forbidden_revisions=[
            "Do not change the scientific question to an easier claim.",
            "Do not lower gates or omit required covariates.",
        ],
        next_agent_instruction="Produce one schema-valid revision for the same question, or return an abandonment disposition.",
        must_preserve=["scientific_question", "gate_thresholds"],
        allowed_contract_changes=["estimand", "disposition"],
    )


def _feedback_for_initial_row(claim: AgenticClaim, row: dict[str, Any]) -> ClaimFeedback:
    if not row.get("draft_success"):
        return _draft_failure_feedback(claim, row)
    if not row.get("estimand_match"):
        return feedback_for_estimand_mismatch(
            claim.claim_id,
            source_verdict=str(row.get("gate_verdict_label") or "draft_mismatch"),
            mismatches=row.get("estimand_mismatches") if isinstance(row.get("estimand_mismatches"), dict) else None,
        )
    verdict = row.get("gate_verdict")
    if isinstance(verdict, dict):
        results = row.get("gate_results") if isinstance(row.get("gate_results"), dict) else None
        return feedback_from_verdict(claim.claim_id, verdict, results)
    return _draft_failure_feedback(claim, {"error": "missing gate verdict"})


def _needs_revision(row: dict[str, Any]) -> bool:
    if not row.get("draft_success") or not row.get("gate_success"):
        return True
    if not row.get("estimand_match"):
        return True
    return str(row.get("gate_verdict_label")) != "confirmed"


def _parse_revision_response(text: str) -> RevisionResponse:
    data = _parse_contract_text(text)
    if not isinstance(data, dict):
        raise ValueError("Revision response did not parse to a mapping")
    if "response_type" in data:
        return RevisionResponse.model_validate(data)
    return RevisionResponse(
        response_type="revised_contract",
        revised_contract=ClaimContract.model_validate(data),
        rationale="Parsed bare ClaimContract as revised_contract.",
    )


def _revision_prompt(
    claim: AgenticClaim,
    original_contract: ClaimContract | None,
    initial_row: dict[str, Any],
    feedback: ClaimFeedback,
    *,
    structured: bool,
) -> str:
    payload = {
        "QUESTION": claim.question,
        "INTENDED_ESTIMAND": claim.intended,
        "ORIGINAL_CONTRACT": original_contract.model_dump(mode="json") if original_contract is not None else None,
        "INITIAL_VERDICT": initial_row.get("gate_verdict"),
        "INITIAL_MISMATCHES": initial_row.get("estimand_mismatches", {}),
        "RESPONSE_SCHEMA": {
            "response_type": "revised_contract | claim_disposition",
            "revised_contract": "ClaimContract object when response_type is revised_contract",
            "disposition_label": "needs_more_data | non_replicated | under_powered | fragile | abandon_claim when response_type is claim_disposition",
            "rationale": "short explanation",
        },
    }
    if structured:
        payload["CONFIRM_FEEDBACK"] = feedback.model_dump(mode="json")
        payload["INSTRUCTIONS"] = (
            "Use CONFIRM_FEEDBACK exactly. Return ONLY YAML or JSON matching RESPONSE_SCHEMA. "
            "Use only numeric values supplied in CONFIRM_FEEDBACK.evidence or INITIAL_VERDICT; do not invent p-values, "
            "effect sizes, sample sizes, or thresholds. "
            "If the feedback says the claim needs new data, prefer a claim_disposition unless the revised contract "
            "uses an actually valid independent replication cohort. If discovery_cohort appears in replication_cohorts, "
            "return a claim_disposition; same-cohort replication is not a valid repair. Do not weaken gates or switch outcomes."
        )
    else:
        payload["INSTRUCTIONS"] = (
            "The previous attempt failed or did not confirm. Return ONLY YAML or JSON matching RESPONSE_SCHEMA. "
            "Try one reasonable revision, but keep the original scientific question."
        )
    return yaml.safe_dump(payload, sort_keys=False)


def _generic_retry_question(claim: AgenticClaim, initial_row: dict[str, Any]) -> str:
    return (
        f"{claim.question}\n\n"
        "The previous attempt failed or did not confirm. Try one corrected ClaimContract for the same scientific question. "
        f"Previous issue: {initial_row.get('error') or initial_row.get('gate_verdict_label') or initial_row.get('error_stage')}."
    )


def _execute_contract_for_revision(contract: ClaimContract, root_by_cohort: dict[str, Path]) -> tuple[str, dict[str, Any], list[str]]:
    data_root = root_by_cohort.get(contract.discovery_cohort)
    if data_root is None:
        raise ValueError(f"No data root contains discovery cohort {contract.discovery_cohort!r}")
    verdict, results, cohort_paths = _execute_contract(contract, data_root)
    return verdict.label, _json_safe(results), [str(path) for path in cohort_paths]


def _feedback_disposition_label(feedback: ClaimFeedback) -> str:
    if feedback.primary_failure == "replication":
        return "non_replicated"
    if feedback.primary_failure == "power":
        return "under_powered"
    if feedback.primary_failure in {"multiverse", "multiplicity", "search_provenance", "confound"}:
        return "fragile"
    if feedback.repairability == "needs_new_data":
        return "needs_more_data"
    return "abandon_claim"


def _cohort_columns(catalog: dict[str, Any], cohort: str) -> set[str]:
    columns: set[str] = set()
    data_root: str | None = None
    for entry in catalog.get("cohorts", []):
        if entry.get("cohort") == cohort:
            columns = {str(column) for column in entry.get("columns", [])}
            data_root = str(entry.get("data_root") or "")
    if data_root:
        try:
            df = _load_canonical(Path(data_root) / f"{cohort}.parquet")
            columns = {column for column in columns if column in df.columns and df[column].notna().any()}
        except Exception:
            pass
    return columns


def _shared_covariates(catalog: dict[str, Any], intended: dict[str, Any], original: ClaimContract) -> list[str]:
    cohorts = [str(intended["discovery_cohort"]), *[str(item) for item in intended.get("replication_cohorts", [])]]
    column_sets = [_cohort_columns(catalog, cohort) for cohort in cohorts]
    shared = set.intersection(*column_sets) if column_sets and all(column_sets) else set()

    outcome = intended.get("outcome")
    outcomes = {str(item) for item in outcome} if isinstance(outcome, list) else {str(outcome)}
    group = intended.get("group") if isinstance(intended.get("group"), dict) else {}
    excluded = {str(intended.get("predictor")), str(group.get("var")), *outcomes, "None"}

    candidates = ["age", "sex", "eTIV", "site", "mean_fd"]
    if shared:
        covariates = [name for name in candidates if name in shared and name not in excluded]
    else:
        covariates = [name for name in original.covariates if name not in excluded]
    return covariates


def _contract_from_intended(
    claim: AgenticClaim,
    catalog: dict[str, Any],
    original: ClaimContract,
) -> ClaimContract:
    intended = claim.intended
    data = original.model_dump(mode="json")
    data["claim_id"] = claim.claim_id
    data["question"] = claim.question
    data["estimand"]["type"] = intended["type"]
    data["estimand"]["outcome"] = intended["outcome"]
    data["estimand"]["predictor"] = intended["predictor"]
    data["estimand"]["group"] = intended.get("group")
    data["estimand"]["direction"] = intended["direction"]
    data["estimand"]["unit"] = intended["unit"]
    data["discovery_cohort"] = intended["discovery_cohort"]
    data["replication_cohorts"] = list(intended.get("replication_cohorts", []))
    data["inclusion"] = intended.get("inclusion")
    covariates = _shared_covariates(catalog, intended, original)
    data["covariates"] = covariates
    data["gates"]["confound"]["require_covariates"] = covariates
    data["search_provenance"]["declared"] = True
    data["search_provenance"]["selection"] = "preregistered"
    data["search_provenance"]["family_size"] = max(1, int(data["search_provenance"].get("family_size", 1)))
    return ClaimContract.model_validate(data)


def _controlled_feedback_response(
    claim: AgenticClaim,
    catalog: dict[str, Any],
    original: ClaimContract | None,
    feedback: ClaimFeedback,
) -> RevisionResponse:
    if feedback.primary_failure == "estimand_mismatch" and original is not None:
        intended = claim.intended
        discovery = str(intended.get("discovery_cohort", ""))
        replications = [str(item) for item in intended.get("replication_cohorts", [])]
        if not discovery or discovery in set(replications):
            return RevisionResponse(
                response_type="claim_disposition",
                disposition_label="needs_more_data",
                rationale="The intended claim lacks an independent replication cohort.",
            )
        contract = _contract_from_intended(claim, catalog, original)
        return RevisionResponse(
            response_type="revised_contract",
            revised_contract=contract,
            rationale="Controlled feedback-following repair restored the intended estimand.",
        )
    return RevisionResponse(
        response_type="claim_disposition",
        disposition_label=_feedback_disposition_label(feedback),
        rationale="Controlled feedback-following response chose the feedback-prescribed disposition.",
    )


def _attempt_revision(
    *,
    model_spec: str,
    claim: AgenticClaim,
    catalog: dict[str, Any],
    root_by_cohort: dict[str, Path],
    initial_row: dict[str, Any],
    feedback: ClaimFeedback,
    arm: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_spec": model_spec,
        "claim_id": claim.claim_id,
        "label_class": claim.label_class,
        "arm": arm,
        "attempted": _needs_revision(initial_row),
        "initial_gate_verdict_label": initial_row.get("gate_verdict_label"),
        "initial_estimand_match": bool(initial_row.get("estimand_match")),
        "feedback": feedback.model_dump(mode="json"),
        "revision_valid": False,
        "policy_ok": None,
        "policy_violations": [],
        "accepted_disposition": False,
        "final_label": initial_row.get("gate_verdict_label"),
        "estimand_match_after": bool(initial_row.get("estimand_match")),
    }
    if not row["attempted"]:
        row["revision_valid"] = True
        return row

    original_payload = initial_row.get("drafted_contract")
    original = ClaimContract.model_validate(original_payload) if isinstance(original_payload, dict) else None

    try:
        llm = make_llm(_runtime_model_spec(model_spec))
        if arm == "generic_retry":
            revised = draft_contract(_generic_retry_question(claim, initial_row), catalog, llm=llm)
            response = RevisionResponse(
                response_type="revised_contract",
                revised_contract=revised,
                rationale="Generic retry returned a ClaimContract.",
            )
            raw_text = None
        elif _is_controlled_feedback_model(model_spec):
            response = _controlled_feedback_response(claim, catalog, original, feedback)
            raw_text = yaml.safe_dump(response.model_dump(mode="json"), sort_keys=False)
        else:
            system = (
                "You revise CONFIRM claim contracts. Return only YAML or JSON matching the requested response schema. "
                "Never weaken gates, omit required covariates, or change outcomes after seeing results."
            )
            raw_text = llm.complete(system, _revision_prompt(claim, original, initial_row, feedback, structured=True))
            response = _parse_revision_response(raw_text)
        row["raw_revision_response"] = raw_text
        row["revision_response"] = response.model_dump(mode="json")
    except Exception as exc:
        row.update({"error_stage": "revision_draft", "error": str(exc), "final_label": "revision_error"})
        return row

    if original is None and response.response_type == "revised_contract":
        policy = None
        row["policy_ok"] = True
        row["policy_violations"] = []
        row["policy_warnings"] = ["No original contract was available; accepted as draft recovery and checked by execution/estimand match."]
        row["accepted_disposition"] = False
    else:
        policy = validate_revision_response(original, response, feedback)
        row["policy_ok"] = policy.ok
        row["policy_violations"] = policy.violations
        row["policy_warnings"] = policy.warnings
        row["accepted_disposition"] = policy.accepted_disposition
        if not policy.ok:
            row["final_label"] = "policy_violation"
            return row
        if response.response_type == "claim_disposition":
            row["revision_valid"] = True
            row["final_label"] = response.disposition_label
            return row

    assert response.revised_contract is not None
    matched, mismatches = _estimand_comparison(response.revised_contract, claim.intended)
    row["estimand_match_after"] = matched
    row["estimand_mismatches_after"] = mismatches
    if not matched:
        row["policy_ok"] = False
        row["policy_violations"] = [
            *list(row.get("policy_violations") or []),
            "Revised contract still does not match the intended estimand.",
        ]
        row["final_label"] = "policy_violation"
        return row
    try:
        final_label, gate_results, cohort_paths = _execute_contract_for_revision(response.revised_contract, root_by_cohort)
        row["revision_valid"] = True
        row["final_label"] = final_label
        row["gate_results_after"] = gate_results
        row["cohort_paths_after"] = cohort_paths
    except Exception as exc:
        row.update({"error_stage": "revision_execute", "error": str(exc), "final_label": "execution_error"})
    return row


def _initial_arm_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_spec": row["model_spec"],
        "claim_id": row["claim_id"],
        "label_class": row["label_class"],
        "arm": "initial",
        "attempted": True,
        "draft_success": bool(row.get("draft_success")),
        "estimand_match_after": bool(row.get("estimand_match")),
        "gate_success": bool(row.get("gate_success")),
        "final_label": row.get("gate_verdict_label") or row.get("error_stage") or "not_run",
        "policy_ok": None,
        "policy_violations": [],
        "revision_valid": bool(row.get("draft_success") and row.get("gate_success")),
        "accepted_disposition": False,
    }


def _is_negative_label(label_class: str) -> bool:
    return label_class in {"known_null", "fragile"}


def _appropriate(row: dict[str, Any]) -> bool:
    final_label = str(row.get("final_label"))
    label_class = str(row.get("label_class"))
    if label_class == "known_positive":
        return final_label == "confirmed"
    if _is_negative_label(label_class):
        return final_label not in {"confirmed", "policy_violation", "revision_error", "execution_error", "not_run"}
    return final_label not in {"policy_violation", "revision_error", "execution_error", "not_run"}


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    arm_rows = [row for row in rows if row["arm"] == arm]
    attempted = [row for row in arm_rows if row.get("attempted")]
    revision_attempts = [row for row in arm_rows if arm != "initial" and row.get("attempted")]
    valid = [row for row in revision_attempts if row.get("revision_valid")]
    policy_violations = [
        row
        for row in revision_attempts
        if row.get("policy_ok") is False and row.get("policy_violations")
    ]
    initial_mismatches = [
        row
        for row in revision_attempts
        if row.get("initial_estimand_match") is False
    ]
    repaired_mismatches = [row for row in initial_mismatches if row.get("estimand_match_after") is True]
    false_confirms = [
        row for row in attempted if _is_negative_label(str(row.get("label_class"))) and str(row.get("final_label")) == "confirmed"
    ]
    appropriate = [row for row in attempted if _appropriate(row)]
    return {
        "arm": arm,
        "n_rows": len(arm_rows),
        "attempted_count": len(attempted),
        "revision_attempt_count": len(revision_attempts),
        "valid_revision_count": len(valid),
        "valid_revision_rate": len(valid) / len(revision_attempts) if revision_attempts else math.nan,
        "estimand_match_improvement_count": len(repaired_mismatches),
        "estimand_match_improvement_denominator": len(initial_mismatches),
        "estimand_match_improvement_rate": (
            len(repaired_mismatches) / len(initial_mismatches) if initial_mismatches else math.nan
        ),
        "appropriate_resolution_count": len(appropriate),
        "appropriate_resolution_rate": len(appropriate) / len(attempted) if attempted else math.nan,
        "false_confirmation_count": len(false_confirms),
        "false_confirmation_claims": [row["claim_id"] for row in false_confirms],
        "policy_violation_count": len(policy_violations),
        "policy_violation_rate": len(policy_violations) / len(revision_attempts) if revision_attempts else math.nan,
    }


def _run_one_model(
    model_spec: str,
    claims: list[AgenticClaim],
    catalog: dict[str, Any],
    root_by_cohort: dict[str, Path],
) -> dict[str, Any]:
    initial_model_spec = _runtime_model_spec(model_spec)
    initial_rows = [_run_claim_for_model(initial_model_spec, claim, catalog, root_by_cohort) for claim in claims]
    if initial_model_spec != model_spec:
        for row in initial_rows:
            row["initial_model_spec"] = initial_model_spec
            row["model_spec"] = model_spec
    feedback_by_claim = {
        claim.claim_id: _feedback_for_initial_row(claim, row)
        for claim, row in zip(claims, initial_rows)
    }
    arm_rows: list[dict[str, Any]] = [_initial_arm_row(row) for row in initial_rows]
    for arm in ["generic_retry", "structured_feedback"]:
        for claim, initial in zip(claims, initial_rows):
            feedback = feedback_by_claim[claim.claim_id]
            arm_rows.append(
                _attempt_revision(
                    model_spec=model_spec,
                    claim=claim,
                    catalog=catalog,
                    root_by_cohort=root_by_cohort,
                    initial_row=initial,
                    feedback=feedback,
                    arm=arm,
                )
            )
    return {
        "model_spec": model_spec,
        "initial_claims": initial_rows,
        "feedback": {claim_id: feedback.model_dump(mode="json") for claim_id, feedback in feedback_by_claim.items()},
        "arm_rows": arm_rows,
        "summary": {arm: _arm_summary(arm_rows, arm) for arm in ["initial", "generic_retry", "structured_feedback"]},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(args.smri_root), Path(args.cluster_root)]
    catalog, root_by_cohort = _merge_catalogs(roots)
    models = [spec.strip() for spec in args.models.split(",") if spec.strip()]
    selected_claims = [claim for claim in CLAIMS if not args.claim_id or claim.claim_id in set(args.claim_id)]
    if args.limit is not None:
        selected_claims = selected_claims[: args.limit]
    if not selected_claims:
        raise ValueError("No claims selected")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    payloads: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, max(1, len(models)))) as executor:
        futures = {
            executor.submit(_run_one_model, model, selected_claims, catalog, root_by_cohort): model
            for model in models
        }
        for future in as_completed(futures):
            payloads.append(future.result())
    payloads.sort(key=lambda item: models.index(item["model_spec"]))

    by_arm: dict[str, list[dict[str, Any]]] = {"initial": [], "generic_retry": [], "structured_feedback": []}
    for payload in payloads:
        for row in payload["arm_rows"]:
            by_arm[row["arm"]].append(row)
    summary = {arm: _arm_summary(rows, arm) for arm, rows in by_arm.items()}

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "description": "One-step agent repair benchmark with generic retry versus structured CONFIRM feedback.",
        "command": {
            "models": models,
            "smri_root": str(args.smri_root),
            "cluster_root": str(args.cluster_root),
            "limit": args.limit,
            "claim_id": args.claim_id,
            "max_workers": args.max_workers,
        },
        "summary": summary,
        "models": payloads,
    }
    path = out_dir / f"agentic_feedback_benchmark_{run_id}.json"
    latest = out_dir / "agentic_feedback_benchmark.json"
    path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    shutil.copyfile(path, latest)
    print(f"wrote {path}")
    print(f"wrote {latest}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--smri-root", default="data/prepared_data/smri_disease")
    parser.add_argument("--cluster-root", default="data/prepared_data/cluster_recovered")
    parser.add_argument("--out-dir", default="review-stage/agentic-feedback")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--claim-id", action="append", default=None)
    parser.add_argument("--max-workers", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
