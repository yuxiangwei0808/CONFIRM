"""Build a claim-search source payload from CONFIRM gate result JSONs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from confirm.contract import ClaimContract
from confirm.evidence_partitions import canonical_base_cohort, infer_target_family, load_evidence_manifest


DEFAULT_INPUTS = (
    "review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json",
)


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _gate_payload(row: dict[str, Any], contract: ClaimContract) -> dict[str, Any]:
    embedded_verdict = row.get("gate_verdict")
    if isinstance(embedded_verdict, dict):
        embedded_gates = embedded_verdict.get("gates")
        if isinstance(embedded_gates, dict):
            boolean_gates = {
                str(key): bool(value)
                for key, value in embedded_gates.items()
                if isinstance(value, bool)
            }
            if boolean_gates:
                return boolean_gates

    gate_state = row.get("gate_state")
    if isinstance(gate_state, dict):
        return {str(key): bool(value) for key, value in gate_state.items() if isinstance(value, bool)}

    exec_only = _bool_or_none(row.get("exec_only"))
    confound = _bool_or_none(row.get("+confound"))
    power = _bool_or_none(row.get("+power"))
    multiverse = _bool_or_none(row.get("+multiverse"))
    replication = _bool_or_none(row.get("+replication"))
    confound_completeness = _bool_or_none(row.get("confound_valid"))
    if confound_completeness is None:
        confound_detail = row.get("confound_completeness")
        if isinstance(confound_detail, dict) and isinstance(confound_detail.get("passed"), bool):
            confound_completeness = bool(confound_detail["passed"])

    search = contract.search_provenance
    search_ok = bool(search.declared and search.selection in {"preregistered", "discovery_only"})
    return {
        "search_provenance": search_ok,
        "multiplicity": bool(exec_only) if exec_only is not None else row.get("final_label") == "confirmed",
        "confound": bool(confound) if confound is not None else True,
        "confound_completeness": bool(confound_completeness) if confound_completeness is not None else True,
        "power": bool(power) if power is not None else row.get("final_label") == "confirmed",
        "multiverse": bool(multiverse) if multiverse is not None else row.get("final_label") == "confirmed",
        "replication": bool(replication) if replication is not None else row.get("final_label") == "confirmed",
    }


def _result_payload(row: dict[str, Any], contract: ClaimContract) -> dict[str, Any]:
    existing = row.get("gate_results")
    if isinstance(existing, dict):
        results = dict(existing)
        results.setdefault("contract", contract.model_dump(mode="json"))
        results.setdefault("source_row_summary", {})
        if isinstance(results["source_row_summary"], dict):
            results["source_row_summary"].update(
                {
                    "source_final_label": row.get("final_label"),
                    "source_rationale": row.get("rationale"),
                    "scoring_label": row.get("scoring_label"),
                    "label_class": row.get("label_class"),
                    "ground_truth": row.get("ground_truth"),
                    "label_authority": row.get("label_authority"),
                    "modality": row.get("modality"),
                    "n_discovery": row.get("n_discovery"),
                    "n_replication": row.get("n_replication"),
                }
            )
        return results

    primary = row.get("primary_effect")
    if not isinstance(primary, dict):
        primary = {
            "beta": row.get("best_beta"),
            "p": row.get("best_p"),
            "standardized_effect": row.get("best_standardized_effect"),
            "n": row.get("n_discovery"),
        }
    multiverse = {
        "fraction_consistent": row.get("multiverse_fraction_consistent"),
        "passed": row.get("+multiverse"),
        "specs": row.get("multiverse_specs") if isinstance(row.get("multiverse_specs"), list) else [],
    }
    results: dict[str, Any] = {
        "contract": contract.model_dump(mode="json"),
        "primary": primary,
        "power": row.get("power") if isinstance(row.get("power"), dict) else {},
        "multiverse": multiverse,
        "replication": row.get("replication") if isinstance(row.get("replication"), dict) else {},
        "confound_completeness": row.get("confound_completeness") if isinstance(row.get("confound_completeness"), dict) else {},
        "source_row_summary": {
            "source_final_label": row.get("final_label"),
            "source_rationale": row.get("rationale"),
            "scoring_label": row.get("scoring_label"),
            "label_authority": row.get("label_authority"),
            "modality": row.get("modality"),
            "n_discovery": row.get("n_discovery"),
            "n_replication": row.get("n_replication"),
        },
    }
    if isinstance(row.get("primary_region_table"), dict):
        results["regions"] = row["primary_region_table"]
    return results


def _source_row(row: dict[str, Any], source_path: Path) -> dict[str, Any] | None:
    contract_payload = row.get("contract")
    if not isinstance(contract_payload, dict):
        return None
    contract = ClaimContract.model_validate(contract_payload)
    label = str(row.get("final_label") or "unknown")
    target_family = row.get("target_family") or infer_target_family(contract, row)
    return {
        "claim_id": str(row.get("claim_id") or contract.claim_id),
        "target_family": target_family,
        "source_mode": row.get("source_mode"),
        "model_spec": row.get("model_spec"),
        "question": contract.question,
        "draft_success": True,
        "gate_success": True,
        "estimand_match": True,
        "gate_verdict_label": label,
        "gate_verdict": {
            "label": label,
            "abstained": label != "confirmed",
            "rationale": str(row.get("rationale") or ""),
            "gates": _gate_payload(row, contract),
        },
        "gate_results": _result_payload(row, contract),
        "drafted_contract": contract.model_dump(mode="json"),
        "source_result_path": str(source_path),
        "source_label_authority": row.get("label_authority"),
        "source_scoring_label": row.get("scoring_label") or row.get("label_class"),
        "source_label_class": row.get("label_class"),
        "source_ground_truth": row.get("ground_truth"),
        "synthetic_failure_family": row.get("synthetic_failure_family") or row.get("family"),
        "failure_family": row.get("family"),
        "source_modality": row.get("modality"),
        "source_citation": row.get("source_citation"),
        "label_basis": row.get("label_basis"),
        "notes": row.get("notes"),
    }


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("claims") if isinstance(payload.get("claims"), list) else []
    executable: list[dict[str, Any]] = []
    non_executable: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            converted = _source_row(row, path)
        except Exception as exc:
            non_executable.append(
                {
                    "source_path": str(path),
                    "claim_id": str(row.get("claim_id")),
                    "reason": f"contract conversion failed: {exc}",
                    "final_label": row.get("final_label"),
                }
            )
            continue
        if converted is None:
            final_label = str(row.get("final_label") or "")
            if final_label in {"skipped_n", "error"}:
                non_executable.append(
                    {
                        "source_path": str(path),
                        "claim_id": str(row.get("claim_id")),
                        "reason": f"unscored row: {final_label}",
                        "final_label": row.get("final_label"),
                    }
                )
                continue
            non_executable.append(
                {
                    "source_path": str(path),
                    "claim_id": str(row.get("claim_id")),
                    "reason": "missing embedded ClaimContract",
                    "final_label": row.get("final_label"),
                }
            )
        else:
            executable.append(converted)
    return executable, non_executable


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [Path(item) for item in args.input]
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    loaded_rows: list[dict[str, Any]] = []
    non_executable: list[dict[str, Any]] = []
    excluded_by_evidence_policy: list[dict[str, Any]] = []
    duplicate_count = 0
    evidence_manifest = load_evidence_manifest(args.evidence_manifest)
    external_only_bases = evidence_manifest.external_only_bases() if evidence_manifest is not None else set()
    for path in inputs:
        executable, missing = _load_rows(path)
        non_executable.extend(missing)
        for row in executable:
            row.setdefault("model_spec", args.model_spec)
            loaded_rows.append(row)
            if evidence_manifest is not None:
                try:
                    contract = ClaimContract.model_validate(row["drafted_contract"])
                    bases = {
                        canonical_base_cohort(contract.discovery_cohort),
                        *[canonical_base_cohort(cohort) for cohort in contract.replication_cohorts],
                    }
                    external_only_source_bases = {
                        base
                        for base in bases
                        for external_base in external_only_bases
                        if base == external_base or base.startswith(f"{external_base}_")
                    }
                    if bool(args.exclude_external_only_sources) and external_only_source_bases:
                        excluded_by_evidence_policy.append(
                            {
                                "source_path": str(path),
                                "claim_id": row["claim_id"],
                                "reason": "source cohort is reserved for external evaluation",
                                "target_family": row.get("target_family"),
                            }
                        )
                        continue
                    if bool(args.require_excluded_evidence) and not evidence_manifest.has_excluded_evidence_for_contract(contract):
                        excluded_by_evidence_policy.append(
                            {
                                "source_path": str(path),
                                "claim_id": row["claim_id"],
                                "reason": "no target-level excluded holdout/external evidence",
                                "target_family": row.get("target_family"),
                            }
                        )
                        continue
                except Exception as exc:  # noqa: BLE001
                    excluded_by_evidence_policy.append(
                        {
                            "source_path": str(path),
                            "claim_id": row.get("claim_id"),
                            "reason": f"evidence-policy check failed: {exc}",
                            "target_family": row.get("target_family"),
                        }
                    )
                    continue
            key = (str(row["source_result_path"]), str(row["claim_id"]))
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            rows.append(row)

    failed = [row for row in rows if row.get("gate_verdict_label") != "confirmed"]
    labels = Counter(str(row.get("gate_verdict_label")) for row in rows)
    executable_by_source = Counter(str(row.get("source_result_path")) for row in rows)
    non_executable_by_reason = Counter(str(row.get("reason")) for row in non_executable)
    input_by_target = Counter(str(row.get("target_family") or "unknown") for row in loaded_rows)
    eligible_by_target = Counter(str(row.get("target_family") or "unknown") for row in rows)
    failed_by_target = Counter(str(row.get("target_family") or "unknown") for row in failed)
    excluded_by_target = Counter(str(row.get("target_family") or "unknown") for row in excluded_by_evidence_policy)
    exclusion_reason_by_target: dict[str, dict[str, int]] = {}
    for item in excluded_by_evidence_policy:
        target = str(item.get("target_family") or "unknown")
        reason = str(item.get("reason") or "unknown")
        exclusion_reason_by_target.setdefault(target, {})
        exclusion_reason_by_target[target][reason] = exclusion_reason_by_target[target].get(reason, 0) + 1
    source = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Claim-search source built from frozen Stage 2 CONFIRM gate results.",
        "source_model_spec": args.model_spec,
        "input_result_paths": [str(path) for path in inputs],
        "models": [
            {
                "model_spec": args.model_spec,
                "source_path": ";".join(str(path) for path in inputs),
                "initial_claims": rows,
            }
        ],
        "non_executable_reported_rows": non_executable,
        "excluded_by_evidence_policy": excluded_by_evidence_policy,
    }
    summary = {
        "created_at": source["created_at"],
        "source": str(args.out),
        "input_result_paths": [str(path) for path in inputs],
        "total_executable_initial_claims": len(rows),
        "failed_or_mismatched_claims": len(failed),
        "label_counts": dict(labels),
        "executable_by_source": dict(executable_by_source),
        "non_executable_reported_rows": len(non_executable),
        "non_executable_by_reason": dict(non_executable_by_reason),
        "excluded_by_evidence_policy": len(excluded_by_evidence_policy),
        "excluded_by_evidence_policy_examples": excluded_by_evidence_policy[:10],
        "input_claims_by_target_family": dict(input_by_target),
        "eligible_claims_by_target_family": dict(eligible_by_target),
        "failed_claims_by_target_family": dict(failed_by_target),
        "excluded_claims_by_target_family": dict(excluded_by_target),
        "exclusion_reasons_by_target_family": exclusion_reason_by_target,
        "duplicate_count": duplicate_count,
        "failed_claim_ids": [row["claim_id"] for row in failed],
        "non_executable_examples": non_executable[:10],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(source, indent=2), encoding="utf-8")
    summary_path = out.parent / "claim_source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return {"source": source, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=None, help="CONFIRM gate result JSON. Repeatable.")
    parser.add_argument("--out", default="review-stage/claim-search-gpt55-main/source/claim_search_source.json")
    parser.add_argument("--model-spec", default="benchmark/initial-claims-all-gpt55")
    parser.add_argument("--evidence-manifest", default=None)
    parser.add_argument("--require-excluded-evidence", action="store_true")
    parser.set_defaults(exclude_external_only_sources=True)
    parser.add_argument("--exclude-external-only-sources", dest="exclude_external_only_sources", action="store_true")
    parser.add_argument("--include-external-only-sources", dest="exclude_external_only_sources", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = list(DEFAULT_INPUTS)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
