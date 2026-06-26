"""Run CONFIRM's agentic loop over a focused multi-LLM claim set."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from confirm.agent import _execute_contract, build_data_catalog, draft_contract, interpret
from confirm.contract import ClaimContract
from confirm.llm import make_llm
from confirm.verdict import Verdict

DEFAULT_MODELS = (
    "openai:gpt-5-mini",
    "openai:gpt-4o",
    "anthropic:claude-haiku-4-5",
    "anthropic:claude-sonnet-4-5",
    "openrouter:deepseek/deepseek-chat",
    "openrouter:qwen/qwen-2.5-72b-instruct",
)
ANTI_HALLUCINATION_SUFFIX = "Numeric statements not present in the computed result bundle were removed."


@dataclass(frozen=True)
class AgenticClaim:
    claim_id: str
    question: str
    intended: dict[str, Any]
    label_class: str


CLAIMS: tuple[AgenticClaim, ...] = (
    AgenticClaim(
        claim_id="ad_hippocampal_atrophy_adni_oasis3",
        label_class="known_positive",
        question=(
            "In ADNI as discovery and OASIS3 as replication, are Dementia participants lower than CN "
            "in smri_hippocampus after age, sex, and eTIV adjustment?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "smri_hippocampus",
            "predictor": "dx",
            "group": {"var": "dx", "case": "Dementia", "control": "CN"},
            "direction": "negative",
            "unit": "scalar",
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
        },
    ),
    AgenticClaim(
        claim_id="ad_entorhinal_atrophy_adni_oasis3",
        label_class="known_positive",
        question=(
            "In ADNI as discovery and OASIS3 as replication, are Dementia participants lower than CN "
            "in smri_entorhinal after age, sex, and eTIV adjustment?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "smri_entorhinal",
            "predictor": "dx",
            "group": {"var": "dx", "case": "Dementia", "control": "CN"},
            "direction": "negative",
            "unit": "scalar",
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
        },
    ),
    AgenticClaim(
        claim_id="ad_wholebrain_atrophy_adni_oasis3",
        label_class="known_positive",
        question=(
            "In ADNI as discovery and OASIS3 as replication, are Dementia participants lower than CN "
            "in smri_wholebrain after age, sex, and eTIV adjustment?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "smri_wholebrain",
            "predictor": "dx",
            "group": {"var": "dx", "case": "Dementia", "control": "CN"},
            "direction": "negative",
            "unit": "scalar",
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
        },
    ),
    AgenticClaim(
        claim_id="ad_brainwide_smri_adni_oasis3",
        label_class="known_positive",
        question=(
            "In ADNI as discovery and OASIS3 as replication, is there a brain-wide sMRI atrophy pattern "
            "for Dementia versus CN across shared smri_ regions after age, sex, and eTIV adjustment?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "smri_",
            "outcome_aliases": ["smri_", "smri", "smri_*"],
            "predictor": "dx",
            "group": {"var": "dx", "case": "Dementia", "control": "CN"},
            "direction": "negative",
            "unit": "brainwide",
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
        },
    ),
    AgenticClaim(
        claim_id="sz_fc_within_cobre_fbirn",
        label_class="known_positive",
        question=(
            "In COBRE as discovery and FBIRN as replication, do SZ participants have lower "
            "fc_within_network than HC after age and sex adjustment?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "fc_within_network",
            "predictor": "dx",
            "group": {"var": "dx", "case": "SZ", "control": "HC"},
            "direction": "negative",
            "unit": "scalar",
            "discovery_cohort": "COBRE",
            "replication_cohorts": ["FBIRN"],
        },
    ),
    AgenticClaim(
        claim_id="brain_aging_hippocampus_adni_oasis3_cn",
        label_class="known_positive",
        question=(
            "Among CN participants only, using ADNI as discovery and OASIS3 as replication, "
            "is older age associated with lower smri_hippocampus after sex and eTIV adjustment?"
        ),
        intended={
            "type": "association",
            "outcome": "smri_hippocampus",
            "predictor": "age",
            "group": None,
            "direction": "negative",
            "unit": "scalar",
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
            "inclusion": 'dx == "CN"',
        },
    ),
    AgenticClaim(
        claim_id="injected_null_site_abide1",
        label_class="known_null",
        question=(
            "As a site-confound null/control in ABIDE1, is fc_mean_abs different between site NYU "
            "and site UCLA_1 after age and sex adjustment, using ABIDE1 as the replication check?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "fc_mean_abs",
            "predictor": "site",
            "group": {"var": "site", "case": "NYU", "control": "UCLA_1"},
            "direction": "two_sided",
            "unit": "scalar",
            "discovery_cohort": "ABIDE1",
            "replication_cohorts": ["ABIDE1"],
        },
    ),
    AgenticClaim(
        claim_id="asd_fc_mean_abs_abide1",
        label_class="fragile",
        question=(
            "In ABIDE1, do ASD participants differ from HC in fc_mean_abs after age and sex adjustment, "
            "using ABIDE1 as the replication check because no second ASD cohort is in this catalog?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "fc_mean_abs",
            "predictor": "dx",
            "group": {"var": "dx", "case": "ASD", "control": "HC"},
            "direction": "two_sided",
            "unit": "scalar",
            "discovery_cohort": "ABIDE1",
            "replication_cohorts": ["ABIDE1"],
        },
    ),
    AgenticClaim(
        claim_id="sex_smri_hippocampus_adni_oasis3_cn",
        label_class="fragile",
        question=(
            "Among CN participants only, using ADNI as discovery and OASIS3 as replication, "
            "do female participants have lower smri_hippocampus than male participants after age and eTIV adjustment?"
        ),
        intended={
            "type": "group_diff",
            "outcome": "smri_hippocampus",
            "predictor": "sex",
            "group": {"var": "sex", "case": "F", "control": "M"},
            "direction": "negative",
            "unit": "scalar",
            "discovery_cohort": "ADNI",
            "replication_cohorts": ["OASIS3"],
            "inclusion": 'dx == "CN"',
        },
    ),
)


def _json_safe(data: Any) -> Any:
    if hasattr(data, "to_dict"):
        return _json_safe(data.to_dict())
    if data.__class__.__module__.startswith("numpy") and hasattr(data, "item"):
        return _json_safe(data.item())
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_safe(value) for value in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data


def _safe_model_name(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec).strip("_") or "model"


def _merge_catalogs(roots: list[Path]) -> tuple[dict[str, Any], dict[str, Path]]:
    cohorts: list[dict[str, Any]] = []
    root_by_cohort: dict[str, Path] = {}
    for root in roots:
        catalog = build_data_catalog(root)
        for entry in catalog.get("cohorts", []):
            merged = dict(entry)
            merged["data_root"] = str(root)
            cohorts.append(merged)
            cohort = merged.get("cohort")
            if cohort:
                root_by_cohort[str(cohort)] = root
    return {
        "data_dir": "merged",
        "data_roots": [str(root) for root in roots],
        "cohorts": cohorts,
    }, root_by_cohort


def _group_dict(contract: ClaimContract) -> dict[str, str] | None:
    group = contract.estimand.group
    return group.model_dump() if group is not None else None


def _outcome_matches(actual: Any, intended: dict[str, Any]) -> bool:
    expected = intended["outcome"]
    aliases = [expected, *intended.get("outcome_aliases", [])]
    if actual in aliases:
        return True
    if intended.get("unit") != "brainwide":
        return False
    if isinstance(actual, list) and isinstance(expected, str) and expected.endswith("_"):
        return bool(actual) and all(str(item).startswith(expected) for item in actual)
    if isinstance(actual, str) and isinstance(expected, str) and expected.endswith("_"):
        return actual == expected.rstrip("_")
    return False


def _estimand_comparison(contract: ClaimContract, intended: dict[str, Any]) -> tuple[bool, dict[str, dict[str, Any]]]:
    actual = {
        "type": contract.estimand.type,
        "outcome": contract.estimand.outcome,
        "predictor": contract.estimand.predictor,
        "group": _group_dict(contract),
        "direction": contract.estimand.direction,
        "unit": contract.estimand.unit,
        "discovery_cohort": contract.discovery_cohort,
        "replication_cohorts": contract.replication_cohorts,
    }
    expected = {key: intended.get(key) for key in actual}
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if key == "outcome":
            matched = _outcome_matches(actual_value, intended)
        elif key == "replication_cohorts":
            matched = sorted(actual_value) == sorted(expected_value or [])
        else:
            matched = actual_value == expected_value
        if not matched:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    return not mismatches, mismatches


def _interpret_target(results: dict[str, Any]) -> Any:
    if "regions" in results:
        return results["regions"]
    if "primary" in results:
        return results["primary"]
    return results


def _run_claim_for_model(
    model_spec: str,
    claim: AgenticClaim,
    catalog: dict[str, Any],
    root_by_cohort: dict[str, Path],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_spec": model_spec,
        "claim_id": claim.claim_id,
        "label_class": claim.label_class,
        "question": claim.question,
        "intended_estimand": claim.intended,
        "draft_success": False,
        "estimand_match": False,
        "gate_success": False,
        "gate_verdict_label": None,
        "anti_hallucination_ok": None,
        "anti_hallucination_removed_number": False,
    }

    try:
        llm = make_llm(model_spec)
    except Exception as exc:
        row.update({"error_stage": "make_llm", "error": str(exc)})
        return row

    try:
        contract = draft_contract(claim.question, catalog, llm=llm)
        row["draft_success"] = True
        row["drafted_contract"] = contract.model_dump(mode="json")
        matched, mismatches = _estimand_comparison(contract, claim.intended)
        row["estimand_match"] = matched
        row["estimand_mismatches"] = mismatches
    except Exception as exc:
        row.update({"error_stage": "draft_contract", "error": str(exc)})
        return row

    try:
        data_root = root_by_cohort.get(contract.discovery_cohort)
        if data_root is None:
            raise ValueError(f"No data root contains discovery cohort {contract.discovery_cohort!r}")
        verdict, results, cohort_paths = _execute_contract(contract, data_root)
    except Exception as exc:
        verdict = Verdict(
            label="fragile",
            abstained=True,
            rationale=f"execution_error: {exc}",
            gates={"execution": False, "reason": "execution_error", "error": str(exc)},
        )
        results = {
            "contract": contract.model_dump(mode="json"),
            "verdict": verdict,
            "execution_error": {"reason": "execution_error", "error": str(exc)},
        }
        cohort_paths = []
        row["governance_reason"] = "execution_error"
        row["execution_error"] = str(exc)

    row["gate_success"] = True
    row["gate_verdict"] = verdict.to_dict()
    row["gate_verdict_label"] = verdict.label
    row["cohort_paths"] = [str(path) for path in cohort_paths]
    row["gate_results"] = _json_safe(results)
    if row.get("governance_reason") == "execution_error":
        row["interpretation"] = verdict.rationale
        row["anti_hallucination_ok"] = True
        return row

    try:
        narrative = interpret(verdict, _interpret_target(results), llm=llm)
        removed = ANTI_HALLUCINATION_SUFFIX in narrative
        row["interpretation"] = narrative
        row["anti_hallucination_ok"] = not removed
        row["anti_hallucination_removed_number"] = removed
    except Exception as exc:
        row.update({"error_stage": "interpret", "interpret_error": str(exc), "anti_hallucination_ok": False})
    return row


def _model_summary(model_spec: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_claims = len(rows)
    draft_success = sum(bool(row.get("draft_success")) for row in rows)
    estimand_match = sum(bool(row.get("estimand_match")) for row in rows)
    gate_success = sum(bool(row.get("gate_success")) for row in rows)
    anti_catches = sum(bool(row.get("anti_hallucination_removed_number")) for row in rows)
    return {
        "model_spec": model_spec,
        "n_claims": n_claims,
        "draft_success_count": draft_success,
        "draft_success_rate": draft_success / n_claims if n_claims else math.nan,
        "estimand_match_count": estimand_match,
        "estimand_match_rate": estimand_match / draft_success if draft_success else math.nan,
        "gate_success_count": gate_success,
        "gate_success_rate": gate_success / draft_success if draft_success else math.nan,
        "anti_hallucination_catch_count": anti_catches,
    }


def _cross_model_summary(model_payloads: list[dict[str, Any]], claims: list[AgenticClaim]) -> dict[str, Any]:
    rows = [row for payload in model_payloads for row in payload["claims"]]
    by_claim: dict[str, Any] = {}
    for claim in claims:
        claim_rows = [row for row in rows if row["claim_id"] == claim.claim_id]
        valid = {
            row["model_spec"]: row["gate_verdict_label"]
            for row in claim_rows
            if row.get("draft_success") and row.get("gate_verdict_label")
        }
        agreement = None
        if len(valid) >= 2:
            agreement = len(set(valid.values())) == 1
        by_claim[claim.claim_id] = {
            "label_class": claim.label_class,
            "intended_estimand": claim.intended,
            "verdicts": {
                row["model_spec"]: row.get("gate_verdict_label") or row.get("error_stage") or "not_run"
                for row in claim_rows
            },
            "valid_gate_verdicts": valid,
            "agreement": agreement,
        }

    comparable = [item for item in by_claim.values() if item["agreement"] is not None]
    agreement_count = sum(bool(item["agreement"]) for item in comparable)
    catch_count = sum(bool(row.get("anti_hallucination_removed_number")) for row in rows)
    return {
        "per_model": {payload["model_spec"]: payload["summary"] for payload in model_payloads},
        "per_claim": by_claim,
        "cross_model_verdict_agreement": (
            agreement_count / len(comparable) if comparable else None
        ),
        "cross_model_verdict_agreement_count": agreement_count,
        "cross_model_verdict_agreement_denominator": len(comparable),
        "anti_hallucination_catch_count": catch_count,
    }


def _run_one_model(
    model_spec: str,
    claims: list[AgenticClaim],
    catalog: dict[str, Any],
    root_by_cohort: dict[str, Path],
    out_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    rows = [_run_claim_for_model(model_spec, claim, catalog, root_by_cohort) for claim in claims]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "model_spec": model_spec,
        "summary": _model_summary(model_spec, rows),
        "claims": rows,
    }
    safe_name = _safe_model_name(model_spec)
    run_path = out_dir / f"agentic_multillm_{run_id}_{safe_name}.json"
    model_path = out_dir / f"{safe_name}.json"
    run_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    shutil.copyfile(run_path, model_path)
    payload["path"] = str(model_path)
    payload["run_path"] = str(run_path)
    return payload


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
            executor.submit(_run_one_model, model, selected_claims, catalog, root_by_cohort, out_dir, run_id): model
            for model in models
        }
        for future in as_completed(futures):
            payloads.append(future.result())

    payloads.sort(key=lambda item: models.index(item["model_spec"]))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "command": {
            "models": models,
            "smri_root": str(args.smri_root),
            "cluster_root": str(args.cluster_root),
            "limit": args.limit,
            "claim_id": args.claim_id,
            "max_workers": args.max_workers,
        },
        "model_outputs": [
            {"model_spec": item["model_spec"], "path": item["path"], "run_path": item["run_path"]}
            for item in payloads
        ],
        "catalog": {
            "data_roots": catalog["data_roots"],
            "cohorts": [
                {"cohort": entry.get("cohort"), "n": entry.get("n"), "data_root": entry.get("data_root")}
                for entry in catalog["cohorts"]
            ],
        },
        **_cross_model_summary(payloads, selected_claims),
    }
    summary_path = out_dir / f"agentic_multillm_summary_{run_id}.json"
    latest_path = out_dir / "agentic_multillm_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    shutil.copyfile(summary_path, latest_path)

    print(f"wrote {summary_path}")
    for item in payloads:
        print(f"wrote {item['path']}")
        print(f"wrote {item['run_path']}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--smri-root", default="data/prepared_data/smri_disease")
    parser.add_argument("--cluster-root", default="data/prepared_data/cluster_recovered")
    parser.add_argument("--out-dir", default="review-stage/agentic-multillm")
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
