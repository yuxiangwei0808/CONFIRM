"""Derive fixed CONFIRM evidence-tier decisions from frozen gate vectors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench.benchmark import BenchmarkDataset
from confirm.verdict import (
    EVIDENCE_TIER_REQUIRED_GATES,
    MinimumEvidenceTier,
    classify_support,
)
from nbs.claim_search_analysis_common import (
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)

ANALYSIS_VERSION = "confirm-evidence-tiers-v1"
MINIMUM_TIERS: tuple[MinimumEvidenceTier, ...] = (
    "discovery",
    "replicated",
    "confirmed",
)


def _stratum(
    benchmark_track: str,
    reference_basis: str,
    dataset_id: str,
) -> str:
    if benchmark_track == "scientific":
        return "internal_scientific"
    if benchmark_track == "synthetic_stress":
        return "synthetic_control"
    if benchmark_track == "external_transfer":
        kind = (
            "literature"
            if reference_basis == "literature"
            else "constructed_control"
        )
        return f"external_{kind}_{dataset_id}"
    return f"{benchmark_track}_{reference_basis}"


def build_decisions(package_dir: Path) -> list[dict[str, Any]]:
    """Create one decision per task and minimum tier."""

    dataset = BenchmarkDataset(package_dir)
    cases = {case.benchmark_case_id: case for case in dataset.cases}
    references = {
        reference.benchmark_case_id: reference
        for reference in dataset.references
    }
    tasks = {task.task_id: task for task in dataset.tasks}
    rows: list[dict[str, Any]] = []
    for outcome in sorted(dataset.outcomes, key=lambda row: row.task_id):
        case = cases[outcome.benchmark_case_id]
        reference = references[outcome.benchmark_case_id]
        task = tasks[outcome.task_id]
        gates = (
            outcome.gate_verdict.get("gates")
            if isinstance(outcome.gate_verdict, dict)
            else None
        )
        available = outcome.status == "completed" and isinstance(gates, dict)
        for minimum_tier in MINIMUM_TIERS:
            if available:
                decision = classify_support(gates, minimum_tier)
                supported = decision.supported
                achieved = decision.achieved_evidence_tier
                failed = list(decision.failed_required_gates)
            else:
                supported = False
                achieved = "unsupported"
                failed = ["gate_vector_unavailable"]
            if minimum_tier == "confirmed" and available:
                strict_supported = outcome.confirm_outcome == "confirmed"
                if supported != strict_supported:
                    raise ValueError(
                        "Strict evidence tier does not reproduce frozen verdict: "
                        f"{outcome.task_id}"
                    )
            rows.append(
                {
                    "task_id": outcome.task_id,
                    "benchmark_case_id": outcome.benchmark_case_id,
                    "semantic_cluster_id": case.semantic_cluster_id,
                    "benchmark_track": case.benchmark_track,
                    "target_family": case.target_family,
                    "dataset_id": task.dataset_id,
                    "unit": task.contract.estimand.unit,
                    "reference_basis": reference.basis,
                    "reference_disposition": reference.disposition,
                    "score_eligible": reference.score_eligible,
                    "stratum": _stratum(
                        case.benchmark_track,
                        reference.basis,
                        task.dataset_id,
                    ),
                    "minimum_evidence_tier": minimum_tier,
                    "achieved_evidence_tier": achieved,
                    "available": available,
                    "supported": supported,
                    "failed_required_gates": json.dumps(failed),
                }
            )
    return rows


def summarize_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize support without mixing reference strata."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                row["minimum_evidence_tier"],
                row["stratum"],
                row["unit"],
                row["reference_disposition"],
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        minimum_tier, stratum, unit, disposition = key
        eligible = [row for row in group if row["score_eligible"]]
        available = [row for row in eligible if row["available"]]
        supported_count = sum(row["supported"] for row in available)
        summaries.append(
            {
                "minimum_evidence_tier": minimum_tier,
                "stratum": stratum,
                "unit": unit,
                "reference_disposition": disposition,
                "task_count": len(group),
                "score_eligible_count": len(eligible),
                "available_count": len(available),
                "supported_count": supported_count,
                "supported_rate": (
                    supported_count / len(available) if available else None
                ),
            }
        )
    return summaries


def primary_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report the three predeclared sensitivity and safety quantities."""

    definitions = (
        (
            "confirmable_reference_recall",
            lambda row: (
                row["stratum"] == "internal_scientific"
                and row["reference_basis"] == "literature"
                and row["reference_disposition"] == "confirm"
            ),
        ),
        (
            "literature_abstention_unsafe_support",
            lambda row: (
                row["stratum"] == "internal_scientific"
                and row["reference_basis"] == "literature"
                and row["reference_disposition"] == "abstain"
            ),
        ),
        (
            "synthetic_control_support",
            lambda row: (
                row["stratum"] == "synthetic_control"
                and row["reference_disposition"] == "abstain"
            ),
        ),
    )
    output: list[dict[str, Any]] = []
    for minimum_tier in MINIMUM_TIERS:
        tier_rows = [
            row
            for row in rows
            if row["minimum_evidence_tier"] == minimum_tier
            and row["score_eligible"]
            and row["available"]
        ]
        for metric, predicate in definitions:
            selected = [row for row in tier_rows if predicate(row)]
            supported = sum(row["supported"] for row in selected)
            output.append(
                {
                    "minimum_evidence_tier": minimum_tier,
                    "metric": metric,
                    "supported_count": supported,
                    "denominator": len(selected),
                    "rate": supported / len(selected) if selected else None,
                }
            )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = Path(args.package_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = build_decisions(package_dir)
    summaries = summarize_decisions(decisions)
    metrics = primary_metrics(decisions)
    decision_path = out_dir / "evidence_tier_decisions.csv"
    summary_path = out_dir / "evidence_tier_summary.csv"
    metrics_path = out_dir / "primary_metrics.csv"
    write_csv_atomic(decision_path, decisions)
    write_csv_atomic(summary_path, summaries)
    write_csv_atomic(metrics_path, metrics)

    result = {
        "version": ANALYSIS_VERSION,
        "task_count": len(decisions) // len(MINIMUM_TIERS),
        "decision_count": len(decisions),
        "minimum_evidence_tiers": list(MINIMUM_TIERS),
        "strict_verdict_reconciliation_errors": 0,
        "primary_metrics": metrics,
        "interpretation_restrictions": [
            "The tiers are fixed gate policies, not calibrated operating points.",
            "The analysis favors sensitivity or specificity; it does not maximize either.",
            "Reference strata and external datasets are not pooled.",
            "Unresolved references are excluded from accuracy metrics.",
            "No gate result or threshold was recomputed.",
        ],
    }
    result_path = out_dir / "evidence_tier_results.json"
    write_json_atomic(result_path, result)

    inputs = [
        package_dir / "cases.jsonl",
        package_dir / "references.jsonl",
        package_dir / "tasks.jsonl",
        package_dir / "outcomes.jsonl",
    ]
    outputs = [decision_path, summary_path, metrics_path, result_path]
    implementation_path = Path(__file__).resolve()
    manifest = {
        "version": ANALYSIS_VERSION,
        "implementation": {
            "path": str(implementation_path.relative_to(implementation_path.parents[1])),
            "sha256": sha256_file(implementation_path),
        },
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in outputs
        ],
        "parameters": {
            "minimum_evidence_tiers": list(MINIMUM_TIERS),
            "required_gates": {
                tier: list(EVIDENCE_TIER_REQUIRED_GATES[tier])
                for tier in MINIMUM_TIERS
            },
        },
        "interpretation_restrictions": result["interpretation_restrictions"],
    }
    write_json_atomic(out_dir / "analysis_manifest.json", manifest)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        default="review-stage/neuroclaimbench-v2.1/compact",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/evidence-tiers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
