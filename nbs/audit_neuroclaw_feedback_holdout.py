"""Re-check recovered claims on evidence that played no part in finding them.

Both feedback arms are scored by CONFIRM, and one of them is coached by CONFIRM,
so support on the search data cannot separate a better claim from a better-aimed
one. This audit re-executes every recovered candidate against its predeclared
holdout partition under unchanged gates.

The holdout partitions were consulted during method development, so this measures
retrospective concordance rather than prospective confirmation. It is
nevertheless independent of the feedback signal, which is the specific
circularity at issue.

Scope note: this uses the predeclared holdout pairing and the unchanged gates,
but not the full preflight used elsewhere in the project, so it omits the overlap
and unit diagnostics that pipeline adds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from confirm.contract import ClaimContract
from confirm.evidence_partitions import load_evidence_manifest
from confirm.excluded_evidence import (
    ExcludedEvidenceUnavailableError,
    execute_contract,
    mapped_contract_for_evidence,
)

PROTOCOL_VERSION = "neuroclaimbench-v2.1-neuroclaw-feedback-holdout-v1"
ARMS = ("no_feedback", "self_critique", "confirm_diagnosis")


def _recovered_candidates(arm_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in arm_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state = row.get("state") or {}
        supported = set(state.get("internally_supported_candidate_ids") or [])
        if not supported:
            continue
        for candidate in state.get("candidate_history") or []:
            if candidate.get("candidate_id") not in supported:
                continue
            contract = candidate.get("proposed_contract")
            if not contract:
                continue
            out.append(
                {
                    "parent_claim_id": row["claim_id"],
                    "candidate_id": candidate["candidate_id"],
                    "contract": contract,
                }
            )
    return out


def _baseline_confirmations(path: Path) -> list[dict[str, Any]]:
    """Claims CONFIRM confirmed on the agent's first attempt, before any feedback."""

    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("confirm_support") or not row.get("drafted_contract"):
            continue
        out.append(
            {
                "parent_claim_id": row["claim_id"],
                "candidate_id": row["claim_id"],
                "contract": row["drafted_contract"],
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    manifest = load_evidence_manifest(args.evidence_manifest)
    if manifest is None:
        raise SystemExit(f"No evidence manifest at {args.evidence_manifest}")
    evidence_roots = [Path(args.evidence_root)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        if arm == "no_feedback":
            drafted = Path(args.main_dir) / "drafted_outcomes.jsonl"
            items = _baseline_confirmations(drafted) if drafted.exists() else []
        else:
            arm_path = Path(args.main_dir) / f"arm_{arm}.jsonl"
            items = _recovered_candidates(arm_path) if arm_path.exists() else []
        for item in items:
            source = ClaimContract.model_validate(item["contract"])
            record = {
                "arm": arm,
                "parent_claim_id": item["parent_claim_id"],
                "candidate_id": item["candidate_id"],
                "holdout_status": "",
                "holdout_label": "",
                "holdout_supported": False,
                "reason": "",
            }
            try:
                mapped, _discovery, _reps, set_id = mapped_contract_for_evidence(
                    source,
                    manifest,
                    "holdout",
                )
                result = execute_contract(
                    mapped,
                    evidence_roots,
                    evidence_scope="holdout",
                    source_contract=source,
                    evidence_set_id=set_id,
                )
                record["holdout_status"] = "evaluated"
                record["holdout_label"] = result["final_label"]
                record["holdout_supported"] = result["final_label"] == "confirmed"
            except (ExcludedEvidenceUnavailableError, FileNotFoundError, ValueError) as exc:
                record["holdout_status"] = "unavailable"
                record["reason"] = str(exc)[:200]
            rows.append(record)

    if not rows:
        raise SystemExit("No recovered candidates found to audit")

    with (out_dir / "holdout_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "arms": {}}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        evaluated = [r for r in arm_rows if r["holdout_status"] == "evaluated"]
        supported = [r for r in evaluated if r["holdout_supported"]]
        summary["arms"][arm] = {
            "recovered_candidates": len(arm_rows),
            "holdout_evaluated": len(evaluated),
            "holdout_unavailable": len(arm_rows) - len(evaluated),
            "holdout_supported": len(supported),
            "holdout_supported_parents": len({r["parent_claim_id"] for r in supported}),
        }
    (out_dir / "holdout_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    for arm, stats in summary["arms"].items():
        print(
            f"{arm:20} recovered {stats['recovered_candidates']:>3}  "
            f"holdout-evaluated {stats['holdout_evaluated']:>3}  "
            f"unavailable {stats['holdout_unavailable']:>3}  "
            f"holdout-supported {stats['holdout_supported']:>3} "
            f"across {stats['holdout_supported_parents']} parents"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-dir",
        default="review-stage/neuroclaimbench-v2.1/neuroclaw-feedback-v1",
    )
    parser.add_argument(
        "--out-dir",
        default="review-stage/neuroclaimbench-v2.1/neuroclaw-feedback-v1",
    )
    parser.add_argument(
        "--evidence-manifest",
        default="data/prepared_data/evidence_partitions/manifest.json",
    )
    parser.add_argument(
        "--evidence-root",
        default="data/prepared_data/evidence_partitions/cohorts",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
