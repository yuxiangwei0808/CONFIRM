"""Reconcile the matched structured-diagnosis and generic-retry control arms."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from nbs.claim_search_analysis_common import read_result_header, sha256_json
from confirm.provenance import claim_search_implementation_hashes, mapping_sha256

MODES = ("structured_diagnosis", "generic_retry")
PAIR_FIELDS = (
    "generated_candidate_count",
    "schema_valid_candidate_count",
    "valid_candidate_count",
    "current_data_evaluated_count",
    "provisional_internal_pass_count",
    "final_multiplicity_adjusted_internal_pass_count",
    "multiplicity_retraction_count",
    "execution_error_count",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload.get("rows") or []]


def _recorded_search_implementation_sha256(payload: dict[str, Any]) -> str | None:
    provenance = payload.get("provenance") or {}
    value = provenance.get("search_implementation_hashes_sha256")
    if isinstance(value, str) and value:
        return value
    hashes = provenance.get("search_implementation_hashes")
    if isinstance(hashes, dict) and hashes:
        return mapping_sha256(hashes)
    return None


def _llm_call_counts(artifact: Path, payload: dict[str, Any]) -> dict[str, int]:
    provenance_path = artifact.parent / "run_provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        completed = provenance.get("rendered_prompt_record_count")
        superseded = provenance.get("superseded_transient_prompt_record_count")
        total = provenance.get("total_prompt_attempt_record_count")
        if completed is not None or total is not None:
            completed_count = int(completed if completed is not None else total)
            superseded_count = int(superseded or 0)
            return {
                "completed_trace_llm_call_count": completed_count,
                "superseded_transient_attempt_count": superseded_count,
                "total_llm_attempt_count": int(
                    total if total is not None else completed_count + superseded_count
                ),
            }
    completed_count = sum(
        len((state or {}).get("llm_candidate_prompts") or [])
        for state in payload.get("states") or []
    )
    return {
        "completed_trace_llm_call_count": completed_count,
        "superseded_transient_attempt_count": 0,
        "total_llm_attempt_count": completed_count,
    }


def preflight_current_implementation(structured_artifact: str | Path) -> dict[str, Any]:
    """Fail before LLM work when a modern sweep artifact used different search code."""

    payload = read_result_header(Path(structured_artifact))
    recorded = _recorded_search_implementation_sha256(payload)
    current = mapping_sha256(claim_search_implementation_hashes())
    if recorded is None:
        return {
            "status": "legacy_search_fingerprint_unavailable",
            "current_search_implementation_sha256": current,
        }
    if recorded != current:
        raise ValueError(
            "Structured control artifact was produced by different claim-search code. "
            "Use a structured sweep arm produced by the current implementation."
        )
    return {
        "status": "current_search_implementation_match",
        "current_search_implementation_sha256": current,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else ["claim_id"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(
    root: str | Path,
    *,
    structured_artifact: str | Path | None = None,
    generic_artifact: str | Path | None = None,
    expected_parent_count: int = 215,
) -> dict[str, Any]:
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "structured_diagnosis": (
            Path(structured_artifact)
            if structured_artifact
            else output / "structured_diagnosis" / "iterative_candidate_replay.json"
        ),
        "generic_retry": (
            Path(generic_artifact)
            if generic_artifact
            else output / "generic_retry" / "iterative_candidate_replay.json"
        ),
    }
    payloads = {mode: read_result_header(path) for mode, path in artifacts.items()}
    for mode, payload in payloads.items():
        if payload.get("status") != "completed":
            raise ValueError(f"Control arm is incomplete: {mode}")
        if (payload.get("config") or {}).get("feedback_mode") != mode:
            raise ValueError(f"Control arm has the wrong feedback mode: {mode}")
        if int(payload.get("completed_search_count") or 0) != expected_parent_count:
            raise ValueError(
                f"Control arm {mode} has {payload.get('completed_search_count')} parents; "
                f"expected {expected_parent_count}."
            )

    source_hashes = {
        str(((payload.get("provenance") or {}).get("source") or {}).get("sha256"))
        for payload in payloads.values()
    }
    models = {str(payload.get("llm_model")) for payload in payloads.values()}
    budgets = {
        (
            int((payload.get("config") or {}).get("max_rounds") or 0),
            int((payload.get("config") or {}).get("max_candidates_per_round") or 0),
            int((payload.get("config") or {}).get("llm_schema_retries") or 0),
        )
        for payload in payloads.values()
    }
    schemas = {
        str((payload.get("provenance") or {}).get("schema_sha256"))
        for payload in payloads.values()
    }
    partition_hashes = {
        str((payload.get("provenance") or {}).get("partition_hashes_sha256"))
        for payload in payloads.values()
    }
    manifests = {
        str(((payload.get("provenance") or {}).get("evidence_manifest") or {}).get("sha256"))
        for payload in payloads.values()
    }
    search_implementations = {
        value
        for payload in payloads.values()
        if (value := _recorded_search_implementation_sha256(payload)) is not None
    }
    all_search_fingerprints_present = all(
        _recorded_search_implementation_sha256(payload) is not None
        for payload in payloads.values()
    )
    full_implementations = {
        mode: sha256_json((payload.get("provenance") or {}).get("implementation_hashes") or {})
        for mode, payload in payloads.items()
    }
    if len(source_hashes) != 1 or "None" in source_hashes:
        raise ValueError(f"Control arms do not share one source hash: {source_hashes}")
    if len(models) != 1 or "None" in models:
        raise ValueError(f"Control arms do not share one model: {models}")
    if len(budgets) != 1:
        raise ValueError(f"Control arms do not share one budget: {budgets}")
    if (
        len(schemas) != 1
        or len(partition_hashes) != 1
        or len(manifests) != 1
        or (all_search_fingerprints_present and len(search_implementations) != 1)
        or "None" in schemas
        or "None" in partition_hashes
        or "None" in manifests
    ):
        raise ValueError(
            "Control arms do not share one schema, implementation, partition set, and evidence manifest."
        )

    rows_by_mode = {
        mode: {str(row["claim_id"]): row for row in _load_rows(artifacts[mode])}
        for mode in MODES
    }
    claim_sets = {mode: set(rows) for mode, rows in rows_by_mode.items()}
    if claim_sets[MODES[0]] != claim_sets[MODES[1]]:
        raise ValueError("Control arms do not contain identical parent claim IDs.")

    max_rounds, max_candidates, schema_retries = next(iter(budgets))
    paired_rows: list[dict[str, Any]] = []
    for claim_id in sorted(claim_sets[MODES[0]]):
        structured = rows_by_mode["structured_diagnosis"][claim_id]
        generic = rows_by_mode["generic_retry"][claim_id]
        row: dict[str, Any] = {
            "claim_id": claim_id,
            "target_family": structured.get("target_family"),
            "source_mode": structured.get("source_mode"),
        }
        for field in PAIR_FIELDS:
            structured_value = int(structured.get(field) or 0)
            generic_value = int(generic.get(field) or 0)
            row[f"structured_{field}"] = structured_value
            row[f"generic_{field}"] = generic_value
            row[f"difference_{field}"] = structured_value - generic_value
        row["structured_parent_supported"] = int(
            int(structured.get("final_multiplicity_adjusted_internal_pass_count") or 0) > 0
        )
        row["generic_parent_supported"] = int(
            int(generic.get("final_multiplicity_adjusted_internal_pass_count") or 0) > 0
        )
        paired_rows.append(row)

    support_cells: Counter[str] = Counter()
    stratified: dict[str, dict[str, Counter[str]]] = {
        "target_family": defaultdict(Counter),
        "source_mode": defaultdict(Counter),
    }
    for row in paired_rows:
        structured_supported = bool(row["structured_parent_supported"])
        generic_supported = bool(row["generic_parent_supported"])
        if structured_supported and generic_supported:
            cell = "both"
        elif structured_supported:
            cell = "structured_only"
        elif generic_supported:
            cell = "generic_only"
        else:
            cell = "neither"
        support_cells[cell] += 1
        for dimension in stratified:
            value = str(row.get(dimension) or "unknown")
            stratified[dimension][value]["parent_count"] += 1
            stratified[dimension][value]["structured_supported_parent_count"] += int(
                structured_supported
            )
            stratified[dimension][value]["generic_supported_parent_count"] += int(
                generic_supported
            )

    call_counts_by_mode = {
        mode: _llm_call_counts(artifacts[mode], payloads[mode])
        for mode in MODES
    }
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Matched descriptive control; one GPT-5.5 realization, not a causal estimate.",
        "source_sha256": next(iter(source_hashes)),
        "search_implementation_hashes_sha256": (
            next(iter(search_implementations))
            if all_search_fingerprints_present
            else None
        ),
        "full_implementation_hashes_sha256_by_arm": full_implementations,
        "implementation_compatibility_status": (
            "exact_search_implementation_match"
            if all_search_fingerprints_present
            else "legacy_search_fingerprint_unavailable"
        ),
        "llm_model": next(iter(models)),
        "budget": {
            "max_rounds": max_rounds,
            "max_candidates_per_round": max_candidates,
            "llm_schema_retries": schema_retries,
        },
        "parent_count": len(paired_rows),
        "arms": {
            mode: {
                "artifact": str(artifacts[mode]),
                "llm_call_count": call_counts_by_mode[mode]["total_llm_attempt_count"],
                **call_counts_by_mode[mode],
                "summary": payloads[mode].get("summary") or {},
            }
            for mode in MODES
        },
        "paired_parent_support_cells": {
            key: support_cells[key]
            for key in ("both", "structured_only", "generic_only", "neither")
        },
        "parent_support_by_stratum": {
            dimension: {
                value: dict(counts)
                for value, counts in sorted(values.items())
            }
            for dimension, values in stratified.items()
        },
        "paired_difference_totals": {
            field: sum(row[f"difference_{field}"] for row in paired_rows)
            for field in PAIR_FIELDS
        },
        "causal_interpretation_allowed": False,
    }
    _write_json(output / "control_summary.json", summary)
    _write_csv(output / "control_parent_pairs.csv", paired_rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root")
    parser.add_argument("--structured-artifact")
    parser.add_argument("--generic-artifact")
    parser.add_argument("--expected-parent-count", type=int, default=215)
    parser.add_argument("--preflight-current", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_current:
        if not args.structured_artifact:
            parser.error("--preflight-current requires --structured-artifact")
        print(json.dumps(preflight_current_implementation(args.structured_artifact), indent=2))
        return 0
    if not args.out_root:
        parser.error("--out-root is required unless --preflight-current is used")
    summary = run(
        args.out_root,
        structured_artifact=args.structured_artifact,
        generic_artifact=args.generic_artifact,
        expected_parent_count=args.expected_parent_count,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
