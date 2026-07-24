"""Convert nested v7 search checkpoints into normalized JSONL artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from confirm.search_artifacts import (
    CandidateProposal,
    NormalizedLLMCall,
    NormalizedParent,
    read_v7_result_header,
)


RECONCILIATION_FIELDS = {
    "generated_candidate_count": "generated_candidate_count",
    "proposals_returned_count": "proposals_returned_count",
    "schema_valid_candidate_count": "schema_valid_candidate_count",
    "candidate_count": "candidate_count",
    "unique_candidate_count": "unique_candidate_count",
    "duplicate_candidate_count": "duplicate_candidate_count",
    "current_data_evaluated_count": "current_data_evaluated_count",
    "execution_complete_candidate_count": "execution_complete_candidate_count",
    "provisional_internal_pass_count": "provisional_internal_pass_count",
    "final_multiplicity_adjusted_internal_pass_count": (
        "final_multiplicity_adjusted_internal_pass_count"
    ),
    "multiplicity_retraction_count": "multiplicity_retraction_count",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    count = 0
    parts: list[str] = []
    for row in rows:
        value = (
            row.model_dump(mode="json")
            if hasattr(row, "model_dump")
            else row
        )
        parts.append(json.dumps(value, sort_keys=True) + "\n")
        count += 1
    _atomic_text(path, "".join(parts))
    return count


def _checkpoint(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = payload.pop("checkpoint_sha256", None)
    if checksum != _sha256_json(payload):
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    return payload


def _call_key(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(payload["round_index"]),
        int(payload["attempt_index"]),
        int(payload["schema_attempt_index"]),
        int(payload["validation_retry_index"]),
    )


def _normalized_call(
    parent_id: str,
    prompt: dict[str, Any],
    response: dict[str, Any],
) -> NormalizedLLMCall:
    key = _call_key(prompt)
    if key != _call_key(response):
        raise ValueError(
            f"Prompt/response key mismatch for {parent_id}: "
            f"{key} != {_call_key(response)}"
        )
    call_id = (
        f"{parent_id}:r{key[0]}:a{key[1]}:"
        f"s{key[2]}:v{key[3]}"
    )
    return NormalizedLLMCall(
        call_id=call_id,
        parent_claim_id=parent_id,
        round_index=key[0],
        attempt_index=key[1],
        schema_attempt_index=key[2],
        validation_retry_index=key[3],
        is_retry=bool(prompt["is_retry"]),
        retry_kind=str(prompt["retry_kind"]),
        model=str(prompt["model"]),
        system=str(prompt["system"]),
        user=str(prompt["user"]),
        prompt_hash=str(prompt["prompt_hash"]),
        raw_response=str(response.get("raw_response") or ""),
        response_candidate_count=int(
            response.get("candidate_count") or 0
        ),
        parse_error=response.get("parse_error"),
    )


def _normalized_parent(
    checkpoint: dict[str, Any],
) -> NormalizedParent:
    state = checkpoint["state"]
    return NormalizedParent(
        index=int(checkpoint["index"]),
        claim_id=str(checkpoint["claim_id"]),
        row=dict(checkpoint["row"]),
        original_claim=dict(state["original_claim"]),
        source_metadata=dict(state["source_metadata"]),
        failure_localization=state.get("failure_localization"),
        lineage_graph=dict(state["lineage_graph"]),
        used_evidence=list(state["used_evidence"]),
        internally_supported_candidate_ids=list(
            state["internally_supported_candidate_ids"]
        ),
        provisional_supported_candidate_ids=list(
            state["provisional_supported_candidate_ids"]
        ),
        round_failure_contexts=list(state["round_failure_contexts"]),
        round_summaries=list(state["round_summaries"]),
        final_search_family_size=int(state["final_search_family_size"]),
        stopped_reason=str(state["stopped_reason"]),
    )


def _legacy_result_hash(
    frozen_manifest: dict[str, Any],
    rounds: int,
    candidates: int,
) -> str | None:
    for arm in (
        frozen_manifest.get("claim_search", {}).get("arms", [])
    ):
        if (
            int(arm["max_rounds"]) == rounds
            and int(arm["max_candidates_per_round"]) == candidates
        ):
            return str(arm["result"]["sha256"])
    return None


def convert_arm(
    *,
    arm_row: dict[str, Any],
    sweep_root: Path,
    frozen_manifest: dict[str, Any],
) -> dict[str, Any]:
    rounds = int(arm_row["max_rounds"])
    candidates_per_round = int(
        arm_row["max_candidates_per_round"]
    )
    arm_id = f"r{rounds}_c{candidates_per_round}"
    legacy_result = Path(str(arm_row["artifact"]))
    legacy_dir = legacy_result.parent
    checkpoints = sorted(
        (legacy_dir / "checkpoints/parents").glob("parent_*.json")
    )
    if len(checkpoints) != 215:
        raise ValueError(
            f"{arm_id}: expected 215 checkpoints, found {len(checkpoints)}"
        )

    parents: list[NormalizedParent] = []
    calls: list[NormalizedLLMCall] = []
    retained_candidates: list[dict[str, Any]] = []
    unretained_candidates: list[dict[str, Any]] = []
    duplicate_candidates: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    supported_parent_count = 0

    for path in checkpoints:
        checkpoint = _checkpoint(path)
        parent = _normalized_parent(checkpoint)
        parents.append(parent)
        row = checkpoint["row"]
        state = checkpoint["state"]
        for matrix_key, row_key in RECONCILIATION_FIELDS.items():
            totals[matrix_key] += int(row.get(row_key) or 0)
        supported_parent_count += int(
            bool(row.get("parent_with_internal_support"))
        )

        prompts = {
            _call_key(item): item
            for item in state["llm_candidate_prompts"]
        }
        responses = {
            _call_key(item): item
            for item in state["llm_candidate_responses"]
        }
        if prompts.keys() != responses.keys():
            raise ValueError(
                f"{arm_id}/{parent.claim_id}: prompt/response keys differ"
            )
        calls.extend(
            _normalized_call(
                parent.claim_id,
                prompts[key],
                responses[key],
            )
            for key in sorted(prompts)
        )

        for proposal in state["candidate_history"]:
            compact = CandidateProposal.from_v7(proposal)
            retained_candidates.append(
                {
                    "record_type": "retained",
                    "proposal": compact.model_dump(mode="json"),
                }
            )
        for attempt in state["unretained_candidate_attempts"]:
            unretained_candidates.append(
                {
                    "record_type": "unretained",
                    "parent_claim_id": parent.claim_id,
                    **attempt,
                }
            )
        for duplicate in state["duplicate_candidates"]:
            duplicate_candidates.append(
                {
                    "record_type": "duplicate",
                    **duplicate,
                }
            )
        for evaluation in state["evaluations"]:
            record = dict(evaluation)
            proposal = record.pop("proposal")
            if proposal["candidate_id"] != record["candidate_id"]:
                raise ValueError(
                    f"{arm_id}: evaluation proposal ID mismatch"
                )
            evaluations.append(record)

    if [parent.index for parent in parents] != list(range(1, 216)):
        raise ValueError(f"{arm_id}: parent indexes are not contiguous")
    expected = {
        key: int(arm_row.get(key) or 0)
        for key in RECONCILIATION_FIELDS
    }
    observed = dict(totals)
    observed["parents_with_internal_support_count"] = (
        supported_parent_count
    )
    expected["parents_with_internal_support_count"] = int(
        arm_row.get("parents_with_internal_support_count") or 0
    )
    mismatches = {
        key: {"expected": expected[key], "observed": observed.get(key, 0)}
        for key in expected
        if expected[key] != observed.get(key, 0)
    }
    if mismatches:
        raise ValueError(f"{arm_id}: reconciliation failed: {mismatches}")

    out_dir = sweep_root / "normalized" / "arms" / arm_id
    files = {
        "parents": out_dir / "parents.jsonl",
        "llm_calls": out_dir / "llm_calls.jsonl",
        "candidates": out_dir / "candidates.jsonl",
        "evaluations": out_dir / "evaluations.jsonl",
        "parent_summaries": out_dir / "parent_summaries.jsonl",
        "summary": out_dir / "summary.json",
        "run_header": out_dir / "run_header.json",
        "run_provenance": out_dir / "run_provenance.json",
    }
    _write_jsonl(files["parents"], parents)
    _write_jsonl(files["llm_calls"], calls)
    _write_jsonl(
        files["candidates"],
        [
            *retained_candidates,
            *unretained_candidates,
            *duplicate_candidates,
        ],
    )
    _write_jsonl(files["evaluations"], evaluations)
    _write_jsonl(
        files["parent_summaries"],
        [parent.row for parent in parents],
    )
    summary = {
        "arm_id": arm_id,
        "max_rounds": rounds,
        "max_candidates_per_round": candidates_per_round,
        "legacy_summary": arm_row,
        "reconciled_counts": observed,
    }
    _atomic_text(
        files["summary"],
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        files["run_header"],
        json.dumps(
            read_v7_result_header(legacy_result),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    shutil.copy2(
        legacy_dir / "run_provenance.json",
        files["run_provenance"],
    )
    manifest = {
        "artifact_schema": "claim-search-normalized-v1",
        "arm_id": arm_id,
        "source": {
            "legacy_result_path": str(legacy_result),
            "legacy_result_sha256": _legacy_result_hash(
                frozen_manifest,
                rounds,
                candidates_per_round,
            ),
            "run_provenance_sha256": _file_sha256(
                legacy_dir / "run_provenance.json"
            ),
            "claim_search_source_sha256": str(
                arm_row["source_sha256"]
            ),
            "checkpoint_count": len(checkpoints),
        },
        "counts": {
            "parents": len(parents),
            "llm_calls": len(calls),
            "retained_candidates": len(retained_candidates),
            "unretained_candidates": len(unretained_candidates),
            "duplicate_candidates": len(duplicate_candidates),
            "evaluations": len(evaluations),
        },
        "reconciliation": {
            "status": "exact",
            "counts": observed,
        },
        "files": {},
    }
    for name, path in files.items():
        manifest["files"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    manifest_path = out_dir / "manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        default="review-stage/claim-search-gpt55-sweep-v7",
    )
    parser.add_argument(
        "--frozen-manifest",
        default="benchmark/FROZEN_RUNS.json",
    )
    args = parser.parse_args()
    sweep_root = Path(args.sweep)
    matrix = json.loads(
        (sweep_root / "matrix_summary.json").read_text(encoding="utf-8")
    )
    frozen = json.loads(
        Path(args.frozen_manifest).read_text(encoding="utf-8")
    )
    manifests = [
        convert_arm(
            arm_row=row,
            sweep_root=sweep_root,
            frozen_manifest=frozen,
        )
        for row in matrix["rows"]
    ]
    aggregate = {
        "artifact_schema": "claim-search-normalized-v1",
        "arm_count": len(manifests),
        "arms": [
            {
                "arm_id": item["arm_id"],
                "counts": item["counts"],
                "manifest_sha256": _file_sha256(
                    sweep_root
                    / "normalized/arms"
                    / item["arm_id"]
                    / "manifest.json"
                ),
            }
            for item in manifests
        ],
    }
    _atomic_text(
        sweep_root / "normalized/manifest.json",
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
    )


if __name__ == "__main__":
    main()
