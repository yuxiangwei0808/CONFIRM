"""Compact, result-preserving claim-search artifact models and readers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from confirm.claim_search import LLMCandidateGenerationResponse
from confirm.contract import ClaimContract

# This alias intentionally preserves the exact v7 wire schema and title.
RawCandidateResponseV7 = LLMCandidateGenerationResponse


def read_v7_result_header(
    path: str | Path,
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    """Read the bounded metadata prefix of a v7 monolithic result."""

    source = Path(path)
    data = bytearray()
    boundary = b'\n  "rows":'
    with source.open("rb") as handle:
        while len(data) < max_bytes:
            chunk = handle.read(min(1024 * 1024, max_bytes - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            marker = data.find(boundary)
            if marker >= 0:
                data = data[:marker]
                break
    text = data.decode("utf-8")
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {}
    for key in (
        "status",
        "llm_model",
        "config",
        "provenance",
        "searchable_claim_count",
        "completed_search_count",
        "skipped_search_count",
        "summary",
    ):
        match = re.search(
            rf'(?m)^  {re.escape(json.dumps(key))}:\s*',
            text,
        )
        if match:
            try:
                result[key], _ = decoder.raw_decode(text[match.end() :])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Incomplete result header field {key!r}: {source}"
                ) from exc
    missing = {
        "status",
        "llm_model",
        "config",
        "provenance",
        "summary",
    } - result.keys()
    if missing:
        raise ValueError(
            f"Result header is missing {sorted(missing)}: {source}"
        )
    return result


class CandidateProposal(BaseModel):
    """Compact authoritative representation of one retained proposal."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    parent_claim_id: str
    round_index: int = Field(ge=1)
    proposal_type: str
    proposed_question: str
    proposed_contract: ClaimContract
    rationale: str
    declared_transform: Optional[str] = None
    inferred_transform: Optional[str] = None
    transform_match: Optional[bool] = None
    responds_to_candidate_ids: list[str] = Field(default_factory=list)
    executable_contract_delta: dict[str, Any] = Field(default_factory=dict)
    policy_adjustments: dict[str, Any] = Field(default_factory=dict)
    legacy_self_report: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_v7(cls, payload: dict[str, Any]) -> "CandidateProposal":
        authoritative = {
            "candidate_id",
            "parent_claim_id",
            "round_index",
            "proposal_type",
            "proposed_question",
            "proposed_contract",
            "rationale",
            "declared_transform",
            "inferred_transform",
            "transform_match",
            "responds_to_candidate_ids",
            "executable_contract_delta",
            "policy_adjustments",
        }
        values = {
            key: payload[key]
            for key in authoritative
            if key in payload
        }
        values["declared_transform"] = (
            payload.get("declared_transform")
            or payload.get("transform_type")
        )
        values["legacy_self_report"] = {
            key: value
            for key, value in payload.items()
            if key not in authoritative
        }
        return cls.model_validate(values)

    def to_v7(self) -> dict[str, Any]:
        """Reconstruct the frozen v7 proposal shape for legacy analyses."""

        payload = dict(self.legacy_self_report)
        payload.update(
            {
                "candidate_id": self.candidate_id,
                "parent_claim_id": self.parent_claim_id,
                "round_index": self.round_index,
                "proposal_type": self.proposal_type,
                "proposed_question": self.proposed_question,
                "proposed_contract": self.proposed_contract.model_dump(
                    mode="json"
                ),
                "rationale": self.rationale,
                "declared_transform": self.declared_transform,
                "inferred_transform": self.inferred_transform,
                "transform_match": self.transform_match,
                "responds_to_candidate_ids": (
                    self.responds_to_candidate_ids
                ),
                "executable_contract_delta": (
                    self.executable_contract_delta
                ),
                "policy_adjustments": self.policy_adjustments,
            }
        )
        return payload


class NormalizedParent(BaseModel):
    """One parent lineage without repeated candidates or evaluations."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    claim_id: str
    row: dict[str, Any]
    original_claim: dict[str, Any]
    source_metadata: dict[str, Any]
    failure_localization: Optional[dict[str, Any]] = None
    lineage_graph: dict[str, Any]
    used_evidence: list[str]
    internally_supported_candidate_ids: list[str]
    provisional_supported_candidate_ids: list[str]
    round_failure_contexts: list[dict[str, Any]]
    round_summaries: list[dict[str, Any]]
    final_search_family_size: int
    stopped_reason: str


class NormalizedLLMCall(BaseModel):
    """One prompt/response pair, including retries."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    parent_claim_id: str
    round_index: int
    attempt_index: int
    schema_attempt_index: int
    validation_retry_index: int
    is_retry: bool
    retry_kind: str
    model: str
    system: str
    user: str
    prompt_hash: str
    raw_response: str
    response_candidate_count: int
    parse_error: Optional[str] = None


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_no} is not a JSON object"
                )
            yield value


def normalized_arm_dir(sweep_root: str | Path, arm_id: str) -> Path:
    return Path(sweep_root) / "normalized" / "arms" / arm_id


def read_normalized_parents(
    sweep_root: str | Path,
    arm_id: str,
) -> Iterable[NormalizedParent]:
    path = normalized_arm_dir(sweep_root, arm_id) / "parents.jsonl"
    return (
        NormalizedParent.model_validate(row)
        for row in iter_jsonl(path)
    )


def read_legacy_states_from_normalized_arm(
    arm_dir: str | Path,
) -> list[dict[str, Any]]:
    """Reconstruct v7 state dictionaries without nested on-disk copies."""

    root = Path(arm_dir)
    parents = [
        NormalizedParent.model_validate(row)
        for row in iter_jsonl(root / "parents.jsonl")
    ]
    retained_by_parent: dict[str, list[dict[str, Any]]] = {}
    retained_by_id: dict[str, dict[str, Any]] = {}
    unretained_by_parent: dict[str, list[dict[str, Any]]] = {}
    duplicates_by_parent: dict[str, list[dict[str, Any]]] = {}
    for record in iter_jsonl(root / "candidates.jsonl"):
        record_type = record["record_type"]
        if record_type == "retained":
            proposal = CandidateProposal.model_validate(
                record["proposal"]
            ).to_v7()
            parent_id = str(proposal["parent_claim_id"])
            retained_by_parent.setdefault(parent_id, []).append(proposal)
            retained_by_id[str(proposal["candidate_id"])] = proposal
        elif record_type == "unretained":
            parent_id = str(record["parent_claim_id"])
            payload = dict(record)
            payload.pop("record_type", None)
            payload.pop("parent_claim_id", None)
            unretained_by_parent.setdefault(parent_id, []).append(payload)
        elif record_type == "duplicate":
            parent_id = str(record["parent_claim_id"])
            payload = dict(record)
            payload.pop("record_type", None)
            duplicates_by_parent.setdefault(parent_id, []).append(payload)
        else:
            raise ValueError(f"Unknown candidate record type: {record_type}")

    evaluations_by_parent: dict[str, list[dict[str, Any]]] = {}
    for evaluation in iter_jsonl(root / "evaluations.jsonl"):
        candidate_id = str(evaluation["candidate_id"])
        proposal = retained_by_id.get(candidate_id)
        if proposal is None:
            raise ValueError(
                f"Evaluation references unknown candidate: {candidate_id}"
            )
        record = dict(evaluation)
        record["proposal"] = proposal
        parent_id = str(proposal["parent_claim_id"])
        evaluations_by_parent.setdefault(parent_id, []).append(record)

    prompts_by_parent: dict[str, list[dict[str, Any]]] = {}
    responses_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(root / "llm_calls.jsonl"):
        call = NormalizedLLMCall.model_validate(row)
        common = {
            "round_index": call.round_index,
            "parent_claim_id": call.parent_claim_id,
            "attempt_index": call.attempt_index,
            "schema_attempt_index": call.schema_attempt_index,
            "validation_retry_index": call.validation_retry_index,
            "is_retry": call.is_retry,
            "retry_kind": call.retry_kind,
            "model": call.model,
        }
        prompts_by_parent.setdefault(
            call.parent_claim_id,
            [],
        ).append(
            {
                **common,
                "system": call.system,
                "user": call.user,
                "prompt_hash": call.prompt_hash,
            }
        )
        responses_by_parent.setdefault(
            call.parent_claim_id,
            [],
        ).append(
            {
                **common,
                "raw_response": call.raw_response,
                "candidate_count": call.response_candidate_count,
                "parse_error": call.parse_error,
            }
        )

    states: list[dict[str, Any]] = []
    for parent in parents:
        row = parent.row
        parent_id = parent.claim_id
        states.append(
            {
                "original_claim": parent.original_claim,
                "source_metadata": parent.source_metadata,
                "failure_localization": parent.failure_localization,
                "lineage_graph": parent.lineage_graph,
                "used_evidence": parent.used_evidence,
                "candidate_history": retained_by_parent.get(
                    parent_id, []
                ),
                "unretained_candidate_attempts": (
                    unretained_by_parent.get(parent_id, [])
                ),
                "duplicate_candidates": duplicates_by_parent.get(
                    parent_id, []
                ),
                "evaluations": evaluations_by_parent.get(parent_id, []),
                "internally_supported_candidate_ids": (
                    parent.internally_supported_candidate_ids
                ),
                "round_failure_contexts": parent.round_failure_contexts,
                "round_summaries": parent.round_summaries,
                "provisional_supported_candidate_ids": (
                    parent.provisional_supported_candidate_ids
                ),
                "final_search_family_size": (
                    parent.final_search_family_size
                ),
                "generated_candidate_count": int(
                    row.get("generated_candidate_count") or 0
                ),
                "schema_valid_candidate_count": int(
                    row.get("schema_valid_candidate_count") or 0
                ),
                "unique_candidate_count": int(
                    row.get("unique_candidate_count") or 0
                ),
                "valid_candidate_count": int(
                    row.get("valid_candidate_count") or 0
                ),
                "current_data_evaluated_count": int(
                    row.get("current_data_evaluated_count") or 0
                ),
                "unique_hypotheses_tested_count": int(
                    row.get("unique_hypotheses_tested_count") or 0
                ),
                "llm_candidate_prompts": prompts_by_parent.get(
                    parent_id, []
                ),
                "llm_candidate_responses": responses_by_parent.get(
                    parent_id, []
                ),
                "stopped_reason": parent.stopped_reason,
            }
        )
    return states
