"""Compact public NeuroClaimBench representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.io import read_jsonl
from confirm.contract import ClaimContract

ReferenceDisposition = Literal["confirm", "abstain", "unresolved"]
ReferenceBasis = Literal["literature", "constructed_control", "unresolved"]
ReferenceStrength = Literal[
    "strict",
    "provisional",
    "evidence_gap",
    "constructed",
]


class BenchmarkCase(BaseModel):
    """One canonical scientific or stress-test claim."""

    model_config = ConfigDict(extra="forbid")

    benchmark_case_id: str
    claim_uid: str
    semantic_cluster_id: str
    benchmark_track: str
    target_family: str
    modality: str
    question: str
    question_sha256: str
    contract: Optional[ClaimContract] = None
    contract_sha256: Optional[str] = None
    scientific_core_sha256: str
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    reference_id: str
    task_ids: list[str] = Field(default_factory=list)
    migration_status: str
    alignment_disposition: Optional[str] = None
    pre_v2_contract_sha256: Optional[str] = None


class BenchmarkReference(BaseModel):
    """One reference decision for a canonical benchmark case."""

    model_config = ConfigDict(extra="forbid")

    reference_id: str
    benchmark_case_id: str
    disposition: ReferenceDisposition
    basis: ReferenceBasis
    strength: ReferenceStrength
    score_eligible: bool
    evidence_ids: list[str] = Field(default_factory=list)
    derivation_rule: str
    source_label: str
    source_adjudication_status: str
    agreeing_models: list[str] = Field(default_factory=list)
    agreement_pattern: str = ""
    vote_counts: dict[str, int] = Field(default_factory=dict)


class BenchmarkEvaluationTask(BaseModel):
    """Exact evidence identity used to execute one case."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    benchmark_case_id: str
    dataset_id: str
    contract: ClaimContract
    contract_sha256: str
    discovery_cohort: str
    replication_cohorts: list[str]
    evidence_role: str
    evidence_freshness: str
    partition_paths: list[str]
    partition_hashes: dict[str, str]
    generator_spec: Optional[dict[str, Any]] = None
    generator_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    question_sha256: str
    scientific_core_sha256: str
    code_sha: str
    schema_version: str
    gate_policy_version: str


class TaskOutcome(BaseModel):
    """Compact verdict record; detailed gate bundles live in the audit archive."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    benchmark_case_id: str
    benchmark_claim_id: Optional[str] = None
    status: str
    confirm_outcome: Optional[str] = None
    raw_final_label: Optional[str] = None
    gate_verdict: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    task_fingerprint: str
    result_source: str
    cohort_paths: list[str] = Field(default_factory=list)
    cohort_content_hashes: dict[str, str] = Field(default_factory=dict)
    generated_cohort_hashes: dict[str, str] = Field(default_factory=dict)
    detailed_result_sha256: str


class BenchmarkDataset:
    """Load the compact release schema with no legacy triage vocabulary."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.cases = [
            BenchmarkCase.model_validate(row)
            for row in read_jsonl(self.root / "cases.jsonl")
        ]
        self.references = [
            BenchmarkReference.model_validate(row)
            for row in read_jsonl(self.root / "references.jsonl")
        ]
        self.tasks = [
            BenchmarkEvaluationTask.model_validate(row)
            for row in read_jsonl(self.root / "tasks.jsonl")
        ]
        self.outcomes = [
            TaskOutcome.model_validate(row)
            for row in read_jsonl(self.root / "outcomes.jsonl")
        ]
