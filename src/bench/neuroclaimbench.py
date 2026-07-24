"""NeuroClaimBench schemas, identity rules, and scoring."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confirm.contract import ClaimContract

BenchmarkTrack = Literal["scientific", "synthetic_stress", "external_transfer"]
LabelClass = Literal[
    "known_positive",
    "known_null",
    "fragile",
    "underpowered_small_positive",
    "candidate_unknown",
]
ReferenceDisposition = Literal["confirm", "abstain", "unresolved"]
ReferenceBasis = Literal["literature", "constructed_control"]
Confidence = Literal["low", "medium", "high"]
ConstructMatch = Literal["exact", "partial", "mismatch"]
TriageReferenceLabel = Literal[
    "supported",
    "known_null",
    "fragile_or_mixed",
    "insufficient_evidence",
]
TriageDisposition = Literal["confirm", "abstain", "request_evidence"]
ReferenceStrength = Literal["strict", "provisional", "evidence_gap", "constructed"]
AgreementPattern = Literal[
    "not_applicable",
    "all_three_models",
    "adjudicator_plus_assessor",
    "assessors_only",
    "other",
]
BenchmarkSplit = Literal["scientific", "external_transfer", "synthetic_safety"]
ConfirmOutcome = Literal["confirmed", "abstained", "error", "not_evaluated"]
AlignmentDisposition = Literal[
    "aligned",
    "aligned_with_safety_augmentation",
    "repairable_contract",
    "non_executable",
    "ambiguous_unresolved",
]
AlignmentStatus = Literal["preserved", "safety_augmentation", "mismatch", "not_assessable"]


class FieldAlignment(BaseModel):
    """Outcome-blind comparison of one scientific field."""

    model_config = ConfigDict(extra="forbid")

    field: str
    status: AlignmentStatus
    question_value: Any = None
    contract_value: Any = None
    reason: str


class DeterministicContractRepair(BaseModel):
    """One policy-authorized repair derived without outcome access."""

    model_config = ConfigDict(extra="forbid")

    field_path: str
    old_value: Any = None
    new_value: Any = None
    repair_type: Literal[
        "restore_question_contrast",
        "restore_question_outcome",
        "safety_covariate_augmentation",
    ]
    rationale: str


class GeminiAlignmentAssessment(BaseModel):
    """Advisory semantic alignment assessment with no repair authority."""

    model_config = ConfigDict(extra="forbid")

    aligned: bool
    field_assessments: list[FieldAlignment] = Field(default_factory=list)
    recommended_disposition: AlignmentDisposition
    rationale: str
    model_spec: str = ""
    prompt_sha256: str = ""
    response_sha256: str = ""
    schema_attempts: int = Field(default=1, ge=1)
    call_metadata: dict[str, Any] = Field(default_factory=dict)


class QuestionContractAlignment(BaseModel):
    """Frozen outcome-blind alignment and repair decision for one item."""

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: str
    canonical_question: str
    canonical_question_sha256: str
    executable_contract_sha256: Optional[str] = None
    repaired_contract_sha256: Optional[str] = None
    field_alignments: list[FieldAlignment] = Field(default_factory=list)
    safety_augmentations: list[DeterministicContractRepair] = Field(default_factory=list)
    mismatch_disposition: AlignmentDisposition
    deterministic_repairs: list[DeterministicContractRepair] = Field(default_factory=list)
    repaired_contract: Optional[ClaimContract] = None
    deterministic_preflight: dict[str, Any] = Field(default_factory=dict)
    gemini_assessment: Optional[GeminiAlignmentAssessment] = None
    final_outcome_blind_resolution: AlignmentDisposition
    resolution_reason: str
    policy_version: str


class SourceReference(BaseModel):
    """One source record represented by a canonical benchmark item."""

    model_config = ConfigDict(extra="forbid")

    source_collection: str
    source_id: str
    source_path: str
    source_mode: str = ""
    target_family: str = ""
    prior_label: str = ""
    source_citation: str = ""


class EvaluationTask(BaseModel):
    """One frozen contract/evidence execution task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    benchmark_item_id: str
    contract: ClaimContract
    evidence_role: Literal["source", "holdout", "external", "synthetic_control"]
    dataset_id: str
    discovery_cohort: str
    replication_cohorts: list[str]
    partition_paths: dict[str, str] = Field(default_factory=dict)
    partition_hashes: dict[str, str] = Field(default_factory=dict)
    executable_contract_sha256: str = ""
    scientific_question_sha256: str = ""
    scientific_core_sha256: str = ""
    evidence_freshness: Literal["fresh", "previously_queried", "unknown"] = "unknown"
    generator_spec: Optional[dict[str, Any]] = None
    generator_artifact_hashes: dict[str, str] = Field(default_factory=dict)
    code_sha: str = ""
    schema_version: str = "2.0.0"
    gate_policy_version: str = ""

    @model_validator(mode="after")
    def populate_identity_hashes(self) -> "EvaluationTask":
        expected_contract = exact_contract_hash(self.contract)
        expected_question = scientific_question_hash(self.contract.question)
        expected_core = scientific_core_hash(self.contract)
        if self.executable_contract_sha256 and self.executable_contract_sha256 != expected_contract:
            raise ValueError("Evaluation task executable contract hash is stale")
        if self.scientific_question_sha256 and self.scientific_question_sha256 != expected_question:
            raise ValueError("Evaluation task scientific question hash is stale")
        if self.scientific_core_sha256 and self.scientific_core_sha256 != expected_core:
            raise ValueError("Evaluation task scientific core hash is stale")
        self.executable_contract_sha256 = expected_contract
        self.scientific_question_sha256 = expected_question
        self.scientific_core_sha256 = expected_core
        return self


class EvidenceStudyAssessment(BaseModel):
    """A model's structured assessment of one frozen literature record."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    study_design: Literal["meta_analysis", "multi_cohort", "single_cohort", "review", "other"]
    directness: Literal["direct", "partial", "unrelated"]
    relation: Literal[
        "supports_positive",
        "supports_null",
        "heterogeneous",
        "nonreplicated",
        "design_sensitive",
        "underpowered_small_effect",
        "uninformative",
    ]
    population_match: ConstructMatch
    modality_match: ConstructMatch
    outcome_match: ConstructMatch
    direction_match: ConstructMatch
    independent_group: str = ""
    supporting_text: str = ""


class EvidenceRecord(BaseModel):
    """One frozen PubMed record and its model-specific extractions."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    benchmark_item_id: str
    scientific_question_sha256: str = ""
    pmid: str
    doi: str = ""
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    query: str
    target_family: str
    retrieved_at: str
    model_assessments: dict[str, EvidenceStudyAssessment] = Field(default_factory=dict)


class LabelVote(BaseModel):
    """One independent model vote over a frozen evidence packet."""

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: str
    scientific_question_sha256: str = ""
    model_spec: str
    role: Literal["evidence_assessor", "independent_adjudicator"]
    proposed_label: LabelClass
    construct_match: ConstructMatch
    confidence: Confidence
    evidence_ids: list[str] = Field(default_factory=list, max_length=3)
    paper_assessments: list[EvidenceStudyAssessment] = Field(default_factory=list, max_length=3)
    rationale: str
    prompt_sha256: str
    response_sha256: str
    call_metadata: dict[str, Any] = Field(default_factory=dict)
    schema_attempts: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_evidence_links(self) -> "LabelVote":
        assessed = {row.evidence_id for row in self.paper_assessments}
        cited = set(self.evidence_ids)
        if cited != assessed:
            raise ValueError(
                "Vote evidence_ids and paper_assessments must match exactly: "
                f"missing_assessments={sorted(cited - assessed)}, "
                f"uncited_assessments={sorted(assessed - cited)}"
            )
        return self


class AdjudicationRecord(BaseModel):
    """Deterministic consensus result over independent model votes."""

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: str
    scientific_question_sha256: str = ""
    vote_models: list[str]
    final_label: LabelClass
    reference_disposition: ReferenceDisposition
    adjudication_status: Literal[
        "multi_model_consensus",
        "construction_derived",
        "unresolved",
    ]
    consensus_rule: str
    score_eligible: bool
    unresolved_reason: Optional[str] = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class BenchmarkItem(BaseModel):
    """Canonical public unit in the unified NeuroClaimBench package."""

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: str
    claim_uid: str
    semantic_cluster_id: str
    benchmark_track: BenchmarkTrack
    target_family: str
    modality: str
    question: str
    question_aliases: list[str] = Field(default_factory=list)
    scientific_question_sha256: str = ""
    scientific_core_sha256: str = ""
    contract: Optional[ClaimContract] = None
    exact_contract_sha256: Optional[str] = None
    semantic_claim_sha256: str
    source_references: list[SourceReference]
    aliases: list[str] = Field(default_factory=list)
    evaluation_task_ids: list[str] = Field(default_factory=list)
    migration_status: Literal[
        "ready",
        "pending_contract",
        "superseded_alias",
        "non_executable",
        "ambiguous_unresolved",
    ] = "ready"
    alignment_disposition: Optional[AlignmentDisposition] = None
    alignment_policy_version: str = ""
    pre_v2_contract_sha256: Optional[str] = None
    label_class: LabelClass = "candidate_unknown"
    reference_disposition: ReferenceDisposition = "unresolved"
    adjudication_status: Literal[
        "pending",
        "multi_model_consensus",
        "construction_derived",
        "unresolved",
    ] = "pending"
    score_eligible: bool = False

    @model_validator(mode="after")
    def validate_contract_state(self) -> "BenchmarkItem":
        expected_question = scientific_question_hash(self.question)
        if self.scientific_question_sha256 and self.scientific_question_sha256 != expected_question:
            raise ValueError("Benchmark item scientific question hash is stale")
        self.scientific_question_sha256 = expected_question
        if self.contract is not None:
            expected_contract = exact_contract_hash(self.contract)
            expected_core = scientific_core_hash(self.contract)
            if self.exact_contract_sha256 and self.exact_contract_sha256 != expected_contract:
                raise ValueError("Benchmark item executable contract hash is stale")
            if self.scientific_core_sha256 and self.scientific_core_sha256 != expected_core:
                raise ValueError("Benchmark item scientific core hash is stale")
            self.exact_contract_sha256 = expected_contract
            self.scientific_core_sha256 = expected_core
        if self.migration_status == "ready" and self.contract is None:
            raise ValueError("Ready benchmark items require a frozen contract")
        if self.contract is None and self.exact_contract_sha256 is not None:
            raise ValueError("Items without contracts cannot have an exact contract hash")
        if self.score_eligible and self.reference_disposition == "unresolved":
            raise ValueError("Unresolved items cannot be score eligible")
        return self


def adjudication_claim_payload(item: BenchmarkItem) -> dict[str, Any]:
    """Return the outcome-blind claim fields shown to literature assessors."""

    if item.contract is None:
        raise ValueError(
            "Cannot adjudicate item without a frozen contract: "
            f"{item.benchmark_item_id}"
        )
    contract = item.contract
    return {
        "benchmark_item_id": item.benchmark_item_id,
        "target_family": item.target_family,
        "modality": item.modality,
        "question": item.question,
        "scientific_question_sha256": item.scientific_question_sha256,
        "estimand": contract.estimand.model_dump(mode="json"),
        "covariates": contract.covariates,
        "inclusion": contract.inclusion,
    }


class TriageReferenceProfile(BaseModel):
    """Derived full-corpus reference used for evidence triage analyses."""

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: str
    benchmark_track: BenchmarkTrack
    target_family: str
    source_label: LabelClass
    source_adjudication_status: str
    scientific_question_sha256: str = ""
    triage_label: TriageReferenceLabel
    triage_disposition: TriageDisposition
    reference_basis: ReferenceBasis = "literature"
    reference_strength: ReferenceStrength
    derivation_rule: str
    agreeing_models: list[str] = Field(default_factory=list)
    agreement_pattern: AgreementPattern = "not_applicable"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    vote_counts: dict[str, int] = Field(default_factory=dict)
    score_tracks: list[
        Literal["strict_confirmation", "provisional_confirmation", "evidence_triage", "gate_evaluation"]
    ] = Field(default_factory=list)
    executable: bool


class SimplifiedBenchmarkClaim(BaseModel):
    """Paper-facing benchmark unit with one reference label and one split."""

    model_config = ConfigDict(extra="forbid")

    benchmark_claim_id: str
    benchmark_item_ids: list[str] = Field(min_length=1)
    benchmark_split: BenchmarkSplit
    target_family: str
    modality: str
    question: str
    scientific_question_sha256: str = ""
    semantic_cluster_id: str = ""
    contract: ClaimContract
    contract_sha256: str
    execution_identity_sha256: str
    reference_label: ReferenceDisposition
    score_eligible: bool
    reference_strength: ReferenceStrength
    reference_basis: str
    evaluation_task_ids: list[str] = Field(min_length=1)
    dataset_ids: list[str] = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_score_eligibility(self) -> "SimplifiedBenchmarkClaim":
        expected_question = scientific_question_hash(self.question)
        if self.scientific_question_sha256 and self.scientific_question_sha256 != expected_question:
            raise ValueError("Simplified claim scientific question hash is stale")
        if self.contract_sha256 != exact_contract_hash(self.contract):
            raise ValueError("Simplified claim executable contract hash is stale")
        self.scientific_question_sha256 = expected_question
        if self.score_eligible != (self.reference_label != "unresolved"):
            raise ValueError("score_eligible must be equivalent to a resolved reference label")
        return self


class BenchmarkTaskOutcome(BaseModel):
    """Normalized CONFIRM outcome for one benchmark evaluation task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    benchmark_claim_id: str
    benchmark_item_id: str
    status: Literal["completed", "error"]
    confirm_outcome: Literal["confirmed", "abstained", "error"]
    raw_final_label: str = ""
    result_source: str
    task_fingerprint: str = ""
    error: str = ""
    gate_verdict: dict[str, Any] = Field(default_factory=dict)
    gate_results: dict[str, Any] = Field(default_factory=dict)
    cohort_paths: list[str] = Field(default_factory=list)
    cohort_content_hashes: dict[str, str] = Field(default_factory=dict)
    generated_cohort_hashes: dict[str, str] = Field(default_factory=dict)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def scientific_question_hash(question: str) -> str:
    """Hash canonical scientific wording independently of executable identity."""

    normalized = " ".join(str(question).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def executable_contract_payload(contract: ClaimContract | dict[str, Any]) -> dict[str, Any]:
    payload = contract.model_dump(mode="json") if isinstance(contract, ClaimContract) else dict(contract)
    payload.pop("claim_id", None)
    payload.pop("question", None)
    return payload


def exact_contract_hash(contract: ClaimContract | dict[str, Any]) -> str:
    return sha256_payload(executable_contract_payload(contract))


def semantic_contract_payload(contract: ClaimContract | dict[str, Any]) -> dict[str, Any]:
    payload = executable_contract_payload(contract)
    return {
        "estimand": payload.get("estimand"),
        "covariates": sorted(payload.get("covariates") or []),
        "inclusion": payload.get("inclusion"),
    }


def semantic_contract_hash(contract: ClaimContract | dict[str, Any]) -> str:
    return sha256_payload(semantic_contract_payload(contract))


def scientific_core_payload(contract: ClaimContract | dict[str, Any]) -> dict[str, Any]:
    """Represent the immutable executable scientific core used for alias checks."""

    payload = executable_contract_payload(contract)
    estimand = dict(payload.get("estimand") or {})
    return {
        "type": estimand.get("type"),
        "predictor": estimand.get("predictor"),
        "group": estimand.get("group"),
        "outcome": estimand.get("outcome"),
        "unit": estimand.get("unit"),
        "region_set": estimand.get("region_set"),
        "direction": estimand.get("direction"),
        "discovery_cohort": payload.get("discovery_cohort"),
        "replication_cohorts": payload.get("replication_cohorts"),
    }


def scientific_core_hash(contract: ClaimContract | dict[str, Any]) -> str:
    return sha256_payload(scientific_core_payload(contract))


def unresolved_semantic_hash(*, question: str, target_family: str, source_id: str) -> str:
    return sha256_payload({"question": question.strip(), "target_family": target_family, "source_id": source_id})


def disposition_for_label(label: str) -> ReferenceDisposition:
    if label == "known_positive":
        return "confirm"
    if label in {"known_null", "fragile", "underpowered_small_positive"}:
        return "abstain"
    return "unresolved"


def _vote_evidence_sufficient(vote: LabelVote, label: LabelClass) -> tuple[bool, list[str]]:
    cited = [row for row in vote.paper_assessments if row.evidence_id in set(vote.evidence_ids)]
    direct = [
        row
        for row in cited
        if row.directness == "direct"
        and row.population_match == "exact"
        and row.modality_match == "exact"
        and row.outcome_match == "exact"
    ]
    if label == "known_positive":
        matching = [row for row in direct if row.relation == "supports_positive" and row.direction_match == "exact"]
    elif label == "known_null":
        matching = [row for row in direct if row.relation == "supports_null"]
    elif label == "fragile":
        matching = [row for row in direct if row.relation in {"heterogeneous", "nonreplicated", "design_sensitive"}]
        return bool(matching), sorted({row.evidence_id for row in matching})
    elif label == "underpowered_small_positive":
        matching = [row for row in direct if row.relation == "underpowered_small_effect"]
        return bool(matching), sorted({row.evidence_id for row in matching})
    else:
        return False, []

    strong = [row for row in matching if row.study_design in {"meta_analysis", "multi_cohort"}]
    independent_groups = {row.independent_group or row.evidence_id for row in matching}
    return bool(strong or len(independent_groups) >= 2), sorted({row.evidence_id for row in matching})


def _direct_vote_evidence(vote: LabelVote, label: LabelClass) -> list[str]:
    """Return directly matched evidence without imposing replication sufficiency."""

    relation_by_label = {
        "known_positive": {"supports_positive"},
        "known_null": {"supports_null"},
        "fragile": {"heterogeneous", "nonreplicated", "design_sensitive"},
        "underpowered_small_positive": {"underpowered_small_effect"},
    }
    allowed_relations = relation_by_label.get(label)
    if not allowed_relations:
        return []
    evidence_ids = set(vote.evidence_ids)
    matching = [
        row
        for row in vote.paper_assessments
        if row.evidence_id in evidence_ids
        and row.directness == "direct"
        and row.population_match == "exact"
        and row.modality_match == "exact"
        and row.outcome_match == "exact"
        and row.relation in allowed_relations
        and (label != "known_positive" or row.direction_match == "exact")
    ]
    return sorted({row.evidence_id for row in matching})


def _agreement_pattern(votes: list[LabelVote]) -> AgreementPattern:
    models = {vote.model_spec for vote in votes}
    roles = {vote.role for vote in votes}
    if len(models) >= 3 and roles == {"evidence_assessor", "independent_adjudicator"}:
        return "all_three_models"
    if roles == {"evidence_assessor", "independent_adjudicator"}:
        return "adjudicator_plus_assessor"
    if len(models) >= 2 and roles == {"evidence_assessor"}:
        return "assessors_only"
    return "other" if models else "not_applicable"


def _direct_conflicting_evidence(votes: list[LabelVote], label: LabelClass) -> list[str]:
    conflicting_relations = {
        "known_positive": {
            "supports_null",
            "heterogeneous",
            "nonreplicated",
            "design_sensitive",
            "underpowered_small_effect",
        },
        "known_null": {
            "supports_positive",
            "heterogeneous",
            "nonreplicated",
            "design_sensitive",
            "underpowered_small_effect",
        },
    }.get(label, set())
    if not conflicting_relations:
        return []

    evidence_ids: set[str] = set()
    for vote in votes:
        cited = set(vote.evidence_ids)
        for row in vote.paper_assessments:
            if (
                row.evidence_id in cited
                and row.directness == "direct"
                and row.population_match == "exact"
                and row.modality_match == "exact"
                and row.outcome_match == "exact"
                and row.relation in conflicting_relations
                and (row.relation != "supports_positive" or row.direction_match == "exact")
            ):
                evidence_ids.add(row.evidence_id)
    return sorted(evidence_ids)


def derive_triage_reference(
    item: BenchmarkItem,
    votes: list[LabelVote],
) -> TriageReferenceProfile:
    """Derive strict, provisional, or evidence-gap triage references."""

    vote_counts = Counter(vote.proposed_label for vote in votes)
    alignment_resolved = item.alignment_disposition in {
        None,
        "aligned",
        "aligned_with_safety_augmentation",
        "repairable_contract",
    }
    reference_ready = item.migration_status == "ready" and alignment_resolved
    executable = item.contract is not None and reference_ready
    base = {
        "benchmark_item_id": item.benchmark_item_id,
        "benchmark_track": item.benchmark_track,
        "target_family": item.target_family,
        "source_label": item.label_class,
        "source_adjudication_status": item.adjudication_status,
        "scientific_question_sha256": item.scientific_question_sha256,
        "vote_counts": dict(sorted(vote_counts.items())),
        "executable": executable,
    }
    score_tracks = ["evidence_triage"]
    if executable:
        score_tracks.append("gate_evaluation")

    if not reference_ready:
        if (
            item.migration_status == "non_executable"
            or item.alignment_disposition == "non_executable"
        ):
            derivation_rule = "alignment_non_executable"
        elif (
            item.migration_status == "ambiguous_unresolved"
            or item.alignment_disposition == "ambiguous_unresolved"
        ):
            derivation_rule = "question_contract_alignment_unresolved"
        else:
            derivation_rule = "item_not_ready_for_reference_scoring"
        return TriageReferenceProfile(
            **base,
            triage_label="insufficient_evidence",
            triage_disposition="request_evidence",
            reference_basis=(
                "constructed_control"
                if item.adjudication_status == "construction_derived"
                else "literature"
            ),
            reference_strength="evidence_gap",
            derivation_rule=derivation_rule,
            score_tracks=score_tracks,
        )

    strict_mapping: dict[LabelClass, tuple[TriageReferenceLabel, TriageDisposition]] = {
        "known_positive": ("supported", "confirm"),
        "known_null": ("known_null", "abstain"),
        "fragile": ("fragile_or_mixed", "abstain"),
        "underpowered_small_positive": ("fragile_or_mixed", "abstain"),
        "candidate_unknown": ("insufficient_evidence", "request_evidence"),
    }
    if item.score_eligible:
        triage_label, disposition = strict_mapping[item.label_class]
        agreeing = [vote for vote in votes if vote.proposed_label == item.label_class]
        if item.adjudication_status == "construction_derived":
            return TriageReferenceProfile(
                **base,
                triage_label=triage_label,
                triage_disposition=disposition,
                reference_basis="constructed_control",
                reference_strength="constructed",
                derivation_rule="deterministic_control_construction",
                supporting_evidence_ids=[],
                agreeing_models=[],
                agreement_pattern="not_applicable",
                score_tracks=score_tracks,
            )
        return TriageReferenceProfile(
            **base,
            triage_label=triage_label,
            triage_disposition=disposition,
            reference_basis="literature",
            reference_strength="strict",
            derivation_rule="existing_score_eligible_reference",
            supporting_evidence_ids=sorted(
                {
                    evidence_id
                    for vote in votes
                    if vote.proposed_label == item.label_class
                    for evidence_id in _direct_vote_evidence(vote, item.label_class)
                }
            ),
            agreeing_models=sorted(vote.model_spec for vote in agreeing),
            agreement_pattern=_agreement_pattern(agreeing),
            score_tracks=score_tracks + ["strict_confirmation"],
        )

    provisional_mapping: dict[LabelClass, tuple[TriageReferenceLabel, TriageDisposition]] = {
        "known_positive": ("supported", "confirm"),
        "known_null": ("known_null", "abstain"),
        "fragile": ("fragile_or_mixed", "abstain"),
        "underpowered_small_positive": ("fragile_or_mixed", "abstain"),
    }
    for label in (
        "known_positive",
        "known_null",
        "fragile",
        "underpowered_small_positive",
    ):
        agreeing = [
            vote
            for vote in votes
            if vote.proposed_label == label
            and vote.construct_match == "exact"
            and vote.confidence in {"medium", "high"}
        ]
        evidence_ids = sorted(
            {
                evidence_id
                for vote in agreeing
                for evidence_id in _direct_vote_evidence(vote, label)
            }
        )
        if len(agreeing) >= 2 and evidence_ids:
            triage_label, disposition = provisional_mapping[label]
            conflicting_evidence_ids = _direct_conflicting_evidence(votes, label)
            derivation_rule = "two_model_exact_construct_with_direct_evidence"
            if conflicting_evidence_ids:
                triage_label = "fragile_or_mixed"
                disposition = "abstain"
                derivation_rule = "two_model_reference_with_direct_conflicting_evidence"
            return TriageReferenceProfile(
                **base,
                triage_label=triage_label,
                triage_disposition=disposition,
                reference_basis="literature",
                reference_strength="provisional",
                derivation_rule=derivation_rule,
                agreeing_models=sorted(vote.model_spec for vote in agreeing),
                agreement_pattern=_agreement_pattern(agreeing),
                supporting_evidence_ids=sorted(
                    set(evidence_ids) | set(conflicting_evidence_ids)
                ),
                score_tracks=score_tracks + ["provisional_confirmation"],
            )

    return TriageReferenceProfile(
        **base,
        triage_label="insufficient_evidence",
        triage_disposition="request_evidence",
        reference_basis="literature",
        reference_strength="evidence_gap",
        derivation_rule=(
            "contract_unavailable"
            if item.contract is None
            else "no_two_model_direct_evidence_reference"
        ),
        score_tracks=score_tracks,
    )


def adjudicate_votes(
    benchmark_item_id: str,
    votes: list[LabelVote],
    *,
    adjudicator_model: str,
) -> AdjudicationRecord:
    """Require adjudicator concurrence plus deterministic evidence sufficiency."""

    by_model = {vote.model_spec: vote for vote in votes}
    adjudicator = by_model.get(adjudicator_model)
    if adjudicator is None:
        return _unresolved_adjudication(benchmark_item_id, votes, "missing_independent_adjudicator")
    matching_assessors = [
        vote
        for vote in votes
        if vote.role == "evidence_assessor"
        and vote.proposed_label == adjudicator.proposed_label
        and vote.construct_match == "exact"
        and vote.confidence in {"medium", "high"}
    ]
    if adjudicator.proposed_label == "candidate_unknown":
        return _unresolved_adjudication(benchmark_item_id, votes, "adjudicator_returned_candidate_unknown")
    if adjudicator.construct_match != "exact" or adjudicator.confidence not in {"medium", "high"}:
        return _unresolved_adjudication(benchmark_item_id, votes, "adjudicator_construct_or_confidence_failed")
    if not matching_assessors:
        return _unresolved_adjudication(benchmark_item_id, votes, "no_assessor_agrees_with_adjudicator")

    judge_ok, judge_evidence = _vote_evidence_sufficient(adjudicator, adjudicator.proposed_label)
    agreeing_evidence: set[str] = set()
    assessor_ok = False
    for vote in matching_assessors:
        ok, evidence_ids = _vote_evidence_sufficient(vote, adjudicator.proposed_label)
        assessor_ok = assessor_ok or ok
        agreeing_evidence.update(evidence_ids)
    if not judge_ok or not assessor_ok:
        return _unresolved_adjudication(benchmark_item_id, votes, "evidence_sufficiency_failed")

    label = adjudicator.proposed_label
    evidence_ids = sorted(set(judge_evidence) | agreeing_evidence)
    return AdjudicationRecord(
        benchmark_item_id=benchmark_item_id,
        vote_models=sorted(by_model),
        final_label=label,
        reference_disposition=disposition_for_label(label),
        adjudication_status="multi_model_consensus",
        consensus_rule="independent_adjudicator_agrees_with_assessor_and_evidence_thresholds_pass",
        score_eligible=True,
        supporting_evidence_ids=evidence_ids,
    )


def _unresolved_adjudication(
    benchmark_item_id: str,
    votes: list[LabelVote],
    reason: str,
) -> AdjudicationRecord:
    return AdjudicationRecord(
        benchmark_item_id=benchmark_item_id,
        vote_models=sorted({vote.model_spec for vote in votes}),
        final_label="candidate_unknown",
        reference_disposition="unresolved",
        adjudication_status="unresolved",
        consensus_rule="independent_adjudicator_concurrence_required",
        score_eligible=False,
        unresolved_reason=reason,
    )


def summarize_benchmark(
    items: list[BenchmarkItem],
    observed_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Summarize inventory and optional observed CONFIRM decisions by track."""

    observed = observed_labels or {}
    inventory = {
        "n_items": len(items),
        "track_counts": dict(Counter(item.benchmark_track for item in items)),
        "target_family_counts": dict(Counter(item.target_family for item in items)),
        "label_counts": dict(Counter(item.label_class for item in items)),
        "disposition_counts": dict(Counter(item.reference_disposition for item in items)),
        "migration_status_counts": dict(Counter(item.migration_status for item in items)),
        "score_eligible_count": sum(item.score_eligible for item in items),
    }
    by_track: dict[str, Any] = {}
    for track in sorted({item.benchmark_track for item in items}):
        track_items = [item for item in items if item.benchmark_track == track]
        by_track[track] = _decision_metrics(track_items, observed)
    return {"inventory": inventory, "metrics_by_track": by_track}


def _decision_metrics(items: list[BenchmarkItem], observed: dict[str, str]) -> dict[str, Any]:
    scored = [item for item in items if item.score_eligible and item.reference_disposition != "unresolved"]
    evaluated = [item for item in scored if item.benchmark_item_id in observed]
    confirmable = [item for item in evaluated if item.reference_disposition == "confirm"]
    abstain = [item for item in evaluated if item.reference_disposition == "abstain"]
    confirmed = lambda item: observed.get(item.benchmark_item_id) == "confirmed"
    recall_count = sum(confirmed(item) for item in confirmable)
    unsafe_count = sum(confirmed(item) for item in abstain)
    details: dict[str, dict[str, int]] = defaultdict(lambda: {"confirmed": 0, "denominator": 0})
    for item in evaluated:
        details[item.label_class]["denominator"] += 1
        details[item.label_class]["confirmed"] += int(confirmed(item))
    return {
        "n_items": len(items),
        "n_score_eligible": len(scored),
        "n_evaluated": len(evaluated),
        "adjudication_coverage": _rate(sum(item.score_eligible for item in items), len(items)),
        "confirmable_claim_recall": _rate(recall_count, len(confirmable)),
        "confirmable_claim_recall_count": recall_count,
        "confirmable_claim_recall_denominator": len(confirmable),
        "unsafe_confirmation_rate": _rate(unsafe_count, len(abstain)),
        "unsafe_confirmation_count": unsafe_count,
        "unsafe_confirmation_denominator": len(abstain),
        "label_breakdown": dict(details),
    }


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else math.nan
