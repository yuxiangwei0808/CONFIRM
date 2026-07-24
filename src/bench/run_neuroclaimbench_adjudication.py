"""Run phased multi-model literature adjudication for NeuroClaimBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.neuroclaimbench_v21_compat import (
    AdjudicationRecord,
    BenchmarkItem,
    EvidenceRecord,
    EvidenceStudyAssessment,
    LabelVote,
    adjudicate_votes,
    adjudication_claim_payload,
    sha256_payload,
)
from bench.progress import iter_progress
from bench.pubmed_cache import file_sha256, load_cache_packet
from confirm.llm import (
    LLMClient,
    complete_structured_with_retries,
    make_llm,
)

DEFAULT_PACKAGE = Path("data/neuroclaimbench/v2.1")
DEFAULT_OUT = Path("review-stage/neuroclaimbench-v2.1/adjudication")
DEFAULT_ASSESSORS = ("openai:gpt-5.5", "google:gemini-3.5-flash")
DEFAULT_ADJUDICATOR = "openrouter:anthropic/claude-opus-4.8"
PILOT_SEED = 20260723
ADJUDICATION_POLICY_VERSION = "neuroclaimbench-v2.1-item-consensus-1"
MAX_VOTE_CITATIONS = 3
MAX_RAW_VOTE_CITATIONS = 12


class RawVoteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed_label: Literal[
        "known_positive",
        "known_null",
        "fragile",
        "underpowered_small_positive",
        "candidate_unknown",
    ]
    construct_match: Literal["exact", "partial", "mismatch"]
    confidence: Literal["low", "medium", "high"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=MAX_RAW_VOTE_CITATIONS)
    paper_assessments: list[EvidenceStudyAssessment] = Field(default_factory=list, max_length=MAX_RAW_VOTE_CITATIONS)
    rationale: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            rows.append(row)
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    payloads = [row.model_dump(mode="json") if hasattr(row, "model_dump") else row for row in rows]
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in payloads))


def _load_items(package: Path) -> list[BenchmarkItem]:
    return [BenchmarkItem.model_validate(row) for row in _read_jsonl(package / "benchmark_items.jsonl")]


def _sanitized_claim(item: BenchmarkItem) -> dict[str, Any]:
    """Compatibility wrapper for frozen adjudication prompt construction."""

    return adjudication_claim_payload(item)


def build_vote_prompt(item: BenchmarkItem, records: list[EvidenceRecord], *, reversed_order: bool = False) -> str:
    ordered = list(reversed(records)) if reversed_order else list(records)
    evidence = [
        {
            "evidence_id": record.evidence_id,
            "pmid": record.pmid,
            "doi": record.doi,
            "title": record.title,
            "abstract": record.abstract,
            "journal": record.journal,
            "year": record.year,
        }
        for record in ordered
    ]
    schema_guidance = {
        "proposed_label": [
            "known_positive",
            "known_null",
            "fragile",
            "underpowered_small_positive",
            "candidate_unknown",
        ],
        "construct_match": ["exact", "partial", "mismatch"],
        "confidence": ["low", "medium", "high"],
        "study_design": ["meta_analysis", "multi_cohort", "single_cohort", "review", "other"],
        "directness": ["direct", "partial", "unrelated"],
        "relation": [
            "supports_positive",
            "supports_null",
            "heterogeneous",
            "nonreplicated",
            "design_sensitive",
            "underpowered_small_effect",
            "uninformative",
        ],
    }
    decision_policy = {
        "construct_matching": [
            "Treat dataset names and _DISC, _REP, _HOLDOUT, or similar split suffixes as execution provenance, not as literature constructs.",
            "Match the underlying clinical or demographic population, predictor or group contrast, imaging modality, outcome construct, and direction.",
            "Do not require a paper to use the exact same named dataset, but do require the same population family and scientific construct.",
        ],
        "label_hierarchy": [
            "known_positive: directly matched evidence supports the claimed direction, with either one matched meta-analysis or multi-cohort study, or two independent matched studies.",
            "known_null: the same evidence threshold directly supports a null result; absence of positive evidence is never enough.",
            "fragile: directly matched evidence shows heterogeneity, failed replication, or material design sensitivity; indirect or insufficient evidence alone is not fragility.",
            "underpowered_small_positive: directly matched evidence supports the claimed direction and specifically identifies a small effect with inadequate power; vague weak evidence is not enough.",
            "candidate_unknown: use when construct matching is partial or mismatched, direct evidence is insufficient, or none of the preceding definitions is satisfied.",
        ],
        "tie_breaking": [
            "Apply construct matching before assigning an evidence label.",
            "Use fragile rather than known_positive when directly matched evidence materially conflicts across replications or justified specifications.",
            "Use candidate_unknown rather than guessing between fragile, underpowered_small_positive, and known_null.",
        ],
    }
    return (
        "Independently adjudicate the claim from the frozen literature packet using the supplied decision policy. "
        "Cite only supplied evidence_id values. Cite no more than the three most informative records. Before "
        "returning, count evidence_ids and paper_assessments: each list must contain the same unique zero to three "
        "evidence IDs, with exactly one assessment per cited record and never a fourth assessment. supporting_text "
        "must be a short passage grounded in the supplied abstract. "
        "Do not infer anything about experimental gate results. Return structured JSON only.\n\n"
        + json.dumps(
            {
                "claim": _sanitized_claim(item),
                "decision_policy": decision_policy,
                "allowed_values": schema_guidance,
                "frozen_evidence": evidence,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _call_structured(
    llm: LLMClient,
    *,
    system: str,
    prompt: str,
    response_model: type[BaseModel],
    retries: int,
    validator: Callable[[BaseModel], None] | None = None,
) -> tuple[BaseModel, str, int, list[dict[str, Any]]]:
    """Compatibility wrapper for frozen adjudication retry semantics."""

    return complete_structured_with_retries(
        llm,
        system=system,
        prompt=prompt,
        response_model=response_model,
        retries=retries,
        validator=validator,
    )


def _vote(
    item: BenchmarkItem,
    records: list[EvidenceRecord],
    *,
    model_spec: str,
    role: str,
    retries: int,
    max_tokens: int = 8192,
    reversed_order: bool = False,
) -> tuple[LabelVote, dict[str, Any]]:
    llm = make_llm(model_spec)
    if hasattr(llm, "max_tokens"):
        llm.max_tokens = max_tokens
    prompt = build_vote_prompt(item, records, reversed_order=reversed_order)
    allowed_ids = {record.evidence_id for record in records}

    def validate_packet_links(parsed: BaseModel) -> None:
        response = RawVoteResponse.model_validate(parsed.model_dump(mode="json"))
        response = response.model_copy(
            update={
                "evidence_ids": [evidence_id.strip() for evidence_id in response.evidence_ids if evidence_id.strip()],
                "paper_assessments": [
                    row.model_copy(update={"evidence_id": row.evidence_id.strip()})
                    for row in response.paper_assessments
                    if row.evidence_id.strip()
                ],
            }
        )
        parsed.evidence_ids = response.evidence_ids
        parsed.paper_assessments = response.paper_assessments
        cited = set(response.evidence_ids)
        assessed = {assessment.evidence_id for assessment in response.paper_assessments}
        if len(cited) != len(response.evidence_ids) or len(assessed) != len(response.paper_assessments):
            raise ValueError("evidence_ids and paper_assessments must not contain duplicate evidence IDs")
        missing_assessments = cited - assessed
        uncited_assessments = assessed - cited
        if missing_assessments or (uncited_assessments and not cited):
            raise ValueError(
                "Every cited evidence ID requires one assessment, and assessments cannot replace an empty citation list: "
                f"missing_assessments={sorted(missing_assessments)}, "
                f"uncited_assessments={sorted(uncited_assessments)}"
            )
        unknown = (cited | assessed) - allowed_ids
        if unknown:
            raise ValueError(f"Model cited evidence IDs outside the frozen packet: {sorted(unknown)}")

    parsed, raw, attempts, trace = _call_structured(
        llm,
        system="You are a conservative, evidence-bound neuroimaging claim adjudicator.",
        prompt=prompt,
        response_model=RawVoteResponse,
        retries=retries,
        validator=validate_packet_links,
    )
    response = RawVoteResponse.model_validate(parsed.model_dump(mode="json"))
    raw_payload = json.loads(raw)
    raw_evidence_ids = list(raw_payload.get("evidence_ids") or [])
    raw_assessments = list(raw_payload.get("paper_assessments") or [])
    dropped_blank_evidence_id_count = sum(not str(value).strip() for value in raw_evidence_ids)
    dropped_blank_assessment_count = sum(
        not str(row.get("evidence_id") or "").strip() for row in raw_assessments if isinstance(row, dict)
    )
    original_evidence_ids = list(response.evidence_ids)
    retained_evidence_ids = original_evidence_ids[:MAX_VOTE_CITATIONS]
    assessments_by_id = {row.evidence_id: row for row in response.paper_assessments}
    uncited_assessment_ids = [
        row.evidence_id for row in response.paper_assessments if row.evidence_id not in set(original_evidence_ids)
    ]
    response = response.model_copy(
        update={
            "evidence_ids": retained_evidence_ids,
            "paper_assessments": [assessments_by_id[evidence_id] for evidence_id in retained_evidence_ids],
        }
    )
    normalization_reasons = []
    if len(original_evidence_ids) > MAX_VOTE_CITATIONS:
        normalization_reasons.append("citation_limit")
    if uncited_assessment_ids:
        normalization_reasons.append("uncited_assessments")
    if dropped_blank_evidence_id_count or dropped_blank_assessment_count:
        normalization_reasons.append("blank_evidence_placeholders")
    normalization = {
        "applied": bool(normalization_reasons),
        "reason": "+".join(normalization_reasons) if normalization_reasons else None,
        "original_evidence_ids": original_evidence_ids,
        "retained_evidence_ids": retained_evidence_ids,
        "dropped_uncited_assessment_ids": uncited_assessment_ids,
        "dropped_blank_evidence_id_count": dropped_blank_evidence_id_count,
        "dropped_blank_assessment_count": dropped_blank_assessment_count,
    }
    vote = LabelVote.model_validate(
        {
            "benchmark_item_id": item.benchmark_item_id,
            "scientific_question_sha256": item.scientific_question_sha256,
            "model_spec": model_spec,
            "role": role,
            "proposed_label": response.proposed_label,
            "construct_match": response.construct_match,
            "confidence": response.confidence,
            "evidence_ids": response.evidence_ids,
            "paper_assessments": [row.model_dump(mode="json") for row in response.paper_assessments],
            "rationale": response.rationale,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "call_metadata": dict(getattr(llm, "last_call_metadata", {}) or {}),
            "schema_attempts": attempts,
        }
    )
    return vote, {
        "prompt": prompt,
        "raw_response": raw,
        "trace": trace,
        "normalization": normalization,
        "reversed_order": reversed_order,
    }


def _checkpoint_path(out_dir: Path, item_id: str) -> Path:
    return out_dir / "checkpoints" / f"{item_id}.json"


def _evidence_freeze_path(out_dir: Path, item_id: str) -> Path:
    return out_dir / "evidence_freezes" / f"{item_id}.json"


def _reverse_order_checkpoint_path(out_dir: Path, item_id: str) -> Path:
    return out_dir / "pilot_reverse_order_checkpoints" / f"{item_id}.json"


def _run_fingerprint(item: BenchmarkItem, args: argparse.Namespace) -> str:
    payload = {
        "policy_version": ADJUDICATION_POLICY_VERSION,
        "benchmark_item_id": item.benchmark_item_id,
        "exact_contract_sha256": item.exact_contract_sha256,
        "scientific_question_sha256": item.scientific_question_sha256,
        "assessor_models": list(args.assessor_model),
        "adjudicator_model": args.adjudicator_model,
        "max_records_per_query": args.max_records_per_query,
        "max_evidence_records": args.max_evidence_records,
    }
    manifest_path = Path(args.pubmed_cache_dir) / "cache_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Adjudication requires an audited PubMed cache manifest: {manifest_path}")
    payload["pubmed_cache_manifest_sha256"] = file_sha256(manifest_path)
    return sha256_payload(payload)


def _validate_resumed_payload(payload: dict[str, Any], expected_fingerprint: str, path: Path) -> None:
    observed = payload.get("run_fingerprint")
    if observed != expected_fingerprint:
        raise ValueError(
            f"Refusing stale adjudication artifact {path}: fingerprint {observed!r} "
            f"does not match {expected_fingerprint!r}; use --force or a new output directory"
        )


def _process_item(item: BenchmarkItem, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    fingerprint = _run_fingerprint(item, args)
    checkpoint = _checkpoint_path(out_dir, item.benchmark_item_id)
    if checkpoint.exists() and not args.force:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        _validate_resumed_payload(payload, fingerprint, checkpoint)
        return payload

    freeze_path = _evidence_freeze_path(out_dir, item.benchmark_item_id)
    if freeze_path.exists() and not args.force:
        evidence_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        _validate_resumed_payload(evidence_freeze, fingerprint, freeze_path)
        query_rows = list(evidence_freeze.get("query_generation") or [])
        records = [EvidenceRecord.model_validate(row) for row in evidence_freeze["records"]]
    else:
        packet, manifest, manifest_sha256 = load_cache_packet(
            Path(args.pubmed_cache_dir),
            benchmark_item_id=item.benchmark_item_id,
            exact_contract_sha256=str(item.exact_contract_sha256),
            scientific_question_sha256=item.scientific_question_sha256,
            package_dir=Path(args.package_dir),
            max_records_per_query=args.max_records_per_query,
            max_evidence_records=args.max_evidence_records,
            assessor_models=list(args.assessor_model),
        )
        queries = list(packet.ordered_queries)
        query_rows = list(packet.query_generation)
        records = list(packet.records)
        evidence_freeze = {
            "benchmark_item_id": item.benchmark_item_id,
            "scientific_question_sha256": item.scientific_question_sha256,
            "run_fingerprint": fingerprint,
            "queries": queries,
            "query_generation": query_rows,
            "records": [record.model_dump(mode="json") for record in records],
            "evidence_sha256": sha256_payload([record.model_dump(mode="json") for record in records]),
            "frozen_before_votes": True,
            "retrieval_backend": "pubmed_cache_exact",
            "pubmed_cache_version": manifest.cache_version,
            "pubmed_cache_manifest_sha256": manifest_sha256,
            "pubmed_cache_packet_sha256": packet.packet_sha256,
        }
        _atomic_text(freeze_path, json.dumps(evidence_freeze, indent=2, sort_keys=True) + "\n")
    votes: list[LabelVote] = []
    vote_traces: list[dict[str, Any]] = []
    for model_spec in args.assessor_model:
        try:
            vote, trace = _vote(
                item,
                records,
                model_spec=model_spec,
                role="evidence_assessor",
                retries=args.schema_retries,
                max_tokens=args.llm_max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Adjudication vote failed for item={item.benchmark_item_id}, "
                f"model={model_spec}, role=evidence_assessor: {exc}"
            ) from exc
        votes.append(vote)
        vote_traces.append({"model_spec": model_spec, **trace})
    try:
        judge_vote, judge_trace = _vote(
            item,
            records,
            model_spec=args.adjudicator_model,
            role="independent_adjudicator",
            retries=args.schema_retries,
            max_tokens=args.llm_max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Adjudication vote failed for item={item.benchmark_item_id}, "
            f"model={args.adjudicator_model}, role=independent_adjudicator: {exc}"
        ) from exc
    votes.append(judge_vote)
    vote_traces.append({"model_spec": args.adjudicator_model, **judge_trace})
    payload = {
        "benchmark_item_id": item.benchmark_item_id,
        "run_fingerprint": fingerprint,
        "status": "completed",
        "query_generation": query_rows,
        "evidence_freeze": evidence_freeze,
        "votes": [vote.model_dump(mode="json") for vote in votes],
        "vote_traces": vote_traces,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_text(checkpoint, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def select_pilot_items(
    items: list[BenchmarkItem],
    *,
    seed: int = PILOT_SEED,
    exclude_ids: set[str] | None = None,
) -> list[BenchmarkItem]:
    rng = random.Random(seed)
    excluded = exclude_ids or set()
    selected: list[BenchmarkItem] = []
    targets = ("normative_fmri", "adhd", "asd", "ad_aging", "psychosis")
    for target in targets:
        pool = [
            item
            for item in items
            if item.target_family == target
            and item.contract is not None
            and not item.score_eligible
            and item.benchmark_item_id not in excluded
        ]
        legacy = [item for item in pool if any(ref.source_collection == "legacy_scientific" for ref in item.source_references)]
        literature = [
            item
            for item in pool
            if any(ref.source_collection == "stage2_current" and ref.source_mode == "literature_grounded" for ref in item.source_references)
        ]
        llm = [
            item
            for item in pool
            if any(ref.source_collection == "stage2_current" and ref.source_mode == "llm_proposed" for ref in item.source_references)
        ]
        if len(legacy) < 2:
            raise ValueError(f"Pilot requires two successfully redrafted legacy claims for {target}; found {len(legacy)}")
        rng.shuffle(legacy)
        rng.shuffle(literature)
        rng.shuffle(llm)
        target_rows = legacy[:2]
        selected_ids = {item.benchmark_item_id for item in target_rows}
        for item in literature:
            if len(target_rows) >= 6:
                break
            if item.benchmark_item_id not in selected_ids:
                target_rows.append(item)
                selected_ids.add(item.benchmark_item_id)
        for item in llm:
            if len(target_rows) >= 10:
                break
            if item.benchmark_item_id not in selected_ids:
                target_rows.append(item)
                selected_ids.add(item.benchmark_item_id)
        if len(target_rows) != 10:
            raise ValueError(f"Pilot requires ten executable claims for {target}; found {len(target_rows)}")
        selected.extend(target_rows)
    return sorted(selected, key=lambda row: row.benchmark_item_id)


def _load_pilot_item_ids(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"Development pilot item file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    item_ids = [str(value) for value in payload.get("item_ids") or []]
    if not item_ids:
        raise ValueError(f"Development pilot item file contains no item_ids: {path}")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"Development pilot item file contains duplicate item_ids: {path}")
    return item_ids


def _run_items(items: list[BenchmarkItem], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.max_workers <= 1:
        return [
            _process_item(item, args)
            for item in iter_progress(items, total=len(items), desc="NeuroClaimBench adjudication", enabled=not args.no_progress, unit="claim")
        ]
    rows: dict[str, dict[str, Any]] = {}
    executor_type = ProcessPoolExecutor if args.parallel_backend == "process" else ThreadPoolExecutor
    with executor_type(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_process_item, item, args): item.benchmark_item_id for item in items}
        for future in iter_progress(
            as_completed(futures),
            total=len(futures),
            desc="NeuroClaimBench adjudication",
            enabled=not args.no_progress,
            unit="claim",
        ):
            rows[futures[future]] = future.result()
    return [rows[item.benchmark_item_id] for item in items]


def _nominal_krippendorff_alpha(ratings: list[list[str]]) -> float:
    pairs_equal = 0
    pairs_total = 0
    counts: Counter[str] = Counter()
    for row in ratings:
        values = [value for value in row if value]
        counts.update(values)
        for i, left in enumerate(values):
            for right in values[i + 1 :]:
                pairs_equal += int(left == right)
                pairs_total += 1
    if pairs_total == 0:
        return math.nan
    observed_disagreement = 1.0 - pairs_equal / pairs_total
    total = sum(counts.values())
    if total < 2:
        return math.nan
    expected_agreement = sum(count * (count - 1) for count in counts.values()) / (total * (total - 1))
    expected_disagreement = 1.0 - expected_agreement
    return 1.0 if expected_disagreement == 0 else 1.0 - observed_disagreement / expected_disagreement


def _agreement_summary(ratings: list[list[str]]) -> dict[str, Any]:
    pairwise_equal = 0
    pairwise_total = 0
    unanimous = 0
    for row in ratings:
        values = [value for value in row if value]
        unanimous += int(bool(values) and len(set(values)) == 1)
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                pairwise_equal += int(left == right)
                pairwise_total += 1
    return {
        "unanimous_item_count": unanimous,
        "unanimous_item_rate": unanimous / len(ratings) if ratings else 0.0,
        "raw_pairwise_agreement_count": pairwise_equal,
        "raw_pairwise_comparison_count": pairwise_total,
        "raw_pairwise_agreement_rate": pairwise_equal / pairwise_total if pairwise_total else 0.0,
    }


def _pilot_operationally_accepted(
    *,
    pilot_size: int,
    development_overlap_count: int,
    complete_vote_set_rate: float,
    schema_valid_rate: float,
    fabricated_citation_count: int,
    order_instability_rate: float,
    provenance_complete_rate: float,
) -> bool:
    return (
        pilot_size == 50
        and development_overlap_count == 0
        and complete_vote_set_rate == 1.0
        and schema_valid_rate >= 0.99
        and fabricated_citation_count == 0
        and order_instability_rate <= 0.10
        and provenance_complete_rate == 1.0
    )


def _vote_trace_has_complete_provenance(trace_row: dict[str, Any] | None) -> bool:
    if not trace_row:
        return False
    if not trace_row.get("model_spec") or not trace_row.get("prompt") or not trace_row.get("raw_response"):
        return False
    attempts = list(trace_row.get("trace") or [])
    successful = [attempt for attempt in attempts if attempt.get("schema_valid") is True]
    if not successful:
        return False
    metadata = successful[-1].get("call_metadata") or {}
    return bool(metadata.get("provider") and metadata.get("model"))


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    items = _load_items(Path(args.package_dir))
    if not args.development_pilot_items:
        raise ValueError(
            "The v2.1 validation pilot requires --development-pilot-items so its 50 claims are disjoint "
            "from the rubric-development pilot"
        )
    development_path = Path(args.development_pilot_items)
    development_ids = _load_pilot_item_ids(development_path)
    if len(development_ids) != 50:
        raise ValueError(
            f"The rubric-development pilot must contain exactly 50 item_ids; found {len(development_ids)}"
        )
    pilot = select_pilot_items(items, seed=args.pilot_seed, exclude_ids=set(development_ids))
    selected_ids = [item.benchmark_item_id for item in pilot]
    overlap = sorted(set(selected_ids) & set(development_ids))
    if overlap:
        raise ValueError(f"Validation pilot overlaps the development pilot: {overlap[:5]}")
    out_dir = Path(args.out_dir)
    _atomic_text(
        out_dir / "pilot_items.json",
        json.dumps(
            {
                "policy_version": ADJUDICATION_POLICY_VERSION,
                "pilot_role": "validation",
                "seed": args.pilot_seed,
                "item_ids": selected_ids,
                "development_pilot_items_path": str(development_path),
                "development_pilot_items_sha256": file_sha256(development_path),
                "development_pilot_item_ids": development_ids,
                "development_overlap_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    rows = _run_items(pilot, args)
    return {"phase": "pilot", "n_items": len(rows), "completed": sum(row.get("status") == "completed" for row in rows)}


def run_pilot_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    pilot_payload = json.loads((out_dir / "pilot_items.json").read_text(encoding="utf-8"))
    if pilot_payload.get("policy_version") != ADJUDICATION_POLICY_VERSION:
        raise ValueError(
            "Pilot item policy does not match the active adjudication policy; use a new output directory "
            "and rerun PHASE=pilot"
        )
    item_map = {item.benchmark_item_id: item for item in _load_items(Path(args.package_dir))}
    item_ids = list(pilot_payload["item_ids"])
    development_ids = set(pilot_payload.get("development_pilot_item_ids") or [])
    development_overlap = sorted(set(item_ids) & development_ids)
    rng = random.Random(int(pilot_payload["seed"]))
    reverse_ids = sorted(rng.sample(item_ids, max(1, math.ceil(len(item_ids) * 0.20))))
    reverse_votes: dict[str, LabelVote] = {}
    reverse_traces: dict[str, dict[str, Any]] = {}
    schema_valid = 0
    first_attempt_valid = 0
    total = 0
    fabricated = 0
    provenance_complete = 0
    provenance_total = 0
    ratings: list[list[str]] = []
    original_judge: dict[str, str] = {}
    adjudications: list[AdjudicationRecord] = []
    complete_vote_sets = 0
    for item_id in item_ids:
        payload = json.loads(_checkpoint_path(out_dir, item_id).read_text(encoding="utf-8"))
        votes = [LabelVote.model_validate(row) for row in payload["votes"]]
        ratings.append([vote.proposed_label for vote in votes])
        complete_vote_sets += int(
            len(votes) == 3
            and len({vote.model_spec for vote in votes}) == 3
            and sum(vote.role == "evidence_assessor" for vote in votes) == 2
            and sum(vote.role == "independent_adjudicator" for vote in votes) == 1
        )
        adjudications.append(adjudicate_votes(item_id, votes, adjudicator_model=args.adjudicator_model))
        original_judge[item_id] = next(vote.proposed_label for vote in votes if vote.model_spec == args.adjudicator_model)
        allowed = {row["evidence_id"] for row in payload["evidence_freeze"]["records"]}
        traces_by_model = {row.get("model_spec"): row for row in payload.get("vote_traces") or []}
        for vote in votes:
            total += 1
            schema_valid += 1
            first_attempt_valid += int(vote.schema_attempts == 1)
            fabricated += int(bool(set(vote.evidence_ids) - allowed))
            provenance_total += 1
            provenance_complete += int(_vote_trace_has_complete_provenance(traces_by_model.get(vote.model_spec)))
    for item_id in reverse_ids:
        payload = json.loads(_checkpoint_path(out_dir, item_id).read_text(encoding="utf-8"))
        records = [EvidenceRecord.model_validate(row) for row in payload["evidence_freeze"]["records"]]
        reverse_fingerprint = sha256_payload(
            {
                "source_run_fingerprint": payload["run_fingerprint"],
                "adjudicator_model": args.adjudicator_model,
                "reversed_order": True,
            }
        )
        reverse_checkpoint = _reverse_order_checkpoint_path(out_dir, item_id)
        if reverse_checkpoint.exists():
            reverse_payload = json.loads(reverse_checkpoint.read_text(encoding="utf-8"))
            if reverse_payload.get("reverse_fingerprint") != reverse_fingerprint:
                raise ValueError(f"Refusing stale reverse-order checkpoint: {reverse_checkpoint}")
            vote = LabelVote.model_validate(reverse_payload["vote"])
            trace = dict(reverse_payload["trace"])
        else:
            try:
                vote, trace = _vote(
                    item_map[item_id],
                    records,
                    model_spec=args.adjudicator_model,
                    role="independent_adjudicator",
                    retries=args.schema_retries,
                    max_tokens=args.llm_max_tokens,
                    reversed_order=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Reverse-order audit vote failed for item={item_id}, model={args.adjudicator_model}: {exc}"
                ) from exc
            _atomic_text(
                reverse_checkpoint,
                json.dumps(
                    {
                        "benchmark_item_id": item_id,
                        "reverse_fingerprint": reverse_fingerprint,
                        "vote": vote.model_dump(mode="json"),
                        "trace": trace,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        reverse_votes[item_id] = vote
        reverse_traces[item_id] = {
            "benchmark_item_id": item_id,
            "model_spec": args.adjudicator_model,
            **trace,
        }
        provenance_total += 1
        provenance_complete += int(_vote_trace_has_complete_provenance(reverse_traces[item_id]))
    instability = sum(reverse_votes[item_id].proposed_label != original_judge[item_id] for item_id in reverse_ids)
    alpha = _nominal_krippendorff_alpha(ratings)
    schema_rate = schema_valid / total if total else 0.0
    first_attempt_rate = first_attempt_valid / total if total else 0.0
    provenance_rate = provenance_complete / provenance_total if provenance_total else 0.0
    instability_rate = instability / len(reverse_ids) if reverse_ids else 0.0
    agreement = _agreement_summary(ratings)
    complete_vote_set_rate = complete_vote_sets / len(item_ids) if item_ids else 0.0
    score_eligible_count = sum(row.score_eligible for row in adjudications)
    label_counts = Counter(row.final_label for row in adjudications)
    disposition_counts = Counter(row.reference_disposition for row in adjudications)
    unresolved_reason_counts = Counter(
        row.unresolved_reason for row in adjudications if row.unresolved_reason is not None
    )
    accepted = _pilot_operationally_accepted(
        pilot_size=len(item_ids),
        development_overlap_count=len(development_overlap),
        complete_vote_set_rate=complete_vote_set_rate,
        schema_valid_rate=schema_rate,
        fabricated_citation_count=fabricated,
        order_instability_rate=instability_rate,
        provenance_complete_rate=provenance_rate,
    )
    summary = {
        "phase": "pilot_audit",
        "policy_version": ADJUDICATION_POLICY_VERSION,
        "acceptance_policy": "operational_integrity_with_item_level_consensus",
        "pilot_size": len(item_ids),
        "pilot_role": pilot_payload.get("pilot_role"),
        "development_pilot_item_count": len(development_ids),
        "development_overlap_count": len(development_overlap),
        "development_overlap_item_ids": development_overlap,
        "complete_vote_set_count": complete_vote_sets,
        "complete_vote_set_rate": complete_vote_set_rate,
        "schema_valid_rate": schema_rate,
        "first_attempt_schema_valid_rate": first_attempt_rate,
        "fabricated_citation_count": fabricated,
        "provenance_complete_count": provenance_complete,
        "provenance_expected_count": provenance_total,
        "provenance_complete_rate": provenance_rate,
        "krippendorff_alpha_nominal": alpha,
        "aggregate_agreement_is_descriptive": True,
        **agreement,
        "item_consensus_score_eligible_count": score_eligible_count,
        "item_consensus_score_eligible_rate": score_eligible_count / len(item_ids) if item_ids else 0.0,
        "item_consensus_label_counts": dict(label_counts),
        "item_consensus_disposition_counts": dict(disposition_counts),
        "item_consensus_unresolved_reason_counts": dict(unresolved_reason_counts),
        "reverse_order_item_ids": reverse_ids,
        "claude_order_instability_count": instability,
        "claude_order_instability_rate": instability_rate,
        "accepted": accepted,
        "acceptance_thresholds": {
            "pilot_size": 50,
            "development_overlap_count_max": 0,
            "complete_vote_set_rate_min": 1.0,
            "schema_valid_rate_min": 0.99,
            "fabricated_citation_count_max": 0,
            "claude_order_instability_rate_max": 0.10,
            "provenance_complete_rate_min": 1.0,
        },
        "descriptive_metrics_not_used_for_acceptance": [
            "krippendorff_alpha_nominal",
            "raw_pairwise_agreement_rate",
            "unanimous_item_rate",
            "item_consensus_score_eligible_rate",
        ],
    }
    _write_jsonl(out_dir / "pilot_reverse_order_votes.jsonl", list(reverse_votes.values()))
    _write_jsonl(out_dir / "pilot_reverse_order_traces.jsonl", [reverse_traces[item_id] for item_id in reverse_ids])
    _atomic_text(out_dir / "pilot_audit.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = (
        Path(args.accepted_pilot_audit)
        if args.accepted_pilot_audit
        else Path(args.out_dir) / "pilot_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("policy_version") != ADJUDICATION_POLICY_VERSION:
        raise ValueError("Pilot audit policy does not match the active adjudication policy")
    if not audit.get("accepted"):
        raise ValueError("Pilot acceptance checks failed; full adjudication is blocked")
    package = Path(args.package_dir)
    all_items = _load_items(package)
    items = _required_adjudication_items(all_items, package)
    rows = _run_items(sorted(items, key=lambda row: row.benchmark_item_id), args)
    return {"phase": "full", "n_items": len(rows), "completed": sum(row.get("status") == "completed" for row in rows)}


def _required_adjudication_items(
    items: list[BenchmarkItem],
    package: Path | None = None,
) -> list[BenchmarkItem]:
    split_path = package / "benchmark_splits.json" if package is not None else None
    if split_path is not None and split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        requested = {str(value) for value in split.get("adjudication_candidates") or []}
        by_id = {item.benchmark_item_id: item for item in items}
        missing = requested - set(by_id)
        if missing:
            raise ValueError(f"Adjudication split references missing items: {sorted(missing)[:5]}")
        return sorted(
            (
                by_id[item_id]
                for item_id in requested
                if by_id[item_id].contract is not None
            ),
            key=lambda item: item.benchmark_item_id,
        )
    return sorted(
        (
            item
            for item in items
            if item.contract is not None
            and not item.score_eligible
            and item.benchmark_track in {"scientific", "external_transfer"}
        ),
        key=lambda item: item.benchmark_item_id,
    )


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    package = Path(args.package_dir)
    out_dir = Path(args.out_dir)
    items = _load_items(package)
    item_map = {item.benchmark_item_id: item for item in items}
    required_items = _required_adjudication_items(items, package)
    required_ids = {item.benchmark_item_id for item in required_items}
    checkpoint_paths = {
        path.stem: path for path in sorted((out_dir / "checkpoints").glob("*.json"))
    }
    missing = [item.benchmark_item_id for item in required_items if item.benchmark_item_id not in checkpoint_paths]
    if missing:
        raise ValueError(
            f"Finalize requires completed adjudication checkpoints for all executable scientific/external items; "
            f"missing {len(missing)} (first: {missing[:5]})"
        )
    for item in required_items:
        checkpoint = checkpoint_paths[item.benchmark_item_id]
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"Finalize found incomplete checkpoint: {checkpoint}")
        _validate_resumed_payload(payload, _run_fingerprint(item, args), checkpoint)
    evidence = [
        EvidenceRecord.model_validate(row)
        for row in _read_jsonl(package / "evidence_records.jsonl")
        if str(row.get("benchmark_item_id")) not in required_ids
    ]
    votes = [
        LabelVote.model_validate(row)
        for row in _read_jsonl(package / "label_votes.jsonl")
        if str(row.get("benchmark_item_id")) not in required_ids
    ]
    prompt_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    adjudications = [
        AdjudicationRecord.model_validate(row)
        for row in _read_jsonl(package / "adjudications.jsonl")
        if str(row.get("benchmark_item_id")) not in required_ids
    ]
    for item_id in sorted(required_ids):
        checkpoint = checkpoint_paths[item_id]
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        item_id = str(payload["benchmark_item_id"])
        if item_id not in item_map:
            raise ValueError(f"Adjudication checkpoint references unknown item: {item_id}")
        item_evidence = [EvidenceRecord.model_validate(row) for row in payload["evidence_freeze"]["records"]]
        item_votes = [LabelVote.model_validate(row) for row in payload["votes"]]
        for row in payload.get("query_generation", []):
            prompt_rows.append(
                {
                    "benchmark_item_id": item_id,
                    "stage": "query_expansion",
                    "model_spec": row.get("model_spec"),
                    "prompt": row.get("prompt"),
                }
            )
            response_rows.append(
                {
                    "benchmark_item_id": item_id,
                    "stage": "query_expansion",
                    "model_spec": row.get("model_spec"),
                    "raw_response": row.get("raw_response"),
                    "trace": row.get("trace"),
                }
            )
        for row in payload.get("vote_traces", []):
            prompt_rows.append(
                {
                    "benchmark_item_id": item_id,
                    "stage": "label_vote",
                    "model_spec": row.get("model_spec"),
                    "prompt": row.get("prompt"),
                }
            )
            response_rows.append(
                {
                    "benchmark_item_id": item_id,
                    "stage": "label_vote",
                    "model_spec": row.get("model_spec"),
                    "raw_response": row.get("raw_response"),
                    "trace": row.get("trace"),
                }
            )
        assessments_by_evidence: dict[str, dict[str, EvidenceStudyAssessment]] = {}
        for vote in item_votes:
            for assessment in vote.paper_assessments:
                assessments_by_evidence.setdefault(assessment.evidence_id, {})[vote.model_spec] = assessment
        for record in item_evidence:
            record.model_assessments = assessments_by_evidence.get(record.evidence_id, {})
        evidence.extend(item_evidence)
        votes.extend(item_votes)
        result = adjudicate_votes(item_id, item_votes, adjudicator_model=args.adjudicator_model)
        result = result.model_copy(
            update={"scientific_question_sha256": item_map[item_id].scientific_question_sha256}
        )
        adjudications.append(result)
        item = item_map[item_id]
        item.label_class = result.final_label
        item.reference_disposition = result.reference_disposition
        item.adjudication_status = result.adjudication_status
        item.score_eligible = result.score_eligible
    items = sorted(item_map.values(), key=lambda row: row.benchmark_item_id)
    adjudications = sorted(adjudications, key=lambda row: row.benchmark_item_id)
    _write_jsonl(package / "benchmark_items.jsonl", items)
    _write_jsonl(package / "evidence_records.jsonl", sorted(evidence, key=lambda row: row.evidence_id))
    _write_jsonl(package / "label_votes.jsonl", sorted(votes, key=lambda row: (row.benchmark_item_id, row.model_spec)))
    _write_jsonl(package / "adjudications.jsonl", adjudications)
    _write_jsonl(out_dir / "llm_prompts.jsonl", prompt_rows)
    _write_jsonl(out_dir / "llm_responses.jsonl", response_rows)
    manifest_path = package / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adjudication"] = {
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "assessor_models": list(args.assessor_model),
        "adjudicator_model": args.adjudicator_model,
        "prompt_policy_sha256": sha256_payload({"query": "v1", "vote": ADJUDICATION_POLICY_VERSION}),
        "label_counts": dict(Counter(item.label_class for item in items)),
        "adjudication_status_counts": dict(Counter(item.adjudication_status for item in items)),
        "score_eligible_count": sum(item.score_eligible for item in items),
    }
    for name in ("benchmark_items.jsonl", "evidence_records.jsonl", "label_votes.jsonl", "adjudications.jsonl"):
        manifest.setdefault("output_files", {})[name] = {
            "path": str(package / name),
            "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest(),
        }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest["adjudication"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["pilot", "pilot_audit", "full", "finalize"])
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--assessor-model", action="append", default=None)
    parser.add_argument("--adjudicator-model", default=DEFAULT_ADJUDICATOR)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--max-records-per-query", type=int, default=5)
    parser.add_argument("--max-evidence-records", type=int, default=12)
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument("--pubmed-cache-dir", default="data/neuroclaimbench/pubmed-cache-v2.1")
    parser.add_argument("--pilot-seed", type=int, default=PILOT_SEED)
    parser.add_argument("--development-pilot-items")
    parser.add_argument("--accepted-pilot-audit")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["thread", "process"], default="thread")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.assessor_model is None:
        args.assessor_model = list(DEFAULT_ASSESSORS)
    if len(args.assessor_model) != 2:
        raise ValueError("Exactly two independent assessor models are required")
    if args.adjudicator_model in set(args.assessor_model):
        raise ValueError("The adjudicator model must differ from both assessor models")
    if not (Path(args.pubmed_cache_dir) / "cache_manifest.json").exists():
        raise ValueError("Adjudication requires a completed, audited PubMed cache")
    if args.phase == "pilot":
        return run_pilot(args)
    if args.phase == "pilot_audit":
        return run_pilot_audit(args)
    if args.phase == "full":
        return run_full(args)
    return run_finalize(args)


def main(argv: Optional[list[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
