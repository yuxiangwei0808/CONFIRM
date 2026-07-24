from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from bench import run_neuroclaimbench_adjudication as adjudication_runner
from bench.neuroclaimbench import (
    BenchmarkItem,
    EvidenceRecord,
    EvidenceStudyAssessment,
    LabelVote,
    SourceReference,
    adjudicate_votes,
    exact_contract_hash,
    semantic_contract_hash,
    summarize_benchmark,
)
from bench.run_neuroclaimbench_adjudication import (
    ADJUDICATION_POLICY_VERSION,
    _required_adjudication_items,
    _pilot_operationally_accepted,
    _validate_resumed_payload,
    _vote,
    _vote_trace_has_complete_provenance,
    build_parser as build_adjudication_parser,
    build_vote_prompt,
    select_pilot_items,
)
from bench.run_neuroclaimbench_build import _CorpusBuilder
from bench.run_neuroclaimbench_pubmed_cache import build_query_prompt
from confirm.contract import ClaimContract


def contract(claim_id: str = "claim", question: str = "Is age associated with hippocampal volume?") -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": claim_id,
            "question": question,
            "estimand": {
                "type": "association",
                "outcome": "smri_hippocampus",
                "predictor": "age",
                "group": None,
                "direction": "negative",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["sex"],
            "inclusion": None,
            "discovery_cohort": "UKB_DISC",
            "replication_cohorts": ["UKB_REP"],
            "search_provenance": {"declared": True, "family_size": 1, "selection": "preregistered"},
            "gates": {
                "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
                "confound": {"require_covariates": ["sex"], "motion_check": False},
                "power": {"min_power": 0.8, "ref_effect": None},
                "multiverse": {"min_fraction_consistent": 0.6},
                "replication": {
                    "alpha": 0.05,
                    "require_same_sign": True,
                    "require_ci_overlap": False,
                    "harmonize": "combat",
                    "pattern_corr_min": 0.5,
                    "region_replication_frac_min": 0.5,
                    "dice_min": 0.0,
                },
            },
            "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
        }
    )


def item(item_id: str = "ncb-scientific-1", source_collection: str = "stage2_current", source_mode: str = "llm_proposed", target: str = "normative_fmri") -> BenchmarkItem:
    frozen = contract(item_id)
    exact = exact_contract_hash(frozen)
    semantic = semantic_contract_hash(frozen)
    return BenchmarkItem(
        benchmark_item_id=item_id,
        claim_uid=f"claim-{semantic[:8]}",
        semantic_cluster_id=f"semantic-{semantic[:8]}",
        benchmark_track="scientific",
        target_family=target,
        modality="sMRI",
        question=frozen.question,
        contract=frozen,
        exact_contract_sha256=exact,
        semantic_claim_sha256=semantic,
        source_references=[
            SourceReference(
                source_collection=source_collection,
                source_id=item_id,
                source_path="source.json",
                source_mode=source_mode,
                target_family=target,
            )
        ],
    )


def assessment(evidence_id: str = "e1") -> EvidenceStudyAssessment:
    return EvidenceStudyAssessment(
        evidence_id=evidence_id,
        study_design="meta_analysis",
        directness="direct",
        relation="supports_positive",
        population_match="exact",
        modality_match="exact",
        outcome_match="exact",
        direction_match="exact",
        independent_group="meta",
        supporting_text="Directly supports the association.",
    )


def vote(model: str, role: str, label: str = "known_positive") -> LabelVote:
    paper = assessment()
    return LabelVote.model_validate(
        {
            "benchmark_item_id": "ncb-scientific-1",
            "model_spec": model,
            "role": role,
            "proposed_label": label,
            "construct_match": "exact",
            "confidence": "high",
            "evidence_ids": ["e1"],
            "paper_assessments": [paper.model_dump(mode="json")],
            "rationale": "Direct meta-analysis evidence.",
            "prompt_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        }
    )


def test_contract_hash_ignores_identity_but_semantic_hash_ignores_evidence_pair():
    first = contract("a", "Question A")
    second_payload = first.model_dump(mode="json")
    second_payload["claim_id"] = "b"
    second_payload["question"] = "Question B"
    second = ClaimContract.model_validate(second_payload)
    assert exact_contract_hash(first) == exact_contract_hash(second)

    moved = copy.deepcopy(second_payload)
    moved["discovery_cohort"] = "HCP_DISC"
    moved["replication_cohorts"] = ["HCP_REP"]
    third = ClaimContract.model_validate(moved)
    assert exact_contract_hash(first) != exact_contract_hash(third)
    assert semantic_contract_hash(first) == semantic_contract_hash(third)


def test_random_control_seed_is_part_of_execution_identity():
    builder = _CorpusBuilder({})
    frozen = contract()
    for seed in (1, 2):
        builder.add_ready(
            track="external_transfer",
            target_family="ad_aging",
            question=frozen.question,
            contract=frozen,
            source=SourceReference(
                source_collection="external_random",
                source_id=f"random-{seed}",
                source_path="external.csv",
            ),
            label_class="known_null",
            construction_derived=True,
            evidence_role="synthetic_control",
            generator_spec={"seed": seed},
        )
    assert len(builder.items) == 2
    assert len(builder.tasks) == 2
    assert all(task.contract == frozen for task in builder.tasks.values())


def test_ready_item_requires_contract():
    with pytest.raises(ValueError, match="require a frozen contract"):
        BenchmarkItem(
            benchmark_item_id="x",
            claim_uid="c",
            semantic_cluster_id="s",
            benchmark_track="scientific",
            target_family="asd",
            modality="unknown",
            question="q",
            semantic_claim_sha256="a" * 64,
            source_references=[],
        )


def test_adjudication_requires_claude_concurrence_and_direct_evidence():
    models = [
        vote("openai:gpt-5.5", "evidence_assessor"),
        vote("google:gemini-3.5-flash", "evidence_assessor", "fragile"),
        vote("openrouter:anthropic/claude-opus-4.8", "independent_adjudicator"),
    ]
    result = adjudicate_votes(
        "ncb-scientific-1",
        models,
        adjudicator_model="openrouter:anthropic/claude-opus-4.8",
    )
    assert result.final_label == "known_positive"
    assert result.score_eligible

    disagreeing = [
        vote("openai:gpt-5.5", "evidence_assessor", "fragile"),
        vote("google:gemini-3.5-flash", "evidence_assessor", "fragile"),
        vote("openrouter:anthropic/claude-opus-4.8", "independent_adjudicator", "known_positive"),
    ]
    result = adjudicate_votes(
        "ncb-scientific-1",
        disagreeing,
        adjudicator_model="openrouter:anthropic/claude-opus-4.8",
    )
    assert result.final_label == "candidate_unknown"
    assert result.unresolved_reason == "no_assessor_agrees_with_adjudicator"


def test_prompts_exclude_gate_and_source_result_fields():
    benchmark_item = item()
    query_prompt = build_query_prompt(benchmark_item)
    vote_prompt = build_vote_prompt(benchmark_item, [])
    for prompt in (query_prompt, vote_prompt):
        lower = prompt.lower()
        assert "gate_verdict" not in lower
        assert "source_mode" not in lower
        assert "holdout_supported" not in lower
        assert "primary_effect" not in lower
        assert "final_label" not in lower
    assert "execution provenance" in vote_prompt.lower()
    assert "do not require a paper to use the exact same named dataset" in vote_prompt.lower()
    for label in ("known_positive", "known_null", "fragile", "underpowered_small_positive", "candidate_unknown"):
        assert f"{label}:" in vote_prompt
    assert ADJUDICATION_POLICY_VERSION == "neuroclaimbench-v2.1-item-consensus-1"


def test_vote_retries_when_citations_and_assessments_do_not_match(monkeypatch):
    benchmark_item = item()
    record = EvidenceRecord(
        evidence_id="e1",
        benchmark_item_id=benchmark_item.benchmark_item_id,
        pmid="12345678",
        title="Matched evidence",
        abstract="The abstract directly assesses the claim.",
        query="query",
        target_family=benchmark_item.target_family,
        retrieved_at="2026-07-22T00:00:00+00:00",
    )
    paper = assessment("e1").model_dump(mode="json")
    responses = [
        {
            "proposed_label": "known_positive",
            "construct_match": "exact",
            "confidence": "high",
            "evidence_ids": [],
            "paper_assessments": [paper],
            "rationale": "Direct support.",
        },
        {
            "proposed_label": "known_positive",
            "construct_match": "exact",
            "confidence": "high",
            "evidence_ids": ["e1"],
            "paper_assessments": [paper],
            "rationale": "Direct support.",
        },
    ]

    class RetryLLM:
        max_tokens = 0
        last_call_metadata = {}

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete_structured(self, system: str, user: str, response_model: type) -> str:
            self.prompts.append(user)
            return json.dumps(responses[len(self.prompts) - 1])

    llm = RetryLLM()
    monkeypatch.setattr("bench.run_neuroclaimbench_adjudication.make_llm", lambda _: llm)

    result, _ = _vote(
        benchmark_item,
        [record],
        model_spec="test:model",
        role="evidence_assessor",
        retries=1,
    )

    assert result.evidence_ids == ["e1"]
    assert result.schema_attempts == 2
    assert len(llm.prompts) == 2
    assert "uncited_assessments=['e1']" in llm.prompts[1]


def test_vote_does_not_retry_insufficient_openrouter_credits(monkeypatch):
    benchmark_item = item()

    class NoCreditLLM:
        max_tokens = 0
        last_call_metadata = {}
        calls = 0

        def complete_structured(self, system: str, user: str, response_model: type) -> str:
            self.calls += 1
            raise RuntimeError("Error code: 402 - {'error': {'message': 'Insufficient credits.', 'code': 402}}")

    llm = NoCreditLLM()
    monkeypatch.setattr("bench.run_neuroclaimbench_adjudication.make_llm", lambda _: llm)

    with pytest.raises(RuntimeError, match="Non-retryable LLM provider error"):
        _vote(
            benchmark_item,
            [],
            model_spec="openrouter:anthropic/claude-opus-4.8",
            role="independent_adjudicator",
            retries=3,
        )

    assert llm.calls == 1


def test_vote_deterministically_limits_four_valid_citations(monkeypatch):
    benchmark_item = item()
    records = [
        EvidenceRecord(
            evidence_id=f"e{index}",
            benchmark_item_id=benchmark_item.benchmark_item_id,
            pmid=f"1234567{index}",
            title=f"Evidence {index}",
            abstract="The abstract directly assesses the claim.",
            query="query",
            target_family=benchmark_item.target_family,
            retrieved_at="2026-07-22T00:00:00+00:00",
        )
        for index in range(1, 5)
    ]
    payload = {
        "proposed_label": "known_positive",
        "construct_match": "exact",
        "confidence": "high",
        "evidence_ids": [record.evidence_id for record in records],
        "paper_assessments": [assessment(record.evidence_id).model_dump(mode="json") for record in records],
        "rationale": "Direct support.",
    }

    class FourCitationLLM:
        max_tokens = 0
        last_call_metadata = {}

        def __init__(self) -> None:
            self.calls = 0

        def complete_structured(self, system: str, user: str, response_model: type) -> str:
            self.calls += 1
            return json.dumps(payload)

    llm = FourCitationLLM()
    monkeypatch.setattr("bench.run_neuroclaimbench_adjudication.make_llm", lambda _: llm)

    result, trace = _vote(
        benchmark_item,
        records,
        model_spec="test:model",
        role="independent_adjudicator",
        retries=3,
    )

    assert result.evidence_ids == ["e1", "e2", "e3"]
    assert [row.evidence_id for row in result.paper_assessments] == ["e1", "e2", "e3"]
    assert result.schema_attempts == 1
    assert llm.calls == 1
    assert trace["normalization"] == {
        "applied": True,
        "reason": "citation_limit",
        "original_evidence_ids": ["e1", "e2", "e3", "e4"],
        "retained_evidence_ids": ["e1", "e2", "e3"],
        "dropped_uncited_assessment_ids": [],
        "dropped_blank_evidence_id_count": 0,
        "dropped_blank_assessment_count": 0,
    }


def test_vote_drops_fourth_uncited_assessment_without_retry(monkeypatch):
    benchmark_item = item()
    records = [
        EvidenceRecord(
            evidence_id=f"e{index}",
            benchmark_item_id=benchmark_item.benchmark_item_id,
            pmid=f"2234567{index}",
            title=f"Evidence {index}",
            abstract="The abstract directly assesses the claim.",
            query="query",
            target_family=benchmark_item.target_family,
            retrieved_at="2026-07-22T00:00:00+00:00",
        )
        for index in range(1, 5)
    ]
    payload = {
        "proposed_label": "known_positive",
        "construct_match": "exact",
        "confidence": "high",
        "evidence_ids": ["e1", "e2", "e3"],
        "paper_assessments": [assessment(record.evidence_id).model_dump(mode="json") for record in records],
        "rationale": "Direct support.",
    }

    class ExtraAssessmentLLM:
        max_tokens = 0
        last_call_metadata = {}
        calls = 0

        def complete_structured(self, system: str, user: str, response_model: type) -> str:
            self.calls += 1
            return json.dumps(payload)

    llm = ExtraAssessmentLLM()
    monkeypatch.setattr("bench.run_neuroclaimbench_adjudication.make_llm", lambda _: llm)

    result, trace = _vote(
        benchmark_item,
        records,
        model_spec="test:model",
        role="independent_adjudicator",
        retries=3,
    )

    assert result.evidence_ids == ["e1", "e2", "e3"]
    assert [row.evidence_id for row in result.paper_assessments] == ["e1", "e2", "e3"]
    assert llm.calls == 1
    assert trace["normalization"]["reason"] == "uncited_assessments"
    assert trace["normalization"]["dropped_uncited_assessment_ids"] == ["e4"]


def test_vote_drops_blank_no_citation_placeholders(monkeypatch):
    benchmark_item = item()
    payload = {
        "proposed_label": "candidate_unknown",
        "construct_match": "partial",
        "confidence": "low",
        "evidence_ids": [""],
        "paper_assessments": [assessment("").model_dump(mode="json")],
        "rationale": "No directly matched evidence.",
    }

    class BlankCitationLLM:
        max_tokens = 0
        last_call_metadata = {}
        calls = 0

        def complete_structured(self, system: str, user: str, response_model: type) -> str:
            self.calls += 1
            return json.dumps(payload)

    llm = BlankCitationLLM()
    monkeypatch.setattr("bench.run_neuroclaimbench_adjudication.make_llm", lambda _: llm)

    result, trace = _vote(
        benchmark_item,
        [],
        model_spec="test:model",
        role="independent_adjudicator",
        retries=3,
    )

    assert result.evidence_ids == []
    assert result.paper_assessments == []
    assert llm.calls == 1
    assert trace["normalization"]["reason"] == "blank_evidence_placeholders"
    assert trace["normalization"]["dropped_blank_evidence_id_count"] == 1
    assert trace["normalization"]["dropped_blank_assessment_count"] == 1


def test_pilot_audit_preserves_reverse_order_provenance(tmp_path, monkeypatch):
    benchmark_item = item()
    package = tmp_path / "package"
    out = tmp_path / "out"
    package.mkdir()
    (package / "benchmark_items.jsonl").write_text(benchmark_item.model_dump_json() + "\n", encoding="utf-8")
    out.mkdir()
    (out / "pilot_items.json").write_text(
        json.dumps(
            {
                "policy_version": ADJUDICATION_POLICY_VERSION,
                "pilot_role": "validation",
                "seed": 20260723,
                "item_ids": [benchmark_item.benchmark_item_id],
                "development_pilot_item_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = EvidenceRecord(
        evidence_id="e1",
        benchmark_item_id=benchmark_item.benchmark_item_id,
        pmid="12345678",
        title="Matched evidence",
        abstract="The abstract directly assesses the claim.",
        query="query",
        target_family=benchmark_item.target_family,
        retrieved_at="2026-07-22T00:00:00+00:00",
    )
    votes = [
        vote("openai:gpt-5.5", "evidence_assessor"),
        vote("google:gemini-3.5-flash", "evidence_assessor"),
        vote("openrouter:anthropic/claude-opus-4.8", "independent_adjudicator"),
    ]

    def trace(model_spec: str, reversed_order: bool = False) -> dict:
        return {
            "model_spec": model_spec,
            "prompt": "prompt",
            "raw_response": "{}",
            "trace": [
                {
                    "schema_valid": True,
                    "call_metadata": {"provider": "test", "model": model_spec},
                }
            ],
            "reversed_order": reversed_order,
        }

    checkpoint = out / "checkpoints" / f"{benchmark_item.benchmark_item_id}.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_fingerprint": "source-fingerprint",
                "votes": [row.model_dump(mode="json") for row in votes],
                "vote_traces": [trace(row.model_spec) for row in votes],
                "evidence_freeze": {"records": [record.model_dump(mode="json")]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reverse_vote = vote("openrouter:anthropic/claude-opus-4.8", "independent_adjudicator")
    reverse_trace = trace(reverse_vote.model_spec, reversed_order=True)
    reverse_trace.pop("model_spec")
    monkeypatch.setattr(adjudication_runner, "_vote", lambda *args, **kwargs: (reverse_vote, reverse_trace))

    result = adjudication_runner.run_pilot_audit(
        SimpleNamespace(
            out_dir=str(out),
            package_dir=str(package),
            pilot_seed=20260723,
            adjudicator_model="openrouter:anthropic/claude-opus-4.8",
            schema_retries=1,
            llm_max_tokens=1024,
        )
    )

    assert result["accepted"] is False
    assert result["aggregate_agreement_is_descriptive"] is True
    assert "krippendorff_alpha_min" not in result["acceptance_thresholds"]
    assert result["provenance_complete_count"] == result["provenance_expected_count"] == 4
    saved = json.loads((out / "pilot_reverse_order_traces.jsonl").read_text(encoding="utf-8"))
    assert saved["benchmark_item_id"] == benchmark_item.benchmark_item_id
    assert saved["reversed_order"] is True
    assert _vote_trace_has_complete_provenance(saved)
    assert (out / "pilot_reverse_order_checkpoints" / f"{benchmark_item.benchmark_item_id}.json").exists()


def test_pilot_selection_is_ten_per_target_with_legacy_and_both_current_modes():
    rows = []
    for target in ("normative_fmri", "adhd", "asd", "ad_aging", "psychosis"):
        for index in range(2):
            rows.append(item(f"{target}-legacy-{index}", "legacy_scientific", "literature", target))
        for index in range(4):
            rows.append(item(f"{target}-lit-{index}", "stage2_current", "literature_grounded", target))
        for index in range(6):
            rows.append(item(f"{target}-llm-{index}", "stage2_current", "llm_proposed", target))
    selected = select_pilot_items(rows)
    assert len(selected) == 50
    for target in ("normative_fmri", "adhd", "asd", "ad_aging", "psychosis"):
        target_rows = [row for row in selected if row.target_family == target]
        assert len(target_rows) == 10
        assert sum(any(ref.source_collection == "legacy_scientific" for ref in row.source_references) for row in target_rows) == 2


def test_pilot_acceptance_uses_operational_checks_not_aggregate_agreement():
    assert _pilot_operationally_accepted(
        pilot_size=50,
        development_overlap_count=0,
        complete_vote_set_rate=1.0,
        schema_valid_rate=1.0,
        fabricated_citation_count=0,
        order_instability_rate=0.10,
        provenance_complete_rate=1.0,
    )


def test_validation_pilot_excludes_development_item_ids():
    rows = []
    development_ids = set()
    for target in ("normative_fmri", "adhd", "asd", "ad_aging", "psychosis"):
        for index in range(4):
            row = item(f"{target}-legacy-{index}", "legacy_scientific", "literature", target)
            rows.append(row)
            if index < 2:
                development_ids.add(row.benchmark_item_id)
        for index in range(8):
            row = item(f"{target}-lit-{index}", "stage2_current", "literature_grounded", target)
            rows.append(row)
            if index < 2:
                development_ids.add(row.benchmark_item_id)
        for index in range(10):
            row = item(f"{target}-llm-{index}", "stage2_current", "llm_proposed", target)
            rows.append(row)
            if index < 2:
                development_ids.add(row.benchmark_item_id)

    selected = select_pilot_items(rows, exclude_ids=development_ids)

    assert len(selected) == 50
    assert not ({row.benchmark_item_id for row in selected} & development_ids)


def test_pilot_selection_does_not_repeat_cross_source_aliases():
    rows = []
    for target in ("normative_fmri", "adhd", "asd", "ad_aging", "psychosis"):
        alias = item(f"{target}-alias", "legacy_scientific", "literature", target)
        alias.source_references.append(
            SourceReference(
                source_collection="stage2_current",
                source_id=f"{target}-stage2-alias",
                source_path="stage2.json",
                source_mode="literature_grounded",
                target_family=target,
            )
        )
        rows.append(alias)
        rows.append(item(f"{target}-legacy-2", "legacy_scientific", "literature", target))
        for index in range(4):
            rows.append(item(f"{target}-lit-{index}", "stage2_current", "literature_grounded", target))
        for index in range(6):
            rows.append(item(f"{target}-llm-{index}", "stage2_current", "llm_proposed", target))
    selected = select_pilot_items(rows)
    assert len(selected) == len({row.benchmark_item_id for row in selected}) == 50


def test_resume_rejects_stale_adjudication_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="Refusing stale adjudication artifact"):
        _validate_resumed_payload({"run_fingerprint": "old"}, "new", tmp_path / "checkpoint.json")


def test_finalize_scope_excludes_controls_and_pending_items():
    scientific = item("scientific")
    control = item("control")
    control.benchmark_track = "synthetic_stress"
    control.label_class = "known_null"
    control.reference_disposition = "abstain"
    control.adjudication_status = "construction_derived"
    control.score_eligible = True
    pending = item("pending")
    pending.contract = None
    pending.exact_contract_sha256 = None
    pending.migration_status = "pending_contract"
    required = _required_adjudication_items([control, pending, scientific])
    assert [row.benchmark_item_id for row in required] == ["scientific"]


def test_adjudication_parser_accepts_process_backend():
    args = build_adjudication_parser().parse_args(
        ["--phase", "pilot", "--parallel-backend", "process", "--max-workers", "4"]
    )
    assert args.parallel_backend == "process"
    assert args.max_workers == 4


def test_summary_never_pools_tracks():
    scientific = item("science")
    scientific.label_class = "known_positive"
    scientific.reference_disposition = "confirm"
    scientific.adjudication_status = "multi_model_consensus"
    scientific.score_eligible = True
    stress = item("stress")
    stress.benchmark_track = "synthetic_stress"
    stress.label_class = "known_null"
    stress.reference_disposition = "abstain"
    stress.adjudication_status = "construction_derived"
    stress.score_eligible = True
    summary = summarize_benchmark([scientific, stress], {"science": "confirmed", "stress": "fragile"})
    assert summary["metrics_by_track"]["scientific"]["confirmable_claim_recall"] == 1.0
    assert summary["metrics_by_track"]["synthetic_stress"]["unsafe_confirmation_rate"] == 0.0
