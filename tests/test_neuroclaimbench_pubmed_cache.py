from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from bench.neuroclaimbench import BenchmarkItem, LabelVote, SourceReference, exact_contract_hash, semantic_contract_hash
from bench.pubmed_cache import (
    PubMedCacheMissError,
    QueryPlanRow,
    load_cache_packet,
    normalized_query,
    package_snapshot,
    query_key,
)
from bench.pubmed import PubMedRecord
from bench import run_neuroclaimbench_adjudication as adjudication
from bench import run_neuroclaimbench_pubmed_cache as cache_runner
from confirm.contract import ClaimContract


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "cache_claim",
            "question": "Is age negatively associated with hippocampal volume?",
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


def _item() -> BenchmarkItem:
    contract = _contract()
    exact = exact_contract_hash(contract)
    semantic = semantic_contract_hash(contract)
    return BenchmarkItem(
        benchmark_item_id="ncb-scientific-cache-test",
        claim_uid=f"ncb-claim-{semantic[:16]}",
        semantic_cluster_id=f"ncb-sem-{semantic[:16]}",
        benchmark_track="scientific",
        target_family="ad_aging",
        modality="sMRI",
        question=contract.question,
        contract=contract,
        exact_contract_sha256=exact,
        semantic_claim_sha256=semantic,
        source_references=[
            SourceReference(
                source_collection="stage2_current",
                source_id="cache_claim",
                source_path="source.json",
                source_mode="llm_proposed",
                target_family="ad_aging",
            )
        ],
    )


def _write_package(path: Path, item: BenchmarkItem) -> None:
    path.mkdir(parents=True)
    (path / "benchmark_items.jsonl").write_text(item.model_dump_json() + "\n", encoding="utf-8")
    (path / "benchmark_splits.json").write_text(
        json.dumps({"version": "2.0.0", "adjudication_candidates": [item.benchmark_item_id]}) + "\n",
        encoding="utf-8",
    )


def _cache_args(package: Path, cache: Path, phase: str) -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        package_dir=str(package),
        cache_dir=str(cache),
        assessor_model=["openai:gpt-5.5", "google:gemini-3.5-flash"],
        schema_retries=0,
        max_records_per_query=2,
        max_evidence_records=2,
        assessor_models=["openai:gpt-5.5", "google:gemini-3.5-flash"],
        pubmed_email="",
        pubmed_api_key="",
        pubmed_timeout=1.0,
        request_retries=0,
        retry_delay=0.0,
        fetch_batch_size=100,
        max_workers=1,
        no_progress=True,
    )


def _write_plan(package: Path, cache: Path, item: BenchmarkItem) -> QueryPlanRow:
    package_sha, package_files = package_snapshot(package)
    row = QueryPlanRow(
        benchmark_item_id=item.benchmark_item_id,
        exact_contract_sha256=str(item.exact_contract_sha256),
        target_family=item.target_family,
        modality=item.modality,
        package_sha256=package_sha,
        plan_fingerprint="f" * 64,
        base_query="query one",
        ordered_queries=["query one", "query two"],
        assessor_models=["openai:gpt-5.5", "google:gemini-3.5-flash"],
        query_prompt_sha256="p" * 64,
        query_generation=[],
    )
    cache.mkdir(parents=True)
    (cache / "query_plan.jsonl").write_text(row.model_dump_json() + "\n", encoding="utf-8")
    state = {
        "package_sha256": package_sha,
        "package_files": package_files,
        "max_records_per_query": 2,
        "max_evidence_records": 2,
    }
    (cache / ".work").mkdir()
    (cache / ".work" / "plan_state.json").write_text(json.dumps(state), encoding="utf-8")
    return row


def _record(pmid: str, query: str = "cache") -> PubMedRecord:
    return PubMedRecord(
        pmid=pmid,
        title=f"Article {pmid}",
        abstract=f"Abstract for {pmid}.",
        journal="Journal",
        year="2025",
        doi=f"10.1/{pmid}",
        mesh_terms=["Hippocampus"],
        query=query,
        target_family="cache",
        modality="metadata_abstract",
        retrieved_at="2026-07-22T00:00:00+00:00",
    )


def test_query_key_is_whitespace_stable_and_parameter_sensitive():
    assert normalized_query("age   AND\nMRI") == "age AND MRI"
    assert query_key("age   AND MRI", retmax=5) == query_key("age AND MRI", retmax=5)
    assert query_key("age AND MRI", retmax=5) != query_key("age AND MRI", retmax=10)


def test_rate_limiter_enforces_configured_interval():
    now = [0.0]
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    limiter = cache_runner.RequestRateLimiter(10.0, clock=lambda: now[0], sleeper=sleep)
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert sleeps == pytest.approx([0.1, 0.1])


def test_plan_covers_exact_package_candidates(tmp_path: Path, monkeypatch):
    package = tmp_path / "package"
    cache = tmp_path / "cache"
    item = _item()
    _write_package(package, item)

    def expand(item: BenchmarkItem, model: str, retries: int):
        query = f"{model} expansion"
        prompt = cache_runner.build_query_prompt(item)
        return [query], {
            "model_spec": model,
            "prompt": prompt,
            "raw_response": json.dumps({"queries": [query]}),
            "schema_attempts": 1,
            "trace": [],
        }

    monkeypatch.setattr(cache_runner, "_query_expansions", expand)
    args = _cache_args(package, cache, "plan")
    result = cache_runner.run_plan(args)
    assert result == {"phase": "plan", "candidate_count": 1, "planned_count": 1}
    plans = [QueryPlanRow.model_validate_json(line) for line in (cache / "query_plan.jsonl").read_text().splitlines()]
    assert [row.benchmark_item_id for row in plans] == [item.benchmark_item_id]
    assert plans[0].ordered_queries[1:] == [
        "openai:gpt-5.5 expansion",
        "google:gemini-3.5-flash expansion",
    ]


def test_fetch_resumes_without_repeating_completed_search_and_stops_early(tmp_path: Path, monkeypatch):
    package = tmp_path / "package"
    cache = tmp_path / "cache"
    item = _item()
    _write_package(package, item)
    _write_plan(package, cache, item)
    args = _cache_args(package, cache, "fetch")
    query_calls: list[str] = []
    fail_once = [True]

    def search(query: str, **_: object) -> list[str]:
        query_calls.append(query)
        return ["1", "2"]

    def fetch(ids: list[str], **_: object) -> list[PubMedRecord]:
        if fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("temporary fetch failure")
        return [_record(pmid) for pmid in ids]

    monkeypatch.setattr(cache_runner, "search_pubmed_ids", search)
    monkeypatch.setattr(cache_runner, "fetch_pubmed_records", fetch)
    with pytest.raises(RuntimeError, match="rerun PHASE=fetch"):
        cache_runner.run_fetch(args)
    assert query_calls == ["query one"]

    result = cache_runner.run_fetch(args)
    assert result["query_result_count"] == 1
    assert result["article_count"] == 2
    assert query_calls == ["query one"]
    cache_runner.run_freeze(_cache_args(package, cache, "freeze"))
    audit = cache_runner.run_audit(_cache_args(package, cache, "audit"))
    assert audit["counts"]["evidence_packets"] == 1
    assert audit["counts"]["evidence_records"] == 2

    packet, _, _ = load_cache_packet(
        cache,
        benchmark_item_id=item.benchmark_item_id,
        exact_contract_sha256=str(item.exact_contract_sha256),
        package_dir=package,
        max_records_per_query=2,
        max_evidence_records=2,
        assessor_models=["openai:gpt-5.5", "google:gemini-3.5-flash"],
    )
    assert [record.pmid for record in packet.records] == ["1", "2"]
    assert all(record.query == "query one" for record in packet.records)
    with pytest.raises(PubMedCacheMissError, match="assessor models differ"):
        load_cache_packet(
            cache,
            benchmark_item_id=item.benchmark_item_id,
            exact_contract_sha256=str(item.exact_contract_sha256),
            package_dir=package,
            max_records_per_query=2,
            max_evidence_records=2,
            assessor_models=["different:model", "google:gemini-3.5-flash"],
        )


def test_cache_backed_adjudication_never_calls_query_or_pubmed_network(tmp_path: Path, monkeypatch):
    package = tmp_path / "package"
    cache = tmp_path / "cache"
    out = tmp_path / "adjudication"
    item = _item()
    _write_package(package, item)
    _write_plan(package, cache, item)
    cache_args = _cache_args(package, cache, "fetch")
    monkeypatch.setattr(cache_runner, "search_pubmed_ids", lambda query, **kwargs: ["1", "2"])
    monkeypatch.setattr(
        cache_runner,
        "fetch_pubmed_records",
        lambda ids, **kwargs: [_record(pmid) for pmid in ids],
    )
    cache_runner.run_fetch(cache_args)
    cache_runner.run_freeze(_cache_args(package, cache, "freeze"))
    cache_runner.run_audit(_cache_args(package, cache, "audit"))

    def fake_vote(item: BenchmarkItem, records: list, *, model_spec: str, role: str, **kwargs):
        return (
            LabelVote(
                benchmark_item_id=item.benchmark_item_id,
                model_spec=model_spec,
                role=role,
                proposed_label="candidate_unknown",
                construct_match="partial",
                confidence="low",
                rationale="Insufficient evidence.",
                prompt_sha256="a" * 64,
                response_sha256="b" * 64,
            ),
            {"prompt": "p", "raw_response": "{}", "trace": []},
        )

    monkeypatch.setattr(adjudication, "_vote", fake_vote)
    args = argparse.Namespace(
        out_dir=str(out),
        package_dir=str(package),
        assessor_model=["openai:gpt-5.5", "google:gemini-3.5-flash"],
        adjudicator_model="openrouter:anthropic/claude-opus-4.8",
        schema_retries=0,
        llm_max_tokens=1024,
        max_records_per_query=2,
        max_evidence_records=2,
        pubmed_cache_dir=str(cache),
        force=False,
    )
    result = adjudication._process_item(item, args)
    assert result["status"] == "completed"
    assert result["evidence_freeze"]["retrieval_backend"] == "pubmed_cache_exact"
    assert result["evidence_freeze"]["pubmed_cache_packet_sha256"]


def test_cache_miss_is_typed(tmp_path: Path):
    with pytest.raises(PubMedCacheMissError, match="pubmed_cache_miss"):
        load_cache_packet(
            tmp_path / "missing",
            benchmark_item_id="missing",
            exact_contract_sha256="x",
            package_dir=tmp_path,
            max_records_per_query=5,
            max_evidence_records=12,
            assessor_models=["openai:gpt-5.5", "google:gemini-3.5-flash"],
        )
