"""Build a frozen local PubMed evidence cache for NeuroClaimBench."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from bench.io import read_jsonl
from bench.neuroclaimbench_v21_compat import (
    BenchmarkItem,
    EvidenceRecord,
    adjudication_claim_payload,
    sha256_payload,
)
from bench.progress import iter_progress
from bench.pubmed import fetch_pubmed_records, search_pubmed_ids
from bench.pubmed_cache import (
    CACHE_VERSION,
    ArticleFetchStatus,
    CachedPubMedArticle,
    EvidencePacket,
    PubMedCacheManifest,
    PubMedQueryResult,
    QueryPlanRow,
    clear_cache_loader,
    file_sha256,
    normalized_query,
    package_snapshot,
    packet_payload_hash,
    query_key,
)
from bench.run_neuroclaimbench_adjudication import DEFAULT_ASSESSORS
from confirm.llm import complete_structured_with_retries, make_llm

DEFAULT_PACKAGE = Path("data/neuroclaimbench/v2.1")
DEFAULT_CACHE = Path("data/neuroclaimbench/pubmed-cache-v2.1")
T = TypeVar("T")


class QueryExpansionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1, max_length=3)


def _base_query(item: BenchmarkItem) -> str:
    claim = adjudication_claim_payload(item)
    estimand = claim["estimand"]
    outcome = estimand["outcome"]
    outcome_text = " ".join(outcome) if isinstance(outcome, list) else str(outcome)
    outcome_text = outcome_text.replace("smri_", "").replace("fc_", "functional connectivity ").replace("_", " ")
    target_terms = {
        "normative_fmri": "(aging OR age OR sex OR cognition)",
        "adhd": "(ADHD OR attention deficit hyperactivity disorder)",
        "asd": "(autism OR autism spectrum disorder)",
        "ad_aging": "(Alzheimer disease OR dementia OR cognitive aging)",
        "psychosis": "(schizophrenia OR psychosis)",
    }.get(item.target_family, item.target_family.replace("_", " "))
    modality = "functional MRI" if "fmri" in item.modality.lower() else "MRI"
    return f"{target_terms} AND ({outcome_text}) AND ({modality})"


def build_query_prompt(item: BenchmarkItem) -> str:
    return (
        "Generate up to three concise PubMed query expansions for the scientific claim below. "
        "Preserve the population, predictor or group contrast, outcome construct, modality, and direction. "
        "Do not mention CONFIRM, gate outcomes, p-values, or validation datasets. Return structured JSON only.\n\n"
        + json.dumps(
            {
                "claim": adjudication_claim_payload(item),
                "base_query": _base_query(item),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _query_expansions(item: BenchmarkItem, model_spec: str, retries: int) -> tuple[list[str], dict[str, Any]]:
    llm = make_llm(model_spec)
    prompt = build_query_prompt(item)
    parsed, raw, attempts, trace = complete_structured_with_retries(
        llm,
        system="You generate reproducible PubMed queries for neuroimaging evidence retrieval.",
        prompt=prompt,
        response_model=QueryExpansionResponse,
        retries=retries,
    )
    response = QueryExpansionResponse.model_validate(parsed.model_dump(mode="json"))
    queries: list[str] = []
    for query in response.queries:
        cleaned = normalized_query(query)
        if cleaned and cleaned not in queries:
            queries.append(cleaned)
    return queries[:3], {
        "model_spec": model_spec,
        "prompt": prompt,
        "raw_response": raw,
        "schema_attempts": attempts,
        "trace": trace,
    }


def _seed_pmids(item: BenchmarkItem) -> list[str]:
    pmids: list[str] = []
    for reference in item.source_references:
        for match in re.findall(r"(?:pmid\s*[:=]?\s*)?(\d{7,9})", reference.source_citation, flags=re.IGNORECASE):
            if match not in pmids:
                pmids.append(match)
    return pmids


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    payloads = [row.model_dump(mode="json") if hasattr(row, "model_dump") else row for row in rows]
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in payloads))


def _load_items(package_dir: Path) -> tuple[list[BenchmarkItem], list[str]]:
    items = [
        BenchmarkItem.model_validate(row)
        for row in read_jsonl(package_dir / "benchmark_items.jsonl")
    ]
    split = json.loads((package_dir / "benchmark_splits.json").read_text(encoding="utf-8"))
    candidate_ids = [str(value) for value in split["adjudication_candidates"]]
    by_id = {item.benchmark_item_id: item for item in items}
    missing = [item_id for item_id in candidate_ids if item_id not in by_id]
    if missing:
        raise ValueError(f"Benchmark split references missing items: {missing[:5]}")
    selected = [by_id[item_id] for item_id in candidate_ids]
    if any(item.contract is None or not item.exact_contract_sha256 for item in selected):
        raise ValueError("All adjudication candidates must have frozen contracts")
    return selected, candidate_ids


def _ensure_not_finalized(cache_dir: Path) -> None:
    if (cache_dir / "cache_manifest.json").exists():
        raise ValueError(f"PubMed cache is immutable after audit: {cache_dir}; use a new versioned directory")


def _plan_fingerprint(
    item: BenchmarkItem,
    *,
    package_sha256: str,
    assessor_models: list[str],
    prompt_sha256: str,
) -> str:
    return sha256_payload(
        {
            "cache_version": CACHE_VERSION,
            "benchmark_item_id": item.benchmark_item_id,
            "exact_contract_sha256": item.exact_contract_sha256,
            "scientific_question_sha256": item.scientific_question_sha256,
            "package_sha256": package_sha256,
            "assessor_models": assessor_models,
            "query_prompt_sha256": prompt_sha256,
        }
    )


def _build_plan_item(
    item: BenchmarkItem,
    *,
    package_sha256: str,
    assessor_models: list[str],
    schema_retries: int,
    cache_dir: Path,
) -> QueryPlanRow:
    prompt = build_query_prompt(item)
    prompt_sha256 = sha256_payload(prompt)
    fingerprint = _plan_fingerprint(
        item,
        package_sha256=package_sha256,
        assessor_models=assessor_models,
        prompt_sha256=prompt_sha256,
    )
    checkpoint = cache_dir / ".work" / "query_plans" / f"{item.benchmark_item_id}.json"
    if checkpoint.exists():
        row = QueryPlanRow.model_validate_json(checkpoint.read_text(encoding="utf-8"))
        if row.plan_fingerprint != fingerprint:
            raise ValueError(f"Stale query-plan checkpoint: {checkpoint}")
        return row

    generation: list[dict[str, Any]] = []
    expansions = []
    for model_spec in assessor_models:
        model_queries, trace = _query_expansions(item, model_spec, schema_retries)
        generation.append(trace)
        for query in model_queries:
            cleaned = normalized_query(query)
            if cleaned and cleaned not in expansions:
                expansions.append(cleaned)
    ordered = [normalized_query(_base_query(item))]
    ordered.extend(query for query in expansions if query not in ordered)
    row = QueryPlanRow(
        benchmark_item_id=item.benchmark_item_id,
        exact_contract_sha256=str(item.exact_contract_sha256),
        scientific_question_sha256=item.scientific_question_sha256,
        target_family=item.target_family,
        modality=item.modality,
        package_sha256=package_sha256,
        plan_fingerprint=fingerprint,
        base_query=ordered[0],
        ordered_queries=ordered,
        seed_pmids=_seed_pmids(item),
        assessor_models=assessor_models,
        query_prompt_sha256=prompt_sha256,
        query_generation=generation,
    )
    _atomic_text(checkpoint, row.model_dump_json(indent=2) + "\n")
    return row


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    _ensure_not_finalized(cache_dir)
    package_dir = Path(args.package_dir)
    items, candidate_ids = _load_items(package_dir)
    package_sha256, package_files = package_snapshot(package_dir)
    assessor_models = list(args.assessor_model or DEFAULT_ASSESSORS)
    results: dict[str, QueryPlanRow] = {}

    def work(item: BenchmarkItem) -> QueryPlanRow:
        return _build_plan_item(
            item,
            package_sha256=package_sha256,
            assessor_models=assessor_models,
            schema_retries=args.schema_retries,
            cache_dir=cache_dir,
        )

    if args.max_workers <= 1:
        for item in iter_progress(items, total=len(items), desc="PubMed query plan", enabled=not args.no_progress, unit="claim"):
            row = work(item)
            results[item.benchmark_item_id] = row
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(work, item): item.benchmark_item_id for item in items}
            for future in iter_progress(
                as_completed(futures), total=len(futures), desc="PubMed query plan", enabled=not args.no_progress, unit="claim"
            ):
                row = future.result()
                results[row.benchmark_item_id] = row
    ordered_rows = [results[item_id] for item_id in candidate_ids]
    _write_jsonl(cache_dir / "query_plan.jsonl", ordered_rows)
    state = {
        "cache_version": CACHE_VERSION,
        "package_sha256": package_sha256,
        "package_files": package_files,
        "assessor_models": assessor_models,
        "max_records_per_query": args.max_records_per_query,
        "max_evidence_records": args.max_evidence_records,
        "planned_at": _utc_now(),
    }
    _atomic_text(cache_dir / ".work" / "plan_state.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    return {
        "phase": "plan",
        "candidate_count": len(ordered_rows),
        "planned_count": len(ordered_rows),
    }


class RequestRateLimiter:
    def __init__(self, requests_per_second: float, *, clock: Callable[[], float] = time.monotonic, sleeper: Callable[[float], None] = time.sleep) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self.clock = clock
        self.sleeper = sleeper
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        with self.lock:
            now = self.clock()
            delay = max(0.0, self.next_allowed - now)
            if delay:
                self.sleeper(delay)
                now = self.clock()
            self.next_allowed = max(now, self.next_allowed) + self.interval


def _retry_request(
    operation: Callable[[], T],
    *,
    limiter: RequestRateLimiter,
    retries: int,
    retry_delay: float,
) -> T:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            limiter.wait()
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_delay * (2**attempt))
    assert last_error is not None
    raise last_error


def _load_work_query_results(cache_dir: Path) -> dict[str, PubMedQueryResult]:
    rows: dict[str, PubMedQueryResult] = {}
    for path in sorted((cache_dir / ".work" / "query_results").glob("*.json")):
        row = PubMedQueryResult.model_validate_json(path.read_text(encoding="utf-8"))
        rows[row.query_key] = row
    return rows


def _load_work_article_statuses(cache_dir: Path) -> dict[str, ArticleFetchStatus]:
    rows: dict[str, ArticleFetchStatus] = {}
    for path in sorted((cache_dir / ".work" / "article_status").glob("*.json")):
        row = ArticleFetchStatus.model_validate_json(path.read_text(encoding="utf-8"))
        rows[row.pmid] = row
    return rows


def _selected_articles(
    plan: QueryPlanRow,
    query_results: dict[str, PubMedQueryResult],
    statuses: dict[str, ArticleFetchStatus],
    *,
    retmax: int,
    limit: int,
) -> list[tuple[CachedPubMedArticle, str]]:
    selected: list[tuple[CachedPubMedArticle, str]] = []
    seen: set[str] = set()

    def add(pmids: Iterable[str], source_query: str) -> None:
        for pmid in pmids:
            status = statuses.get(pmid)
            if pmid in seen or status is None or status.status != "available" or status.article is None:
                continue
            seen.add(pmid)
            selected.append((status.article, source_query))
            if len(selected) >= limit:
                return

    add(plan.seed_pmids, "frozen_source_citation")
    for query in plan.ordered_queries:
        if len(selected) >= limit:
            break
        result = query_results.get(query_key(query, retmax=retmax))
        if result is None:
            break
        add(result.pmids, query)
    return selected[:limit]


def _records_for_plan(
    plan: QueryPlanRow,
    query_results: dict[str, PubMedQueryResult],
    statuses: dict[str, ArticleFetchStatus],
    *,
    retmax: int,
    limit: int,
) -> list[EvidenceRecord]:
    selected = _selected_articles(plan, query_results, statuses, retmax=retmax, limit=limit)
    return [
        EvidenceRecord(
            evidence_id=f"ncb-evidence-{plan.benchmark_item_id[-12:]}-{article.pmid}",
            benchmark_item_id=plan.benchmark_item_id,
            scientific_question_sha256=plan.scientific_question_sha256,
            pmid=article.pmid,
            doi=article.doi,
            title=article.title,
            abstract=article.abstract,
            journal=article.journal,
            year=article.year,
            query=source_query,
            target_family=plan.target_family,
            retrieved_at=article.retrieved_at,
        )
        for article, source_query in selected
    ]


def _fetch_query_batch(
    queries: list[str],
    *,
    args: argparse.Namespace,
    limiter: RequestRateLimiter,
) -> tuple[list[PubMedQueryResult], list[dict[str, Any]]]:
    email = args.pubmed_email or os.getenv("NCBI_EMAIL", "")
    api_key = args.pubmed_api_key or os.getenv("NCBI_API_KEY", "")

    def work(query: str) -> PubMedQueryResult:
        pmids = _retry_request(
            lambda: search_pubmed_ids(
                query,
                max_records=args.max_records_per_query,
                email=email,
                api_key=api_key,
                timeout=args.pubmed_timeout,
            ),
            limiter=limiter,
            retries=args.request_retries,
            retry_delay=args.retry_delay,
        )
        return PubMedQueryResult(
            query_key=query_key(query, retmax=args.max_records_per_query),
            query=normalized_query(query),
            retmax=args.max_records_per_query,
            pmids=pmids,
            retrieved_at=_utc_now(),
        )

    successes: list[PubMedQueryResult] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(work, query): query for query in queries}
        for future in iter_progress(
            as_completed(futures), total=len(futures), desc="PubMed searches", enabled=not args.no_progress, unit="query"
        ):
            query = futures[future]
            try:
                successes.append(future.result())
            except Exception as exc:
                failures.append({"kind": "query", "query": query, "error": str(exc), "failed_at": _utc_now()})
    return successes, failures


def _fetch_article_batches(
    pmids: list[str],
    *,
    args: argparse.Namespace,
    limiter: RequestRateLimiter,
) -> tuple[list[ArticleFetchStatus], list[dict[str, Any]]]:
    email = args.pubmed_email or os.getenv("NCBI_EMAIL", "")
    api_key = args.pubmed_api_key or os.getenv("NCBI_API_KEY", "")
    chunks = [pmids[index : index + args.fetch_batch_size] for index in range(0, len(pmids), args.fetch_batch_size)]

    def work(chunk: list[str]) -> list[ArticleFetchStatus]:
        fetched_at = _utc_now()
        records = _retry_request(
            lambda: fetch_pubmed_records(
                chunk,
                query="pubmed_cache_batch",
                target_family="cache",
                modality="metadata_abstract",
                email=email,
                api_key=api_key,
                timeout=args.pubmed_timeout,
                retrieved_at=fetched_at,
            ),
            limiter=limiter,
            retries=args.request_retries,
            retry_delay=args.retry_delay,
        )
        by_pmid = {record.pmid: record for record in records}
        statuses: list[ArticleFetchStatus] = []
        for pmid in chunk:
            record = by_pmid.get(pmid)
            article = None
            if record is not None:
                article = CachedPubMedArticle(
                    pmid=record.pmid,
                    title=record.title,
                    abstract=record.abstract,
                    journal=record.journal,
                    year=record.year,
                    doi=record.doi,
                    mesh_terms=record.mesh_terms,
                    retrieved_at=record.retrieved_at,
                )
            statuses.append(
                ArticleFetchStatus(
                    pmid=pmid,
                    status="available" if article is not None else "unavailable",
                    article=article,
                    fetched_at=fetched_at,
                )
            )
        return statuses

    statuses: list[ArticleFetchStatus] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(work, chunk): chunk for chunk in chunks}
        for future in iter_progress(
            as_completed(futures), total=len(futures), desc="PubMed article batches", enabled=not args.no_progress, unit="batch"
        ):
            chunk = futures[future]
            try:
                statuses.extend(future.result())
            except Exception as exc:
                failures.append({"kind": "article_batch", "pmids": chunk, "error": str(exc), "failed_at": _utc_now()})
    return statuses, failures


def run_fetch(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    _ensure_not_finalized(cache_dir)
    plans = [
        QueryPlanRow.model_validate(row)
        for row in read_jsonl(cache_dir / "query_plan.jsonl")
    ]
    plan_state = json.loads((cache_dir / ".work" / "plan_state.json").read_text(encoding="utf-8"))
    if plan_state.get("max_records_per_query") != args.max_records_per_query:
        raise ValueError("Fetch max_records_per_query does not match the query-plan state")
    if plan_state.get("max_evidence_records") != args.max_evidence_records:
        raise ValueError("Fetch max_evidence_records does not match the query-plan state")
    rate = 10.0 if (args.pubmed_api_key or os.getenv("NCBI_API_KEY")) else 3.0
    limiter = RequestRateLimiter(rate)
    query_results = _load_work_query_results(cache_dir)
    statuses = _load_work_article_statuses(cache_dir)
    reused_query_count = 0
    reused_article_count = 0
    reuse_cache_from = getattr(args, "reuse_cache_from", None)
    if reuse_cache_from:
        reuse_dir = Path(reuse_cache_from)
        manifest_path = reuse_dir / "cache_manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Reuse cache is not audited: {reuse_dir}")
        manifest = PubMedCacheManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        reuse_manifest_sha256 = file_sha256(manifest_path)
        for name, metadata in manifest.output_files.items():
            if file_sha256(reuse_dir / name) != metadata["sha256"]:
                raise ValueError(f"Reuse cache hash mismatch: {reuse_dir / name}")
        current_query_keys = {
            query_key(query, retmax=args.max_records_per_query)
            for plan in plans
            for query in plan.ordered_queries
        }
        for payload in read_jsonl(reuse_dir / "query_results.jsonl"):
            row = PubMedQueryResult.model_validate(payload)
            if row.query_key not in current_query_keys or row.query_key in query_results:
                continue
            query_results[row.query_key] = row
            _atomic_text(
                cache_dir / ".work" / "query_results" / f"{row.query_key}.json",
                row.model_dump_json(indent=2) + "\n",
            )
            reused_query_count += 1
        required_pmids = {
            pmid
            for result in query_results.values()
            for pmid in result.pmids
        } | {pmid for plan in plans for pmid in plan.seed_pmids}
        for payload in read_jsonl(reuse_dir / "articles.jsonl"):
            article = CachedPubMedArticle.model_validate(payload)
            if article.pmid not in required_pmids or article.pmid in statuses:
                continue
            status = ArticleFetchStatus(
                pmid=article.pmid,
                status="available",
                article=article,
                fetched_at=article.retrieved_at,
            )
            statuses[article.pmid] = status
            _atomic_text(
                cache_dir / ".work" / "article_status" / f"{article.pmid}.json",
                status.model_dump_json(indent=2) + "\n",
            )
            reused_article_count += 1
    failures: list[dict[str, Any]] = []
    fetch_state_path = cache_dir / ".work" / "fetch_state.json"
    if fetch_state_path.exists():
        fetch_state = json.loads(fetch_state_path.read_text(encoding="utf-8"))
    else:
        fetch_state = {"retrieval_started_at": _utc_now()}
        _atomic_text(fetch_state_path, json.dumps(fetch_state, indent=2, sort_keys=True) + "\n")
    fetch_state["reuse_cache_from"] = reuse_cache_from or None
    fetch_state["reuse_cache_manifest_sha256"] = (
        reuse_manifest_sha256 if reuse_cache_from else None
    )
    fetch_state["reused_query_result_count"] = reused_query_count
    fetch_state["reused_article_count"] = reused_article_count

    def persist_status(rows: list[ArticleFetchStatus]) -> None:
        for row in rows:
            statuses[row.pmid] = row
            path = cache_dir / ".work" / "article_status" / f"{row.pmid}.json"
            _atomic_text(path, row.model_dump_json(indent=2) + "\n")

    seed_pmids = sorted({pmid for plan in plans for pmid in plan.seed_pmids if pmid not in statuses})
    if seed_pmids:
        rows, errors = _fetch_article_batches(seed_pmids, args=args, limiter=limiter)
        persist_status(rows)
        failures.extend(errors)
        if errors:
            _write_jsonl(cache_dir / "fetch_failures.jsonl", failures)
            raise RuntimeError(f"PubMed cache fetch has {len(failures)} failures; rerun PHASE=fetch to resume")

    cached_result_pmids = sorted(
        {pmid for result in query_results.values() for pmid in result.pmids if pmid not in statuses}
    )
    if cached_result_pmids:
        rows, errors = _fetch_article_batches(cached_result_pmids, args=args, limiter=limiter)
        persist_status(rows)
        failures.extend(errors)
        if errors:
            _write_jsonl(cache_dir / "fetch_failures.jsonl", failures)
            raise RuntimeError(f"PubMed cache fetch has {len(failures)} failures; rerun PHASE=fetch to resume")

    max_queries = max((len(plan.ordered_queries) for plan in plans), default=0)
    for wave in range(max_queries):
        needed: dict[str, str] = {}
        for plan in plans:
            selected = _selected_articles(
                plan,
                query_results,
                statuses,
                retmax=args.max_records_per_query,
                limit=args.max_evidence_records,
            )
            if len(selected) >= args.max_evidence_records or wave >= len(plan.ordered_queries):
                continue
            query = plan.ordered_queries[wave]
            key = query_key(query, retmax=args.max_records_per_query)
            if key not in query_results:
                needed[key] = query
        if not needed:
            continue
        rows, errors = _fetch_query_batch(list(needed.values()), args=args, limiter=limiter)
        wave_failed = bool(errors)
        failures.extend(errors)
        for row in rows:
            query_results[row.query_key] = row
            path = cache_dir / ".work" / "query_results" / f"{row.query_key}.json"
            _atomic_text(path, row.model_dump_json(indent=2) + "\n")
        missing_pmids = sorted({pmid for row in rows for pmid in row.pmids if pmid not in statuses})
        if missing_pmids:
            article_rows, article_errors = _fetch_article_batches(missing_pmids, args=args, limiter=limiter)
            persist_status(article_rows)
            failures.extend(article_errors)
            wave_failed = wave_failed or bool(article_errors)
        if wave_failed:
            break

    fetch_state["retrieval_finished_at"] = _utc_now()
    _atomic_text(fetch_state_path, json.dumps(fetch_state, indent=2, sort_keys=True) + "\n")
    _write_jsonl(cache_dir / "query_results.jsonl", sorted(query_results.values(), key=lambda row: row.query_key))
    articles = sorted(
        (status.article for status in statuses.values() if status.status == "available" and status.article is not None),
        key=lambda row: row.pmid,
    )
    _write_jsonl(cache_dir / "articles.jsonl", articles)
    _write_jsonl(cache_dir / "fetch_failures.jsonl", failures)
    if failures:
        raise RuntimeError(f"PubMed cache fetch has {len(failures)} failures; rerun PHASE=fetch to resume")
    return {
        "phase": "fetch",
        "query_result_count": len(query_results),
        "article_count": len(articles),
        "unavailable_pmid_count": sum(status.status == "unavailable" for status in statuses.values()),
        "reused_query_result_count": reused_query_count,
        "reused_article_count": reused_article_count,
        "package_sha256": plan_state["package_sha256"],
    }


def run_freeze(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    _ensure_not_finalized(cache_dir)
    plans = [
        QueryPlanRow.model_validate(row)
        for row in read_jsonl(cache_dir / "query_plan.jsonl")
    ]
    plan_state = json.loads((cache_dir / ".work" / "plan_state.json").read_text(encoding="utf-8"))
    if plan_state.get("max_records_per_query") != args.max_records_per_query:
        raise ValueError("Freeze max_records_per_query does not match the query-plan state")
    if plan_state.get("max_evidence_records") != args.max_evidence_records:
        raise ValueError("Freeze max_evidence_records does not match the query-plan state")
    failures = read_jsonl(cache_dir / "fetch_failures.jsonl")
    if failures:
        raise ValueError(f"Cannot freeze cache with {len(failures)} fetch failures")
    query_results = {
        row.query_key: row
        for row in (
            PubMedQueryResult.model_validate(value)
            for value in read_jsonl(cache_dir / "query_results.jsonl")
        )
    }
    statuses = _load_work_article_statuses(cache_dir)
    packets: list[EvidencePacket] = []
    for plan in iter_progress(plans, total=len(plans), desc="Freeze evidence packets", enabled=not args.no_progress, unit="claim"):
        records = _records_for_plan(
            plan,
            query_results,
            statuses,
            retmax=args.max_records_per_query,
            limit=args.max_evidence_records,
        )
        packets.append(
            EvidencePacket(
                benchmark_item_id=plan.benchmark_item_id,
                exact_contract_sha256=plan.exact_contract_sha256,
                scientific_question_sha256=plan.scientific_question_sha256,
                package_sha256=plan.package_sha256,
                plan_fingerprint=plan.plan_fingerprint,
                ordered_queries=plan.ordered_queries,
                query_generation=plan.query_generation,
                records=records,
                packet_sha256=packet_payload_hash(records),
            )
        )
    _write_jsonl(cache_dir / "evidence_packets.jsonl", packets)
    summary = {
        "phase": "freeze",
        "packet_count": len(packets),
        "record_count": sum(len(packet.records) for packet in packets),
        "full_packet_count": sum(len(packet.records) == args.max_evidence_records for packet in packets),
        "short_packet_count": sum(len(packet.records) < args.max_evidence_records for packet in packets),
    }
    _atomic_text(cache_dir / ".work" / "freeze_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _source_code_hash() -> str:
    root = Path(__file__).resolve().parent
    names = ("pubmed_cache.py", "run_neuroclaimbench_pubmed_cache.py", "run_neuroclaimbench_adjudication.py")
    return sha256_payload({name: file_sha256(root / name) for name in names})


def _model_routes(plans: list[QueryPlanRow]) -> list[dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for plan in plans:
        for generation in plan.query_generation:
            model_spec = str(generation.get("model_spec") or "")
            for attempt in generation.get("trace") or []:
                metadata = attempt.get("call_metadata") or {}
                route = {
                    "model_spec": model_spec,
                    "provider": metadata.get("provider"),
                    "reported_model": metadata.get("model"),
                    "routed_provider": metadata.get("routed_provider"),
                }
                routes[sha256_payload(route)] = route
    return [routes[key] for key in sorted(routes)]


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    manifest_path = cache_dir / "cache_manifest.json"
    if manifest_path.exists():
        manifest = PubMedCacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        for name, metadata in manifest.output_files.items():
            if file_sha256(cache_dir / name) != metadata["sha256"]:
                raise ValueError(f"Finalized PubMed cache hash mismatch: {name}")
        return {"phase": "audit", "status": manifest.status, "counts": manifest.counts, "already_finalized": True}
    package_dir = Path(args.package_dir)
    items, candidate_ids = _load_items(package_dir)
    package_sha256, package_files = package_snapshot(package_dir)
    plans = [
        QueryPlanRow.model_validate(row)
        for row in read_jsonl(cache_dir / "query_plan.jsonl")
    ]
    query_results = {
        row.query_key: row
        for row in (
            PubMedQueryResult.model_validate(value)
            for value in read_jsonl(cache_dir / "query_results.jsonl")
        )
    }
    articles = [
        CachedPubMedArticle.model_validate(row)
        for row in read_jsonl(cache_dir / "articles.jsonl")
    ]
    packets = [
        EvidencePacket.model_validate(row)
        for row in read_jsonl(cache_dir / "evidence_packets.jsonl")
    ]
    failures = read_jsonl(cache_dir / "fetch_failures.jsonl")
    statuses = _load_work_article_statuses(cache_dir)
    if failures:
        raise ValueError(f"Cannot audit cache with {len(failures)} fetch failures")
    if [plan.benchmark_item_id for plan in plans] != candidate_ids:
        raise ValueError("Query plan does not exactly match benchmark adjudication candidates")
    if [packet.benchmark_item_id for packet in packets] != candidate_ids:
        raise ValueError("Evidence packets do not exactly match benchmark adjudication candidates")
    item_by_id = {item.benchmark_item_id: item for item in items}
    if any(plan.package_sha256 != package_sha256 for plan in plans):
        raise ValueError("Query plan package hash mismatch")
    configured_models = list(args.assessor_model or DEFAULT_ASSESSORS)
    if any(plan.assessor_models != configured_models for plan in plans):
        raise ValueError("Query plan assessor models do not match the audit configuration")
    if any(plan.exact_contract_sha256 != item_by_id[plan.benchmark_item_id].exact_contract_sha256 for plan in plans):
        raise ValueError("Query plan contract hash mismatch")
    if any(
        plan.scientific_question_sha256
        and plan.scientific_question_sha256
        != item_by_id[plan.benchmark_item_id].scientific_question_sha256
        for plan in plans
    ):
        raise ValueError("Query plan scientific-question hash mismatch")
    for result in query_results.values():
        if result.query_key != query_key(result.query, retmax=result.retmax, sort=result.sort):
            raise ValueError(f"Query-result key mismatch: {result.query_key}")
        if result.retmax != args.max_records_per_query:
            raise ValueError(f"Query-result retmax mismatch: {result.query_key}")
    plan_by_id = {plan.benchmark_item_id: plan for plan in plans}
    for packet in packets:
        if packet.packet_sha256 != packet_payload_hash(packet.records):
            raise ValueError(f"Evidence packet hash mismatch: {packet.benchmark_item_id}")
        rebuilt = _records_for_plan(
            plan_by_id[packet.benchmark_item_id],
            query_results,
            statuses,
            retmax=args.max_records_per_query,
            limit=args.max_evidence_records,
        )
        if packet.records != rebuilt:
            raise ValueError(f"Evidence packet does not match deterministic reconstruction: {packet.benchmark_item_id}")
        if len(packet.records) < args.max_evidence_records:
            plan = plan_by_id[packet.benchmark_item_id]
            if any(query_key(query, retmax=args.max_records_per_query) not in query_results for query in plan.ordered_queries):
                raise ValueError(f"Short packet has unqueried planned searches: {packet.benchmark_item_id}")
    required_pmids = {
        pmid
        for plan in plans
        for pmid in plan.seed_pmids
    } | {pmid for result in query_results.values() for pmid in result.pmids}
    missing_statuses = sorted(required_pmids - set(statuses))
    if missing_statuses:
        raise ValueError(f"PubMed article fetch status is missing for {len(missing_statuses)} PMIDs")
    canonical_articles = {article.pmid: article for article in articles}
    work_articles = {
        pmid: status.article
        for pmid, status in statuses.items()
        if status.status == "available" and status.article is not None
    }
    if canonical_articles != work_articles:
        raise ValueError("Canonical articles.jsonl does not match resumable article checkpoints")
    output_names = ("query_plan.jsonl", "query_results.jsonl", "articles.jsonl", "evidence_packets.jsonl", "fetch_failures.jsonl")
    fetch_state = json.loads((cache_dir / ".work" / "fetch_state.json").read_text(encoding="utf-8"))
    manifest = PubMedCacheManifest(
        created_at=_utc_now(),
        package_sha256=package_sha256,
        package_files=package_files,
        assessor_models=configured_models,
        model_routes=_model_routes(plans),
        query_prompt_sha256s=sorted({plan.query_prompt_sha256 for plan in plans}),
        parameters={
            "database": "pubmed",
            "sort": "relevance",
            "max_records_per_query": args.max_records_per_query,
            "max_evidence_records": args.max_evidence_records,
            "content_scope": "metadata_and_abstracts",
            "fetch_batch_size": args.fetch_batch_size,
            "request_retries": args.request_retries,
            "rate_limit_without_api_key": 3,
            "rate_limit_with_api_key": 10,
            "reuse_cache_from": fetch_state.get("reuse_cache_from"),
            "reuse_cache_manifest_sha256": fetch_state.get("reuse_cache_manifest_sha256"),
            "reused_query_result_count": fetch_state.get("reused_query_result_count", 0),
            "reused_article_count": fetch_state.get("reused_article_count", 0),
        },
        counts={
            "adjudication_candidates": len(candidate_ids),
            "query_plans": len(plans),
            "unique_query_results": len(query_results),
            "unique_articles": len(articles),
            "evidence_packets": len(packets),
            "evidence_records": sum(len(packet.records) for packet in packets),
            "short_packets": sum(len(packet.records) < args.max_evidence_records for packet in packets),
            "fetch_failures": 0,
        },
        retrieval_started_at=fetch_state.get("retrieval_started_at"),
        retrieval_finished_at=fetch_state.get("retrieval_finished_at"),
        output_files={name: {"path": str(cache_dir / name), "sha256": file_sha256(cache_dir / name)} for name in output_names},
        source_code_sha256=_source_code_hash(),
        interpretation_restrictions=[
            "cache contains PubMed metadata and abstracts only",
            "query results use the frozen PubMed relevance ordering",
            "adjudication requires this audited cache and performs no PubMed retrieval",
        ],
    )
    _atomic_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    clear_cache_loader()
    return {"phase": "audit", "status": "audited", "counts": manifest.counts, "already_finalized": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["plan", "fetch", "freeze", "audit"])
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--assessor-model", action="append", default=None)
    parser.add_argument("--schema-retries", type=int, default=3)
    parser.add_argument("--max-records-per-query", type=int, default=5)
    parser.add_argument("--max-evidence-records", type=int, default=12)
    parser.add_argument("--pubmed-email", default="")
    parser.add_argument("--pubmed-api-key", default="")
    parser.add_argument("--pubmed-timeout", type=float, default=30.0)
    parser.add_argument("--request-retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--fetch-batch-size", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--reuse-cache-from")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.assessor_model is None:
        args.assessor_model = list(DEFAULT_ASSESSORS)
    if len(args.assessor_model) != 2:
        raise ValueError("Exactly two query-expansion assessor models are required")
    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if args.phase == "plan":
        return run_plan(args)
    if args.phase == "fetch":
        return run_fetch(args)
    if args.phase == "freeze":
        return run_freeze(args)
    return run_audit(args)


def main(argv: list[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps({"status": "completed", "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
