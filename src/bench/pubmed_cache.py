"""Typed local PubMed snapshot artifacts for NeuroClaimBench adjudication."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.neuroclaimbench_v21_compat import (
    EvidenceRecord,
    sha256_payload,
)

CACHE_VERSION = "pubmed-cache-v1"


class PubMedCacheMissError(RuntimeError):
    """Raised when adjudication cannot resolve a frozen cache packet."""

    error_code = "pubmed_cache_miss"


class QueryPlanRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: str
    exact_contract_sha256: str
    scientific_question_sha256: str = ""
    target_family: str
    modality: str
    package_sha256: str
    plan_fingerprint: str
    base_query: str
    ordered_queries: list[str]
    seed_pmids: list[str] = Field(default_factory=list)
    assessor_models: list[str]
    query_prompt_sha256: str
    query_generation: list[dict[str, Any]]


class PubMedQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_key: str
    database: Literal["pubmed"] = "pubmed"
    query: str
    retmax: int
    sort: Literal["relevance"] = "relevance"
    pmids: list[str]
    retrieved_at: str


class CachedPubMedArticle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    doi: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    retrieved_at: str


class ArticleFetchStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str
    status: Literal["available", "unavailable"]
    article: Optional[CachedPubMedArticle] = None
    fetched_at: str


class EvidencePacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: str
    exact_contract_sha256: str
    scientific_question_sha256: str = ""
    package_sha256: str
    plan_fingerprint: str
    ordered_queries: list[str]
    query_generation: list[dict[str, Any]]
    records: list[EvidenceRecord]
    packet_sha256: str


class PubMedCacheManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_version: Literal["pubmed-cache-v1"] = CACHE_VERSION
    status: Literal["audited"] = "audited"
    created_at: str
    package_sha256: str
    package_files: dict[str, str]
    assessor_models: list[str]
    model_routes: list[dict[str, Any]]
    query_prompt_sha256s: list[str]
    parameters: dict[str, Any]
    counts: dict[str, int]
    retrieval_started_at: Optional[str] = None
    retrieval_finished_at: Optional[str] = None
    output_files: dict[str, dict[str, str]]
    source_code_sha256: str
    interpretation_restrictions: list[str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_snapshot(package_dir: Path) -> tuple[str, dict[str, str]]:
    names = ("benchmark_items.jsonl", "benchmark_splits.json")
    hashes = {name: file_sha256(package_dir / name) for name in names}
    return sha256_payload(hashes), hashes


def normalized_query(query: str) -> str:
    return " ".join(query.split())


def query_key(query: str, *, retmax: int, sort: str = "relevance") -> str:
    return sha256_payload(
        {
            "database": "pubmed",
            "query": normalized_query(query),
            "retmax": retmax,
            "sort": sort,
        }
    )


def packet_payload_hash(records: list[EvidenceRecord]) -> str:
    return sha256_payload([record.model_dump(mode="json") for record in records])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise PubMedCacheMissError(f"pubmed_cache_miss: missing cache artifact {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(payload)
    return rows


@lru_cache(maxsize=4)
def _load_cache(cache_dir_text: str) -> tuple[PubMedCacheManifest, dict[str, EvidencePacket], str]:
    cache_dir = Path(cache_dir_text)
    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.exists():
        raise PubMedCacheMissError(f"pubmed_cache_miss: cache is not audited: {manifest_path}")
    manifest = PubMedCacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for name, metadata in manifest.output_files.items():
        path = cache_dir / name
        if not path.exists() or file_sha256(path) != metadata["sha256"]:
            raise PubMedCacheMissError(f"pubmed_cache_miss: cache hash mismatch for {path}")
    packets = [EvidencePacket.model_validate(row) for row in _read_jsonl(cache_dir / "evidence_packets.jsonl")]
    by_item = {packet.benchmark_item_id: packet for packet in packets}
    if len(by_item) != len(packets):
        raise ValueError("Duplicate benchmark_item_id values in evidence_packets.jsonl")
    if len(packets) != manifest.counts.get("evidence_packets"):
        raise PubMedCacheMissError("pubmed_cache_miss: packet count does not match the cache manifest")
    return manifest, by_item, file_sha256(manifest_path)


def load_cache_packet(
    cache_dir: Path,
    *,
    benchmark_item_id: str,
    exact_contract_sha256: str,
    scientific_question_sha256: str = "",
    package_dir: Path,
    max_records_per_query: int,
    max_evidence_records: int,
    assessor_models: list[str],
) -> tuple[EvidencePacket, PubMedCacheManifest, str]:
    manifest, packets, manifest_sha256 = _load_cache(str(cache_dir.resolve()))
    current_package_sha256, current_files = package_snapshot(package_dir)
    if current_package_sha256 != manifest.package_sha256 or current_files != manifest.package_files:
        raise PubMedCacheMissError("pubmed_cache_miss: benchmark package does not match the cache snapshot")
    if manifest.parameters.get("max_records_per_query") != max_records_per_query:
        raise PubMedCacheMissError("pubmed_cache_miss: max_records_per_query differs from the cache")
    if manifest.parameters.get("max_evidence_records") != max_evidence_records:
        raise PubMedCacheMissError("pubmed_cache_miss: max_evidence_records differs from the cache")
    if manifest.assessor_models != assessor_models:
        raise PubMedCacheMissError("pubmed_cache_miss: assessor models differ from the cache query plan")
    packet = packets.get(benchmark_item_id)
    if packet is None:
        raise PubMedCacheMissError(f"pubmed_cache_miss: no packet for {benchmark_item_id}")
    if packet.exact_contract_sha256 != exact_contract_sha256:
        raise PubMedCacheMissError(f"pubmed_cache_miss: contract mismatch for {benchmark_item_id}")
    if (
        scientific_question_sha256
        and packet.scientific_question_sha256
        and packet.scientific_question_sha256 != scientific_question_sha256
    ):
        raise PubMedCacheMissError(f"pubmed_cache_miss: scientific question mismatch for {benchmark_item_id}")
    if packet.package_sha256 != manifest.package_sha256:
        raise PubMedCacheMissError(f"pubmed_cache_miss: package mismatch for {benchmark_item_id}")
    if packet.packet_sha256 != packet_payload_hash(packet.records):
        raise PubMedCacheMissError(f"pubmed_cache_miss: packet hash mismatch for {benchmark_item_id}")
    return packet, manifest, manifest_sha256


def clear_cache_loader() -> None:
    """Clear the process-local cache after tests or a newly finalized snapshot."""

    _load_cache.cache_clear()
