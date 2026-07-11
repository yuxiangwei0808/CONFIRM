"""Build PubMed-grounded initial claim questions for CONFIRM Stage 1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from bench.progress import iter_progress
from bench.run_initial_claim_drafting import (
    DEFAULT_DATA_ROOTS,
    DEFAULT_MODEL,
    DEFAULT_TARGET_FAMILIES,
    TARGET_GUIDANCE,
    _complete_structured,
    _json_safe,
    _make_llm,
    _merge_catalogs,
    _parse_json_model,
    _safe_id,
    _write_jsonl,
)
from confirm.derived_columns import CONFIRM_DX, columns_with_virtuals, confirm_dx_levels
from confirm.llm import LLMClient

DEFAULT_OUT_DIR = Path("review-stage/literature-grounding-gpt55")
DEFAULT_CLAIMS_OUT = Path("data/claims/literature_grounded_claims.csv")


class PubMedRecord(BaseModel):
    """A retrieved PubMed abstract with provenance."""

    model_config = ConfigDict(extra="forbid")

    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    doi: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    query: str
    target_family: str
    modality: str
    retrieved_at: str


class LiteratureClaimSeed(BaseModel):
    """One paper-derived scientific claim seed before local feasibility filtering."""

    model_config = ConfigDict(extra="forbid")

    seed_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    source_pmid: str = ""
    source_doi: str = ""
    source_title: str = ""
    source_year: str = ""
    target_family: Literal["normative_fmri", "adhd", "asd", "ad_aging", "psychosis"]
    outcome_modality: str
    predictor_or_group: str
    outcome_family: str
    expected_direction: Literal["positive", "negative", "two_sided", "mixed", "unknown"]
    covariates: list[str] = Field(default_factory=list)
    candidate_question: str
    evidence_snippet: str
    support_level: Literal["direct", "indirect", "speculative"]
    rationale: str


class LiteratureClaimExtractionResponse(BaseModel):
    """Structured LLM response for claim extraction from one PubMed record."""

    model_config = ConfigDict(extra="forbid")

    claims: list[LiteratureClaimSeed]


class FeasibilityResult(BaseModel):
    """Deterministic result describing whether a seed is locally executable."""

    model_config = ConfigDict(extra="forbid")

    seed_id: str
    pmid: str
    target_family: str
    status: Literal["executable_now", "requires_new_data", "requires_new_feature_adapter", "ambiguous_or_unsupported"]
    reason: str
    discovery_cohort: Optional[str] = None
    replication_cohort: Optional[str] = None
    matched_outcome_examples: list[str] = Field(default_factory=list)
    missing_covariates: list[str] = Field(default_factory=list)
    rejected_pair_reasons: list[str] = Field(default_factory=list)


DEFAULT_PUBMED_QUERIES: dict[str, list[dict[str, str]]] = {
    "normative_fmri": [
        {
            "modality": "fMRI-FC",
            "query": '("functional connectivity" OR fMRI) AND (age OR aging OR sex OR cognition) AND neuroimaging',
        }
    ],
    "adhd": [
        {
            "modality": "fMRI-FC",
            "query": 'ADHD AND ("functional connectivity" OR fMRI) AND ("case control" OR controls)',
        }
    ],
    "asd": [
        {
            "modality": "fMRI-FC",
            "query": 'autism AND ("functional connectivity" OR fMRI) AND ("case control" OR controls)',
        }
    ],
    "ad_aging": [
        {
            "modality": "sMRI",
            "query": '("Alzheimer disease" OR dementia OR aging) AND (hippocampal OR entorhinal OR "brain atrophy") AND MRI',
        }
    ],
    "psychosis": [
        {
            "modality": "fMRI-FC",
            "query": '(schizophrenia OR psychosis) AND ("functional connectivity" OR fMRI) AND controls',
        }
    ],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def _pubmed_url(endpoint: str, params: dict[str, str]) -> str:
    return f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}?{urllib.parse.urlencode(params)}"


def _urlopen_text(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _pubmed_ids(query: str, *, max_records: int, email: str, api_key: str, timeout: float) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_records),
        "sort": "relevance",
        "tool": "confirm_claim_grounding",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    data = json.loads(_urlopen_text(_pubmed_url("esearch.fcgi", params), timeout=timeout))
    return [str(item) for item in data.get("esearchresult", {}).get("idlist", [])]


def _fetch_pubmed_records(
    ids: list[str],
    *,
    query: str,
    target_family: str,
    modality: str,
    email: str,
    api_key: str,
    timeout: float,
    retrieved_at: str,
) -> list[PubMedRecord]:
    if not ids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "tool": "confirm_claim_grounding",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    xml_text = _urlopen_text(_pubmed_url("efetch.fcgi", params), timeout=timeout)
    return _parse_pubmed_xml(xml_text, query=query, target_family=target_family, modality=modality, retrieved_at=retrieved_at)


def _parse_pubmed_xml(
    xml_text: str,
    *,
    query: str,
    target_family: str,
    modality: str,
    retrieved_at: str,
) -> list[PubMedRecord]:
    root = ET.fromstring(xml_text)
    records: list[PubMedRecord] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        title = _text(article.find(".//ArticleTitle"))
        abstract = " ".join(
            part for part in (_text(node) for node in article.findall(".//Abstract/AbstractText")) if part
        )
        if not abstract:
            continue
        doi = ""
        for item in article.findall(".//ArticleIdList/ArticleId"):
            if item.attrib.get("IdType") == "doi":
                doi = _text(item)
                break
        year = _text(article.find(".//JournalIssue/PubDate/Year"))
        if not year:
            medline_date = _text(article.find(".//JournalIssue/PubDate/MedlineDate"))
            year = medline_date[:4] if medline_date else ""
        records.append(
            PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=_text(article.find(".//Journal/Title")),
                year=year,
                doi=doi,
                mesh_terms=[_text(node) for node in article.findall(".//MeshHeading/DescriptorName") if _text(node)],
                query=query,
                target_family=target_family,
                modality=modality,
                retrieved_at=retrieved_at,
            )
        )
    return records


def _text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def retrieve_pubmed_records(args: argparse.Namespace) -> tuple[list[PubMedRecord], list[dict[str, Any]]]:
    """Retrieve PubMed records for configured target families."""

    if args.records_jsonl:
        rows = _read_jsonl(Path(args.records_jsonl))
        records = [PubMedRecord.model_validate(row) for row in rows]
        return records, []

    target_families = args.target_family or list(DEFAULT_TARGET_FAMILIES)
    email = args.pubmed_email or os.getenv("NCBI_EMAIL", "")
    api_key = args.pubmed_api_key or os.getenv("NCBI_API_KEY", "")
    retrieved_at = datetime.now().isoformat(timespec="seconds")
    query_rows: list[dict[str, Any]] = []
    records: list[PubMedRecord] = []
    for target_family in iter_progress(
        target_families,
        total=len(target_families),
        desc="PubMed queries",
        enabled=not args.no_progress,
        unit="target",
    ):
        for query_cfg in DEFAULT_PUBMED_QUERIES[target_family]:
            query = query_cfg["query"]
            modality = query_cfg["modality"]
            ids = _pubmed_ids(
                query,
                max_records=args.max_records_per_query,
                email=email,
                api_key=api_key,
                timeout=args.pubmed_timeout,
            )
            query_rows.append(
                {
                    "target_family": target_family,
                    "modality": modality,
                    "query": query,
                    "max_records": args.max_records_per_query,
                    "pmids": ids,
                    "retrieved_at": retrieved_at,
                }
            )
            records.extend(
                _fetch_pubmed_records(
                    ids,
                    query=query,
                    target_family=target_family,
                    modality=modality,
                    email=email,
                    api_key=api_key,
                    timeout=args.pubmed_timeout,
                    retrieved_at=retrieved_at,
                )
            )
            if args.pubmed_delay > 0:
                time.sleep(args.pubmed_delay)
    return _dedupe_records(records), query_rows


def _dedupe_records(records: list[PubMedRecord]) -> list[PubMedRecord]:
    seen: set[str] = set()
    out: list[PubMedRecord] = []
    for record in records:
        if record.pmid in seen:
            continue
        seen.add(record.pmid)
        out.append(record)
    return out


def _extraction_prompt(record: PubMedRecord, max_claims: int, previous_error: Optional[str]) -> str:
    payload = {
        "task": "Extract literature-grounded scientific claim seeds for later CONFIRM contract drafting.",
        "rules": [
            "Return structured JSON only.",
            f"Return at most {max_claims} claim seeds. Return an empty claims list if the abstract has no testable neuroimaging claim.",
            "Use only information supported by the title/abstract/MeSH fields.",
            "Do not invent p-values, effect sizes, cohorts, or gate results.",
            "Prefer claims with a predictor or case/control contrast, an outcome modality/family, and an expected direction.",
            "Do not rewrite unsupported modalities into local fMRI-FC or sMRI terms. Preserve whether a claim needs DTI/DWI, structural connectivity, task fMRI, EEG/MEG, PET, CSF, genetics, blood, or plasma evidence.",
            "The claim seed is not a CONFIRM result and is not confirmed by this extraction step.",
        ],
        "allowed_target_families": list(DEFAULT_TARGET_FAMILIES),
        "record": record.model_dump(mode="json"),
        "schema_hint": LiteratureClaimExtractionResponse.model_json_schema(),
    }
    if previous_error:
        payload["fix_previous_validation_error"] = previous_error
    return json.dumps(payload, indent=2, sort_keys=True)


def extract_claim_seeds(
    records: list[PubMedRecord],
    llm: LLMClient,
    *,
    max_claims_per_record: int,
    schema_retries: int,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[LiteratureClaimSeed]]:
    """Use an LLM to extract structured claim seeds from PubMed records."""

    system = (
        "You extract PubMed abstracts into literature-grounded CONFIRM claim seeds. "
        "Return JSON only matching the provided schema."
    )
    prompts: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    seeds: list[LiteratureClaimSeed] = []
    for record in iter_progress(records, total=len(records), desc="literature extraction", enabled=progress, unit="paper"):
        last_error: Optional[str] = None
        for attempt in range(schema_retries + 1):
            prompt = _extraction_prompt(record, max_claims_per_record, last_error)
            prompts.append({"pmid": record.pmid, "attempt": attempt, "system": system, "user": prompt})
            raw = _complete_structured(llm, system, prompt, LiteratureClaimExtractionResponse)
            response_row: dict[str, Any] = {"pmid": record.pmid, "attempt": attempt, "raw_response": raw}
            responses.append(response_row)
            try:
                parsed = _parse_json_model(raw, LiteratureClaimExtractionResponse)
                if len(parsed.claims) > max_claims_per_record:
                    raise ValueError(f"Expected at most {max_claims_per_record} claims, got {len(parsed.claims)}.")
                for index, seed in enumerate(parsed.claims, start=1):
                    data = seed.model_dump(mode="json")
                    data["seed_id"] = _safe_id(data["seed_id"] or f"{record.target_family}_{record.pmid}_{index}")
                    data["target_family"] = record.target_family
                    data["source_pmid"] = record.pmid
                    data["source_doi"] = record.doi
                    data["source_title"] = record.title
                    data["source_year"] = record.year
                    seeds.append(LiteratureClaimSeed.model_validate(data))
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                response_row["schema_error"] = last_error
        else:
            responses[-1]["extraction_failed"] = True
    return prompts, responses, _dedupe_seeds(seeds)


def _dedupe_seeds(seeds: list[LiteratureClaimSeed]) -> list[LiteratureClaimSeed]:
    seen: set[str] = set()
    out: list[LiteratureClaimSeed] = []
    for seed in seeds:
        seed_id = seed.seed_id
        if seed_id in seen:
            seed_id = f"{seed_id}_{len(seen) + 1}"
            seed = seed.model_copy(update={"seed_id": seed_id})
        seen.add(seed_id)
        out.append(seed)
    return out


def evaluate_feasibility(
    seeds: list[LiteratureClaimSeed],
    records_by_pmid: dict[str, PubMedRecord],
    catalog: dict[str, Any],
) -> list[FeasibilityResult]:
    """Classify extracted seeds by local executability."""

    results: list[FeasibilityResult] = []
    for seed in seeds:
        record = records_by_pmid.get(seed.source_pmid, _record_fallback(records_by_pmid, seed))
        blocker = _local_feasibility_blocker(seed)
        if blocker is not None:
            results.append(
                FeasibilityResult(
                    seed_id=seed.seed_id,
                    pmid=seed.source_pmid or (record.pmid if record else ""),
                    target_family=seed.target_family,
                    status="requires_new_feature_adapter",
                    reason=blocker,
                )
            )
            continue
        compatible = _compatible_cohorts(seed, catalog)
        if not compatible:
            status = "requires_new_feature_adapter" if _known_modality(seed) else "ambiguous_or_unsupported"
            results.append(
                FeasibilityResult(
                    seed_id=seed.seed_id,
                    pmid=seed.source_pmid or (record.pmid if record else ""),
                    target_family=seed.target_family,
                    status=status,
                    reason="No target-family cohort has locally prepared outcome columns matching this modality/family.",
                )
            )
            continue
        pair = _choose_discovery_replication(seed, compatible)
        if pair is None:
            rejected = _pair_rejection_reasons(seed, compatible)
            results.append(
                FeasibilityResult(
                    seed_id=seed.seed_id,
                    pmid=seed.source_pmid or (record.pmid if record else ""),
                    target_family=seed.target_family,
                    status="requires_new_data",
                    reason=(
                        "No discovery/replication pair passed pair-level executable checks: "
                        + "; ".join(rejected or ["requires compatible discovery and replication cohorts"])
                    ),
                    matched_outcome_examples=compatible[0]["matched_idps"][:10],
                    rejected_pair_reasons=rejected,
                )
            )
            continue
        discovery, replication, shared_outcomes = pair
        common_columns = set(discovery.get("columns", [])) & set(replication.get("columns", []))
        predictor_issue = _predictor_mapping_issue(seed, common_columns)
        if predictor_issue is not None:
            results.append(
                FeasibilityResult(
                    seed_id=seed.seed_id,
                    pmid=seed.source_pmid or (record.pmid if record else ""),
                    target_family=seed.target_family,
                    status="requires_new_feature_adapter",
                    reason=predictor_issue,
                    discovery_cohort=str(discovery["cohort"]),
                    replication_cohort=str(replication["cohort"]),
                    matched_outcome_examples=shared_outcomes[:25],
                )
            )
            continue
        missing_covariates = sorted(
            set(_standard_covariates(seed))
            - common_columns
            - {"site"}
        )
        if not [cov for cov in _standard_covariates(seed) if cov not in missing_covariates]:
            results.append(
                FeasibilityResult(
                    seed_id=seed.seed_id,
                    pmid=seed.source_pmid or (record.pmid if record else ""),
                    target_family=seed.target_family,
                    status="requires_new_feature_adapter",
                    reason="No local standard covariates are shared by the selected discovery/replication pair.",
                    discovery_cohort=str(discovery["cohort"]),
                    replication_cohort=str(replication["cohort"]),
                    matched_outcome_examples=shared_outcomes[:25],
                )
            )
            continue
        results.append(
            FeasibilityResult(
                seed_id=seed.seed_id,
                pmid=seed.source_pmid or (record.pmid if record else ""),
                target_family=seed.target_family,
                status="executable_now",
                reason="Local discovery and replication evidence have compatible outcome columns; unavailable optional covariates should be omitted during contract drafting.",
                discovery_cohort=str(discovery["cohort"]),
                replication_cohort=str(replication["cohort"]),
                matched_outcome_examples=shared_outcomes[:25],
                missing_covariates=missing_covariates,
            )
        )
    return results


def _record_fallback(records_by_pmid: dict[str, PubMedRecord], seed: LiteratureClaimSeed) -> Optional[PubMedRecord]:
    if len(records_by_pmid) == 1:
        return next(iter(records_by_pmid.values()))
    for record in records_by_pmid.values():
        if record.target_family == seed.target_family:
            return record
    return None


def _known_modality(seed: LiteratureClaimSeed) -> bool:
    return _outcome_prefix(seed) is not None


def _local_feasibility_blocker(seed: LiteratureClaimSeed) -> Optional[str]:
    outcome_text = f"{seed.outcome_modality} {seed.outcome_family}".lower()
    predictor_text = f"{seed.predictor_or_group} {seed.candidate_question}".lower()
    unsupported_outcome_patterns = [
        (r"\b(dti|dwi|diffusion|tractography)\b", "diffusion MRI/DTI/DWI outcomes are not present in the local derivative tables."),
        (r"\b(structural connectivity|white[- ]matter integrity|cingulum bundle)\b", "structural-connectivity or white-matter tract outcomes need a new feature adapter."),
        (r"\b(c-bsf|structure[- ]function coupling)\b", "structure-function coupling is not represented by the local FC or sMRI columns."),
        (r"\b(eeg|meg)\b", "EEG/MEG outcomes are outside the local neuroimaging derivative tables."),
        (r"\b(task[- ]fmri|task[- ]based|task dependent|task-dependent|during .*tasks?|n-back|activation|bold response)\b", "task-fMRI activation/connectivity is not represented by the local resting-state FC descriptors."),
        (r"\b(effective connectivity|dynamic causal model(?:ling)?|dcm|causal connectivity|spectral dynamic causal)\b", "effective/causal connectivity is not represented by the local FC columns."),
        (r"\b(dynamic functional connectivity|dfc|fmri[- ]dynamics|connectivity dynamics|ivfc|mcd|mean correlational distance|individual variability|heterogeneity)\b", "dynamic, heterogeneity, or individual-variability FC metrics are not represented by the local static FC descriptors."),
        (r"\b(graph theory|graph metric|modularity|centrality|efficiency|system segregation|network segregation|topological)\b", "graph-theory or topology metrics are not present as local FC columns."),
        (r"\b(cerebellar|cerebello|thalamus|thalamocortical|subcortical|midbrain|frontostriatal|fronto[- ]striato[- ]thalamic)\b", "cerebellar, thalamic, subcortical, or midbrain outcomes are not represented by the local FC network columns."),
        (r"\b(microbleed|microbleeds|lesion|lesions|white[- ]matter hyperintensit|wmh)\b", "lesion, microbleed, and white-matter-hyperintensity outcomes are not present in the local derivative tables."),
    ]
    unsupported_predictor_patterns = [
        (r"\b(graph theory|graph metric|modularity|centrality|efficiency|system segregation|network segregation|topological)\b", "graph-theory or topology predictors are not present as local scalar predictors."),
        (r"\b(csf|p[- ]?tau|t[- ]?tau|\btau\b|tdp[- ]?43|amyloid|aβ|abeta)\b", "CSF/amyloid/tau/TDP-43 predictors are not available in the local benchmark columns."),
        (r"\b(dopamine|dopaminergic|pet dopamine|striatal dopamine)\b", "PET dopamine predictors are not available in the local benchmark columns."),
        (r"\b(apoe|genetic|genotype|polygenic)\b", "genetic predictors are not available in the local benchmark columns."),
        (r"\b(blood|plasma|serum|cortisol|inflammatory marker)\b", "blood/plasma predictors are not available in the local benchmark columns."),
        (r"\b(symptom severity|clinical severity|positive symptoms?|negative symptoms?|panss|psychosis[- ]related dimensions?)\b", "Symptom-severity predictors are not available in the local benchmark columns."),
        (r"\b(ivfc|mcd|mean correlational distance|individual variability|heterogeneity)\b", "Individual-variability FC predictors are not available as scalar predictors in the local benchmark columns."),
    ]
    for pattern, reason in unsupported_outcome_patterns:
        if re.search(pattern, outcome_text):
            return reason
    for pattern, reason in unsupported_predictor_patterns:
        if re.search(pattern, predictor_text):
            return reason
    if _imaging_as_predictor_with_behavior_outcome(seed):
        return "Brain-to-behavior claims with imaging predictors and behavioral/cognitive outcomes are not executable by the current imaging-outcome CONFIRM contract."
    return None


def _imaging_as_predictor_with_behavior_outcome(seed: LiteratureClaimSeed) -> bool:
    predictor_text = seed.predictor_or_group.lower()
    outcome_text = f"{seed.outcome_modality} {seed.outcome_family}".lower()
    predictor_is_imaging = bool(
        re.search(r"\b(fmri|functional connectivity|connectivity|smri|mri|hippocamp|brain volume|network)\b", predictor_text)
    )
    outcome_is_behavior = bool(
        re.search(r"\b(cognition|cognitive|behavior|behaviour|performance|executive function|symptom|clinical score)\b", outcome_text)
    )
    return predictor_is_imaging and outcome_is_behavior


def _predictor_mapping_issue(seed: LiteratureClaimSeed, common_columns: set[str]) -> Optional[str]:
    """Return why the seed's predictor cannot be represented by local columns."""

    if _basic_predictor_supported(seed):
        return None
    if _diagnosis_contrast_supported(seed):
        return None
    if _mentions_local_column(seed.predictor_or_group, common_columns):
        return None
    return (
        "The requested predictor/contrast is not locally executable. Stage 0 only marks seeds executable "
        "when the predictor maps to age/sex, the target diagnosis contrast, or an exact shared local column."
    )


def _basic_predictor_supported(seed: LiteratureClaimSeed) -> bool:
    text = seed.predictor_or_group.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return normalized in {"age", "sex"} or normalized.startswith("age ") or normalized.startswith("sex ")


def _diagnosis_contrast_supported(seed: LiteratureClaimSeed) -> bool:
    text = seed.predictor_or_group.lower()
    predictor_role_text = re.sub(r"\bcognitively normal\b", "normal", text)
    target_terms = {
        "adhd": ["adhd", "attention deficit"],
        "asd": ["asd", "autism", "autistic"],
        "ad_aging": ["alzheimer", "dementia"],
        "psychosis": ["psychosis", "schizophrenia", "schizoaffective"],
    }.get(seed.target_family, [])
    if not target_terms or not any(term in predictor_role_text for term in target_terms):
        return False
    contrast_terms = ["diagnosis", "case", "control", "healthy", "versus", "compared", "patients", "group"]
    if not any(term in predictor_role_text for term in contrast_terms):
        return False
    non_diagnostic_predictor_terms = [
        "neurofeedback",
        "intervention",
        "treatment",
        "sham",
        "control condition",
        "mindfulness",
        "stroke",
        "sedentary",
        "actigraphy",
        "symptom",
        "score",
        "rating",
        "behavior",
        "behaviour",
        "cognition",
        "cognitive",
        "iq",
        "functioning",
        "pathology",
        "biomarker",
        "connectivity",
        "subtype",
        "membership",
    ]
    return not any(term in predictor_role_text for term in non_diagnostic_predictor_terms)


def _mentions_local_column(text: str, common_columns: set[str]) -> bool:
    normalized_text = _normalize_column_text(text)
    for column in common_columns:
        if column in {"subject_id", "cohort", "site"}:
            continue
        normalized_column = _normalize_column_text(column)
        if normalized_column and re.search(rf"\b{re.escape(normalized_column)}\b", normalized_text):
            return True
    return False


def _normalize_column_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _compatible_cohorts(seed: LiteratureClaimSeed, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = _outcome_prefix(seed)
    if prefix is None:
        return []
    guidance = TARGET_GUIDANCE[seed.target_family]
    allowed_bases = {_base_cohort(name) for name in guidance["cohort_examples"]}
    compatible: list[dict[str, Any]] = []
    for cohort in catalog.get("cohorts", []):
        name = str(cohort.get("cohort", ""))
        if allowed_bases and _base_cohort(name) not in allowed_bases:
            continue
        idps = [str(item) for item in cohort.get("idps", [])]
        matches = _matching_idps(seed, idps, prefix)
        if not matches:
            continue
        dx_levels = cohort.get("dx_levels", [])
        columns = columns_with_virtuals(name, [str(item) for item in cohort.get("columns", [])], dx_levels)
        if _needs_dx(seed) and "dx" not in columns and not cohort.get("dx_levels"):
            continue
        row = dict(cohort)
        row["columns"] = columns
        row["matched_idps"] = matches
        compatible.append(row)
    return compatible


def _outcome_prefix(seed: LiteratureClaimSeed) -> Optional[str]:
    text = f"{seed.outcome_modality} {seed.outcome_family}".lower()
    if any(token in text for token in ["fmri", "functional connectivity", "connectivity", "fnc", "fc"]):
        return "fc"
    if any(token in text for token in ["smri", "structural", "hippocamp", "entorhinal", "temporal", "atrophy", "volume"]):
        return "smri"
    if "pet" in text or "fdg" in text:
        return "pet"
    return None


def _matching_idps(seed: LiteratureClaimSeed, idps: list[str], prefix: str) -> list[str]:
    text = seed.outcome_family.lower()
    missing_capability = _missing_requested_outcome_capability(seed, idps)
    if missing_capability is not None:
        return []
    if "thick" in text:
        return [col for col in idps if "thick" in col.lower()]
    if "surface area" in text or "area" in text:
        return [col for col in idps if "area" in col.lower()]
    if "hippocamp" in text:
        specific = [col for col in idps if "hippocampus" in col.lower()]
        if specific:
            return specific
    if "entorhinal" in text:
        specific = [col for col in idps if "entorhinal" in col.lower()]
        if specific:
            return specific
    if "middle temporal" in text or "midtemp" in text:
        specific = [col for col in idps if "midtemp" in col.lower() or "middle_temporal" in col.lower()]
        if specific:
            return specific
    if "whole" in text and "brain" in text:
        specific = [col for col in idps if "wholebrain" in col.lower() or "whole_brain" in col.lower()]
        if specific:
            return specific
    if prefix == "fc":
        return [col for col in idps if col.startswith("fc_")]
    return [col for col in idps if col.startswith(f"{prefix}_")]


def _missing_requested_outcome_capability(seed: LiteratureClaimSeed, idps: list[str]) -> Optional[str]:
    """Detect requested outcome measurements that are absent from local IDP names."""

    text = f"{seed.outcome_modality} {seed.outcome_family}".lower()
    available = " ".join(_normalize_column_text(col) for col in idps)
    capability_requirements = [
        (
            r"\b(qmri|quantitative mri|r1|r2\*?|mtsat|proton density|microstructure)\b",
            ["qmri", "r1", "r2", "mtsat", "proton density", "microstructure"],
        ),
        (
            r"\b(multiscale entropy|\bmse\b|complexity|optimal frequency|connectivity distance|geodesic|hierarchy|similarity index|degree centrality|\bfdc\b)\b",
            ["entropy", "mse", "complexity", "frequency", "distance", "geodesic", "hierarchy", "similarity", "centrality", "fdc"],
        ),
    ]
    for pattern, required_terms in capability_requirements:
        if re.search(pattern, text) and not any(term in available for term in required_terms):
            return pattern
    return None


def _needs_dx(seed: LiteratureClaimSeed) -> bool:
    text = f"{seed.predictor_or_group} {seed.candidate_question}".lower()
    return any(token in text for token in ["diagnosis", "case", "control", "adhd", "asd", "autism", "schiz", "psychosis", "dementia", "alzheimer"])


def _standard_covariates(seed: LiteratureClaimSeed) -> list[str]:
    covariates = {cov for cov in seed.covariates if cov}
    text = f"{seed.outcome_modality} {seed.outcome_family}".lower()
    if "smri" in text or "structural" in text or "volume" in text:
        covariates.update({"age", "sex", "eTIV"})
    elif "fmri" in text or "connect" in text:
        covariates.update({"age", "sex", "site"})
    return sorted(covariates)


def _shared_matched_outcomes(discovery: dict[str, Any], replication: dict[str, Any]) -> list[str]:
    return sorted(set(discovery.get("matched_idps", [])) & set(replication.get("matched_idps", [])))


def _feature_family(columns: list[str]) -> str:
    values = set(columns)
    if any(col.startswith("fc_fc_") for col in values):
        return "fc_edges"
    if any(col.startswith("fc_") for col in values):
        return "fc_summary"
    if any(col.startswith("smri_") for col in values):
        return "smri"
    if any(col.startswith("pet_") for col in values):
        return "pet"
    return "unknown"


def _pair_group_labels_compatible(seed: LiteratureClaimSeed, discovery: dict[str, Any], replication: dict[str, Any]) -> bool:
    if not _needs_dx(seed):
        return True
    for cohort in (discovery, replication):
        levels = confirm_dx_levels(str(cohort.get("cohort", "")), cohort.get("dx_levels", []))
        if set(levels) != {"case", "control"}:
            return False
    return True


def _pair_rejection_reason(seed: LiteratureClaimSeed, discovery: dict[str, Any], replication: dict[str, Any]) -> str | None:
    shared = _shared_matched_outcomes(discovery, replication)
    if not shared:
        families = sorted({_feature_family(list(discovery.get("matched_idps", []))), _feature_family(list(replication.get("matched_idps", [])))})
        if len(set(families)) > 1:
            return "incompatible_feature_family"
        return "no_shared_outcomes"
    if not _pair_group_labels_compatible(seed, discovery, replication):
        return "incompatible_group_labels"
    return None


def _pair_rejection_reasons(seed: LiteratureClaimSeed, compatible: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    discovery = [row for row in compatible if str(row.get("cohort", "")).endswith("_DISC")]
    replication = [row for row in compatible if str(row.get("cohort", "")).endswith("_REP")]
    for disc in discovery or compatible:
        for rep in replication or compatible:
            if str(rep.get("cohort")) == str(disc.get("cohort")):
                continue
            reason = _pair_rejection_reason(seed, disc, rep)
            if reason is not None:
                reasons.append(reason)
    return sorted(set(reasons))


def _choose_discovery_replication(seed: LiteratureClaimSeed, compatible: list[dict[str, Any]]) -> Optional[tuple[dict[str, Any], dict[str, Any], list[str]]]:
    discovery = [row for row in compatible if str(row.get("cohort", "")).endswith("_DISC")]
    replication = [row for row in compatible if str(row.get("cohort", "")).endswith("_REP")]
    candidates: list[tuple[tuple[int, int, str, str], dict[str, Any], dict[str, Any], list[str]]] = []
    for disc in discovery or compatible:
        for rep in replication or compatible:
            if str(rep.get("cohort")) == str(disc.get("cohort")):
                continue
            reason = _pair_rejection_reason(seed, disc, rep)
            if reason is not None:
                continue
            shared = _shared_matched_outcomes(disc, rep)
            cross_dataset = _base_cohort(str(rep.get("cohort"))) != _base_cohort(str(disc.get("cohort")))
            score = (
                0 if cross_dataset else 1,
                -len(shared),
                str(disc.get("cohort")),
                str(rep.get("cohort")),
            )
            candidates.append((score, disc, rep, shared))
    if not candidates:
        return None
    _, disc, rep, shared = sorted(candidates, key=lambda item: item[0])[0]
    return disc, rep, shared


def _base_cohort(name: str) -> str:
    return re.sub(r"_(DISC|REP|HOLDOUT|EXTERNAL).*", "", name)


def _claim_question(seed: LiteratureClaimSeed, feasibility: FeasibilityResult) -> str:
    direction = "two_sided" if seed.expected_direction in {"mixed", "unknown"} else seed.expected_direction
    covariates = ";".join(_feasible_covariates(seed, feasibility))
    return (
        f"For literature-grounded {seed.predictor_or_group} effect on {seed.outcome_family} "
        f"in {seed.outcome_modality}, using {_base_cohort(feasibility.discovery_cohort or '')} as discovery "
        f"and {_base_cohort(feasibility.replication_cohort or '')} as replication, test the claim with expected "
        f"direction {direction} and adjust for {covariates} when available."
    )


def _feasible_covariates(seed: LiteratureClaimSeed, feasibility: FeasibilityResult) -> list[str]:
    missing = set(feasibility.missing_covariates)
    return [cov for cov in _standard_covariates(seed) if cov not in missing]


def _claim_rows(
    seeds: list[LiteratureClaimSeed],
    records_by_pmid: dict[str, PubMedRecord],
    feasibility: list[FeasibilityResult],
) -> list[dict[str, Any]]:
    seed_by_id = {seed.seed_id: seed for seed in seeds}
    rows: list[dict[str, Any]] = []
    for result in feasibility:
        if result.status != "executable_now":
            continue
        seed = seed_by_id[result.seed_id]
        record = records_by_pmid.get(result.pmid, _record_fallback(records_by_pmid, seed))
        citation = f"PMID:{result.pmid}"
        if record and record.doi:
            citation += f"; DOI:{record.doi}"
        allowed_covariates = _feasible_covariates(seed, result)
        if not allowed_covariates:
            continue
        group_metadata = _group_metadata(seed, result)
        rows.append(
            {
                "claim_id": _safe_id(f"pubmed_{result.pmid}_{seed.seed_id}"),
                "target_family": seed.target_family,
                "source_mode": "literature_grounded",
                "question": _claim_question(seed, result),
                "label_class": "candidate_unknown",
                "label_basis": "pubmed_literature",
                "source_citation": citation,
                "notes": (
                    f"support_level={seed.support_level}; source_title={record.title if record else ''}; "
                    f"feasibility_reason={result.reason}; evidence_snippet={seed.evidence_snippet}"
                ),
                "include_in_main": True,
                "discovery_cohort": result.discovery_cohort or "",
                "replication_cohorts": result.replication_cohort or "",
                "allowed_covariates": ";".join(allowed_covariates),
                "shared_outcome_columns_sample": ";".join(result.matched_outcome_examples[:25]),
                "shared_outcome_prefixes": ";".join(_shared_outcome_prefixes(result.matched_outcome_examples)),
                "group_var": group_metadata["group_var"],
                "case_label": group_metadata["case_label"],
                "control_label": group_metadata["control_label"],
                "source_pmid": result.pmid,
                "source_seed_id": seed.seed_id,
            }
        )
    return rows


def _shared_outcome_prefixes(outcomes: list[str]) -> list[str]:
    prefixes: list[str] = []
    if any(col.startswith("fc_fc_") for col in outcomes):
        prefixes.append("fc_fc_")
    if any(col.startswith("fc_") for col in outcomes):
        prefixes.append("fc_")
    if any(col.startswith("smri_") for col in outcomes):
        prefixes.append("smri_")
    if any(col.startswith("pet_") for col in outcomes):
        prefixes.append("pet_")
    return list(dict.fromkeys(prefixes))


def _group_metadata(seed: LiteratureClaimSeed, feasibility: FeasibilityResult) -> dict[str, str]:
    if _needs_dx(seed):
        return {"group_var": CONFIRM_DX, "case_label": "case", "control_label": "control"}
    return {"group_var": "", "case_label": "", "control_label": ""}


def _write_claim_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "claim_id",
        "target_family",
        "source_mode",
        "question",
        "label_class",
        "label_basis",
        "source_citation",
        "notes",
        "include_in_main",
        "discovery_cohort",
        "replication_cohorts",
        "allowed_covariates",
        "shared_outcome_columns_sample",
        "shared_outcome_prefixes",
        "group_var",
        "case_label",
        "control_label",
        "source_pmid",
        "source_seed_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_feasibility_csv(path: Path, rows: list[FeasibilityResult]) -> None:
    fieldnames = [
        "seed_id",
        "pmid",
        "target_family",
        "status",
        "reason",
        "discovery_cohort",
        "replication_cohort",
        "matched_outcome_examples",
        "missing_covariates",
        "rejected_pair_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.model_dump(mode="json")
            data["matched_outcome_examples"] = ";".join(data["matched_outcome_examples"])
            data["missing_covariates"] = ";".join(data["missing_covariates"])
            data["rejected_pair_reasons"] = ";".join(data["rejected_pair_reasons"])
            writer.writerow(data)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_roots = [Path(item) for item in (args.data_root or [str(path) for path in DEFAULT_DATA_ROOTS])]
    catalog = _merge_catalogs(data_roots)
    if not catalog["cohorts"]:
        raise ValueError(f"No readable cohort parquet files found in data roots: {data_roots}")

    records, query_rows = retrieve_pubmed_records(args)
    llm = _make_llm(args.model, args.llm_max_tokens)
    prompts, responses, seeds = extract_claim_seeds(
        records,
        llm,
        max_claims_per_record=args.max_claims_per_record,
        schema_retries=args.schema_retries,
        progress=not args.no_progress,
    )
    records_by_pmid = {record.pmid: record for record in records}
    feasibility = evaluate_feasibility(seeds, records_by_pmid, catalog)
    claim_rows = _claim_rows(seeds, records_by_pmid, feasibility)

    _write_jsonl(out_dir / "pubmed_queries.jsonl", query_rows)
    _write_jsonl(out_dir / "pubmed_records.jsonl", [record.model_dump(mode="json") for record in records])
    _write_jsonl(out_dir / "llm_literature_extraction_prompts.jsonl", prompts)
    _write_jsonl(out_dir / "llm_literature_extraction_responses.jsonl", responses)
    _write_jsonl(out_dir / "extracted_claim_seeds.jsonl", [seed.model_dump(mode="json") for seed in seeds])
    _write_jsonl(out_dir / "literature_claim_feasibility.jsonl", [row.model_dump(mode="json") for row in feasibility])
    _write_feasibility_csv(out_dir / "literature_claim_feasibility.csv", feasibility)
    _write_claim_csv(Path(args.claims_out), claim_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_spec": args.model,
        "data_roots": [str(root) for root in data_roots],
        "n_pubmed_records": len(records),
        "n_extracted_claim_seeds": len(seeds),
        "n_executable_claim_questions": len(claim_rows),
        "status_counts": dict(Counter(row.status for row in feasibility)),
        "target_family_counts": dict(Counter(seed.target_family for seed in seeds)),
        "claims_out": str(args.claims_out),
    }
    (out_dir / "literature_grounding_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'literature_grounding_summary.json'}")
    print(f"wrote {args.claims_out}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-family", action="append", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--claims-out", default=str(DEFAULT_CLAIMS_OUT))
    parser.add_argument("--records-jsonl", default=None, help="Use pre-fetched PubMed records instead of querying PubMed.")
    parser.add_argument("--max-records-per-query", type=int, default=20)
    parser.add_argument("--max-claims-per-record", type=int, default=3)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument("--pubmed-email", default="")
    parser.add_argument("--pubmed-api-key", default="")
    parser.add_argument("--pubmed-timeout", type=float, default=30.0)
    parser.add_argument("--pubmed-delay", type=float, default=0.34)
    parser.add_argument("--data-root", action="append", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
