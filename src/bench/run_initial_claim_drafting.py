"""Generate initial CONFIRM claim questions and draft frozen contracts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bench.progress import iter_progress
from confirm.agent import DOMAIN_PRIOR_SYSTEM_PROMPT, _EXAMPLE_CONTRACT, _contract_prompt, _parse_contract_text, build_data_catalog
from confirm.candidate_preflight import CandidatePreflightContext, CandidatePreflightResult
from confirm.contract import ClaimContract
from confirm.derived_columns import columns_with_virtuals
from confirm.evidence_partitions import canonical_base_cohort, is_excluded_evidence_cohort
from confirm.llm import (
    LLMClient,
    complete_structured,
    make_llm,
    parse_structured,
)

DEFAULT_MODEL = "openai:gpt-5.5"
DEFAULT_TARGET_FAMILIES = ("normative_fmri", "adhd", "asd", "ad_aging", "psychosis")
DEFAULT_FIXED_CLAIMS = Path("data/claims/literature_grounded_claims.csv")
DEFAULT_SYNTHETIC_CLAIMS = Path("data/claims/synthetic_stress_claims.csv")
DEFAULT_DATA_ROOTS = (
    Path("data/prepared_data/evidence_partitions/benchmark_ready/cohorts"),
)
FIXED_SOURCE_MODES = {"literature", "literature_grounded", "inventory"}
_PREFLIGHT_CONTEXT_CACHE: dict[tuple[str, ...], CandidatePreflightContext] = {}
_SOURCE_COHORT_ALIASES = {
    "BSNIP1": "BSNIP",
    "SZJH": "JH",
}
_UNSUPPORTED_CONTRACT_TERM_PATTERNS: list[tuple[str, str]] = [
    (r"\b(dti|dwi|diffusion|tractography)\b", "diffusion_or_dti"),
    (r"\b(task[- ]fmri|task[- ]based|n-back|activation|bold response)\b", "task_fmri"),
    (r"\b(effective connectivity|dynamic causal model(?:ling)?|dcm|causal connectivity|spectral dynamic causal)\b", "effective_connectivity"),
    (r"\b(dynamic functional connectivity|dfc|ivfc|mcd|mean correlational distance|individual variability|heterogeneity)\b", "dynamic_or_variability_fc"),
    (r"\b(graph theory|graph metric|modularity|centrality|efficiency|system segregation|network segregation|topological)\b", "graph_metric"),
    (r"\b(cerebellar|cerebello|thalamus|thalamocortical|subcortical|midbrain|frontostriatal|fronto[- ]striato[- ]thalamic)\b", "unsupported_anatomy"),
    (r"\b(csf|p[- ]?tau|t[- ]?tau|\btau\b|tdp[- ]?43|amyloid|aβ|abeta|dopamine|dopaminergic)\b", "unsupported_biomarker"),
    (r"\b(brain[- ]to[- ]behavior|executive function|behavior|behaviour|clinical score|symptom severity|positive symptoms?|negative symptoms?|panss)\b", "unsupported_behavioral_or_clinical"),
]
_DIAGNOSIS_TARGET_FAMILIES = {"adhd", "asd", "ad_aging", "psychosis"}
_DIAGNOSIS_GROUP_VARS = {"dx", "diagnosis", "diagnosis_group", "group", "confirm_dx"}


class DraftContractError(ValueError):
    """Raised when contract drafting exhausts retries while preserving trace artifacts."""

    def __init__(self, message: str, prompts: list[dict[str, Any]], responses: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.prompts = prompts
        self.responses = responses


TARGET_GUIDANCE: dict[str, dict[str, Any]] = {
    "normative_fmri": {
        "description": "Normative fMRI associations or group contrasts in non-disease cohorts.",
        "cohort_examples": ["UKB_DISC", "UKB_REP", "HCP_DISC", "HCP_REP", "HCP_Aging_DISC", "HCP_Aging_REP", "ABCD_DISC", "ABCD_REP"],
        "outcome_modality": "fMRI connectivity descriptors",
        "typical_predictors": ["age", "sex", "cognition"],
    },
    "adhd": {
        "description": "ADHD case/control or symptom-linked fMRI derivative claims.",
        "cohort_examples": ["ADHD200_DISC", "ADHD200_REP", "ABCD_DISC", "ABCD_REP"],
        "outcome_modality": "fMRI connectivity or dynamics descriptors",
        "typical_predictors": ["dx", "had_adhd"],
    },
    "asd": {
        "description": "ASD case/control fMRI derivative claims.",
        "cohort_examples": ["ABIDE1_DISC", "ABIDE1_REP", "ABIDE2_DISC", "ABIDE2_REP"],
        "outcome_modality": "fMRI connectivity descriptors",
        "typical_predictors": ["dx"],
    },
    "ad_aging": {
        "description": "Alzheimer's disease, dementia, or aging structural/PET/fMRI derivative claims.",
        "cohort_examples": ["ADNI_DISC", "ADNI_REP", "OASIS3_DISC", "OASIS3_REP", "ADNI_fMRI_DISC", "OASIS3_fMRI_REP"],
        "outcome_modality": "sMRI, PET, or fMRI disease/aging descriptors",
        "typical_predictors": ["dx", "age", "sex"],
    },
    "psychosis": {
        "description": "Schizophrenia or psychosis case/control fMRI connectivity claims.",
        "cohort_examples": ["COBRE_DISC", "COBRE_REP", "FBIRN_DISC", "FBIRN_REP", "ChineseSZ_DISC", "BSNIP2_REP"],
        "outcome_modality": "fMRI connectivity descriptors",
        "typical_predictors": ["dx"],
    },
}


class ClaimQuestion(BaseModel):
    """A natural-language claim source before contract drafting."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    target_family: str
    source_mode: Literal["literature", "literature_grounded", "inventory", "llm_proposed", "synthetic_stress"]
    question: str
    label_class: str = "candidate_unknown"
    label_basis: str = "llm_proposed"
    source_citation: str = ""
    notes: str = ""
    include_in_main: bool = True
    discovery_cohort: str = ""
    replication_cohorts: str = ""
    allowed_covariates: str = ""
    shared_outcome_columns_sample: str = ""
    shared_outcome_prefixes: str = ""
    group_var: str = ""
    case_label: str = ""
    control_label: str = ""
    source_pmid: str = ""
    source_seed_id: str = ""


class ProposedQuestion(BaseModel):
    """One LLM-proposed initial claim question."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    target_family: str
    question: str
    scientific_rationale: str
    expected_modality: str
    suggested_cohort_family: str

    @model_validator(mode="after")
    def validate_target(self) -> "ProposedQuestion":
        if self.target_family not in TARGET_GUIDANCE:
            raise ValueError(f"Unknown target_family: {self.target_family}")
        return self


class QuestionGenerationResponse(BaseModel):
    """Structured response for LLM-generated claim questions."""

    model_config = ConfigDict(extra="forbid")

    questions: list[ProposedQuestion]


def _json_safe(data: Any) -> Any:
    if hasattr(data, "to_dict"):
        return _json_safe(data.to_dict())
    if hasattr(data, "model_dump"):
        return _json_safe(data.model_dump(mode="json"))
    if data.__class__.__module__.startswith("numpy") and hasattr(data, "item"):
        return _json_safe(data.item())
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_safe(value) for value in data]
    return data


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip().lower()).strip("_")
    return cleaned or "claim"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_claim_csv(path: Path) -> list[ClaimQuestion]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [{key: value for key, value in row.items()} for row in csv.DictReader(handle)]
    claims: list[ClaimQuestion] = []
    for row in rows:
        row["include_in_main"] = str(row.get("include_in_main", "true")).strip().lower() not in {"0", "false", "no", "off"}
        claims.append(ClaimQuestion.model_validate(row))
    return claims


def load_claim_questions(
    mode: str,
    fixed_claims: Path,
    synthetic_claims: Path,
    *,
    include_synthetic_stress: bool = False,
) -> list[ClaimQuestion]:
    """Load fixed claim questions for the requested source mode."""

    fixed = _read_claim_csv(fixed_claims)
    synthetic = _read_claim_csv(synthetic_claims)
    selected: list[ClaimQuestion] = []
    if mode in {"literature", "all"}:
        selected.extend(row for row in fixed if row.source_mode == "literature")
    if mode in {"literature_grounded", "all"}:
        selected.extend(row for row in fixed if row.source_mode == "literature_grounded")
    if mode in {"inventory", "all"}:
        selected.extend(row for row in fixed if row.source_mode == "inventory")
    if include_synthetic_stress:
        selected.extend(synthetic)
    return selected


def _catalog_summary(catalog: dict[str, Any], target_family: str) -> dict[str, Any]:
    guidance = TARGET_GUIDANCE[target_family]
    examples = set(guidance["cohort_examples"])
    rows = []
    for cohort in catalog.get("cohorts", []):
        name = str(cohort.get("cohort", ""))
        if name in examples or any(name.startswith(base.split("_", 1)[0]) for base in examples):
            idps = [str(item) for item in cohort.get("idps", [])]
            rows.append(
                {
                    "cohort": name,
                    "n": cohort.get("n"),
                    "dx_levels": cohort.get("dx_levels", []),
                    "idp_examples": idps[:40],
                    "n_idps": len(idps),
                    "data_root": cohort.get("data_root"),
                }
            )
    return {"target_family": target_family, "target_guidance": guidance, "candidate_cohorts": rows}


def _merge_catalogs(roots: list[Path]) -> dict[str, Any]:
    cohorts: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        catalog = build_data_catalog(root)
        for entry in catalog.get("cohorts", []):
            if is_excluded_evidence_cohort(str(entry.get("cohort", ""))):
                continue
            merged = dict(entry)
            merged["data_root"] = str(root)
            cohorts.append(merged)
    return {"data_dir": "merged", "data_roots": [str(root) for root in roots], "cohorts": cohorts}


def _question_generation_prompt(target_family: str, catalog: dict[str, Any], exact_questions: int, previous_error: str | None) -> str:
    payload = {
        "task": "Generate scientifically plausible natural-language CONFIRM claim questions. Do not draft contracts.",
        "target_family": target_family,
        "exact_questions": exact_questions,
        "constraints": [
            "Return structured JSON only.",
            f"Return exactly {exact_questions} questions for this target_family. Not fewer and not more.",
            "Every question must be executable from the available cohort metadata.",
            "Questions must be specific enough for a later ClaimContract: predictor/contrast, outcome family, cohorts, direction when justified, and covariates.",
            "Do not invent p-values, effect sizes, cohorts, columns, or gate results.",
            "These are initial claims before CONFIRM gate evaluation, not feedback-loop follow-up claims.",
        ],
        "catalog": _catalog_summary(catalog, target_family),
    }
    if previous_error:
        payload["fix_previous_validation_error"] = previous_error
    return json.dumps(payload, indent=2, sort_keys=True)


def _complete_structured(llm: LLMClient, system: str, user: str, response_model: type[BaseModel]) -> str:
    """Compatibility wrapper for the frozen Stage 1 implementation."""

    return complete_structured(llm, system, user, response_model)


def _parse_json_model(text: str, response_model: type[BaseModel]) -> BaseModel:
    """Compatibility wrapper for the frozen Stage 1 implementation."""

    return parse_structured(text, response_model)


def _make_llm(model_spec: str, max_tokens: int) -> LLMClient:
    llm = make_llm(model_spec)
    if hasattr(llm, "max_tokens"):
        setattr(llm, "max_tokens", max_tokens)
    return llm


def _preflight_context_from_catalog(catalog: dict[str, Any]) -> CandidatePreflightContext | None:
    roots = tuple(str(root) for root in catalog.get("data_roots", []) if str(root))
    if not roots:
        return None
    cached = _PREFLIGHT_CONTEXT_CACHE.get(roots)
    if cached is not None:
        return cached
    context = CandidatePreflightContext.from_roots(roots)
    if not context.cohorts:
        return None
    _PREFLIGHT_CONTEXT_CACHE[roots] = context
    return context


def _split_metadata_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in re.split(r"\s*(?:;|,|\band\b)\s*", str(text), flags=re.IGNORECASE) if item.strip()]


def _question_discovery_cohort(question: ClaimQuestion) -> str:
    return question.discovery_cohort.strip()


def _question_replication_cohorts(question: ClaimQuestion) -> list[str]:
    return _split_metadata_list(question.replication_cohorts)


def _catalog_entry(catalog: dict[str, Any], cohort: str) -> dict[str, Any] | None:
    cohorts = [entry for entry in catalog.get("cohorts", []) if "cohort" in entry]
    for entry in cohorts:
        if str(entry.get("cohort")) == cohort:
            return dict(entry)
    expected = _cohort_key(cohort)
    for entry in cohorts:
        if _cohort_key(str(entry.get("cohort", ""))) == expected:
            return dict(entry)
    return None


def _literature_grounded_pair_catalog(question: ClaimQuestion, catalog: dict[str, Any]) -> dict[str, Any]:
    discovery = _question_discovery_cohort(question)
    replications = _question_replication_cohorts(question)
    if question.source_mode != "literature_grounded" or not discovery:
        return catalog
    selected: list[dict[str, Any]] = []
    for cohort in [discovery, *replications]:
        entry = _catalog_entry(catalog, cohort)
        if entry is None:
            entry = {"cohort": cohort, "available": False, "columns": [], "idps": [], "dx_levels": []}
        selected.append(entry)
    return {
        "data_dir": catalog.get("data_dir", "merged"),
        "data_roots": list(catalog.get("data_roots", [])),
        "cohorts": selected,
    }


def _common_values(cohorts: list[dict[str, Any]], field: str) -> list[str]:
    value_sets = [set(str(item) for item in cohort.get(field, []) or []) for cohort in cohorts]
    if not value_sets:
        return []
    return sorted(set.intersection(*value_sets))


def _common_columns_for_question(question: ClaimQuestion, pair_catalog: dict[str, Any]) -> list[str]:
    return _common_values([row for row in pair_catalog.get("cohorts", []) if row.get("available", True)], "columns")


def _common_idps_for_question(question: ClaimQuestion, pair_catalog: dict[str, Any]) -> list[str]:
    return _common_values([row for row in pair_catalog.get("cohorts", []) if row.get("available", True)], "idps")


def _metadata_outcome_sample(question: ClaimQuestion, common_idps: list[str]) -> list[str]:
    sample = _split_metadata_list(question.shared_outcome_columns_sample)
    if sample:
        common = set(common_idps)
        return [item for item in sample if item in common] or sample
    return common_idps[:25]


def _metadata_outcome_prefixes(question: ClaimQuestion, common_idps: list[str]) -> list[str]:
    prefixes = _split_metadata_list(question.shared_outcome_prefixes)
    if prefixes:
        return prefixes
    out: list[str] = []
    if any(col.startswith("fc_fc_") for col in common_idps):
        out.append("fc_fc_")
    if any(col.startswith("fc_") for col in common_idps):
        out.append("fc_")
    if any(col.startswith("smri_") for col in common_idps):
        out.append("smri_")
    if any(col.startswith("pet_") for col in common_idps):
        out.append("pet_")
    return list(dict.fromkeys(out))


def _allowed_covariates_for_question(question: ClaimQuestion, common_columns: list[str]) -> list[str]:
    allowed = _split_metadata_list(question.allowed_covariates)
    if allowed:
        common = set(common_columns)
        return [cov for cov in allowed if cov in common]
    default = ["age", "sex", "site", "eTIV"]
    common = set(common_columns)
    return [cov for cov in default if cov in common]


def _cohort_details_for_pair(pair_catalog: dict[str, Any], common_columns: list[str], common_idps: list[str]) -> dict[str, Any]:
    common_filter_columns = [col for col in common_columns if col not in set(common_idps)]
    return {
        str(cohort.get("cohort")): {
            "available": bool(cohort.get("available", True)),
            "n": cohort.get("n"),
            "dx_levels": cohort.get("dx_levels", []),
            "common_filter_columns": common_filter_columns[:80],
            "shared_idp_examples": common_idps[:80],
            "n_shared_idps": len(common_idps),
            "data_root": cohort.get("data_root"),
        }
        for cohort in pair_catalog.get("cohorts", [])
    }


def _literature_grounded_contract_prompt(question: ClaimQuestion, catalog: dict[str, Any], previous_error: str | None = None) -> str:
    pair_catalog = _literature_grounded_pair_catalog(question, catalog)
    discovery = _question_discovery_cohort(question)
    replications = _question_replication_cohorts(question)
    common_columns = _common_columns_for_question(question, pair_catalog)
    common_idps = _common_idps_for_question(question, pair_catalog)
    allowed_covariates = _allowed_covariates_for_question(question, common_columns)
    shared_outcome_sample = _metadata_outcome_sample(question, common_idps)
    shared_outcome_prefixes = _metadata_outcome_prefixes(question, common_idps)
    forbidden_covariates = [cov for cov in ["eTIV", "motion"] if cov not in set(allowed_covariates)]
    group_hint = None
    if question.group_var and question.case_label and question.control_label:
        group_hint = {"var": question.group_var, "case": question.case_label, "control": question.control_label}
    instructions = [
        "Output ONLY one YAML or JSON object matching ClaimContract; no prose or markdown fences.",
        f"Set discovery_cohort exactly to {discovery!r}. Do not choose a different discovery cohort.",
        f"Set replication_cohorts exactly to {replications!r}. Do not add, drop, or replace replication cohorts.",
        "Use only columns common to every selected cohort.",
        f"covariates and gates.confound.require_covariates are a closed subset of ALLOWED_COVARIATES: {allowed_covariates}.",
        "Do not invent covariates, interactions, task variables, motion variables, or literature-only variables.",
        "Do not use eTIV unless it appears in ALLOWED_COVARIATES.",
        "For scalar claims, estimand.outcome must be one exact shared IDP column from SHARED_OUTCOME_COLUMNS_SAMPLE or COHORT_DETAILS.",
        "For brainwide fMRI-FC claims, use an observed shared prefix such as fc_fc_ only when listed in SHARED_OUTCOME_PREFIXES; never append a suffix unless that exact suffix is present in shared outcome examples.",
        "Use estimand.type='association' only for numeric predictors; use group_diff for diagnosis, sex, ASD/HC, ADHD/control, SZ/HC, or dementia/control contrasts.",
        "inclusion must be null or a simple pandas-query subgroup filter over common filter columns, such as sex == \"M\" or age >= 65.",
        "Keep paper evidence snippets as provenance only; do not introduce unsupported modalities or measurements into the contract.",
        "Use the same gate thresholds and reporting_language_allowed shape as EXAMPLE_CONTRACT.",
    ]
    payload = {
        "QUESTION": question.question,
        "SOURCE_PROVENANCE": {
            "source_mode": question.source_mode,
            "source_pmid": question.source_pmid,
            "source_seed_id": question.source_seed_id,
            "source_citation": question.source_citation,
        },
        "IMMUTABLE_COHORTS": {"discovery_cohort": discovery, "replication_cohorts": replications},
        "ALLOWED_COVARIATES": allowed_covariates,
        "FORBIDDEN_COVARIATES": forbidden_covariates,
        "SHARED_OUTCOME_COLUMNS_SAMPLE": shared_outcome_sample[:80],
        "SHARED_OUTCOME_PREFIXES": shared_outcome_prefixes,
        "GROUP_HINT": group_hint,
        "COHORT_DETAILS": _cohort_details_for_pair(pair_catalog, common_columns, common_idps),
        "INSTRUCTIONS": instructions,
        "EXAMPLE_CONTRACT": _EXAMPLE_CONTRACT,
    }
    if previous_error:
        payload["FIX_THIS_VALIDATION_ERROR_FROM_YOUR_LAST_OUTPUT"] = previous_error
    return yaml.safe_dump(payload, sort_keys=False)


def _contract_prompt_for_question(question: ClaimQuestion, catalog: dict[str, Any], previous_error: str | None = None) -> str:
    if question.source_mode == "literature_grounded" and _question_discovery_cohort(question):
        return _literature_grounded_contract_prompt(question, catalog, previous_error)
    question_text = question.question
    if question.source_mode == "llm_proposed":
        question_text += (
            "\n\nStage 1 drafting constraints: "
            "For ADHD, ASD, Alzheimer/dementia, schizophrenia, or psychosis case/control contrasts, "
            "use the normalized virtual column confirm_dx with case='case' and control='control' when available; "
            "do not use raw dx labels. For fMRI/FC outcomes, do not include eTIV; use only available covariates "
            "such as age, sex, and site."
        )
    return _contract_prompt(question_text, catalog, previous_error)


def _format_preflight_error(result: CandidatePreflightResult) -> str:
    return "semantic_preflight_error: " + "; ".join(result.violations)


def _target_retry_constraints(question: ClaimQuestion) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if question.target_family in _DIAGNOSIS_TARGET_FAMILIES:
        constraints["diagnosis_contrast"] = {
            "predictor": "confirm_dx",
            "group": {"var": "confirm_dx", "case": "case", "control": "control"},
            "instruction": "Use confirm_dx for target-family case/control contrasts when available; do not use raw dx labels.",
        }
    constraints["fmri_fc_covariates"] = {
        "allowed_when_available": ["age", "sex", "site"],
        "forbidden": ["eTIV"],
        "instruction": "For fMRI/FC outcomes, omit eTIV.",
    }
    return constraints


def _retry_hints_for_preflight(question: ClaimQuestion, result: CandidatePreflightResult, catalog: dict[str, Any]) -> dict[str, Any]:
    hints: dict[str, Any] = {"violations": list(result.violations), "target_constraints": _target_retry_constraints(question)}
    if question.source_mode == "literature_grounded":
        pair_catalog = _literature_grounded_pair_catalog(question, catalog)
        common_columns = _common_columns_for_question(question, pair_catalog)
        common_idps = _common_idps_for_question(question, pair_catalog)
        allowed_covariates = _allowed_covariates_for_question(question, common_columns)
        hints.update(
            {
                "fixed_discovery_cohort": _question_discovery_cohort(question),
                "fixed_replication_cohorts": _question_replication_cohorts(question),
                "allowed_covariates": allowed_covariates,
                "shared_outcome_columns_sample": _metadata_outcome_sample(question, common_idps)[:25],
                "shared_outcome_prefixes": _metadata_outcome_prefixes(question, common_idps),
                "group_hint": {
                    "var": question.group_var,
                    "case": question.case_label,
                    "control": question.control_label,
                }
                if question.group_var and question.case_label and question.control_label
                else None,
            }
        )
    return hints


def _format_retry_hints(hints: dict[str, Any]) -> str:
    return "retry_with_constraints: " + json.dumps(_json_safe(hints), sort_keys=True)


def _pre_draft_semantic_issue(question: ClaimQuestion, catalog: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    if question.source_mode != "literature_grounded":
        return None
    discovery = _question_discovery_cohort(question)
    replications = _question_replication_cohorts(question)
    if not discovery or not replications:
        return (
            "requires_better_stage0_seed",
            "Literature-grounded row is missing fixed discovery or replication cohort metadata.",
            {"fixed_discovery_cohort": discovery, "fixed_replication_cohorts": replications},
        )
    pair_catalog = _literature_grounded_pair_catalog(question, catalog)
    unavailable = [str(row.get("cohort")) for row in pair_catalog.get("cohorts", []) if not row.get("available", True)]
    if unavailable:
        return (
            "unsupported_local_columns",
            f"Fixed cohorts are not available in the configured data roots: {unavailable}.",
            {"unavailable_cohorts": unavailable},
        )
    common_columns = _common_columns_for_question(question, pair_catalog)
    common_idps = _common_idps_for_question(question, pair_catalog)
    declared_shared = _split_metadata_list(question.shared_outcome_columns_sample)
    if not declared_shared or not common_idps:
        return (
            "unsupported_local_columns",
            "Literature-grounded row has no shared outcome columns for its fixed discovery/replication pair.",
            {
                "fixed_discovery_cohort": discovery,
                "fixed_replication_cohorts": replications,
                "shared_outcome_columns_sample": declared_shared,
                "common_outcome_count": len(common_idps),
            },
        )
    if question.group_var or question.case_label or question.control_label:
        if not (question.group_var and question.case_label and question.control_label):
            return (
                "invalid_group_contrast",
                "Literature-grounded row has incomplete group contrast metadata.",
                {"group_var": question.group_var, "case_label": question.case_label, "control_label": question.control_label},
            )
        if question.group_var not in set(common_columns):
            return (
                "invalid_group_contrast",
                f"Group variable {question.group_var!r} is not common to all fixed cohorts.",
                {
                    "group_var": question.group_var,
                    "case_label": question.case_label,
                    "control_label": question.control_label,
                    "common_columns_sample": common_columns[:50],
                },
            )
    return None


def _cohort_key(cohort: str) -> str:
    base = re.sub(r"\s+", "_", str(cohort).strip())
    base = canonical_base_cohort(base.upper())
    base = re.sub(r"_(SPLIT|SITE_A|SITE_B|SITES)$", "", base)
    key = re.sub(r"[^A-Z0-9]+", "", base)
    return _SOURCE_COHORT_ALIASES.get(key, key)


def _source_cohort_keys(text: str | None) -> set[str]:
    if not text:
        return set()
    keys: set[str] = set()
    for part in re.split(r"\s*(?:/|;|,|\band\b)\s*", text, flags=re.IGNORECASE):
        cleaned = re.sub(r"\b(split|site\s+[ab]|sites?)\b", "", part, flags=re.IGNORECASE).strip()
        if not cleaned:
            continue
        key = _cohort_key(cleaned)
        if key:
            keys.add(key)
    return keys


def _explicit_source_cohorts(question: ClaimQuestion) -> tuple[set[str], set[str]]:
    if question.source_mode == "literature_grounded":
        expected_discovery = _source_cohort_keys(question.discovery_cohort)
        expected_replication = _source_cohort_keys(question.replication_cohorts)
        if expected_discovery or expected_replication:
            return expected_discovery, expected_replication
    match = re.search(
        r"\busing\s+(?P<discovery>.+?)\s+as discovery(?:\s+and\s+(?P<replication>.+?)\s+as replication)?(?:,|$)",
        question.question,
        flags=re.IGNORECASE,
    )
    if match is None:
        return set(), set()
    return _source_cohort_keys(match.group("discovery")), _source_cohort_keys(match.group("replication"))


def _source_preservation_violations(question: ClaimQuestion, contract: ClaimContract) -> list[str]:
    if question.source_mode not in FIXED_SOURCE_MODES:
        return []
    expected_discovery, expected_replication = _explicit_source_cohorts(question)
    violations: list[str] = []
    actual_discovery = _cohort_key(contract.discovery_cohort)
    actual_replication = {_cohort_key(cohort) for cohort in contract.replication_cohorts}
    if expected_discovery and actual_discovery not in expected_discovery:
        violations.append(
            "Source preservation: discovery_cohort "
            f"{contract.discovery_cohort!r} does not match explicit fixed-source discovery cohorts "
            f"{sorted(expected_discovery)}."
        )
    if expected_replication:
        missing = sorted(expected_replication - actual_replication)
        extra = sorted(actual_replication - expected_replication)
        if missing:
            violations.append(
                "Source preservation: replication_cohorts are missing explicit fixed-source replication cohorts "
                f"{missing}."
            )
        if extra:
            violations.append(
                "Source preservation: replication_cohorts include cohorts not named by the fixed source "
                f"{extra}."
            )
    return violations


def _retry_hints_for_source_preservation(question: ClaimQuestion, violations: list[str]) -> dict[str, Any]:
    hints: dict[str, Any] = {"violations": list(violations)}
    if question.source_mode == "literature_grounded":
        hints.update(
            {
                "fixed_discovery_cohort": _question_discovery_cohort(question),
                "fixed_replication_cohorts": _question_replication_cohorts(question),
                "instruction": "Use exactly these fixed literature-grounded cohorts. Do not choose replacement cohorts.",
            }
        )
    return hints


def _canonicalize_contract_payload(question: ClaimQuestion, payload: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Apply deterministic, provenance-preserving cleanup before schema/preflight validation."""

    data = json.loads(json.dumps(payload))
    estimand = data.get("estimand")
    if not isinstance(estimand, dict):
        return data
    _canonicalize_diagnosis_group(question, estimand, data, catalog)
    _drop_inappropriate_etiv(data, catalog)
    return data


def _canonicalize_diagnosis_group(question: ClaimQuestion, estimand: dict[str, Any], data: dict[str, Any], catalog: dict[str, Any]) -> None:
    if question.target_family not in _DIAGNOSIS_TARGET_FAMILIES:
        return
    if estimand.get("type") != "group_diff":
        return
    group = estimand.get("group")
    if not isinstance(group, dict):
        return
    predictor = str(estimand.get("predictor", "")).strip()
    group_var = str(group.get("var", "")).strip()
    if predictor not in _DIAGNOSIS_GROUP_VARS and group_var not in _DIAGNOSIS_GROUP_VARS:
        return
    if not _selected_cohorts_share_column(data, catalog, "confirm_dx"):
        return
    estimand["predictor"] = "confirm_dx"
    estimand["group"] = {"var": "confirm_dx", "case": "case", "control": "control"}
    _remove_items(data, ["confirm_dx"], from_covariates=True, from_required_covariates=True)


def _drop_inappropriate_etiv(data: dict[str, Any], catalog: dict[str, Any]) -> None:
    if "eTIV" not in set(data.get("covariates") or []) | set(data.get("gates", {}).get("confound", {}).get("require_covariates", []) or []):
        return
    if _payload_outcome_is_fc(data) or not _selected_cohorts_share_column(data, catalog, "eTIV"):
        _remove_items(data, ["eTIV"], from_covariates=True, from_required_covariates=True)


def _payload_outcome_is_fc(data: dict[str, Any]) -> bool:
    estimand = data.get("estimand") if isinstance(data.get("estimand"), dict) else {}
    outcomes = estimand.get("outcome")
    if not isinstance(outcomes, list):
        outcomes = [outcomes]
    text = " ".join(str(item) for item in outcomes if item is not None)
    text += " " + str(data.get("question", ""))
    return bool(re.search(r"\b(fmri|functional connectivity|rs[- ]?fmri|fc_|fc-fc)\b", text, flags=re.IGNORECASE))


def _selected_cohorts_share_column(data: dict[str, Any], catalog: dict[str, Any], column: str) -> bool:
    cohorts = [data.get("discovery_cohort"), *(data.get("replication_cohorts") or [])]
    if not cohorts:
        return False
    for cohort in cohorts:
        entry = _catalog_entry(catalog, str(cohort))
        if entry is None:
            return False
        columns = columns_with_virtuals(
            str(entry.get("cohort", cohort)),
            [str(item) for item in entry.get("columns", [])],
            entry.get("dx_levels", []),
        )
        if column not in set(columns):
            return False
    return True


def _remove_items(data: dict[str, Any], values: list[str], *, from_covariates: bool, from_required_covariates: bool) -> None:
    banned = set(values)
    if from_covariates and isinstance(data.get("covariates"), list):
        data["covariates"] = [item for item in data["covariates"] if item not in banned]
    if from_required_covariates:
        confound = data.get("gates", {}).get("confound") if isinstance(data.get("gates"), dict) else None
        if isinstance(confound, dict) and isinstance(confound.get("require_covariates"), list):
            confound["require_covariates"] = [item for item in confound["require_covariates"] if item not in banned]


def generate_llm_questions(
    llm: LLMClient,
    catalog: dict[str, Any],
    target_family: str,
    *,
    count: int,
    schema_retries: int,
) -> tuple[list[ClaimQuestion], list[dict[str, Any]], list[dict[str, Any]]]:
    """Ask an LLM for initial claim questions for one target family."""

    system = (
        "You generate initial CONFIRM benchmark claim questions. "
        "The goal is a diverse set of executable scientific questions, not claims tuned to pass gates. "
        "Return JSON only matching the provided schema."
    )
    prompts: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    last_error: str | None = None
    for attempt in range(schema_retries + 1):
        prompt = _question_generation_prompt(target_family, catalog, count, last_error)
        prompts.append({"target_family": target_family, "attempt": attempt, "system": system, "user": prompt})
        raw = _complete_structured(llm, system, prompt, QuestionGenerationResponse)
        responses.append({"target_family": target_family, "attempt": attempt, "raw_response": raw})
        try:
            parsed = _parse_json_model(raw, QuestionGenerationResponse)
            questions = list(parsed.questions)
            if len(questions) != count:
                raise ValueError(f"Expected exactly {count} questions for {target_family}, got {len(questions)}.")
            wrong_targets = sorted({item.target_family for item in questions if item.target_family != target_family})
            if wrong_targets:
                raise ValueError(f"All questions must use target_family {target_family!r}; got {wrong_targets}.")
            out: list[ClaimQuestion] = []
            seen: set[str] = set()
            for index, item in enumerate(questions, start=1):
                claim_id = _safe_id(item.claim_id)
                if claim_id in seen:
                    claim_id = f"{target_family}_llm_{index:03d}_{claim_id}"
                seen.add(claim_id)
                out.append(
                    ClaimQuestion(
                        claim_id=claim_id,
                        target_family=target_family,
                        source_mode="llm_proposed",
                        question=item.question,
                        label_class="candidate_unknown",
                        label_basis="llm_proposed",
                        source_citation="llm_proposed",
                        notes=f"{item.scientific_rationale} Suggested cohort family: {item.suggested_cohort_family}",
                        include_in_main=True,
                    )
                )
            return out, prompts, responses
        except Exception as exc:
            last_error = str(exc)
            responses[-1]["schema_error"] = last_error
    raise ValueError(f"LLM failed to generate valid questions for {target_family}: {last_error}")


def draft_contract_with_trace(
    question: ClaimQuestion,
    catalog: dict[str, Any],
    llm: LLMClient,
    *,
    schema_retries: int,
    preflight_context: CandidatePreflightContext | None = None,
    system_prompt: str | None = None,
) -> tuple[ClaimContract, list[dict[str, Any]], list[dict[str, Any]]]:
    """Draft one ClaimContract with prompt/response trace and validation retries.

    ``system_prompt`` overrides the default domain-prior system prompt. It is used
    by the claim-generation integration study to drive drafting with another
    agent's persona while keeping the same contract schema, catalog, preflight,
    and source-preservation checks.
    """

    active_system = system_prompt or DOMAIN_PRIOR_SYSTEM_PROMPT
    prompts: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    last_error: str | None = None
    context = preflight_context if preflight_context is not None else _preflight_context_from_catalog(catalog)
    for attempt in range(schema_retries + 1):
        prompt = _contract_prompt_for_question(question, catalog, last_error)
        prompts.append({"claim_id": question.claim_id, "attempt": attempt, "system": active_system, "user": prompt})
        raw = _complete_structured(llm, active_system, prompt, ClaimContract)
        responses.append({"claim_id": question.claim_id, "attempt": attempt, "raw_response": raw})
        try:
            parsed_payload = _canonicalize_contract_payload(question, _parse_contract_text(raw), catalog)
            contract = ClaimContract.model_validate(parsed_payload)
            data = contract.model_dump(mode="json")
            data["claim_id"] = _safe_id(question.claim_id)
            data["question"] = question.question
            final_contract = ClaimContract.model_validate(data)
            excluded_cohorts = [
                cohort
                for cohort in [final_contract.discovery_cohort, *final_contract.replication_cohorts]
                if is_excluded_evidence_cohort(cohort)
            ]
            if excluded_cohorts:
                message = f"initial contracts cannot use excluded evaluation cohorts: {excluded_cohorts}"
                responses[-1]["evidence_policy_error"] = message
                last_error = message
                raise ValueError(message)
            source_violations = _source_preservation_violations(question, final_contract)
            if source_violations:
                responses[-1]["source_preservation_error"] = "source_preservation_error: " + "; ".join(source_violations)
                hints = _retry_hints_for_source_preservation(question, source_violations)
                responses[-1]["retry_hints"] = hints
                last_error = responses[-1]["source_preservation_error"] + "\n" + _format_retry_hints(hints)
                raise ValueError(responses[-1]["source_preservation_error"])
            if context is not None:
                preflight = context.validate_contract(final_contract)
                responses[-1]["preflight"] = preflight.model_dump(mode="json")
                if not preflight.ok:
                    responses[-1]["preflight_error"] = _format_preflight_error(preflight)
                    hints = _retry_hints_for_preflight(question, preflight, catalog)
                    responses[-1]["retry_hints"] = hints
                    last_error = responses[-1]["preflight_error"] + "\n" + _format_retry_hints(hints)
                    raise ValueError(responses[-1]["preflight_error"])
            return final_contract, prompts, responses
        except Exception as exc:
            if "retry_hints" not in responses[-1]:
                last_error = str(exc)
            responses[-1]["schema_error"] = last_error
            responses[-1]["validation_error"] = last_error
    raise DraftContractError(
        f"LLM failed to draft a valid executable ClaimContract for {question.claim_id}: {last_error}",
        prompts,
        responses,
    )


def _question_record(question: ClaimQuestion) -> dict[str, Any]:
    return question.model_dump(mode="json")


def _draft_record(question: ClaimQuestion, contract: ClaimContract, model_spec: str) -> dict[str, Any]:
    return {
        **question.model_dump(mode="json"),
        "model_spec": model_spec,
        "draft_success": True,
        "drafted_contract": contract.model_dump(mode="json"),
    }


def _draft_disposition_from_violations(violations: list[str]) -> str:
    text = " ".join(violations).lower()
    if "missing outcome columns" in text:
        return "unsupported_local_columns"
    if "missing analysis columns" in text:
        return "unsupported_local_columns"
    if "missing group levels" in text or "group" in text and "level" in text:
        return "invalid_group_contrast"
    if "too few complete rows" in text:
        return "no_complete_cases"
    return "requires_better_stage0_seed"


def _last_semantic_failure(responses: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    for response in reversed(responses):
        if "preflight" in response:
            preflight = response.get("preflight")
            if isinstance(preflight, dict):
                violations = [str(item) for item in preflight.get("violations", [])]
                return _draft_disposition_from_violations(violations), response.get("retry_hints")
        if "source_preservation_error" in response:
            return "requires_better_stage0_seed", response.get("retry_hints")
    return None, None


def _draft_question_worker(task: tuple[int, dict[str, Any], dict[str, Any], str, int, int]) -> tuple[
    int,
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    index, question_payload, catalog, model_spec, schema_retries, llm_max_tokens = task
    question = ClaimQuestion.model_validate(question_payload)
    try:
        pre_draft_issue = _pre_draft_semantic_issue(question, catalog)
        if pre_draft_issue is not None:
            disposition, message, hints = pre_draft_issue
            return (
                index,
                None,
                [],
                [],
                {
                    **question.model_dump(mode="json"),
                    "model_spec": model_spec,
                    "draft_success": False,
                    "error_stage": "semantic_preflight",
                    "draft_disposition": disposition,
                    "error": message,
                    "retry_hints": hints,
                },
            )
        llm = _make_llm(model_spec, llm_max_tokens)
        preflight_context = _preflight_context_from_catalog(catalog)
        contract, prompts, responses = draft_contract_with_trace(
            question,
            catalog,
            llm,
            schema_retries=schema_retries,
            preflight_context=preflight_context,
        )
        validation = {"claim_id": question.claim_id, "draft_success": True}
        if responses and "preflight" in responses[-1]:
            validation["preflight"] = responses[-1]["preflight"]
        return index, _draft_record(question, contract, model_spec), prompts, responses, validation
    except DraftContractError as exc:
        disposition, retry_hints = _last_semantic_failure(exc.responses)
        error_stage = "semantic_preflight" if question.source_mode == "literature_grounded" and disposition else "draft_contract"
        row = {
            **question.model_dump(mode="json"),
            "model_spec": model_spec,
            "draft_success": False,
            "error_stage": error_stage,
            "error": str(exc),
        }
        if disposition:
            row["draft_disposition"] = disposition
        if retry_hints:
            row["retry_hints"] = retry_hints
        return (
            index,
            None,
            exc.prompts,
            exc.responses,
            row,
        )
    except Exception as exc:
        return (
            index,
            None,
            [],
            [],
            {
                **question.model_dump(mode="json"),
                "model_spec": model_spec,
                "draft_success": False,
                "error_stage": "draft_contract",
                "error": str(exc),
            },
        )


def _draft_all_contracts(
    questions: list[ClaimQuestion],
    catalog: dict[str, Any],
    *,
    model_spec: str,
    schema_retries: int,
    llm_max_tokens: int,
    max_workers: int,
    parallel_backend: str,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [
        (index, question.model_dump(mode="json"), catalog, model_spec, schema_retries, llm_max_tokens)
        for index, question in enumerate(questions)
    ]
    ordered: dict[int, tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    if max_workers <= 1:
        for task in iter_progress(tasks, total=len(tasks), desc="contract drafting", enabled=progress, unit="claim"):
            index, draft, prompts, responses, validation = _draft_question_worker(task)
            ordered[index] = (draft, prompts, responses, validation)
    else:
        executor_cls = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
        with executor_cls(max_workers=max_workers) as executor:
            futures = {executor.submit(_draft_question_worker, task): task[0] for task in tasks}
            for future in iter_progress(as_completed(futures), total=len(futures), desc="contract drafting", enabled=progress, unit="claim"):
                index, draft, prompts, responses, validation = future.result()
                ordered[index] = (draft, prompts, responses, validation)

    drafted: list[dict[str, Any]] = []
    contract_prompts: list[dict[str, Any]] = []
    contract_responses: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for index in range(len(questions)):
        draft, prompts, responses, validation = ordered[index]
        contract_prompts.extend(prompts)
        contract_responses.extend(responses)
        if draft is not None:
            drafted.append(draft)
        validation_rows.append(validation)
    return drafted, contract_prompts, contract_responses, validation_rows


def _accepted_contract_audit(drafted: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for row in drafted:
        contract = row.get("drafted_contract", {})
        payload = {
            "question": row.get("question"),
            "contract_question": contract.get("question") if isinstance(contract, dict) else None,
            "estimand": contract.get("estimand") if isinstance(contract, dict) else None,
            "covariates": contract.get("covariates") if isinstance(contract, dict) else None,
            "inclusion": contract.get("inclusion") if isinstance(contract, dict) else None,
        }
        text = json.dumps(_json_safe(payload), sort_keys=True).lower()
        term_hits = [
            {"term_class": term_class, "pattern": pattern}
            for pattern, term_class in _UNSUPPORTED_CONTRACT_TERM_PATTERNS
            if re.search(pattern, text, flags=re.IGNORECASE)
        ]
        if term_hits:
            hits.append(
                {
                    "claim_id": row.get("claim_id"),
                    "target_family": row.get("target_family"),
                    "term_hits": term_hits,
                }
            )
    return {
        "n_drafted_contracts": len(drafted),
        "unsupported_contract_hit_count": len(hits),
        "unsupported_contract_hits": hits,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_families = args.target_family or list(DEFAULT_TARGET_FAMILIES)
    data_roots = [Path(item) for item in (args.data_root or [str(path) for path in DEFAULT_DATA_ROOTS])]
    catalog = _merge_catalogs(data_roots)
    if not catalog["cohorts"]:
        raise ValueError(f"No readable cohort parquet files found in data roots: {data_roots}")

    llm = _make_llm(args.model, args.llm_max_tokens)
    fixed_questions = load_claim_questions(
        args.mode,
        Path(args.fixed_claims),
        Path(args.synthetic_claims),
        include_synthetic_stress=bool(args.include_synthetic_stress),
    )

    question_prompts: list[dict[str, Any]] = []
    question_responses: list[dict[str, Any]] = []
    llm_questions: list[ClaimQuestion] = []
    if args.mode in {"llm_proposed", "all"}:
        for target_family in iter_progress(
            target_families,
            total=len(target_families),
            desc="question generation",
            enabled=not args.no_progress,
            unit="target",
        ):
            generated, prompts, responses = generate_llm_questions(
                llm,
                catalog,
                target_family,
                count=args.num_claims_per_target,
                schema_retries=args.schema_retries,
            )
            llm_questions.extend(generated)
            question_prompts.extend(prompts)
            question_responses.extend(responses)

    questions = [*fixed_questions, *llm_questions]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("No claim questions selected")

    drafted, contract_prompts, contract_responses, validation = _draft_all_contracts(
        questions,
        catalog,
        model_spec=args.model,
        schema_retries=args.schema_retries,
        llm_max_tokens=args.llm_max_tokens,
        max_workers=args.max_workers,
        parallel_backend=args.parallel_backend,
        progress=not args.no_progress,
    )

    _write_jsonl(out_dir / "claim_questions.jsonl", [_question_record(item) for item in questions])
    _write_jsonl(out_dir / "llm_question_prompts.jsonl", question_prompts)
    _write_jsonl(out_dir / "llm_question_responses.jsonl", question_responses)
    _write_jsonl(out_dir / "llm_contract_prompts.jsonl", contract_prompts)
    _write_jsonl(out_dir / "llm_contract_responses.jsonl", contract_responses)
    _write_jsonl(out_dir / "drafted_contracts.jsonl", drafted)
    (out_dir / "draft_validation.json").write_text(json.dumps(_json_safe(validation), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    accepted_audit = _accepted_contract_audit(drafted)
    (out_dir / "accepted_contract_audit.json").write_text(
        json.dumps(_json_safe(accepted_audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "model_spec": args.model,
        "num_claims_per_target": args.num_claims_per_target,
        "llm_max_tokens": args.llm_max_tokens,
        "max_workers": args.max_workers,
        "parallel_backend": args.parallel_backend,
        "target_families": list(target_families),
        "data_roots": [str(root) for root in data_roots],
        "fixed_claims": str(args.fixed_claims),
        "synthetic_claims": str(args.synthetic_claims),
        "n_questions": len(questions),
        "n_drafted_contracts": len(drafted),
        "n_draft_errors": len(validation) - len(drafted),
        "unsupported_contract_hit_count": accepted_audit["unsupported_contract_hit_count"],
        "question_counts_by_source_mode": dict(Counter(item.source_mode for item in questions)),
        "question_counts_by_target_family": dict(Counter(item.target_family for item in questions)),
    }
    (out_dir / "draft_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'drafted_contracts.jsonl'}")
    print(f"wrote {out_dir / 'draft_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["literature", "literature_grounded", "inventory", "llm_proposed", "all"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-claims-per-target", type=int, default=50)
    parser.add_argument("--target-family", action="append", default=None)
    parser.add_argument("--out-dir", default="review-stage/initial-claims-all-gpt55")
    parser.add_argument("--fixed-claims", default=str(DEFAULT_FIXED_CLAIMS))
    parser.add_argument("--synthetic-claims", default=str(DEFAULT_SYNTHETIC_CLAIMS))
    parser.add_argument("--include-synthetic-stress", action="store_true")
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--llm-max-tokens", type=int, default=8192)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", action="append", default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
