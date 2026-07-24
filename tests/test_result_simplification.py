from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bench.benchmark import (
    BenchmarkCase,
    BenchmarkEvaluationTask,
    BenchmarkReference,
    TaskOutcome,
)
from confirm.claim_search import (
    CLAIM_CANDIDATE_SYSTEM_PROMPT,
    LLMCandidateGenerationResponse,
)
from confirm.contract import ClaimContract
from confirm.search_artifacts import (
    CandidateProposal,
    RawCandidateResponseV7,
)


ROOT = Path(__file__).resolve().parents[1]
V7_PROMPT_SHA256 = (
    "2d458df69c45aa133c8008fc97fd5e794bfc391b81669a80c370179bc53e4668"
)
V7_SCHEMA_SHA256 = (
    "7b10d62321945ce66335a75061f7bf544b3923ce9b5210090e7ffe4f730939b5"
)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "claim",
            "question": "Do cases differ from controls?",
            "estimand": {
                "type": "group_diff",
                "outcome": "smri_metric",
                "predictor": "confirm_dx",
                "group": {
                    "var": "confirm_dx",
                    "case": "case",
                    "control": "control",
                },
                "direction": "negative",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["age", "sex"],
            "inclusion": None,
            "discovery_cohort": "DATA_DISC",
            "replication_cohorts": ["DATA_REP"],
            "search_provenance": {
                "declared": True,
                "family_size": 1,
                "selection": "preregistered",
            },
            "gates": {
                "multiplicity": {
                    "method": "fdr_bh",
                    "alpha": 0.05,
                    "family_size": 1,
                },
                "confound": {
                    "require_covariates": ["age", "sex"],
                    "motion_check": False,
                },
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
            "reporting_language_allowed": [
                "confirmed",
                "non_replicated",
                "under_powered",
                "fragile",
            ],
        }
    )


def test_v7_prompt_and_wire_schema_are_frozen() -> None:
    assert (
        hashlib.sha256(
            CLAIM_CANDIDATE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
        == V7_PROMPT_SHA256
    )
    assert RawCandidateResponseV7 is LLMCandidateGenerationResponse
    assert (
        _sha256_json(RawCandidateResponseV7.model_json_schema())
        == V7_SCHEMA_SHA256
    )


def test_compact_candidate_preserves_legacy_self_reports() -> None:
    payload = {
        "candidate_id": "candidate",
        "parent_claim_id": "parent",
        "round_index": 1,
        "proposal_type": "exploratory_followup_claim",
        "proposed_question": "Is the connected outcome lower in cases?",
        "proposed_contract": _contract().model_dump(mode="json"),
        "rationale": "Connected follow-up.",
        "transform_type": "alternative_outcome",
        "domain_core": {"outcome_modality": "sMRI"},
        "preservation_check": {"direction_preserved": True},
    }
    compact = CandidateProposal.from_v7(payload)
    restored = compact.to_v7()
    assert restored["domain_core"] == payload["domain_core"]
    assert restored["preservation_check"] == payload["preservation_check"]
    assert restored["transform_type"] == "alternative_outcome"
    assert restored["proposed_contract"] == payload["proposed_contract"]


def test_public_benchmark_schema_has_no_legacy_triage_models() -> None:
    schema_text = json.dumps(
        {
            model.__name__: model.model_json_schema()
            for model in (
                BenchmarkCase,
                BenchmarkReference,
                BenchmarkEvaluationTask,
                TaskOutcome,
            )
        },
        sort_keys=True,
    )
    assert "TriageReferenceProfile" not in schema_text
    assert "SimplifiedBenchmarkClaim" not in schema_text


def test_active_analysis_uses_only_public_benchmark_model() -> None:
    for relative in (
        "nbs/analyze_neuroclaimbench_v21.py",
        "nbs/analyze_neuroclaimbench_gate_attribution.py",
        "nbs/analyze_neuroclaimbench_v21_feedback_crosswalk.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "bench.benchmark" in text
        assert "TriageReferenceProfile" not in text
        assert "SimplifiedBenchmarkClaim" not in text


def test_public_launchers_guard_api_and_keep_local_phases_local() -> None:
    build = (
        ROOT / "scripts/launch_neuroclaimbench_build.sh"
    ).read_text(encoding="utf-8")
    analyze = (
        ROOT / "scripts/launch_neuroclaimbench_analyze.sh"
    ).read_text(encoding="utf-8")
    assert "ALLOW_API" in build
    assert "require_api" in build
    assert "ALLOW_API" not in analyze
    assert "run_neuroclaimbench_adjudication" not in analyze
    assert "run_neuroclaimbench_pubmed_cache" not in analyze


def test_public_runtime_does_not_import_private_runner_helpers() -> None:
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from bench.run_" not in text or " import _" not in text
