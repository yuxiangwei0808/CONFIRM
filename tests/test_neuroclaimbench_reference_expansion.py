from __future__ import annotations

import json
from argparse import Namespace

import pytest

from bench.neuroclaimbench import (
    BenchmarkItem,
    EvidenceStudyAssessment,
    LabelVote,
    SourceReference,
    derive_triage_reference,
    exact_contract_hash,
    semantic_contract_hash,
)
from bench.run_neuroclaimbench_reference_expansion import run
from confirm.contract import ClaimContract


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "claim",
            "question": "Is age associated with lower hippocampal volume?",
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
                    "require_covariates": ["sex"],
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


def _item(
    *,
    label: str = "candidate_unknown",
    status: str = "unresolved",
    score_eligible: bool = False,
) -> BenchmarkItem:
    contract = _contract()
    exact = exact_contract_hash(contract)
    semantic = semantic_contract_hash(contract)
    disposition = (
        "confirm"
        if label == "known_positive"
        else "abstain"
        if label in {"known_null", "fragile", "underpowered_small_positive"}
        else "unresolved"
    )
    return BenchmarkItem(
        benchmark_item_id="ncb-scientific-test",
        claim_uid=f"ncb-claim-{semantic[:16]}",
        semantic_cluster_id=f"ncb-sem-{semantic[:16]}",
        benchmark_track="scientific",
        target_family="normative_fmri",
        modality="sMRI",
        question=contract.question,
        contract=contract,
        exact_contract_sha256=exact,
        semantic_claim_sha256=semantic,
        source_references=[
            SourceReference(
                source_collection="stage2_current",
                source_id="claim",
                source_path="results.json",
                source_mode="llm_proposed",
                target_family="normative_fmri",
            )
        ],
        label_class=label,
        reference_disposition=disposition,
        adjudication_status=status,
        score_eligible=score_eligible,
    )


def _vote(
    model: str,
    *,
    label: str,
    relation: str,
    direct: bool = True,
    construct_match: str = "exact",
) -> LabelVote:
    assessment = EvidenceStudyAssessment(
        evidence_id="e1",
        study_design="single_cohort",
        directness="direct" if direct else "partial",
        relation=relation,
        population_match="exact",
        modality_match="exact",
        outcome_match="exact",
        direction_match="exact",
        independent_group="cohort",
        supporting_text="Matched evidence.",
    )
    return LabelVote(
        benchmark_item_id="ncb-scientific-test",
        model_spec=model,
        role=(
            "independent_adjudicator"
            if model.startswith("openrouter:")
            else "evidence_assessor"
        ),
        proposed_label=label,
        construct_match=construct_match,
        confidence="high",
        evidence_ids=["e1"],
        paper_assessments=[assessment],
        rationale="Evidence assessment.",
        prompt_sha256="a" * 64,
        response_sha256="b" * 64,
    )


def test_strict_reference_passes_through_unchanged():
    item = _item(
        label="known_positive",
        status="multi_model_consensus",
        score_eligible=True,
    )
    profile = derive_triage_reference(item, [])
    assert profile.triage_label == "supported"
    assert profile.reference_strength == "strict"
    assert profile.triage_disposition == "confirm"
    assert "strict_confirmation" in profile.score_tracks


def test_two_models_and_one_direct_study_create_provisional_support():
    item = _item()
    votes = [
        _vote("openai:gpt-5.5", label="known_positive", relation="supports_positive"),
        _vote(
            "google:gemini-3.5-flash",
            label="known_positive",
            relation="supports_positive",
        ),
        _vote(
            "openrouter:anthropic/claude-opus-4.8",
            label="candidate_unknown",
            relation="uninformative",
        ),
    ]
    profile = derive_triage_reference(item, votes)
    assert profile.triage_label == "supported"
    assert profile.reference_strength == "provisional"
    assert profile.supporting_evidence_ids == ["e1"]
    assert len(profile.agreeing_models) == 2
    assert profile.agreement_pattern == "assessors_only"


def test_direct_conflicting_evidence_becomes_provisional_mixed():
    item = _item()
    votes = [
        _vote("openai:gpt-5.5", label="known_positive", relation="supports_positive"),
        _vote(
            "google:gemini-3.5-flash",
            label="known_positive",
            relation="supports_positive",
        ),
        _vote(
            "openrouter:anthropic/claude-opus-4.8",
            label="fragile",
            relation="heterogeneous",
        ),
    ]

    profile = derive_triage_reference(item, votes)

    assert profile.triage_label == "fragile_or_mixed"
    assert profile.triage_disposition == "abstain"
    assert profile.reference_strength == "provisional"
    assert profile.derivation_rule == "two_model_reference_with_direct_conflicting_evidence"
    assert profile.agreement_pattern == "assessors_only"


def test_agreement_without_direct_evidence_remains_evidence_gap():
    item = _item()
    votes = [
        _vote(
            "openai:gpt-5.5",
            label="known_positive",
            relation="supports_positive",
            direct=False,
        ),
        _vote(
            "google:gemini-3.5-flash",
            label="known_positive",
            relation="supports_positive",
            direct=False,
        ),
    ]
    profile = derive_triage_reference(item, votes)
    assert profile.triage_label == "insufficient_evidence"
    assert profile.reference_strength == "evidence_gap"
    assert profile.triage_disposition == "request_evidence"


def test_provisional_null_requires_direct_null_evidence():
    item = _item()
    votes = [
        _vote("openai:gpt-5.5", label="known_null", relation="supports_null"),
        _vote(
            "google:gemini-3.5-flash",
            label="known_null",
            relation="supports_null",
        ),
    ]
    profile = derive_triage_reference(item, votes)
    assert profile.triage_label == "known_null"
    assert profile.reference_strength == "provisional"
    assert profile.triage_disposition == "abstain"


@pytest.mark.parametrize(
    ("migration_status", "alignment_disposition", "derivation_rule"),
    [
        (
            "ambiguous_unresolved",
            "ambiguous_unresolved",
            "question_contract_alignment_unresolved",
        ),
        ("non_executable", "non_executable", "alignment_non_executable"),
    ],
)
def test_nonready_alignment_cannot_be_promoted_by_model_votes(
    migration_status: str,
    alignment_disposition: str,
    derivation_rule: str,
):
    item = _item().model_copy(
        update={
            "migration_status": migration_status,
            "alignment_disposition": alignment_disposition,
        }
    )
    votes = [
        _vote("openai:gpt-5.5", label="known_positive", relation="supports_positive"),
        _vote(
            "google:gemini-3.5-flash",
            label="known_positive",
            relation="supports_positive",
        ),
    ]

    profile = derive_triage_reference(item, votes)

    assert profile.triage_label == "insufficient_evidence"
    assert profile.triage_disposition == "request_evidence"
    assert profile.reference_strength == "evidence_gap"
    assert profile.derivation_rule == derivation_rule
    assert profile.vote_counts == {"known_positive": 2}
    assert profile.score_tracks == ["evidence_triage"]
    assert not profile.executable


def test_runner_writes_non_destructive_reference_artifacts(tmp_path):
    package = tmp_path / "package"
    output = tmp_path / "output"
    package.mkdir()
    item = _item()
    votes = [
        _vote("openai:gpt-5.5", label="fragile", relation="heterogeneous"),
        _vote(
            "google:gemini-3.5-flash",
            label="fragile",
            relation="heterogeneous",
        ),
    ]
    (package / "benchmark_items.jsonl").write_text(
        item.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (package / "label_votes.jsonl").write_text(
        "".join(vote.model_dump_json() + "\n" for vote in votes),
        encoding="utf-8",
    )

    summary = run(
        Namespace(
            package_dir=str(package),
            out_dir=str(output),
            results=[],
        )
    )

    assert summary["inventory"]["n_items"] == 1
    assert summary["inventory"]["triage_label_counts"] == {
        "fragile_or_mixed": 1
    }
    assert summary["inventory"]["scored_reference_count"] == 1
    assert "confirmation_reference_count" not in summary["inventory"]
    profile = json.loads(
        (output / "triage_reference_profiles.jsonl").read_text(encoding="utf-8")
    )
    assert profile["reference_strength"] == "provisional"
    assert profile["agreement_pattern"] == "assessors_only"
    assert (output / "triage_reference_profiles.csv").exists()
    assert (output / "triage_summary.json").exists()
