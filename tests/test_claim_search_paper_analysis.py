from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from nbs.analyze_claim_search_evidence import _case_cell, _matched_rows, _summary_rows
from nbs.analyze_claim_search_sweep import _project_lineage, _transform_failure_rows
from nbs.candidate_novelty_review import (
    BLINDED_COLUMNS,
    ForcedChoiceBatch,
    _assert_blinded,
    _blind_item,
    novelty_metrics,
    run_forced_choice_review,
)
from nbs.claim_search_analysis_common import clustered_binary_interval, merge_analysis_manifest


def _contract(claim_id: str, outcome: str) -> dict:
    return {
        "claim_id": claim_id,
        "question": f"Question about {outcome}",
        "estimand": {
            "type": "association",
            "outcome": outcome,
            "predictor": "age",
            "direction": "positive",
            "unit": "scalar",
            "region_set": None,
        },
        "covariates": ["sex"],
        "inclusion": None,
        "discovery_cohort": "DISC",
        "replication_cohorts": ["REP"],
        "search_provenance": {"family_size": 1},
        "gates": {"multiplicity": {"family_size": 1}},
        "reporting_language_allowed": ["confirmed"],
    }


def _exposure(*, supported: bool = True, retracted: bool = False) -> dict:
    parent = _contract("parent", "smri_a")
    candidate = _contract("candidate", "smri_b")
    return {
        "exposure_id": "exposure",
        "lineage_event_id": "r3_c5:parent",
        "arm_id": "r3_c5",
        "parent_claim_id": "parent",
        "candidate_id": "candidate",
        "target_family": "ad_aging",
        "source_mode": "llm_proposed",
        "round_index": 1,
        "generation_status": "retained",
        "inferred_transform": "alternative_outcome",
        "executable_contract_delta": {"estimand.outcome": {"parent": "smri_a", "candidate": "smri_b"}},
        "parent_contract": parent,
        "effective_contract": candidate,
        "validation_ok": True,
        "current_data_evaluated": True,
        "final_internal_supported": supported,
        "provisional_internal_supported": supported or retracted,
        "multiplicity_retracted": retracted,
        "exact_contract_id": "exact",
        "semantic_cluster_id": "semantic",
    }


def test_parent_clustered_bootstrap_is_deterministic():
    first = clustered_binary_interval([0, 1, 1, 0], resamples=200, seed=20260721)
    second = clustered_binary_interval([0, 1, 1, 0], resamples=200, seed=20260721)

    assert first == second
    assert first[0] == 0.5


def test_analysis_manifest_merges_phase_sections(tmp_path):
    manifest = tmp_path / "analysis_manifest.json"
    output = tmp_path / "output.csv"
    output.write_text("value\n1\n", encoding="utf-8")

    merge_analysis_manifest(
        manifest,
        section_name="sweep_analysis",
        section_payload={"rows": 1},
        inputs=[],
        outputs=[output],
        restrictions=("sweep restriction",),
    )
    merge_analysis_manifest(
        manifest,
        section_name="retrospective_evidence_analysis",
        section_payload={"rows": 2},
        inputs=[],
        outputs=[output],
        restrictions=("evidence restriction",),
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["sweep_analysis"] == {"rows": 1}
    assert payload["retrospective_evidence_analysis"] == {"rows": 2}
    assert payload["interpretation_restrictions"] == [
        "evidence restriction",
        "sweep restriction",
    ]


def test_novelty_metrics_detect_material_parent_relative_change():
    row = novelty_metrics([_exposure()])[0]

    assert row["novelty_class"] == "materially_different_connected_hypothesis"
    assert row["outcome_set_jaccard"] == 0.0
    assert row["fixed_estimand_assessability"] == "not_assessable"


def test_novelty_metrics_ignore_adaptive_family_size_bookkeeping():
    exposure = _exposure()
    exposure["effective_contract"] = json.loads(json.dumps(exposure["parent_contract"]))
    exposure["effective_contract"]["claim_id"] = "candidate"
    exposure["effective_contract"]["question"] = "Reworded question"
    exposure["effective_contract"]["search_provenance"]["family_size"] = 25
    exposure["effective_contract"]["search_provenance"]["selection"] = "adaptive_followup"
    exposure["effective_contract"]["gates"]["multiplicity"]["family_size"] = 25

    row = novelty_metrics([exposure])[0]

    assert row["changed_executable_path_count"] == 0
    assert row["novelty_class"] == "no_op"


def test_blinded_packet_rejects_outcome_metadata_columns():
    exposure = _exposure()
    item = _blind_item(exposure, item_id="item_001")
    _assert_blinded([item])
    leaked = {**item, "support_status": "supported"}

    with pytest.raises(ValueError, match="unexpected columns"):
        _assert_blinded([leaked])


def test_forced_choice_review_is_strict_balanced_and_resumable(tmp_path):
    packet = []
    key_rows = []
    for pair_index in range(1, 51):
        pair_id = f"pair_{pair_index:03d}"
        for offset, group in enumerate(("control_structured", "control_generic")):
            item_id = f"item_{2 * pair_index - 1 + offset:03d}"
            packet.append(
                _blind_item(_exposure(), item_id=item_id, pair_id=pair_id)
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "review_group": group,
                    "parent_claim_id": f"parent_{pair_index:03d}",
                }
            )
    with (tmp_path / "forced_choice_candidate_packet.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=BLINDED_COLUMNS)
        writer.writeheader()
        writer.writerows(packet)
    (tmp_path / "forced_choice_candidate_key.jsonl").write_text(
        "\n".join(json.dumps(row) for row in key_rows)
        + "\n",
        encoding="utf-8",
    )

    class FakeStructuredClient:
        def __init__(self, model: str):
            self.model = model
            self.calls = 0
            self.last_call_metadata = {
                "provider": model.split(":", 1)[0],
                "model": model,
            }

        def complete_structured(self, _system, prompt, _schema):
            self.calls += 1
            pair_ids = [pair["pair_id"] for pair in json.loads(prompt)["pairs"]]
            return json.dumps(
                {
                    "decisions": [
                        {
                            "pair_id": pair_id,
                            "preferred_candidate": "A",
                            "reason": "Candidate A is the better connected follow-up.",
                        }
                        for pair_id in pair_ids
                    ]
                }
            )

    models = (
        "google:test-reviewer",
        "openrouter:anthropic/test-reviewer",
        "openrouter:deepseek/test-reviewer",
    )
    clients = {model: FakeStructuredClient(model) for model in models}
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        reviewer_model=list(models),
        batch_size=10,
        max_output_tokens=4096,
        schema_retries=1,
    )

    first = run_forced_choice_review(args, clients=clients)
    second = run_forced_choice_review(args, clients=clients)

    assert first["decision_count"] == 150
    assert first["checkpoint_reuse_count"] == 0
    assert second["checkpoint_reuse_count"] == 15
    assert sum(client.calls for client in clients.values()) == 15
    assignments = [
        json.loads(line)
        for line in (tmp_path / "forced_choice_assignment_key.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for model in models:
        model_rows = [row for row in assignments if row["model_spec"] == model]
        assert sum(row["structured_label"] == "A" for row in model_rows) == 25
        assert sum(row["structured_label"] == "B" for row in model_rows) == 25
    for pair_index in range(1, 51):
        pair_id = f"pair_{pair_index:03d}"
        pair_rows = [row for row in assignments if row["pair_id"] == pair_id]
        structured_as_a = sum(row["structured_label"] == "A" for row in pair_rows)
        assert structured_as_a in {1, 2}


def test_forced_choice_schema_has_only_four_decisions():
    schema = ForcedChoiceBatch.model_json_schema()
    properties = schema["$defs"]["ForcedChoiceDecision"]["properties"]

    assert properties["preferred_candidate"]["enum"] == ["A", "B", "tie", "neither"]


def test_evidence_summary_keeps_external_sets_separate_and_matches_parent_cells():
    evidence = []
    for evidence_set, supported in (("NACC", 1), ("CNP", 0)):
        evidence.append(
            {
                "lineage_event_id": "r3_c5:parent",
                "arm_id": "r3_c5",
                "parent_claim_id": "parent",
                "candidate_id": "candidate",
                "target_family": "ad_aging",
                "source_mode": "llm_proposed",
                "evidence_kind": "external",
                "evidence_set_id": evidence_set,
                "compatible": 1,
                "evaluated": 1,
                "supported": supported,
                "execution_error": 0,
                "preflight_status": "eligible",
            }
        )
    evidence.append(
        {
            "lineage_event_id": "r3_c5:parent",
            "arm_id": "r3_c5",
            "parent_claim_id": "parent",
            "candidate_id": "candidate",
            "target_family": "ad_aging",
            "source_mode": "llm_proposed",
            "evidence_kind": "holdout",
            "evidence_set_id": None,
            "compatible": 1,
            "evaluated": 1,
            "supported": 1,
            "execution_error": 0,
            "preflight_status": "eligible",
            "matched_parent_holdout": 1,
            "parent_holdout_evaluated": 1,
            "parent_holdout_supported": 0,
        }
    )
    lineages = [{"arm_id": "r3_c5", "parent_claim_id": "parent", "internally_supported_candidate_ids": ["candidate"]}]

    summary = _summary_rows(evidence, lineages)
    external_sets = {
        row["dimension_value"]
        for row in summary
        if row["evidence_kind"] == "external" and row["dimension"] == "evidence_set"
    }
    matched = _matched_rows(evidence)

    assert external_sets == {"NACC", "CNP"}
    assert matched[0]["matched_outcome_cell"] == "candidate_only"


def test_case_selection_is_stable_under_row_reordering():
    exposure = _exposure(retracted=True)
    evidence = [
        {
            "arm_id": "r3_c5",
            "parent_claim_id": parent,
            "candidate_id": f"candidate_{parent}",
            "lineage_event_id": f"r3_c5:{parent}",
            "target_family": "ad_aging",
            "evidence_kind": "holdout",
            "evaluated": 1,
            "supported": 0,
            "candidate_only_holdout_support": 0,
            "matched_parent_holdout": 1,
            "parent_holdout_supported": 0,
        }
        for parent in ("b", "a")
    ]
    lineages = [
        {
            "arm_id": "r10_c10",
            "parent_claim_id": "asd_parent",
            "target_family": "asd",
            "internally_supported_candidate_ids": [],
        }
    ]

    forward = _case_cell(evidence, [exposure], lineages)
    reverse = _case_cell(list(reversed(evidence)), [exposure], lineages)

    assert json.dumps(forward, sort_keys=True) == json.dumps(reverse, sort_keys=True)


def test_failure_transition_reads_nested_round_failure_contexts():
    exposure = _exposure(supported=False)
    lineages = [
        {
            "lineage_event_id": "r3_c5:parent",
            "arm_id": "r3_c5",
            "parent_claim_id": "parent",
            "failure_localization": {"failed_gates": ["multiplicity", "replication"]},
            "round_failure_contexts": [
                {
                    "round_index": 1,
                    "failed_candidates": [
                        {"candidate_id": "candidate", "failed_gates": ["replication"]}
                    ],
                }
            ],
        }
    ]

    rows = _transform_failure_rows([exposure], lineages, resamples=20, seed=7)
    transitions = [row for row in rows if row["dimension"] == "parent_failure_transition"]

    assert {row["parent_failed_gate"] for row in transitions} == {"multiplicity", "replication"}
    assert next(row for row in transitions if row["parent_failed_gate"] == "multiplicity")[
        "candidate_satisfied_parent_gate"
    ] == 1


def test_lineage_projection_keeps_compact_failure_transition_fields():
    lineage = {
        "arm_id": "r3_c5",
        "lineage_event_id": "r3_c5:parent",
        "parent_claim_id": "parent",
        "round_failure_contexts": [
            {
                "round_index": 1,
                "failed_candidates": [
                    {
                        "candidate_id": "candidate",
                        "failed_gates": ["replication"],
                        "effective_contract": {"large": "payload"},
                    }
                ],
            }
        ],
    }

    projected = _project_lineage(lineage)

    assert projected["round_failure_contexts"][0]["failed_candidates"] == [
        {"candidate_id": "candidate", "failed_gates": ["replication"]}
    ]
