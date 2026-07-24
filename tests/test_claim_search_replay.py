from __future__ import annotations

import json

import pandas as pd
import pytest

from bench import run_iterative_claim_search_replay as replay
from tests.test_claim_search import _FakeCandidateLLM, _contract, _results, _verdict


class _CountingLLM(_FakeCandidateLLM):
    model = "fake:counting"

    def __init__(self, counter: dict[str, int]) -> None:
        self.counter = counter

    def complete(self, system, user):
        self.counter["calls"] += 1
        return super().complete(system, user)


def test_replay_resume_skips_completed_parent_without_repeating_llm_calls(tmp_path, monkeypatch):
    contract = _contract()
    data_root = tmp_path / "cohorts"
    data_root.mkdir()
    frame = pd.DataFrame(
        {
            "subject_id": [f"sub-{index}" for index in range(40)],
            "site": ["site1", "site2"] * 20,
            "age": [65 + index % 5 for index in range(40)],
            "sex": ["F", "M"] * 20,
            "dx": ["Dementia", "CN"] * 20,
            "smri_hippocampus": [1.0 + index * 0.01 for index in range(40)],
            "smri_entorhinal": [2.0 + index * 0.01 for index in range(40)],
        }
    )
    frame.to_parquet(data_root / "ADNI.parquet")
    frame.to_parquet(data_root / "OASIS3.parquet")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_spec": "source-model",
                        "initial_claims": [
                            {
                                "claim_id": contract.claim_id,
                                "draft_success": True,
                                "gate_success": True,
                                "estimand_match": True,
                                "gate_verdict_label": "fragile",
                                "gate_verdict": _verdict(),
                                "gate_results": _results(contract),
                                "drafted_contract": contract.model_dump(mode="json"),
                                "target_family": "ad_aging",
                                "source_mode": "llm_proposed",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    counter = {"calls": 0}
    monkeypatch.setattr(replay, "make_llm", lambda _: _CountingLLM(counter))
    out_dir = tmp_path / "out"
    argv = [
        "--input",
        str(source),
        "--out-dir",
        str(out_dir),
        "--llm",
        "fake:counting",
        "--max-rounds",
        "1",
        "--max-candidates",
        "2",
        "--candidate-evaluation",
        "off",
        "--data-root",
        str(data_root),
        "--checkpoint-every",
        "1",
        "--expected-parent-count",
        "1",
        "--no-progress",
    ]

    first = replay.run(replay.build_parser().parse_args(argv))
    first_call_count = counter["calls"]
    parent_checkpoints = list((out_dir / "checkpoints" / "parents").glob("parent_*.json"))
    assert len(parent_checkpoints) == 1

    # Simulate a checkpoint written by the older implementation-sensitive identity.
    provenance_path = out_dir / "run_provenance.json"
    prior_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    legacy_identity = {
        **prior_provenance["resume_identity"],
        "implementation_hashes_sha256": "legacy-implementation-hash",
    }
    legacy_identity_sha256 = replay._sha256_json(legacy_identity)
    prior_provenance["resume_identity"] = legacy_identity
    prior_provenance["resume_identity_sha256"] = legacy_identity_sha256
    provenance_path.write_text(json.dumps(prior_provenance), encoding="utf-8")
    checkpoint_payload = json.loads(parent_checkpoints[0].read_text(encoding="utf-8"))
    checkpoint_payload.pop("checkpoint_sha256")
    checkpoint_payload["resume_identity_sha256"] = legacy_identity_sha256
    checkpoint_payload["checkpoint_sha256"] = replay._sha256_json(checkpoint_payload)
    parent_checkpoints[0].write_text(json.dumps(checkpoint_payload), encoding="utf-8")

    (out_dir / "iterative_candidate_replay.json").unlink()
    (out_dir / "iterative_candidate_replay.csv").unlink()
    second = replay.run(replay.build_parser().parse_args(argv))

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert first_call_count > 0
    assert counter["calls"] == first_call_count
    assert second["completed_search_count"] == 1
    ledger = second["states"][0]["source_metadata"]["source_evidence_ledger"]
    assert [item["role"] for item in ledger] == ["discovery", "replication"]
    assert [item["cohort"] for item in ledger] == ["ADNI", "OASIS3"]
    assert all(item["partition_hash"] for item in ledger)
    assert {item["partition_hash_kind"] for item in ledger} == {"content_sha256"}
    assert second["provenance"]["implementation_hashes"]
    assert second["provenance"]["search_implementation_hashes"]
    assert second["provenance"]["search_implementation_hashes_sha256"]
    assert "implementation_hashes_sha256" not in second["provenance"]["resume_identity"]
    assert legacy_identity_sha256 in second["provenance"][
        "compatible_resume_identity_sha256s"
    ]
    assert second["provenance"]["compatible_resume_identities"][0][
        "resume_identity_sha256"
    ] == legacy_identity_sha256
    assert second["provenance"]["compatible_resume_identities"][0][
        "implementation_hashes"
    ]
    assert second["provenance"]["partition_hashes_sha256"] == second["provenance"][
        "resume_identity"
    ]["partition_hashes_sha256"]

    checkpoint_payload = json.loads(parent_checkpoints[0].read_text(encoding="utf-8"))
    checkpoint_payload.pop("checkpoint_sha256")
    checkpoint_payload["state"].update(
        {
            "candidate_history": [],
            "evaluations": [],
            "internally_supported_candidate_ids": [],
            "generated_candidate_count": 0,
            "schema_valid_candidate_count": 0,
            "unique_candidate_count": 0,
            "valid_candidate_count": 0,
            "current_data_evaluated_count": 0,
            "llm_candidate_prompts": [{"prompt": "transport-failed"}],
            "llm_candidate_responses": [
                {"candidate_count": 0, "parse_error": "Connection error."}
            ],
            "stopped_reason": "candidate_generation_failed",
        }
    )
    checkpoint_payload["row"]["stopped_reason"] = "candidate_generation_failed"
    checkpoint_payload["checkpoint_sha256"] = replay._sha256_json(checkpoint_payload)
    parent_checkpoints[0].write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    calls_before_transient_retry = counter["calls"]

    third = replay.run(replay.build_parser().parse_args(argv))

    assert counter["calls"] > calls_before_transient_retry
    assert third["status"] == "completed"
    assert third["provenance"]["superseded_transient_prompt_record_count"] == 1
    assert third["provenance"]["total_prompt_attempt_record_count"] == (
        third["provenance"]["rendered_prompt_record_count"] + 1
    )

    checkpoint_payload = json.loads(parent_checkpoints[0].read_text(encoding="utf-8"))
    checkpoint_payload["row"]["target_family"] = "tampered"
    parent_checkpoints[0].write_text(json.dumps(checkpoint_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint hash"):
        replay.run(replay.build_parser().parse_args(argv))


def test_resume_identity_compatibility_ignores_only_implementation_hash():
    current = {
        "source_sha256": "source",
        "config": {"max_rounds": 3},
        "llm_model": "openai:gpt-5.5",
    }
    legacy = {**current, "implementation_hashes_sha256": "implementation"}

    assert replay._resume_identities_compatible(current, legacy)
    assert not replay._resume_identities_compatible(
        current,
        {**legacy, "source_sha256": "different-source"},
    )
