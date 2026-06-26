from __future__ import annotations

import argparse
import json

from bench.run_feedback_replay import run


def test_feedback_replay_writes_stable_outputs(tmp_path):
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "\n".join(
            [
                "claim_id,final_label,rationale,search_selection",
                "confirmed_claim,confirmed,All gates passed.,preregistered",
                "confounded_claim,fragile,Failed gates: confound; predictor is nested in a declared confound.,preregistered",
                "fishing_claim,fragile,Failed gates: multiplicity,discovery_only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    result = run(argparse.Namespace(input=[str(audit)], out_dir=str(out_dir)))

    assert result["summary"]["n_feedback"] == 3
    assert result["summary"]["n_abstentions"] == 2
    assert result["summary"]["feedback_coverage"] == 1.0
    assert (out_dir / "feedback_replay.json").exists()
    assert (out_dir / "feedback_replay.csv").exists()

    payload = json.loads((out_dir / "feedback_replay.json").read_text(encoding="utf-8"))
    failures = {item["claim_id"]: item["primary_failure"] for item in payload["feedback"]}
    assert failures["confounded_claim"] == "confound"
    assert failures["fishing_claim"] == "search_provenance"
