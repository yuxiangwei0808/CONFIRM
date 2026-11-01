"""Generate the visual guide's examples from CONFIRM's synthetic demo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from confirm.demo import run_demo
from confirm.execution import jsonable

SITE = ROOT / "demo/site/guide"
DESCRIPTIONS = {
    "supported": {
        "title": "Repeats in a new sample",
        "short": "Passes all five checks",
        "question": "Does the imaging measurement decrease with age?",
        "setup": "The measurement decreases with age in both independently simulated samples.",
        "lesson": "The association passes all five checks, including the test in a separate sample.",
    },
    "site_confounded": {
        "title": "Explained by scanning site",
        "short": "Site was left out of the plan",
        "question": "Does the imaging measurement increase with age?",
        "setup": "The simulated measurement differs by scanning site, and participants at one site are older. The initial plan leaves out site.",
        "lesson": "The apparent link with age comes from a site difference that the plan does not account for.",
    },
    "nonreplicating": {
        "title": "Does not repeat",
        "short": "Fails the replication check",
        "question": "Does the imaging measurement decrease with age?",
        "setup": "The simulated measurement decreases with age in the first sample. The second sample has no age effect.",
        "lesson": "Screen accepts the claim. Replicate and Full withhold support because the association fails the test in the second sample.",
    },
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def build_examples() -> None:
    with tempfile.TemporaryDirectory(prefix="confirm-guide-") as temporary:
        root = Path(temporary) / "demo"
        summary = run_demo(root)
        cohorts = {name: pd.read_parquet(root / f"cohorts/{name}.parquet")
                   for name in ("DEMO_DISC", "DEMO_REP")}
        examples = []
        for key, description in DESCRIPTIONS.items():
            receipt = json.loads((root / f"results/{key}/receipt.json").read_text())
            results = receipt["results"]
            contract = results["contract"]
            record = {
                "example": key, "synthetic": True,
                "framework_version": receipt["environment"]["packages"]["confirm"],
                "contract_sha256": receipt["contract"]["sha256"],
                "cohort_sha256": [entry["sha256"] for entry in receipt["cohorts"]],
                "environment": receipt["environment"]["packages"], "results": results,
            }
            write_json(SITE / f"data/{key}-record.json", record)
            write_json(SITE / f"data/{key}-plan.json", contract)
            points = {}
            for cohort, frame in cohorts.items():
                sample = frame.sample(n=180, random_state=17).sort_index()
                points[cohort] = [[round(float(row.age), 3),
                                   round(float(row[contract["estimand"]["outcome"]]), 5), row.site]
                                  for _, row in sample.iterrows()]
            examples.append({"id": key, **description, "contract": contract,
                             "contract_sha256": record["contract_sha256"],
                             "points": points, "results": results,
                             "support": summary["results"][key]["support"]})
        write_json(SITE / "data/examples.json", {
            "synthetic": True, "llm_calls": 0,
            "seeds": {"discovery": 1701, "replication": 1702},
            "observations_per_cohort": 600, "displayed_points_per_cohort": 180,
            "framework_version": record["framework_version"], "examples": examples,
        })


def extract_followup(source: Path) -> None:
    """Export aggregate fields from one recorded agent checkpoint."""
    raw = source.read_bytes()
    checkpoint = json.loads(raw)
    state = checkpoint["state"]
    assert checkpoint["claim_id"] == "neg_site_confound_ukb_s1"
    assert len(state["evaluations"]) == 2
    candidates = []
    for evaluation in state["evaluations"]:
        proposal = evaluation["proposal"]
        result = evaluation["gate_results"]
        primary = result.get("primary")
        aggregate = ({key: primary[key] for key in ["beta", "ci_low", "ci_high", "p", "n"]}
                     if primary else None)
        region_list = (result.get("regions") or {}).get("regions", [])
        candidates.append({
            "id": evaluation["candidate_id"], "question": proposal["proposed_question"],
            "type": proposal["proposal_type"], "contract": proposal["proposed_contract"],
            "validation": {key: evaluation["validation"][key] for key in ["ok", "violations", "warnings"]},
            "evaluated": evaluation["evaluated"], "label": evaluation["final_label"],
            "supported": evaluation["current_data_supported"], "primary": aggregate,
            "region_count": len(region_list),
            "significant_regions": sum(r["significant"] for r in region_list),
            "gates": {key: value for key, value in result["verdict"]["gates"].items()
                      if type(value) is bool},
            "final_family_size": evaluation["final_family_size"],
        })
    write_json(SITE / "data/followup.json", {
        "kind": "recorded_benchmark_control", "source_sha256": hashlib.sha256(raw).hexdigest(),
        "claim_id": checkpoint["claim_id"], "model": checkpoint["row"]["candidate_generator_model_spec"],
        "parent_contract": state["original_claim"],
        "parent_label": state["failure_localization"]["source_verdict"],
        "failed_parent_gates": state["failure_localization"]["failed_gates"],
        "candidates": candidates, "rounds": state["round_summaries"],
        "final_family_size": state["final_search_family_size"],
        "stopped_reason": state["stopped_reason"],
        "description": "A recorded agent run on a site-confounded benchmark control. The export contains plans, validation decisions, and aggregate results. It contains no participant records or model prompts.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-source", type=Path,
                        help="Optional original checkpoint for refreshing the recorded follow-up export")
    args = parser.parse_args()
    build_examples()
    if args.followup_source:
        extract_followup(args.followup_source)
    print("Visual-guide data generated in demo/site/guide/data")
