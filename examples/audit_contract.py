"""Evaluate a plan supplied by another agent using CONFIRM's public API."""
import argparse
import json
from pathlib import Path
from confirm import load_contract
from confirm.execution import evaluate_contract

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--contract", default="outputs/demo/contracts/supported.yaml")
parser.add_argument("--data-dir", type=Path, default=Path("outputs/demo/cohorts"))
args = parser.parse_args()
contract = load_contract(args.contract)
verdict, results, inputs = evaluate_contract(
    contract, args.data_dir, ref_effect=contract.gates.power.ref_effect,
    minimum_evidence_tier="confirmed",
)
print(json.dumps({"decision": results["support_decision"].to_dict(),
                  "diagnosis": verdict.rationale}, indent=2))
