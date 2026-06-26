"""Evaluate CONFIRM as a post-hoc layer over NeuroClaw decisions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.metrics import exact_binomial_ci


def neuroclaw_binary(decision: str | None) -> str:
    """Map extracted NeuroClaw decisions to the paper's binary baseline."""

    return "CONFIRMS" if str(decision or "").strip().upper() == "CONFIRMS" else "NO-EFFECT"


def confirm_layer_decision(neuroclaw_decision: str | None, confirm_final_label: str | None) -> str:
    """Return the NeuroClaw->CONFIRM-layer decision."""

    if neuroclaw_binary(neuroclaw_decision) == "CONFIRMS" and str(confirm_final_label) == "confirmed":
        return "CONFIRMS"
    return "ABSTAIN"


def _rate(count: int, denominator: int) -> dict[str, Any]:
    ci = exact_binomial_ci(count, denominator)
    return {
        "count": int(count),
        "denominator": int(denominator),
        "rate": float(count / denominator) if denominator else float("nan"),
        "ci95_exact": [float(ci[0]), float(ci[1])],
    }


def _metric(rows: list[dict[str, Any]], label_classes: set[str], decision_key: str) -> dict[str, Any]:
    subset = [row for row in rows if row["label_class"] in label_classes]
    count = sum(row[decision_key] == "CONFIRMS" for row in subset)
    return _rate(count, len(subset))


def compute_layer_result(data: dict[str, Any]) -> dict[str, Any]:
    """Compute NeuroClaw-alone vs NeuroClaw+CONFIRM-layer FCR/TPR."""

    rows: list[dict[str, Any]] = []
    converted: list[str] = []
    for source in data.get("per_claim", []):
        label_class = str(source.get("label_class"))
        neuroclaw_alone = neuroclaw_binary(source.get("neuroclaw_decision"))
        layer = confirm_layer_decision(source.get("neuroclaw_decision"), source.get("confirm_final_label"))
        row = {
            "claim_id": source.get("claim_id"),
            "label_class": label_class,
            "neuroclaw_raw_decision": source.get("neuroclaw_decision"),
            "neuroclaw_alone_decision": neuroclaw_alone,
            "confirm_final_label": source.get("confirm_final_label"),
            "confirm_layer_decision": layer,
        }
        rows.append(row)
        if label_class in {"known_null", "fragile"} and neuroclaw_alone == "CONFIRMS" and layer != "CONFIRMS":
            converted.append(str(source.get("claim_id")))

    negative = {"known_null", "fragile"}
    positive = {"known_positive"}
    summary = {
        "neuroclaw_alone_FCR": _metric(rows, negative, "neuroclaw_alone_decision"),
        "neuroclaw_confirm_layer_FCR": _metric(rows, negative, "confirm_layer_decision"),
        "neuroclaw_alone_TPR": _metric(rows, positive, "neuroclaw_alone_decision"),
        "neuroclaw_confirm_layer_TPR": _metric(rows, positive, "confirm_layer_decision"),
    }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "CONFIRM post-hoc layer over NeuroClaw blinded decisions.",
        "source_model": data.get("model"),
        "n_claims": len(rows),
        **summary,
        "neuroclaw_false_confirms_converted_to_abstentions": converted,
        "claims": rows,
    }


def write_outputs(result: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "confirm_layer_result.json"
    csv_path = out_dir / "confirm_layer_result.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "claim_id",
            "label_class",
            "neuroclaw_raw_decision",
            "neuroclaw_alone_decision",
            "confirm_final_label",
            "confirm_layer_decision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["claims"])
    return json_path, csv_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = compute_layer_result(source)
    json_path, csv_path = write_outputs(result, Path(args.out_dir))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="review-stage/round5-neuroclaw/neuroclaw_comparison.json")
    parser.add_argument("--out-dir", default="review-stage/confirm-layer")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
