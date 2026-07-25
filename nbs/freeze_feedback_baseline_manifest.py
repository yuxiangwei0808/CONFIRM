"""Freeze the artifact identities used by the feedback-method comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nbs.claim_search_analysis_common import sha256_file, write_json_atomic


def _provenance(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(
    *,
    method: str,
    track: str,
    root: Path,
    provenance_path: Path,
    expected_parents: int,
    budget: str,
) -> dict[str, Any]:
    provenance = _provenance(provenance_path)
    source = provenance.get("source") or {}
    return {
        "method": method,
        "track": track,
        "budget": budget,
        "artifact_root": str(root),
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "source_path": source.get("path"),
        "source_sha256": source.get("sha256"),
        "prompt_sha256": provenance.get("prompt_sha256"),
        "schema_sha256": provenance.get("schema_sha256"),
        "llm_model": provenance.get("llm_model"),
        "expected_parent_count": expected_parents,
        "excluded_evidence_query_count": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    structured = Path(args.structured_dir)
    generic = Path(args.generic_root)
    self_refine = Path(args.self_refine_root)
    entries = [
        _entry(
            method="failure_specific",
            track="scientific",
            root=structured,
            provenance_path=structured / "run_provenance.json",
            expected_parents=215,
            budget="R3/C5",
        ),
        _entry(
            method="failure_blind",
            track="scientific",
            root=generic,
            provenance_path=generic / "run_provenance.json",
            expected_parents=215,
            budget="R3/C5",
        ),
        _entry(
            method="self_refine",
            track="scientific",
            root=self_refine / "scientific",
            provenance_path=self_refine
            / "scientific/replay/run_provenance.json",
            expected_parents=215,
            budget="R3/C5",
        ),
        _entry(
            method="self_refine",
            track="known_negative",
            root=self_refine / "safety",
            provenance_path=self_refine / "safety/replay/run_provenance.json",
            expected_parents=150,
            budget="R3/C5",
        ),
    ]
    scientific_hashes = {
        entry["source_sha256"]
        for entry in entries
        if entry["track"] == "scientific"
    }
    if len(scientific_hashes) != 1 or None in scientific_hashes:
        raise ValueError(
            f"Scientific feedback arms do not share one frozen source: {scientific_hashes}"
        )
    for entry in entries:
        if entry["llm_model"] != "openai:gpt-5.5":
            raise ValueError(
                f"Unexpected model for {entry['method']}/{entry['track']}: "
                f"{entry['llm_model']}"
            )
    payload = {
        "version": "feedback-baseline-arm-manifest-v1",
        "outcome_blind": True,
        "arms": entries,
        "interpretation": [
            "scientific_arms_are_matched_at_r3_c5",
            "known_negative_self_refine_uses_all_150_parents",
            "excluded_evidence_is_not_queried_during_source_search",
            "one_gpt55_realization_descriptive_only",
        ],
    }
    destination = Path(args.out)
    write_json_atomic(destination, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structured-dir",
        default=(
            "review-stage/claim-search-gpt55-sweep-v7/"
            "normalized/arms/r3_c5"
        ),
    )
    parser.add_argument(
        "--generic-root",
        default=(
            "review-stage/claim-search-gpt55-control-r3-c5-v7/"
            "generic_retry"
        ),
    )
    parser.add_argument(
        "--self-refine-root",
        default="review-stage/claim-search-gpt55-self-refine-r3-c5-v1",
    )
    parser.add_argument(
        "--out",
        default=(
            "review-stage/claim-search-gpt55-feedback-baselines-v1/"
            "feedback_arm_manifest.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    payload = run(build_parser().parse_args(argv))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
