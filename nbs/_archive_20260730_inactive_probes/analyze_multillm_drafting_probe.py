"""Summarise the multi-model drafting probe.

Every model drafts a contract for the same fixed question set, and the
unchanged gates then score each drafted contract. The probe therefore separates
two things that a single-model run cannot: how much the drafted contract varies
with the model, and how much the resulting verdict varies. A governance layer
should absorb the former without propagating it into the latter.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

SUPPORTED_VERDICTS = {"confirmed"}

# Synthetic-stress prompts describe a planted null that exists only in the frozen
# control contract. A freely drafting model re-specifies the estimand (dropping the
# random-group predictor, or substituting a real contrast), so a supported verdict
# on these rows measures drafting drift and is not a false confirmation of the
# planted null. They are reported separately and excluded from agreement metrics.
DRAFTER_RESPECIFIED_MODE = "synthetic_stress"
COMPARABLE_MODE = "literature_grounded"


def _read_audit(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _read_drafts(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["claim_id"]] = row
    return rows


# Free text carries no decision weight; comparing it would report spurious differences.
IGNORED_CONTRACT_FIELDS = {"question", "claim_id", "reporting_language_allowed"}

# Fields that decide the verdict, compared separately so the probe can say *which*
# part of the contract the drafter varied.
CONTRACT_FACETS = ("estimand", "covariates", "gates", "search_provenance", "inclusion")


def _hashable(value):
    """Canonicalise a contract fragment. Lists in this schema are semantically
    unordered sets, so sort them to avoid reporting ordering as a difference."""
    if isinstance(value, list):
        items = [_hashable(v) for v in value]
        try:
            return tuple(sorted(items, key=repr))
        except TypeError:
            return tuple(items)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _facet_of(contract: dict | None, facet: str):
    if not isinstance(contract, dict):
        return None
    return _hashable(contract.get(facet))


def _full_fingerprint(contract: dict | None):
    if not isinstance(contract, dict):
        return None
    return _hashable({k: v for k, v in contract.items() if k not in IGNORED_CONTRACT_FIELDS})


def _estimand_of(contract: dict | None) -> tuple | None:
    """Reduce a drafted contract to the fields that define what was tested."""
    if not isinstance(contract, dict):
        return None
    est = contract.get("estimand") or {}
    group = est.get("group") or {}
    return _hashable(
        [
            est.get("type"),
            est.get("outcome"),
            est.get("direction"),
            est.get("unit"),
            group.get("var"),
            group.get("case"),
            group.get("control"),
            contract.get("discovery_cohort"),
            contract.get("replication_cohorts"),
        ]
    )


def collect(root: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    per_model: dict[str, list[dict]] = {}
    drafts: dict[str, dict] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        audit = model_dir / "gates" / "claim_gate_audit.csv"
        if not audit.exists():
            continue
        rows = _read_audit(audit)
        if not rows:
            continue
        model = rows[0].get("model_spec") or model_dir.name
        per_model[model] = rows
        drafts[model] = _read_drafts(model_dir / "drafted_contracts.jsonl")
    return per_model, drafts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="review-stage/multillm-probe-v2")
    parser.add_argument("--out-dir", default="review-stage/multillm-probe-v2/_summary")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"no probe output at {root}", file=sys.stderr)
        return 1
    per_model, drafts = collect(root)
    if not per_model:
        print(f"no completed model runs under {root}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"models with completed runs: {len(per_model)}\n")
    print("Literature-grounded questions (the comparable set):")
    print(f"{'model':40s} {'drafted':>9s} {'gated':>7s} {'confirmed':>10s}   {'synth-respec':>12s}")
    rows_out = []
    for model, rows in per_model.items():
        lit = [r for r in rows if r.get("source_mode") == COMPARABLE_MODE]
        synth = [r for r in rows if r.get("source_mode") == DRAFTER_RESPECIFIED_MODE]
        drafted = sum(r.get("draft_success") == "True" for r in lit)
        gated = sum(r.get("gate_success") == "True" for r in lit)
        confirmed = sum(r.get("gate_verdict_label") in SUPPORTED_VERDICTS for r in lit)
        synth_supported = sum(r.get("gate_verdict_label") in SUPPORTED_VERDICTS for r in synth)
        print(
            f"{model:40s} {drafted:>4d}/{len(lit):<4d} {gated:>7d} {confirmed:>10d}   "
            f"{synth_supported:>4d}/{len(synth):<4d}"
        )
        rows_out.append(
            {
                "model_spec": model,
                "literature_questions": len(lit),
                "draft_success": drafted,
                "gate_success": gated,
                "confirmed": confirmed,
                "synthetic_questions": len(synth),
                "synthetic_supported_after_respecification": synth_supported,
            }
        )

    # Verdict agreement across models, restricted to the comparable question set.
    per_claim: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for model, rows in per_model.items():
        for r in rows:
            if r.get("source_mode") != COMPARABLE_MODE:
                continue
            per_claim[r["claim_id"]][model] = r.get("gate_verdict_label") or "none"
    complete = {c: v for c, v in per_claim.items() if len(v) == len(per_model)}
    unanimous = [c for c, v in complete.items() if len(set(v.values())) == 1]
    print(
        f"\nverdict agreement: {len(unanimous)}/{len(complete)} claims unanimous "
        f"across {len(per_model)} models"
    )

    # Contract agreement, on the whole contract rather than the estimand alone.
    same_contract_ids: list[str] = []
    diff_contract_ids: list[str] = []
    facet_diffs: collections.Counter = collections.Counter()
    for claim_id in complete:
        prints = set()
        ok = True
        for model in per_model:
            row = drafts.get(model, {}).get(claim_id)
            fp = _full_fingerprint((row or {}).get("drafted_contract"))
            if fp is None:
                ok = False
                break
            prints.add(fp)
        if not ok:
            continue
        if len(prints) == 1:
            same_contract_ids.append(claim_id)
            continue
        diff_contract_ids.append(claim_id)
        for facet in CONTRACT_FACETS:
            vals = {
                _facet_of((drafts.get(m, {}).get(claim_id) or {}).get("drafted_contract"), facet)
                for m in per_model
            }
            if len(vals) > 1:
                facet_diffs[facet] += 1
    comparable = len(same_contract_ids) + len(diff_contract_ids)
    print(f"contract agreement: {len(same_contract_ids)}/{comparable} claims drafted identically")
    if facet_diffs:
        print("  contract fields that diverge (of the claims that differ):")
        for facet, n in facet_diffs.most_common():
            print(f"    {facet:20s} {n}/{len(diff_contract_ids)}")
    same_estimand_ids, diff_estimand_ids = same_contract_ids, diff_contract_ids

    # The decisive split: when every model drafts the same contract, the gates are
    # deterministic and should agree. Residual disagreement there would indicate
    # nondeterminism in the pipeline rather than drafting variation.
    def _unanimous_rate(ids: list[str]) -> tuple[int, int]:
        agree = sum(1 for c in ids if len(set(complete[c].values())) == 1)
        return agree, len(ids)

    same_agree, same_n = _unanimous_rate(same_contract_ids)
    diff_agree, diff_n = _unanimous_rate(diff_contract_ids)
    print(
        f"\n  verdict agreement | identical contract : {same_agree}/{same_n}"
        + (f" ({same_agree / same_n:.0%})" if same_n else "")
        + "   <- gate determinism"
    )
    print(
        f"  verdict agreement | differing contract : {diff_agree}/{diff_n}"
        + (f" ({diff_agree / diff_n:.0%})" if diff_n else "")
        + "   <- drafting variation"
    )

    # With more than two models an all-models-identical contract is vanishingly rare,
    # so the determinism check is computed over model pairs instead.
    import itertools

    pair_same_agree = pair_same_n = pair_diff_agree = pair_diff_n = 0
    for m1, m2 in itertools.combinations(sorted(per_model), 2):
        for claim_id, verdicts in complete.items():
            c1 = (drafts.get(m1, {}).get(claim_id) or {}).get("drafted_contract")
            c2 = (drafts.get(m2, {}).get(claim_id) or {}).get("drafted_contract")
            f1, f2 = _full_fingerprint(c1), _full_fingerprint(c2)
            if f1 is None or f2 is None:
                continue
            agree = verdicts.get(m1) == verdicts.get(m2)
            if f1 == f2:
                pair_same_n += 1
                pair_same_agree += agree
            else:
                pair_diff_n += 1
                pair_diff_agree += agree
    print("\n  pairwise over model pairs:")
    print(
        f"    identical contract : {pair_same_agree}/{pair_same_n}"
        + (f" ({pair_same_agree / pair_same_n:.0%})" if pair_same_n else "")
        + "   <- gate determinism"
    )
    print(
        f"    differing contract : {pair_diff_agree}/{pair_diff_n}"
        + (f" ({pair_diff_agree / pair_diff_n:.0%})" if pair_diff_n else "")
        + "   <- drafting variation"
    )

    disagreements = sorted(set(complete) - set(unanimous))
    if disagreements:
        print(f"\nclaims with a split verdict ({len(disagreements)}):")
        for claim_id in disagreements[:15]:
            verdicts = collections.Counter(complete[claim_id].values())
            print(f"  {claim_id[:52]:54s} {dict(verdicts)}")

    summary = {
        "models": rows_out,
        "claims_scored_by_all_models": len(complete),
        "unanimous_verdicts": len(unanimous),
        "identical_contract_claims": len(same_contract_ids),
        "contract_comparable_claims": comparable,
        "diverging_contract_fields": dict(facet_diffs),
        "verdict_agreement_given_identical_contract": [same_agree, same_n],
        "verdict_agreement_given_differing_contract": [diff_agree, diff_n],
        "pairwise_agreement_identical_contract": [pair_same_agree, pair_same_n],
        "pairwise_agreement_differing_contract": [pair_diff_agree, pair_diff_n],
        "split_verdict_claims": disagreements,
        "interpretation_restrictions": [
            "Every model drafts from the same fixed question set; gates are unchanged across models.",
            "Verdict agreement measures reporting stability, not correctness.",
            "Agreement metrics cover literature-grounded questions only.",
            "Synthetic-stress prompts are re-specified by a freely drafting model, so a supported "
            "verdict there is drafting drift and not a false confirmation of the planted null; the "
            "planted null exists only in the frozen control contract.",
        ],
    }
    (out_dir / "multillm_probe_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "multillm_probe_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {out_dir}/multillm_probe_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
