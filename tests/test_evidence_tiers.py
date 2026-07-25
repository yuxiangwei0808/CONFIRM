from __future__ import annotations

from itertools import product
from pathlib import Path

from bench import run_drafted_contract_gates as stage2
from confirm import execution
from confirm.contract import ClaimContract
from confirm.verdict import (
    EVIDENCE_TIER_REQUIRED_GATES,
    Verdict,
    classify_support,
)


def _gates(**overrides: bool) -> dict[str, bool]:
    gates = {
        "search_provenance": True,
        "confound": True,
        "confound_completeness": True,
        "multiplicity": True,
        "power": True,
        "multiverse": True,
        "replication": True,
    }
    gates.update(overrides)
    return gates


def _contract() -> ClaimContract:
    return ClaimContract.model_validate(
        {
            "claim_id": "tier_test",
            "question": "Is age associated with the scalar outcome?",
            "estimand": {
                "type": "association",
                "outcome": "outcome",
                "predictor": "age",
                "group": None,
                "direction": "two_sided",
                "unit": "scalar",
                "region_set": None,
            },
            "covariates": ["sex"],
            "inclusion": None,
            "discovery_cohort": "DISC",
            "replication_cohorts": ["REP"],
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
                    "harmonize": "none",
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


def test_all_gates_pass_supports_every_minimum_tier() -> None:
    assert classify_support(_gates(), "discovery").supported
    assert classify_support(_gates(), "replicated").supported
    decision = classify_support(_gates(), "confirmed")
    assert decision.supported
    assert decision.achieved_evidence_tier == "confirmed"


def test_replicated_support_does_not_require_power_or_multiverse() -> None:
    gates = _gates(power=False, multiverse=False)

    assert classify_support(gates, "discovery").supported
    assert classify_support(gates, "replicated").supported
    strict = classify_support(gates, "confirmed")
    assert not strict.supported
    assert strict.achieved_evidence_tier == "replicated_supported"
    assert strict.failed_required_gates == ("power", "multiverse")


def test_discovery_support_does_not_satisfy_higher_tiers() -> None:
    gates = _gates(replication=False)

    discovery = classify_support(gates, "discovery")
    replicated = classify_support(gates, "replicated")
    strict = classify_support(gates, "confirmed")

    assert discovery.supported
    assert discovery.achieved_evidence_tier == "discovery_supported"
    assert not replicated.supported
    assert not strict.supported


def test_core_gate_failures_and_missing_gates_fail_closed() -> None:
    for gate in EVIDENCE_TIER_REQUIRED_GATES["discovery"]:
        decision = classify_support(_gates(**{gate: False}), "discovery")
        assert not decision.supported
        assert decision.achieved_evidence_tier == "unsupported"

    missing = _gates()
    del missing["confound_completeness"]
    assert not classify_support(missing, "discovery").supported


def test_brainwide_diagnostic_fields_do_not_replace_compound_replication_gate() -> None:
    gates = {
        **_gates(),
        "pattern_corr": False,
        "region_replication_fraction": False,
        "dice": False,
    }

    assert classify_support(gates, "replicated").supported
    assert classify_support(gates, "confirmed").supported


def test_confirmed_tier_matches_original_all_gate_conjunction_exhaustively() -> None:
    required = EVIDENCE_TIER_REQUIRED_GATES["confirmed"]
    for values in product((False, True), repeat=len(required)):
        gates = dict(zip(required, values))
        assert classify_support(gates).supported is all(values)


def test_public_execution_embeds_support_decision(monkeypatch, tmp_path: Path) -> None:
    contract = _contract()
    for cohort in ("DISC", "REP"):
        (tmp_path / f"{cohort}.parquet").touch()

    verdict = Verdict(
        label="under_powered",
        abstained=True,
        rationale="Failed gates: power",
        gates=_gates(power=False),
    )
    monkeypatch.setattr(execution, "load_canonical", lambda path: object())
    monkeypatch.setattr(
        execution,
        "run_scalar_contract",
        lambda contract, discovery, replications, ref_effect: (
            verdict,
            {"verdict": verdict},
        ),
    )

    strict_verdict, results, _ = execution.evaluate_contract(
        contract,
        tmp_path,
        minimum_evidence_tier="discovery",
    )

    assert strict_verdict.label == "under_powered"
    assert results["support_decision"].supported
    assert (
        results["support_decision"].achieved_evidence_tier
        == "replicated_supported"
    )


def test_stage2_records_tier_without_replacing_strict_label() -> None:
    contract = _contract()
    verdict = Verdict(
        label="under_powered",
        abstained=True,
        rationale="Failed gates: power",
        gates=_gates(power=False),
    )
    decision = classify_support(verdict.gates, "discovery")

    row = stage2._gate_row(
        {"claim_id": contract.claim_id},
        contract,
        verdict,
        {"support_decision": decision, "verdict": verdict},
        [Path("DISC.parquet"), Path("REP.parquet")],
    )

    assert row["final_label"] == "under_powered"
    assert row["gate_verdict_label"] == "under_powered"
    assert row["reported_supported"] is True
    assert row["minimum_evidence_tier"] == "discovery"
    assert row["achieved_evidence_tier"] == "replicated_supported"


def test_stage2_cli_defaults_to_confirmed() -> None:
    parser = stage2.build_parser()

    assert parser.parse_args([]).minimum_evidence_tier == "confirmed"
    assert (
        parser.parse_args(
            ["--minimum-evidence-tier", "replicated"]
        ).minimum_evidence_tier
        == "replicated"
    )
