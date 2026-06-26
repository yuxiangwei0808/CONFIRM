from confirm.contract import ClaimContract
from confirm.results import CohortReplicationResult, EffectResult, MultiverseResult, PowerResult, ReplicationResult
from confirm.verdict import UNVERIFIABLE_SEARCH_PROVENANCE, decide


def _contract(search_provenance=None) -> ClaimContract:
    data = {
        "claim_id": "search_lineage_test",
        "question": "Synthetic lineage test.",
        "estimand": {
            "type": "association",
            "outcome": "smri_x",
            "predictor": "age",
            "group": None,
            "direction": "positive",
        },
        "covariates": ["sex"],
        "inclusion": None,
        "discovery_cohort": "DISC",
        "replication_cohorts": ["REP"],
        "gates": {
            "multiplicity": {"method": "fdr_bh", "alpha": 0.05, "family_size": 1},
            "confound": {"require_covariates": ["sex"], "motion_check": False},
            "power": {"min_power": 0.8, "ref_effect": None},
            "multiverse": {"min_fraction_consistent": 0.6},
            "replication": {"alpha": 0.05, "require_same_sign": True, "require_ci_overlap": False, "harmonize": "none"},
        },
        "reporting_language_allowed": ["confirmed", "non_replicated", "under_powered", "fragile"],
    }
    if search_provenance is not None:
        data["search_provenance"] = search_provenance
    return ClaimContract.model_validate(data)


def _passing_parts(p: float = 0.01):
    effect = EffectResult(
        beta=1.0,
        se=0.1,
        ci_low=0.8,
        ci_high=1.2,
        p=p,
        n=100,
        dof=96.0,
        standardized_effect=0.5,
    )
    multiverse = MultiverseResult(fraction_consistent=1.0, passed=True, specs=[])
    power = PowerResult(
        achieved_power=0.99,
        n_needed_80=20.0,
        shrinkage_factor=1.0,
        shrunken_effect=0.5,
        under_powered=False,
        ref_effect=None,
        rationale="power ok",
    )
    replication = ReplicationResult(
        passed=True,
        reason="passed",
        cohort_results=[CohortReplicationResult("REP", True, "passed", effect)],
        harmonized=False,
    )
    return effect, multiverse, power, replication


def test_legacy_contract_defaults_to_preregistered_single_hypothesis():
    contract = _contract()
    assert contract.search_provenance.declared is True
    assert contract.search_provenance.family_size == 1
    assert contract.search_provenance.selection == "preregistered"
    verdict = decide(*_passing_parts(p=0.01), contract)
    assert verdict.label == "confirmed"


def test_fished_claim_uses_search_family_for_multiplicity():
    contract = _contract({"declared": True, "family_size": 10, "selection": "discovery_only"})
    verdict = decide(*_passing_parts(p=0.01), contract)
    assert verdict.label == "fragile"
    assert verdict.abstained is True
    assert verdict.gates["multiplicity"] is False
    assert verdict.gates["multiplicity_effective_family_size"] == 10


def test_replication_touched_or_unknown_search_provenance_abstains():
    for selection in ["full_data", "unknown"]:
        contract = _contract({"declared": True, "family_size": 1, "selection": selection})
        verdict = decide(*_passing_parts(p=1e-12), contract)
        assert verdict.label == "fragile"
        assert verdict.gates["search_provenance"] is False
        assert UNVERIFIABLE_SEARCH_PROVENANCE in verdict.rationale
