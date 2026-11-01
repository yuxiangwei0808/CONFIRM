"""Regression checks for public execution and benchmark failure paths."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from bench.benchmark import BenchmarkDataset
from bench.release_cli import digest, frozen_predictions, run_tasks, score
from confirm.agent import run_claim, run_question
from confirm.brainwide import run_brainwide
from confirm.contract import ClaimContract, PowerGate, load_contract
from confirm.demo import make_example_data
from confirm.execution import cohort_path, evaluate_contract, jsonable
from confirm.multiverse import _apply_brainwide_outlier_handling
from confirm.power import power_check
from confirm.provenance import write_receipt
from confirm.results import EffectResult
from confirm.verdict import decide_brainwide


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    root = tmp_path_factory.mktemp("release-regressions") / "data"
    make_example_data(root)
    return root


@pytest.fixture
def contract(example):
    return load_contract(example / "contracts/supported.yaml")


def test_result_write_failure_produces_scoreable_error(example, contract, tmp_path, monkeypatch):
    dataset = BenchmarkDataset(Path(__file__).resolve().parents[1] / "benchmark/neuroclaimbench-v2.1")
    task = dataset.tasks[0].model_copy(update={
        "contract": contract, "discovery_cohort": contract.discovery_cohort,
        "replication_cohorts": contract.replication_cohorts, "generator_spec": None,
        "partition_hashes": {name: digest(example / f"cohorts/{name}.parquet")
                             for name in [contract.discovery_cohort, *contract.replication_cohorts]},
    })
    dataset.tasks[0] = task
    real_write = Path.write_text
    def fail_result(path, *args, **kwargs):
        if path.name == task.task_id + ".json":
            raise OSError("Result storage unavailable")
        return real_write(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", fail_result)
    output = tmp_path / "run"
    report = run_tasks(dataset, [example / "cohorts"], output, "full", [task.task_id])
    row = json.loads((output / "predictions.jsonl").read_text())
    assert report["errors"] == 1
    assert "supported" not in row and "raw_label" not in row
    assert score(dataset, [row], allow_partial=True)["errors"] == 1


@pytest.mark.parametrize("change", ["missing", "integer", "string"])
def test_incomplete_gate_vectors_are_errors(change):
    gates = {name: True for name in ["search_provenance", "confound", "confound_completeness",
                                   "multiplicity", "power", "multiverse", "replication"]}
    if change == "missing": gates.pop("power")
    elif change == "integer": gates["power"] = 1
    else: gates["power"] = "true"
    data = SimpleNamespace(outcomes=[SimpleNamespace(task_id="t", status="completed",
                            gate_verdict={"gates": gates}, error=None)])
    row = frozen_predictions(data, "screen")[0]
    assert row["status"] == "error" and "supported" not in row


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_power_references_are_rejected(value):
    with pytest.raises(ValueError): PowerGate(min_power=.8, ref_effect=value)


def test_zero_reference_cannot_pass_power(contract):
    effect = EffectResult(beta=1, se=.1, ci_low=.8, ci_high=1.2, p=.001,
                          n=600, dof=595, standardized_effect=.9)
    result = power_check(effect, contract, ref_effect=0)
    assert result.achieved_power == 0 and result.under_powered


def test_nonfinite_power_estimate_cannot_pass(contract, monkeypatch):
    monkeypatch.setattr("confirm.power._achieved_power", lambda *_: float("nan"))
    effect = EffectResult(beta=1, se=.1, ci_low=.8, ci_high=1.2, p=.001,
                          n=600, dof=595, standardized_effect=.9)
    assert power_check(effect, contract).under_powered


def test_power_fallback_uses_equivalent_stable_tails(monkeypatch):
    from confirm.power import _achieved_power
    monkeypatch.setattr("confirm.power.TTestPower.power", lambda *args, **kwargs: float("nan"))
    assert _achieved_power(.5, 33, .05) == pytest.approx(.795365841487504)
    assert _achieved_power(.5, 600, .05) > .999999


def test_public_evaluator_uses_declared_power_reference(example, contract):
    contract = contract.model_copy(update={"gates": contract.gates.model_copy(update={
        "power": contract.gates.power.model_copy(update={"ref_effect": .01})})})
    _, results, _ = evaluate_contract(contract, example / "cohorts")
    assert results["power"].ref_effect == .01
    assert results["power"].under_powered


@pytest.mark.parametrize("alias", [False, True])
def test_replication_cannot_reuse_discovery_file(example, contract, tmp_path, alias):
    root = example / "cohorts"
    name = contract.discovery_cohort
    if alias:
        root = tmp_path
        (root / f"{name}.parquet").symlink_to(example / f"cohorts/{name}.parquet")
        (root / "alias.parquet").symlink_to(example / f"cohorts/{name}.parquet")
        name = "alias"
    contract = contract.model_copy(update={"replication_cohorts": [name]})
    with pytest.raises(ValueError, match="distinct cohort files"):
        evaluate_contract(contract, root)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "nested/cohort"])
def test_cohort_names_cannot_escape_input_directory(tmp_path, name):
    with pytest.raises(ValueError, match="directory components"):
        cohort_path(tmp_path, name)


@pytest.mark.parametrize("mode", ["run", "ask"])
def test_existing_analysis_directory_is_preserved(example, tmp_path, mode):
    output = tmp_path / "previous"
    output.mkdir()
    receipt = output / "receipt.json"
    receipt.write_text('{"existing": true}')
    with pytest.raises(FileExistsError):
        if mode == "run": run_claim(example / "contracts/supported.yaml", example / "cohorts", output)
        else: run_question("age association", example / "cohorts", output, approve=False)
    assert receipt.read_text() == '{"existing": true}'


def test_narration_failure_keeps_completed_analysis(example, contract, tmp_path, monkeypatch):
    class UnavailableNarrator:
        model = "test-narrator"
        def complete(self, *_): raise TimeoutError("Narration timed out")
    monkeypatch.setattr("confirm.agent.get_llm", lambda: UnavailableNarrator())
    monkeypatch.setattr("confirm.agent.draft_contract", lambda *args, **kwargs: contract)
    output = tmp_path / "ask"
    with pytest.raises(TimeoutError, match="Narration timed out"):
        run_question(contract.question, example / "cohorts", output, approve=False)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["results"]["primary"]["n"] == 600
    assert receipt["results"]["support_decision"]["supported"]
    assert receipt["results"]["agent"]["narrative"] is None


def test_receipts_and_execution_results_are_strict_json(tmp_path):
    payload = {"failed": [float("nan"), float("inf"), -float("inf")]}
    path = write_receipt(tmp_path, payload)
    def reject_constant(value): pytest.fail(f"Non-JSON numeric constant: {value}")
    assert json.loads(path.read_text(), parse_constant=reject_constant) == {"failed": [None, None, None]}
    assert json.loads(json.dumps(jsonable(payload), allow_nan=False)) == {"failed": [None, None, None]}


def test_declared_site_mapping_cannot_silently_become_unknown(tmp_path):
    from confirm.ingest.generic_csv import GenericCsvAdapter
    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"id": ["s1"], "age": [25], "sex": ["F"], "scanner": ["A"], "measure": [2.5]}).to_csv(csv, index=False)
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump({"cohort": "test", "columns": {
        "subject_id": "id", "age": "age", "sex": "sex", "site": "scanner_typo"},
        "idps": {"smri_signal": "measure"}}))
    with pytest.raises(ValueError, match="Mapped source columns"):
        GenericCsvAdapter(csv, mapping).to_canonical()


def test_failed_regional_fits_keep_full_correction_family(contract, monkeypatch):
    contract = contract.model_copy(update={"estimand": contract.estimand.model_copy(
        update={"unit": "brainwide", "outcome": "smri_*"})})
    frame = pd.DataFrame({"smri_a": [1, 2], "smri_b": [1, 1], "smri_c": [1, 1]})
    def fit(_, regional, **kwargs):
        if regional.estimand.outcome != "smri_a": raise ValueError("Unidentifiable region")
        return EffectResult(beta=-1, se=.2, ci_low=-1.4, ci_high=-.6, p=.02,
                            n=100, dof=95, standardized_effect=-.5)
    monkeypatch.setattr("confirm.brainwide.fit_effect", fit)
    regions = run_brainwide(frame, contract)
    assert regions.family_size == 3
    assert regions.regions[0].q == pytest.approx(.06)
    assert not regions.regions[0].significant
    verdict = decide_brainwide(regions, SimpleNamespace(passed=True), SimpleNamespace(under_powered=False),
        SimpleNamespace(passed=True, reason="passed", pattern_corr=1, region_replication_fraction=1, dice=1), contract)
    assert not verdict.gates["multiplicity"]
    assert verdict.gates["multiplicity_effective_family_size"] == 3


@pytest.mark.parametrize("outcome", ["smri_", "smri_*", ["smri_a", "smri_b"]])
def test_regional_outlier_handling_resolves_prefixes_and_globs(contract, outcome):
    contract = contract.model_copy(update={"estimand": contract.estimand.model_copy(
        update={"unit": "brainwide", "outcome": outcome})})
    frame = pd.DataFrame({"age": np.arange(100), "smri_a": [0.] * 99 + [1000.], "smri_b": np.ones(100)})
    clipped = _apply_brainwide_outlier_handling(frame, contract, "winsorize")
    filtered = _apply_brainwide_outlier_handling(frame, contract, "drop_z3")
    assert clipped.smri_a.max() < 1000
    assert 99 not in filtered.index
