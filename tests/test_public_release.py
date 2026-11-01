"""Public workflows must work without lab data, model access, or hidden files."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.release_cli import digest, frozen_predictions, run_tasks, score, verify
from confirm.cli import main as confirm_main
from confirm.contract import load_contract
from confirm.demo import run_demo


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark/neuroclaimbench-v2.1"


@pytest.fixture(scope="module")
def benchmark():
    return verify(BENCHMARK)[0]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("quickstart") / "demo"
    run_demo(root)
    return root


def test_demo_explains_distinct_failures(demo):
    results = json.loads((demo / "summary.json").read_text())["results"]
    assert results["supported"]["support"] == {"screen": True, "replicate": True, "full": True}
    assert results["site_confounded"]["support"] == {"screen": False, "replicate": False, "full": False}
    assert "confound_incomplete" in results["site_confounded"]["rationale"]
    assert results["nonreplicating"]["support"] == {"screen": True, "replicate": False, "full": False}


@pytest.mark.parametrize("standard,expected", [("screen", 0), ("replicate", 1), ("full", 1)])
def test_cli_standard_matches_receipt_and_exit(demo, tmp_path, capsys, standard, expected):
    out = tmp_path / standard
    result = confirm_main(["run", "--contract", str(demo / "contracts/nonreplicating.yaml"),
                           "--data-dir", str(demo / "cohorts"), "--out", str(out), "--standard", standard])
    printed = json.loads(capsys.readouterr().out)
    stored = json.loads((out / "receipt.json").read_text())["results"]["support_decision"]
    assert result == expected
    assert printed["support_decision"] == stored
    assert printed["label"] == "non_replicated"
    assert stored["supported"] is (expected == 0)


def test_frozen_inventory_and_record_hashes(benchmark):
    assert len(benchmark.tasks) == 489
    assert len(benchmark.cases) == 544
    original = json.loads((BENCHMARK / "benchmark_manifest.json").read_text())["files"]
    for name in ("cases", "references", "tasks", "outcomes"):
        assert digest(BENCHMARK / f"{name}.jsonl") == original[name]["sha256"]


def test_scoring_keeps_reference_groups_and_denominators(benchmark):
    predictions = [{"task_id": t.task_id, "supported": False} for t in benchmark.tasks]
    report = score(benchmark, predictions)
    assert not report["partial"]
    assert report["completed"] == 489
    science = report["by_track"]["scientific"]["metrics"]
    assert science["literature_recall_combined"]["evaluated"] == 51
    assert science["literature_false_support_combined"]["evaluated"] == 19
    assert science["literature_recall_combined"]["rate"] == 0
    assert report["by_track"]["synthetic_stress"]["metrics"]["control_false_support_combined"]["evaluated"] == 150
    assert report["by_track"]["external_transfer"]["metrics"]["control_false_support_combined"]["evaluated"] == 22


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "string_bool", "error_decision"])
def test_bad_predictions_cannot_silently_affect_scores(benchmark, change):
    rows = frozen_predictions(benchmark, "full")
    if change == "missing": rows.pop()
    elif change == "duplicate": rows.append(rows[0])
    elif change == "unknown": rows[0]["task_id"] = "unknown"
    elif change == "string_bool": rows[0]["supported"] = "false"
    elif change == "error_decision": rows[0]["status"] = "error"
    with pytest.raises(ValueError): score(benchmark, rows)


def test_errors_follow_paper_policy_and_keep_coverage_visible(benchmark):
    ref_by_id = {r.reference_id: r for r in benchmark.references}
    case_by_id = {c.benchmark_case_id: c for c in benchmark.cases}
    task = next(t for t in benchmark.tasks if ref_by_id[case_by_id[t.benchmark_case_id].reference_id].disposition == "confirm")
    report = score(benchmark, [{"task_id": task.task_id, "status": "error", "error": "missing data"}], True)
    assert report["completed"] == 0
    assert report["errors"] == 1
    assert report["missing"] == 488
    assert report["partial"]
    assert report["error_scoring_policy"] == "count_as_withheld"
    metrics = report["by_track"][case_by_id[task.benchmark_case_id].benchmark_track]["metrics"]
    assert metrics["literature_recall_combined"]["evaluated"] == 1
    assert metrics["literature_recall_combined"]["completed"] == 0
    assert metrics["literature_recall_combined"]["rate"] == 0


def test_common_scalar_set_matches_paper_counts(benchmark):
    report = score(benchmark, frozen_predictions(benchmark, "screen"), literature_scope="scalar")
    metrics = report["by_track"]["scientific"]["metrics"]
    assert metrics["literature_recall_combined"]["evaluated"] == 31
    assert metrics["literature_recall_combined"]["supported"] == 19
    assert metrics["literature_false_support_combined"]["evaluated"] == 15
    assert metrics["literature_false_support_combined"]["supported"] == 6
    assert report["by_track"]["synthetic_stress"]["metrics"]["control_false_support_combined"]["evaluated"] == 150


def test_reporting_standards_are_nested(benchmark):
    decisions = {s: {r["task_id"]: r["supported"] for r in frozen_predictions(benchmark, s) if r["status"] == "completed"}
                 for s in ("screen", "replicate", "full")}
    assert len(decisions["full"]) == 489
    for task in benchmark.tasks:
        assert decisions["full"][task.task_id] <= decisions["replicate"][task.task_id] <= decisions["screen"][task.task_id]


def test_runner_verifies_data_before_execution(demo, tmp_path):
    contract = load_contract(demo / "contracts/supported.yaml")
    hashes = {n: digest(demo / f"cohorts/{n}.parquet") for n in [contract.discovery_cohort, *contract.replication_cohorts]}
    task = SimpleNamespace(task_id="demo", contract=contract, discovery_cohort=contract.discovery_cohort,
                           replication_cohorts=contract.replication_cohorts, generator_spec=None, partition_hashes=hashes)
    data = SimpleNamespace(tasks=[task], outcomes=[SimpleNamespace(task_id="demo")])
    good = run_tasks(data, [demo / "cohorts"], tmp_path / "good", "full")
    assert good["completed"] == 1 and good["errors"] == 0
    bad = deepcopy(data)
    bad.tasks[0].partition_hashes[contract.discovery_cohort] = "0" * 64
    failed = run_tasks(bad, [demo / "cohorts"], tmp_path / "bad", "full")
    assert failed["completed"] == 0 and failed["errors"] == 1
    row = json.loads((tmp_path / "bad/predictions.jsonl").read_text())
    assert "supported" not in row


def test_random_control_requires_generated_input_hashes(demo, tmp_path):
    contract = load_contract(demo / "contracts/supported.yaml")
    task = SimpleNamespace(task_id="control", contract=contract, discovery_cohort=contract.discovery_cohort,
        replication_cohorts=contract.replication_cohorts, generator_spec={"type": "independent_random_group_labels"},
        partition_hashes={n: digest(demo / f"cohorts/{n}.parquet") for n in [contract.discovery_cohort, *contract.replication_cohorts]})
    data = SimpleNamespace(tasks=[task], outcomes=[SimpleNamespace(task_id="control", generated_cohort_hashes={})])
    report = run_tasks(data, [demo / "cohorts"], tmp_path / "control", "full")
    assert report["errors"] == 1


def test_verifier_rejects_modified_record(tmp_path):
    import shutil
    copied = tmp_path / "benchmark"
    shutil.copytree(BENCHMARK, copied)
    with (copied / "outcomes.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify(copied)
