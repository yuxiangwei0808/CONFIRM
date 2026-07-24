import json

from nbs.plot_coverage_fcr import generate


def _metric(value: float) -> dict:
    return {
        "FCR": value,
        "FCR_ci95_exact": [max(0.0, value - 0.05), min(1.0, value + 0.05)],
        "FCR_count": 1,
        "FCR_denominator": 4,
        "coverage": 0.5,
        "coverage_ci95_exact": [0.25, 0.75],
        "coverage_count": 2,
        "coverage_denominator": 4,
        "known_positive_recall": 0.5,
        "known_positive_recall_ci95_exact": [0.25, 0.75],
        "known_positive_recall_count": 2,
        "known_positive_recall_denominator": 4,
    }


def test_coverage_fcr_generator_runs_on_combined_results_payload(tmp_path):
    source = tmp_path / "combined_benchmark_results.json"
    rung_names = ["exec_only", "+confound", "+power", "+multiverse", "+replication"]
    summary = {rung: _metric(0.1 + index * 0.02) for index, rung in enumerate(rung_names)}
    source.write_text(
        """{
  "summary_full": %s,
  "summary_main": %s,
  "risk_coverage_full": [%s],
  "risk_coverage_main": [%s]
}"""
        % (
            json.dumps(summary),
            json.dumps(summary),
            json.dumps({"alpha": 0.1, **_metric(0.12)}),
            json.dumps({"alpha": 0.1, **_metric(0.12)}),
        ),
        encoding="utf-8",
    )

    figure_path, table_path = generate(source, tmp_path)

    assert figure_path.exists()
    assert table_path.exists()
    assert figure_path.stat().st_size > 0
    assert table_path.read_text(encoding="utf-8").startswith("section,subset,rung")
