from pathlib import Path

from bench.plot_coverage_fcr import generate


def test_coverage_fcr_generator_runs_on_round3_json(tmp_path):
    source = Path("review-stage/round3-combat/combined_benchmark_results.json")

    figure_path, table_path = generate(source, tmp_path)

    assert figure_path.exists()
    assert table_path.exists()
    assert figure_path.stat().st_size > 0
    assert table_path.read_text(encoding="utf-8").startswith("section,subset,rung")
