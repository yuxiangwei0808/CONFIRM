#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PHASE="${PHASE:-all}"
PACKAGE_DIR="${PACKAGE_DIR:-data/neuroclaimbench/v2.1}"
RESULTS_DIR="${RESULTS_DIR:-review-stage/neuroclaimbench-v2.1/results}"
REFERENCE_DIR="${REFERENCE_DIR:-review-stage/neuroclaimbench-v2.1/reference}"
COMPACT_DIR="${COMPACT_DIR:-review-stage/neuroclaimbench-v2.1/compact}"
ANALYSIS_DIR="${ANALYSIS_DIR:-review-stage/neuroclaimbench-v2.1/analysis}"
GATE_DIR="${GATE_DIR:-review-stage/neuroclaimbench-v2.1/gate-attribution}"
CROSSWALK_DIR="${CROSSWALK_DIR:-review-stage/neuroclaimbench-v2.1/feedback-crosswalk}"
SWEEP="${SWEEP:-review-stage/claim-search-gpt55-sweep-v7}"
RELEASE_DIR="${RELEASE_DIR:-benchmark/neuroclaimbench-v2.1}"
ARCHIVE_DIR="${ARCHIVE_DIR:-review-stage/neuroclaimbench-v2.1/external-archive}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/neuroclaimbench-pycache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$GATE_DIR/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

run_compact() {
  "$PYTHON" -u nbs/compact_neuroclaimbench.py \
    --package-dir "$PACKAGE_DIR" \
    --reference-dir "$REFERENCE_DIR" \
    --results-dir "$RESULTS_DIR" \
    --out "$COMPACT_DIR"
}

run_benchmark() {
  "$PYTHON" -u nbs/analyze_neuroclaimbench_v21.py \
    --package-dir "$PACKAGE_DIR" \
    --compact-dir "$COMPACT_DIR" \
    --results-dir "$RESULTS_DIR" \
    --reference-dir "$REFERENCE_DIR" \
    --alignment-manifest "${ALIGNMENT_MANIFEST:-review-stage/neuroclaimbench-v2.1/alignment/alignment_manifest.json}" \
    --pubmed-cache-dir "${PUBMED_CACHE_DIR:-data/neuroclaimbench/pubmed-cache-v2.1}" \
    --out-dir "$ANALYSIS_DIR"
}

run_gate_attribution() {
  "$PYTHON" -u nbs/analyze_neuroclaimbench_gate_attribution.py \
    --package-dir "$COMPACT_DIR" \
    --out-dir "$GATE_DIR" \
    --bootstrap-resamples "${BOOTSTRAP_RESAMPLES:-2000}" \
    --seed "${SEED:-20260723}"
}

run_crosswalk() {
  "$PYTHON" -u nbs/analyze_neuroclaimbench_v21_feedback_crosswalk.py \
    --package-dir "$COMPACT_DIR" \
    --sweep "$SWEEP" \
    --out-dir "$CROSSWALK_DIR"
}

run_release() {
  "$PYTHON" -u -m bench.run_neuroclaimbench_v21_release \
    --compact-dir "$COMPACT_DIR" \
    --package-dir "$PACKAGE_DIR" \
    --results-dir "$RESULTS_DIR" \
    --reference-dir "$REFERENCE_DIR" \
    --analysis-dir "$ANALYSIS_DIR" \
    --feedback-crosswalk-dir "$CROSSWALK_DIR" \
    --adjudication-dir "${ADJUDICATION_DIR:-review-stage/neuroclaimbench-v2.1/adjudication}" \
    --release-dir "$RELEASE_DIR" \
    --archive-dir "$ARCHIVE_DIR"
}

case "$PHASE" in
  compact) run_compact ;;
  benchmark) run_benchmark ;;
  gate-attribution) run_gate_attribution ;;
  feedback-crosswalk) run_crosswalk ;;
  release) run_release ;;
  all)
    run_compact
    run_benchmark
    run_gate_attribution
    run_crosswalk
    run_release
    ;;
  *)
    echo "Unknown PHASE=$PHASE; expected compact, benchmark, gate-attribution," >&2
    echo "feedback-crosswalk, release, or all." >&2
    exit 2
    ;;
esac

echo "NeuroClaimBench local analysis phase complete: $PHASE"
