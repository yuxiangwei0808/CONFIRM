#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PACKAGE_DIR="${PACKAGE_DIR:-data/neuroclaimbench/v2.1}"
REFERENCE_PROFILES="${REFERENCE_PROFILES:-review-stage/neuroclaimbench-v2.1/reference/triage_reference_profiles.jsonl}"
OUT="${OUT:-review-stage/neuroclaimbench-v2.1/results}"
DATA_ROOTS="${DATA_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts data/prepared_data/evidence_partitions/cohorts review-stage/claim-search-safety-gpt55-r10-c10-v7/data/cohorts}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PROGRESS="${PROGRESS:-on}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/neuroclaimbench-pycache}"

data_args=()
for data_root in $DATA_ROOTS; do
  data_args+=(--data-root "$data_root")
done
progress_args=()
if [[ "$PROGRESS" == "off" ]]; then
  progress_args+=(--no-progress)
fi

"$PYTHON" -u -m bench.run_neuroclaimbench_finalize \
  --phase "${PHASE:-all}" \
  --package-dir "$PACKAGE_DIR" \
  --reference-profiles "$REFERENCE_PROFILES" \
  --out-dir "$OUT" \
  --max-workers "$MAX_WORKERS" \
  --parallel-backend "${PARALLEL_BACKEND:-process}" \
  "${data_args[@]}" "${progress_args[@]}"

echo "NeuroClaimBench local evaluation complete: $OUT"
