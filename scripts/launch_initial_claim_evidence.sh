#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PHASE="${PHASE:-freeze}"
INITIAL_RESULTS="${INITIAL_RESULTS:-review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json}"
OUT="${OUT:-review-stage/initial-claims-gpt55-retrospective-evidence-v1}"
EVIDENCE_MANIFEST="${EVIDENCE_MANIFEST:-data/prepared_data/evidence_partitions/manifest.json}"
EVIDENCE_ROOTS="${EVIDENCE_ROOTS:-data/prepared_data/evidence_partitions/cohorts}"
SOURCE_ROOTS="${SOURCE_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts data/prepared_data/evidence_partitions/cohorts}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
PROGRESS="${PROGRESS:-on}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs"

evidence_args=()
for data_root in $EVIDENCE_ROOTS; do
  evidence_args+=(--evidence-root "$data_root")
done

source_args=()
for data_root in $SOURCE_ROOTS; do
  source_args+=(--source-root "$data_root")
done

cmd=(
  "$PYTHON" -u -m bench.run_initial_claim_evidence
  --phase "$PHASE"
  --initial-results "$INITIAL_RESULTS"
  --out-dir "$OUT"
  --evidence-manifest "$EVIDENCE_MANIFEST"
  --max-workers "$MAX_WORKERS"
  --parallel-backend "$PARALLEL_BACKEND"
  "${evidence_args[@]}"
  "${source_args[@]}"
)

if [[ "$PROGRESS" == "off" ]]; then
  cmd+=(--no-progress)
fi

"${cmd[@]}" 2>&1 | tee "$OUT/logs/${PHASE}.log"

echo "initial-claim evidence phase complete: phase=$PHASE out=$OUT"
