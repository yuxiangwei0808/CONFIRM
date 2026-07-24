#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PHASE="${PHASE:-freeze}"
SWEEP="${SWEEP:-review-stage/claim-search-gpt55-sweep-v7}"
OUT="${OUT:-review-stage/claim-search-gpt55-retrospective-evidence-v3}"
INITIAL_EVIDENCE_DIR="${INITIAL_EVIDENCE_DIR:-review-stage/initial-claims-gpt55-retrospective-evidence-v1}"
EVIDENCE_MANIFEST="${EVIDENCE_MANIFEST:-data/prepared_data/evidence_partitions/manifest.json}"
EVIDENCE_ROOTS="${EVIDENCE_ROOTS:-data/prepared_data/evidence_partitions/cohorts}"
SOURCE_ROOTS="${SOURCE_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts data/prepared_data/evidence_partitions/cohorts}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
PROGRESS="${PROGRESS:-on}"
ALLOW_NONREFERENCE_COUNTS="${ALLOW_NONREFERENCE_COUNTS:-off}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs"

lock_dir="$OUT/.${PHASE}.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "another retrospective-evidence process is already running: phase=$PHASE lock=$lock_dir" >&2
  exit 2
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

evidence_args=()
for data_root in $EVIDENCE_ROOTS; do
  evidence_args+=(--evidence-root "$data_root")
done

source_args=()
for data_root in $SOURCE_ROOTS; do
  source_args+=(--source-root "$data_root")
done

cmd=(
  "$PYTHON" -u -m bench.run_frozen_claim_evidence
  --phase "$PHASE"
  --sweep "$SWEEP"
  --out-dir "$OUT"
  --evidence-manifest "$EVIDENCE_MANIFEST"
  --max-workers "$MAX_WORKERS"
  --parallel-backend "$PARALLEL_BACKEND"
  --initial-evidence-dir "$INITIAL_EVIDENCE_DIR"
  "${evidence_args[@]}"
  "${source_args[@]}"
)

if [[ "$PROGRESS" == "off" ]]; then
  cmd+=(--no-progress)
fi

if [[ "$ALLOW_NONREFERENCE_COUNTS" == "on" ]]; then
  cmd+=(--allow-nonreference-counts)
fi

"${cmd[@]}" 2>&1 | tee "$OUT/logs/${PHASE}.log"

echo "retrospective evidence phase complete: phase=$PHASE out=$OUT"
