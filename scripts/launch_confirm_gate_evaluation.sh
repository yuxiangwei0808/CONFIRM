#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
CONTRACTS="${CONTRACTS:-review-stage/initial-claims-all-gpt55/drafted_contracts.jsonl}"
OUT="${OUT:-review-stage/confirm-gates-all-gpt55}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
PROGRESS="${PROGRESS:-on}"
DATA_ROOTS="${DATA_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts}"
LIMIT="${LIMIT:-}"
MINIMUM_EVIDENCE_TIER="${MINIMUM_EVIDENCE_TIER:-confirmed}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

data_root_args=()
for data_root in $DATA_ROOTS; do
  data_root_args+=(--data-root "$data_root")
done

cmd=(
  "$PYTHON" -m bench.run_drafted_contract_gates
  --contracts "$CONTRACTS"
  --out-dir "$OUT"
  --max-workers "$MAX_WORKERS"
  --parallel-backend "$PARALLEL_BACKEND"
  --minimum-evidence-tier "$MINIMUM_EVIDENCE_TIER"
  "${data_root_args[@]}"
)

if [[ "$PROGRESS" == "off" ]]; then
  cmd+=(--no-progress)
fi

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

"${cmd[@]}"

echo "CONFIRM gate evaluation complete: $OUT"
