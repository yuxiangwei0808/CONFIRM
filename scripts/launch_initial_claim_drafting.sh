#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5.5}"
MODE="${MODE:-all}"
NUM_CLAIMS="${NUM_CLAIMS:-50}"
OUT="${OUT:-review-stage/initial-claims-all-gpt55}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-8192}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
PROGRESS="${PROGRESS:-on}"
FIXED_CLAIMS="${FIXED_CLAIMS:-data/claims/literature_grounded_claims.csv}"
SYNTHETIC_CLAIMS="${SYNTHETIC_CLAIMS:-data/claims/synthetic_stress_claims.csv}"
TARGET_FAMILIES="${TARGET_FAMILIES:-normative_fmri adhd asd ad_aging psychosis}"
DATA_ROOTS="${DATA_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts}"
INCLUDE_SYNTHETIC_STRESS="${INCLUDE_SYNTHETIC_STRESS:-off}"
LIMIT="${LIMIT:-}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

target_args=()
for target_family in $TARGET_FAMILIES; do
  target_args+=(--target-family "$target_family")
done

data_root_args=()
for data_root in $DATA_ROOTS; do
  data_root_args+=(--data-root "$data_root")
done

cmd=(
  "$PYTHON" -m bench.run_initial_claim_drafting
  --mode "$MODE"
  --model "$MODEL"
  --num-claims-per-target "$NUM_CLAIMS"
  --out-dir "$OUT"
  --fixed-claims "$FIXED_CLAIMS"
  --synthetic-claims "$SYNTHETIC_CLAIMS"
  --schema-retries "$SCHEMA_RETRIES"
  --llm-max-tokens "$LLM_MAX_TOKENS"
  --max-workers "$MAX_WORKERS"
  --parallel-backend "$PARALLEL_BACKEND"
  "${target_args[@]}"
  "${data_root_args[@]}"
)

if [[ "$PROGRESS" == "off" ]]; then
  cmd+=(--no-progress)
fi

if [[ "$INCLUDE_SYNTHETIC_STRESS" == "on" ]]; then
  cmd+=(--include-synthetic-stress)
fi

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

"${cmd[@]}"

echo "initial claim drafting complete: $OUT"
