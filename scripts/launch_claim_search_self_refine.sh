#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5.5}"
OUT="${OUT:-review-stage/claim-search-gpt55-self-refine-r3-c5-v1}"
TRACK="${TRACK:-all}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
PROGRESS="${PROGRESS:-on}"
LIMIT="${LIMIT:-}"

SCIENTIFIC_SOURCE="${SCIENTIFIC_SOURCE:-review-stage/claim-search-gpt55-sweep-v7/source/claim_search_source.json}"
SAFETY_ROOT="${SAFETY_ROOT:-review-stage/claim-search-safety-gpt55-r10-c10-v7}"
SAFETY_SOURCE="${SAFETY_SOURCE:-$SAFETY_ROOT/source/claim_search_source.json}"
PARTITION_ROOT="${PARTITION_ROOT:-data/prepared_data/evidence_partitions}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

run_track() {
  local name="$1"
  local source="$2"
  local expected="$3"
  shift 3
  local data_args=("$@")
  local track_out="$OUT/$name"
  mkdir -p "$track_out/logs"

  local command=(
    "$PYTHON" -u -m bench.run_iterative_claim_search_replay
    --input "$source"
    --out-dir "$track_out/replay"
    --llm "$MODEL"
    --max-rounds 3
    --max-candidates 5
    --schema-retries 2
    --feedback-mode generic_retry
    --candidate-strategy self_refine
    --checkpoint-every 10
    --resume on
    --candidate-evaluation on
    --max-workers "$MAX_WORKERS"
    --parallel-backend "$PARALLEL_BACKEND"
    "${data_args[@]}"
  )
  if [[ -n "$LIMIT" ]]; then
    command+=(--limit "$LIMIT" --expected-parent-count "$LIMIT")
  else
    command+=(--expected-parent-count "$expected")
  fi
  if [[ "$PROGRESS" == "off" ]]; then
    command+=(--no-progress)
  fi
  "${command[@]}" 2>&1 | tee "$track_out/logs/self_refine.log"
}

case "$TRACK" in
  scientific)
    run_track scientific "$SCIENTIFIC_SOURCE" 215 \
      --data-root "$PARTITION_ROOT/benchmark_ready/cohorts" \
      --data-root "$PARTITION_ROOT/cohorts" \
      --evidence-manifest "$PARTITION_ROOT/manifest.json"
    ;;
  safety)
    run_track safety "$SAFETY_SOURCE" 150 \
      --data-root "$SAFETY_ROOT/data/cohorts" \
      --evidence-manifest "$SAFETY_ROOT/data/manifest.json"
    ;;
  all)
    run_track scientific "$SCIENTIFIC_SOURCE" 215 \
      --data-root "$PARTITION_ROOT/benchmark_ready/cohorts" \
      --data-root "$PARTITION_ROOT/cohorts" \
      --evidence-manifest "$PARTITION_ROOT/manifest.json"
    run_track safety "$SAFETY_SOURCE" 150 \
      --data-root "$SAFETY_ROOT/data/cohorts" \
      --evidence-manifest "$SAFETY_ROOT/data/manifest.json"
    ;;
  *)
    echo "TRACK must be scientific, safety, or all" >&2
    exit 2
    ;;
esac
