#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5.5}"
ROUNDS="${ROUNDS:-1 3 5 10}"
CANDIDATES="${CANDIDATES:-2 5 10}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
PROGRESS="${PROGRESS:-on}"
OUT="${OUT:-review-stage/claim-search-gpt55-sweep-v7}"
CLAIM_SEARCH_INPUTS="${CLAIM_SEARCH_INPUTS:-review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json}"
EVIDENCE_CONFIG="${EVIDENCE_CONFIG:-configs/evidence_partitions.yml}"
PARTITION_ROOT="${PARTITION_ROOT:-data/prepared_data/evidence_partitions}"
PARTITIONED_BENCHMARK_ROOT="${PARTITIONED_BENCHMARK_ROOT:-$PARTITION_ROOT/benchmark_ready}"
PARTITIONED_COHORT_ROOT="${PARTITIONED_COHORT_ROOT:-$PARTITION_ROOT/cohorts}"
BUILD_EVIDENCE_PARTITIONS="${BUILD_EVIDENCE_PARTITIONS:-off}"
BUILD_SOURCE="${BUILD_SOURCE:-auto}"
FEEDBACK_MODE="${FEEDBACK_MODE:-structured_diagnosis}"
EXPECTED_PARENT_COUNT="${EXPECTED_PARENT_COUNT:-215}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

mkdir -p "$OUT/logs"
read -r -a SOURCE_INPUTS <<< "$CLAIM_SEARCH_INPUTS"
progress_arg=""
if [[ "$PROGRESS" == "off" ]]; then
  progress_arg="--no-progress"
fi

if [[ "$BUILD_EVIDENCE_PARTITIONS" == "on" ]]; then
  "$PYTHON" -m confirm.evidence_partitions \
    --config "$EVIDENCE_CONFIG" \
    --out-root "$PARTITION_ROOT" \
    --check-overlap \
    2>&1 | tee "$OUT/logs/build_evidence_partitions.log"
fi

source_path="$OUT/source/claim_search_source.json"
build_source="$BUILD_SOURCE"
if [[ "$build_source" == "auto" ]]; then
  if [[ -f "$source_path" ]]; then
    build_source="off"
  else
    build_source="on"
  fi
fi

case "$build_source" in
  on)
    build_args=()
    for input_path in "${SOURCE_INPUTS[@]}"; do
      build_args+=(--input "$input_path")
    done
    "$PYTHON" nbs/build_claim_search_source_from_results.py \
      "${build_args[@]}" \
      --out "$source_path" \
      --model-spec benchmark/initial-claims-all-gpt55 \
      --evidence-manifest "$PARTITION_ROOT/manifest.json" \
      --expected-failed-count "$EXPECTED_PARENT_COUNT" \
      2>&1 | tee "$OUT/logs/build_source.log"
    ;;
  off)
    if [[ ! -f "$source_path" ]]; then
      echo "BUILD_SOURCE=off requires an existing frozen source: $source_path" >&2
      exit 1
    fi
    echo "reusing frozen claim-search source: $source_path"
    ;;
  *)
    echo "BUILD_SOURCE must be one of: auto, on, off" >&2
    exit 1
    ;;
esac

for max_rounds in $ROUNDS; do
  for max_candidates in $CANDIDATES; do
    "$PYTHON" -u -m bench.run_iterative_claim_search_replay \
      --input "$source_path" \
      --out-dir "$OUT/matrix/rounds_${max_rounds}/candidates_${max_candidates}" \
      --llm "$MODEL" \
      --max-rounds "$max_rounds" \
      --max-candidates "$max_candidates" \
      --schema-retries "$SCHEMA_RETRIES" \
      --feedback-mode "$FEEDBACK_MODE" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --resume on \
      --expected-parent-count "$EXPECTED_PARENT_COUNT" \
      --candidate-evaluation on \
      --max-workers "$MAX_WORKERS" \
      --parallel-backend "$PARALLEL_BACKEND" \
      --data-root "$PARTITIONED_BENCHMARK_ROOT/cohorts" \
      --data-root "$PARTITIONED_COHORT_ROOT" \
      --evidence-manifest "$PARTITION_ROOT/manifest.json" \
      ${progress_arg:+"$progress_arg"} \
      2>&1 | tee -a "$OUT/logs/rounds_${max_rounds}_candidates_${max_candidates}.log"
  done
done

"$PYTHON" nbs/summarize_claim_search_matrix.py \
  --out-root "$OUT" \
  --expected-rounds "$ROUNDS" \
  --expected-candidates "$CANDIDATES"

echo "claim-search sweep complete: $OUT"
