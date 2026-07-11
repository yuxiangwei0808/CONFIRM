#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5.5}"
MAX_ROUNDS="${MAX_ROUNDS:-1}"
MAX_CANDIDATES="${MAX_CANDIDATES:-2}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
PROGRESS="${PROGRESS:-on}"
OUT="${OUT:-review-stage/claim-search-gpt55-evidence-r${MAX_ROUNDS}-c${MAX_CANDIDATES}-$(date +%Y%m%d)}"
CLAIM_SEARCH_INPUTS="${CLAIM_SEARCH_INPUTS:-review-stage/confirm-gates-all-gpt55/combined_benchmark_results.json}"
EVIDENCE_CONFIG="${EVIDENCE_CONFIG:-configs/evidence_partitions.yml}"
PARTITION_ROOT="${PARTITION_ROOT:-data/prepared_data/evidence_partitions}"
PARTITIONED_BENCHMARK_ROOT="${PARTITIONED_BENCHMARK_ROOT:-$PARTITION_ROOT/benchmark_ready}"
BUILD_EVIDENCE_PARTITIONS="${BUILD_EVIDENCE_PARTITIONS:-on}"

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

build_args=()
for input_path in "${SOURCE_INPUTS[@]}"; do
  build_args+=(--input "$input_path")
done

"$PYTHON" nbs/build_claim_search_source_from_results.py \
  "${build_args[@]}" \
  --out "$OUT/source/claim_search_source.json" \
  --model-spec benchmark/initial-claims-all-gpt55 \
  --evidence-manifest "$PARTITION_ROOT/manifest.json" \
  --require-excluded-evidence \
  2>&1 | tee "$OUT/logs/build_source.log"

"$PYTHON" -u -m bench.run_iterative_claim_search_replay \
  --input "$OUT/source/claim_search_source.json" \
  --out-dir "$OUT/matrix/rounds_${MAX_ROUNDS}/candidates_${MAX_CANDIDATES}" \
  --llm "$MODEL" \
  --max-rounds "$MAX_ROUNDS" \
  --max-candidates "$MAX_CANDIDATES" \
  --schema-retries "$SCHEMA_RETRIES" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --candidate-evaluation on \
  --max-workers "$MAX_WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  --data-root "$PARTITIONED_BENCHMARK_ROOT/cohorts" \
  --data-root "$PARTITION_ROOT/cohorts" \
  --holdout-data-root "$PARTITION_ROOT/cohorts" \
  --evidence-manifest "$PARTITION_ROOT/manifest.json" \
  ${progress_arg:+"$progress_arg"} \
  2>&1 | tee "$OUT/logs/rounds_${MAX_ROUNDS}_candidates_${MAX_CANDIDATES}.log"

"$PYTHON" nbs/summarize_claim_search_matrix.py --out-root "$OUT"

"$PYTHON" -m bench.run_claim_search_case_studies \
  --input "$OUT/matrix/rounds_${MAX_ROUNDS}/candidates_${MAX_CANDIDATES}/iterative_candidate_replay.json" \
  --out-dir "$OUT/case-studies" \
  --limit "${CASE_STUDY_LIMIT:-10}"

echo "claim-search full-scale run complete: $OUT"
