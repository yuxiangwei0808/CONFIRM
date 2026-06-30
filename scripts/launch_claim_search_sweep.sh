#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5}"
ROUNDS="${ROUNDS:-1 3 5 10}"
CANDIDATES="${CANDIDATES:-2 5 10}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
OUT="${OUT:-review-stage/claim-search-gpt5-fullscale-sweep-$(date +%Y%m%d)}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

mkdir -p "$OUT/logs"

"$PYTHON" nbs/build_claim_search_source_from_results.py \
  --input review-stage/round5-combat/combined_benchmark_results.json \
  --input review-stage/negatives-expansion/negatives_expansion_results.json \
  --input review-stage/external-nacc/nacc_external_results.json \
  --input review-stage/external-cnp/CNP_external_results.json \
  --out "$OUT/source/claim_search_source.json" \
  --model-spec benchmark/no-feedback-fullscale \
  2>&1 | tee "$OUT/logs/build_source.log"

for max_rounds in $ROUNDS; do
  for max_candidates in $CANDIDATES; do
    "$PYTHON" -u -m bench.run_iterative_claim_search_replay \
      --input "$OUT/source/claim_search_source.json" \
      --out-dir "$OUT/matrix/rounds_${max_rounds}/candidates_${max_candidates}" \
      --llm "$MODEL" \
      --max-rounds "$max_rounds" \
      --max-candidates "$max_candidates" \
      --schema-retries "$SCHEMA_RETRIES" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --candidate-evaluation on \
      --max-workers "$MAX_WORKERS" \
      --parallel-backend "$PARALLEL_BACKEND" \
      --data-root data/prepared_data/benchmark_ready/cohorts \
      --data-root data/prepared_data/smri_disease \
      --data-root data/prepared_data/cluster_recovered \
      --data-root data/prepared_data/fmri_descriptors \
      --data-root data/canonical \
      2>&1 | tee "$OUT/logs/rounds_${max_rounds}_candidates_${max_candidates}.log"
  done
done

"$PYTHON" nbs/summarize_claim_search_matrix.py --out-root "$OUT"

echo "claim-search sweep complete: $OUT"
