#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5.5}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_CANDIDATES="${MAX_CANDIDATES:-5}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
PROGRESS="${PROGRESS:-on}"
NEGATIVE_LIMIT="${NEGATIVE_LIMIT:-}"
FISHING_FEATURE_LIMIT="${FISHING_FEATURE_LIMIT:-24}"
UNDERPOWERED_COHORT_LIMIT="${UNDERPOWERED_COHORT_LIMIT:-7}"
OUT="${OUT:-review-stage/claim-search-safety-gpt55-$(date +%Y%m%d)}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

mkdir -p "$OUT/logs" "$OUT/data/cohorts"

negative_cmd=(
  "$PYTHON" -m bench.run_negatives_expansion
  --root .
  --out-dir "$OUT/gates"
  --materialize-data-root "$OUT/data/cohorts"
  --fishing-feature-limit "$FISHING_FEATURE_LIMIT"
  --underpowered-cohort-limit "$UNDERPOWERED_COHORT_LIMIT"
)
if [[ -n "$NEGATIVE_LIMIT" ]]; then
  negative_cmd+=(--limit "$NEGATIVE_LIMIT")
fi
"${negative_cmd[@]}" 2>&1 | tee "$OUT/logs/build_known_negatives.log"

"$PYTHON" nbs/build_claim_search_source_from_results.py \
  --input "$OUT/gates/negatives_expansion_results.json" \
  --out "$OUT/source/claim_search_source.json" \
  --model-spec synthetic-stress \
  2>&1 | tee "$OUT/logs/build_safety_source.log"

replay_cmd=(
  "$PYTHON" -u -m bench.run_iterative_claim_search_replay
  --input "$OUT/source/claim_search_source.json"
  --out-dir "$OUT/replay"
  --llm "$MODEL"
  --max-rounds "$MAX_ROUNDS"
  --max-candidates "$MAX_CANDIDATES"
  --schema-retries "$SCHEMA_RETRIES"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --candidate-evaluation on
  --max-workers "$MAX_WORKERS"
  --parallel-backend "$PARALLEL_BACKEND"
  --data-root "$OUT/data/cohorts"
)
if [[ "$PROGRESS" == "off" ]]; then
  replay_cmd+=(--no-progress)
fi
"${replay_cmd[@]}" 2>&1 | tee "$OUT/logs/claim_search_safety.log"

echo "claim-search known-negative safety experiment complete: $OUT"
