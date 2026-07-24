#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5.5}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_CANDIDATES="${MAX_CANDIDATES:-5}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
PROGRESS="${PROGRESS:-on}"
RETRY_TRANSIENT_FAILURES="${RETRY_TRANSIENT_FAILURES:-on}"
OUT="${OUT:-review-stage/claim-search-gpt55-control-r3-c5-v7}"
SWEEP="${SWEEP:-review-stage/claim-search-gpt55-sweep-v7}"
STRUCTURED_ARTIFACT="${STRUCTURED_ARTIFACT:-$SWEEP/matrix/rounds_3/candidates_5/iterative_candidate_replay.json}"
CLAIM_SEARCH_SOURCE="${CLAIM_SEARCH_SOURCE:-$SWEEP/source/claim_search_source.json}"
PARTITION_ROOT="${PARTITION_ROOT:-data/prepared_data/evidence_partitions}"
EXPECTED_PARENT_COUNT="${EXPECTED_PARENT_COUNT:-215}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

mkdir -p "$OUT/logs"

if [[ ! -f "$STRUCTURED_ARTIFACT" ]]; then
  echo "missing completed structured R3/C5 artifact: $STRUCTURED_ARTIFACT" >&2
  exit 1
fi
if [[ ! -f "$CLAIM_SEARCH_SOURCE" ]]; then
  echo "missing exact sweep source: $CLAIM_SEARCH_SOURCE" >&2
  exit 1
fi

"$PYTHON" -m nbs.summarize_claim_search_control \
  --preflight-current \
  --structured-artifact "$STRUCTURED_ARTIFACT"

replay_cmd=(
  "$PYTHON" -u -m bench.run_iterative_claim_search_replay
  --input "$CLAIM_SEARCH_SOURCE"
  --out-dir "$OUT/generic_retry"
  --llm "$MODEL"
  --max-rounds "$MAX_ROUNDS"
  --max-candidates "$MAX_CANDIDATES"
  --schema-retries "$SCHEMA_RETRIES"
  --feedback-mode generic_retry
  --checkpoint-every "$CHECKPOINT_EVERY"
  --resume on
  --retry-transient-generation-failures "$RETRY_TRANSIENT_FAILURES"
  --expected-parent-count "$EXPECTED_PARENT_COUNT"
  --candidate-evaluation on
  --max-workers "$MAX_WORKERS"
  --parallel-backend "$PARALLEL_BACKEND"
  --data-root "$PARTITION_ROOT/benchmark_ready/cohorts"
  --data-root "$PARTITION_ROOT/cohorts"
  --evidence-manifest "$PARTITION_ROOT/manifest.json"
)
if [[ "$PROGRESS" == "off" ]]; then
  replay_cmd+=(--no-progress)
fi
"${replay_cmd[@]}" 2>&1 | tee "$OUT/logs/generic_retry.log"

"$PYTHON" -m nbs.summarize_claim_search_control \
  --out-root "$OUT" \
  --structured-artifact "$STRUCTURED_ARTIFACT" \
  --generic-artifact "$OUT/generic_retry/iterative_candidate_replay.json" \
  --expected-parent-count "$EXPECTED_PARENT_COUNT"

echo "generic-retry control complete and matched to $STRUCTURED_ARTIFACT: $OUT"
