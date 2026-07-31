#!/usr/bin/env bash
# Multi-model drafting probe: every model drafts contracts for the same fixed
# question set, then the unchanged gates score each drafted contract. This
# separates variation introduced by the drafter from variation in the verdict.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-review-stage/multillm-probe-v2}"
DATA_ROOTS="${DATA_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts}"
MAX_WORKERS="${MAX_WORKERS:-4}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-thread}"
LIMIT="${LIMIT:-}"
# Weaker models need more error-fed retries to emit a schema-valid contract; at 2
# they leave recoverable validation failures that shrink the comparable claim set.
SCHEMA_RETRIES="${SCHEMA_RETRIES:-5}"

# Claude is routed through OpenRouter because the direct Anthropic key is rejected.
MODELS="${MODELS:-openai:gpt-5.5 openai:gpt-5.6-luna openai:gpt-5.4 google:gemini-3.5-flash google:gemini-3.5-flash-lite openrouter:anthropic/claude-sonnet-5}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

mkdir -p "$OUT_ROOT"

for model in $MODELS; do
  slug="$(printf '%s' "$model" | tr ':/.' '___')"
  draft_dir="$OUT_ROOT/$slug"
  gate_dir="$OUT_ROOT/$slug/gates"

  if [ -f "$gate_dir/claim_gate_audit.csv" ]; then
    echo "[SKIP] $model already has gate output"
    continue
  fi

  echo "[DRAFT] $model -> $draft_dir"
  # bash 3.2 on macOS treats an empty array as unbound under `set -u`.
  limit_args=()
  if [ -n "$LIMIT" ]; then limit_args=(--limit "$LIMIT"); fi
  "$PYTHON" -B src/bench/run_initial_claim_drafting.py \
    --mode literature_grounded \
    --include-synthetic-stress \
    --model "$model" \
    --out-dir "$draft_dir" \
    --data-root "$DATA_ROOTS" \
    --schema-retries "$SCHEMA_RETRIES" \
    --max-workers "$MAX_WORKERS" \
    --parallel-backend "$PARALLEL_BACKEND" \
    --no-progress \
    ${limit_args[@]+"${limit_args[@]}"}

  echo "[GATES] $model -> $gate_dir"
  "$PYTHON" -B src/bench/run_drafted_contract_gates.py \
    --contracts "$draft_dir/drafted_contracts.jsonl" \
    --out-dir "$gate_dir" \
    --data-root "$DATA_ROOTS" \
    --max-workers "$MAX_WORKERS" \
    --parallel-backend "$PARALLEL_BACKEND" \
    --no-progress

  echo "[DONE] $model"
done

echo "PROBE_COMPLETE"
