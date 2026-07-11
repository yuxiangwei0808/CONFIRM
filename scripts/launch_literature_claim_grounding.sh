#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-openai:gpt-5.5}"
OUT="${OUT:-review-stage/literature-grounding-gpt55}"
CLAIMS_OUT="${CLAIMS_OUT:-data/claims/literature_grounded_claims.csv}"
TARGET_FAMILIES="${TARGET_FAMILIES:-normative_fmri adhd asd ad_aging psychosis}"
DATA_ROOTS="${DATA_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts}"
MAX_RECORDS_PER_QUERY="${MAX_RECORDS_PER_QUERY:-20}"
MAX_CLAIMS_PER_RECORD="${MAX_CLAIMS_PER_RECORD:-3}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-8192}"
PUBMED_EMAIL="${PUBMED_EMAIL:-}"
PUBMED_API_KEY="${PUBMED_API_KEY:-}"
PUBMED_TIMEOUT="${PUBMED_TIMEOUT:-30}"
PUBMED_DELAY="${PUBMED_DELAY:-0.34}"
RECORDS_JSONL="${RECORDS_JSONL:-}"
PROGRESS="${PROGRESS:-on}"

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
  "$PYTHON" -m bench.run_literature_claim_grounding
  --model "$MODEL"
  --out-dir "$OUT"
  --claims-out "$CLAIMS_OUT"
  --max-records-per-query "$MAX_RECORDS_PER_QUERY"
  --max-claims-per-record "$MAX_CLAIMS_PER_RECORD"
  --schema-retries "$SCHEMA_RETRIES"
  --llm-max-tokens "$LLM_MAX_TOKENS"
  --pubmed-timeout "$PUBMED_TIMEOUT"
  --pubmed-delay "$PUBMED_DELAY"
  "${target_args[@]}"
  "${data_root_args[@]}"
)

if [[ -n "$PUBMED_EMAIL" ]]; then
  cmd+=(--pubmed-email "$PUBMED_EMAIL")
fi

if [[ -n "$PUBMED_API_KEY" ]]; then
  cmd+=(--pubmed-api-key "$PUBMED_API_KEY")
fi

if [[ -n "$RECORDS_JSONL" ]]; then
  cmd+=(--records-jsonl "$RECORDS_JSONL")
fi

if [[ "$PROGRESS" == "off" ]]; then
  cmd+=(--no-progress)
fi

"${cmd[@]}"

echo "literature claim grounding complete: $OUT"
echo "Stage 1 input written to: $CLAIMS_OUT"
