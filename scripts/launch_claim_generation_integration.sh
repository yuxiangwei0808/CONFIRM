#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PHASE="${PHASE:-select}"
OUT="${OUT:-review-stage/neuroclaimbench-v2.1/claim-generation-integration-v1}"
QUESTIONS="${QUESTIONS:-review-stage/initial-claims-all-gpt55/claim_questions.jsonl}"
MODEL="${MODEL:-openai:gpt-5.5}"
PER_FAMILY="${PER_FAMILY:-10}"
NEGATIVES_PER_FAMILY="${NEGATIVES_PER_FAMILY:-2}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
MAX_WORKERS="${MAX_WORKERS:-6}"

# The draft phase issues live provider calls unless the offline stand-in model
# is selected. Require an explicit opt-in for real provider spend.
case "$PHASE" in
  draft | all)
    if [[ "$MODEL" != standin* && "${ALLOW_API:-0}" != "1" ]]; then
      echo "PHASE=$PHASE with MODEL=$MODEL needs ALLOW_API=1 (live API cost)." >&2
      echo "Use MODEL=standin for an offline check, or set ALLOW_API=1." >&2
      exit 2
    fi
    ;;
esac

command=(
  "$PYTHON" -m nbs.analyze_claim_generation_integration
  --phase "$PHASE"
  --out-dir "$OUT"
  --questions "$QUESTIONS"
  --model "$MODEL"
  --per-family "$PER_FAMILY"
  --negatives-per-family "$NEGATIVES_PER_FAMILY"
  --schema-retries "$SCHEMA_RETRIES"
  --max-workers "$MAX_WORKERS"
)
if [[ "${PROGRESS:-on}" == "off" ]]; then
  command+=(--no-progress)
fi

PYTHONPATH=src:. "${command[@]}"
