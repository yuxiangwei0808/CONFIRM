#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PHASE="${PHASE:-protocol}"
OUT="${OUT:-review-stage/neuroclaimbench-v2.1/claim-evaluation-baselines-v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-review-stage/neuroclaimbench-v2.1/results/checkpoints}"
PACKAGE_DIR="${PACKAGE_DIR:-benchmark/neuroclaimbench-v2.1}"
MODEL="${MODEL:-openai:gpt-5.5}"
SCHEMA_RETRIES="${SCHEMA_RETRIES:-2}"
MAX_WORKERS="${MAX_WORKERS:-8}"
LIMIT="${LIMIT:-}"

# The LLM judge phases (direct + NeuroClaw personas) issue live provider calls
# unless the offline stand-in model is selected. Require an explicit opt-in.
case "$PHASE" in
  llm_judge | neuroclaw_judge | all)
    if [[ "$MODEL" != standin* && "${ALLOW_API:-0}" != "1" ]]; then
      echo "PHASE=$PHASE with MODEL=$MODEL needs ALLOW_API=1 (live API cost)." >&2
      echo "Use MODEL=standin for an offline check, or set ALLOW_API=1." >&2
      exit 2
    fi
    ;;
esac

command=(
  .venv/bin/python -m nbs.analyze_claim_evaluation_baselines
  --phase "$PHASE"
  --out-dir "$OUT"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --package-dir "$PACKAGE_DIR"
  --model "$MODEL"
  --schema-retries "$SCHEMA_RETRIES"
  --max-workers "$MAX_WORKERS"
)
if [[ -n "$LIMIT" ]]; then
  command+=(--limit "$LIMIT")
fi
if [[ "${PROGRESS:-on}" == "off" ]]; then
  command+=(--no-progress)
fi

PYTHONPATH=src:. "${command[@]}"
