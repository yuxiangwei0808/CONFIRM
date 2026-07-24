#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PHASE="${PHASE:-sweep}"
SWEEP="${SWEEP:-review-stage/claim-search-gpt55-sweep-v7}"
FROZEN_DIR="${FROZEN_DIR:-review-stage/claim-search-gpt55-retrospective-evidence-v3}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$FROZEN_DIR}"
CONTROL_DIR="${CONTROL_DIR:-review-stage/claim-search-gpt55-control-r3-c5-v7}"
OUT="${OUT:-review-stage/claim-search-gpt55-paper-analysis-v1}"
REVIEWER_MODELS="${REVIEWER_MODELS:-google:gemini-3.5-flash openrouter:anthropic/claude-opus-4.8 openrouter:deepseek/deepseek-v4-pro}"
SAFETY_R1C2="${SAFETY_R1C2:-review-stage/claim-search-safety-gpt55-r1-c2-v7/replay/iterative_candidate_replay.json}"
SAFETY_R10C10="${SAFETY_R10C10:-review-stage/claim-search-safety-gpt55-r10-c10-v7/replay/iterative_candidate_replay.json}"

export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/.matplotlib}"
export CONFIRM_LLM_TIMEOUT="${CONFIRM_LLM_TIMEOUT:-300}"

mkdir -p "$OUT" "$MPLCONFIGDIR"

case "$PHASE" in
  sweep)
    "$PYTHON" -u nbs/analyze_claim_search_sweep.py \
      --sweep "$SWEEP" \
      --frozen-dir "$FROZEN_DIR" \
      --out-dir "$OUT" \
      --reference-arm r3_c5 \
      --expected-parent-count 215 \
      --bootstrap-resamples 2000 \
      --seed 20260721 \
      --safety-artifact "$SAFETY_R1C2" \
      --safety-artifact "$SAFETY_R10C10"
    ;;
  evidence)
    "$PYTHON" -u nbs/analyze_claim_search_evidence.py \
      --evidence-dir "$EVIDENCE_DIR" \
      --out-dir "$OUT"
    ;;
  novelty)
    "$PYTHON" -u nbs/candidate_novelty_review.py \
      --phase build \
      --frozen-dir "$FROZEN_DIR" \
      --control-dir "$CONTROL_DIR" \
      --out-dir "$OUT" \
      --seed 20260721
    ;;
  novelty-metrics)
    "$PYTHON" -u nbs/candidate_novelty_review.py \
      --phase metrics \
      --frozen-dir "$FROZEN_DIR" \
      --out-dir "$OUT" \
      --seed 20260721
    ;;
  forced-choice)
    model_args=()
    for model in $REVIEWER_MODELS; do
      model_args+=(--reviewer-model "$model")
    done
    "$PYTHON" -u nbs/candidate_novelty_review.py \
      --phase forced-choice \
      --out-dir "$OUT" \
      --batch-size "${REVIEW_BATCH_SIZE:-10}" \
      --max-output-tokens "${REVIEW_MAX_OUTPUT_TOKENS:-8192}" \
      --schema-retries "${SCHEMA_RETRIES:-2}" \
      "${model_args[@]}"
    ;;
  *)
    echo "unknown PHASE=$PHASE; expected sweep, evidence, novelty-metrics, novelty, or forced-choice" >&2
    exit 2
    ;;
esac

echo "paper analysis phase complete: phase=$PHASE out=$OUT"
