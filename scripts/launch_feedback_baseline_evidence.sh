#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PHASE="${PHASE:-manifest}"
METHOD="${METHOD:-self_refine}"
OUT="${OUT:-review-stage/claim-search-gpt55-feedback-baselines-v1}"
SELF_REFINE_ROOT="${SELF_REFINE_ROOT:-review-stage/claim-search-gpt55-self-refine-r3-c5-v1}"
GENERIC_ROOT="${GENERIC_ROOT:-review-stage/claim-search-gpt55-control-r3-c5-v7/generic_retry}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PROGRESS="${PROGRESS:-on}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "$PHASE" == "manifest" ]]; then
  .venv/bin/python -m nbs.freeze_feedback_baseline_manifest \
    --self-refine-root "$SELF_REFINE_ROOT" \
    --generic-root "$GENERIC_ROOT" \
    --out "$OUT/feedback_arm_manifest.json"
  exit 0
fi

case "$METHOD" in
  failure_blind)
    SWEEP="$GENERIC_ROOT"
    EVIDENCE_MANIFEST="data/prepared_data/evidence_partitions/manifest.json"
    EVIDENCE_ROOT="data/prepared_data/evidence_partitions/cohorts"
    SOURCE_ROOTS="data/prepared_data/evidence_partitions/benchmark_ready/cohorts data/prepared_data/evidence_partitions/cohorts"
    METHOD_OUT="$OUT/evidence/failure_blind"
    ;;
  self_refine)
    SWEEP="$SELF_REFINE_ROOT/scientific"
    EVIDENCE_MANIFEST="data/prepared_data/evidence_partitions/manifest.json"
    EVIDENCE_ROOT="data/prepared_data/evidence_partitions/cohorts"
    SOURCE_ROOTS="data/prepared_data/evidence_partitions/benchmark_ready/cohorts data/prepared_data/evidence_partitions/cohorts"
    METHOD_OUT="$OUT/evidence/self_refine"
    ;;
  self_refine_safety)
    SAFETY_ROOT="review-stage/claim-search-safety-gpt55-r10-c10-v7"
    SWEEP="$SELF_REFINE_ROOT/safety"
    EVIDENCE_MANIFEST="$SAFETY_ROOT/data/manifest.json"
    EVIDENCE_ROOT="$SAFETY_ROOT/data/cohorts"
    SOURCE_ROOTS="$SAFETY_ROOT/data/cohorts"
    METHOD_OUT="$OUT/evidence/self_refine_safety"
    ;;
  *)
    echo "METHOD must be failure_blind, self_refine, or self_refine_safety" >&2
    exit 2
    ;;
esac

command=(
  .venv/bin/python -u -m bench.run_frozen_claim_evidence
  --phase "$PHASE"
  --sweep "$SWEEP"
  --out-dir "$METHOD_OUT"
  --evidence-manifest "$EVIDENCE_MANIFEST"
  --evidence-root "$EVIDENCE_ROOT"
  --max-workers "$MAX_WORKERS"
  --parallel-backend process
  --allow-nonreference-counts
)
for source_root in $SOURCE_ROOTS; do
  command+=(--source-root "$source_root")
done
if [[ "$PROGRESS" == "off" ]]; then
  command+=(--no-progress)
fi
"${command[@]}"
