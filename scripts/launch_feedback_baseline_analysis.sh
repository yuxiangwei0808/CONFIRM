#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-review-stage/claim-search-gpt55-feedback-baselines-v1}"
SELF_REFINE_ROOT="${SELF_REFINE_ROOT:-review-stage/claim-search-gpt55-self-refine-r3-c5-v1}"

PYTHONPATH=src:. .venv/bin/python -m nbs.analyze_feedback_method_baselines \
  --out-dir "$OUT" \
  --self-refine-root "$SELF_REFINE_ROOT"
