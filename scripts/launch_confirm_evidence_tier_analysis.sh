#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PACKAGE_DIR="${PACKAGE_DIR:-review-stage/neuroclaimbench-v2.1/compact}"
OUT="${OUT:-review-stage/neuroclaimbench-v2.1/evidence-tiers}"

export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"

"$PYTHON" -m nbs.analyze_confirm_evidence_tiers \
  --package-dir "$PACKAGE_DIR" \
  --out-dir "$OUT"
