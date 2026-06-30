#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-review-stage}"
HARMONIZE="${HARMONIZE:-combat}"
ROUND_DIR="${ROUND_DIR:-$OUT_ROOT/round5-combat}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [ "$OUT_ROOT" != "review-stage" ]; then
  echo "ERROR: OUT_ROOT must be review-stage because bench.run_nacc_external writes review-stage/external-nacc."
  exit 1
fi

mkdir -p "$ROUND_DIR" "$OUT_ROOT/negatives-expansion" "$OUT_ROOT/external-cnp"

"$PYTHON" -m bench.run_expanded_benchmark \
  --data-root data/prepared_data/benchmark_ready \
  --smri-root data/prepared_data/smri_disease \
  --cluster-root data/prepared_data/cluster_recovered \
  --out-dir "$ROUND_DIR" \
  --harmonize "$HARMONIZE"

"$PYTHON" -m bench.run_multimodal_benchmark \
  --data-root data/prepared_data/benchmark_ready \
  --out-dir "$ROUND_DIR" \
  --harmonize "$HARMONIZE"

"$PYTHON" -m bench.combine_benchmark_results \
  --input "$ROUND_DIR/expanded_benchmark_results.json" \
  --input "$ROUND_DIR/multimodal_benchmark_results.json" \
  --out-dir "$ROUND_DIR"

"$PYTHON" -m bench.run_negatives_expansion \
  --root "$ROOT" \
  --out-dir "$OUT_ROOT/negatives-expansion"

"$PYTHON" -m bench.run_nacc_external

"$PYTHON" -m bench.run_external_generic \
  --cohort data/prepared_data/external/ds000030.parquet \
  --claims data/external_benchmark/ds000030_claims.csv \
  --control-dx CONTROL \
  --cohort-name CNP \
  --out-dir "$OUT_ROOT/external-cnp"

echo "initial CONFIRM experiment stack complete: $OUT_ROOT"
