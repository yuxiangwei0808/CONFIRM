#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-review-stage}"
HARMONIZE="${HARMONIZE:-combat}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PARALLEL_BACKEND="${PARALLEL_BACKEND:-process}"
PROGRESS="${PROGRESS:-on}"
EVIDENCE_CONFIG="${EVIDENCE_CONFIG:-configs/evidence_partitions.yml}"
PARTITION_ROOT="${PARTITION_ROOT:-data/prepared_data/evidence_partitions}"
PARTITIONED_BENCHMARK_ROOT="${PARTITIONED_BENCHMARK_ROOT:-$PARTITION_ROOT/benchmark_ready}"
MAIN_DIR="${MAIN_DIR:-$OUT_ROOT/curated-gate-benchmark-combat}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-off}"
RUN_EXTERNAL="${RUN_EXTERNAL:-off}"
SYNTHETIC_DIR="${SYNTHETIC_DIR:-$OUT_ROOT/curated-gate-synthetic-stress}"
NACC_DIR="${NACC_DIR:-$OUT_ROOT/curated-gate-external-nacc}"
CNP_DIR="${CNP_DIR:-$OUT_ROOT/curated-gate-external-cnp}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$MAIN_DIR"

progress_arg=""
if [[ "$PROGRESS" == "off" ]]; then
  progress_arg="--no-progress"
fi

echo "[stage] build evidence partitions -> $PARTITION_ROOT"
"$PYTHON" -m confirm.evidence_partitions \
  --config "$EVIDENCE_CONFIG" \
  --out-root "$PARTITION_ROOT" \
  --check-overlap

echo "[stage] build partitioned benchmark-ready layer -> $PARTITIONED_BENCHMARK_ROOT"
"$PYTHON" nbs_data/build_partitioned_benchmark_ready_layer.py \
  --benchmark-root data/prepared_data/benchmark_ready \
  --evidence-manifest "$PARTITION_ROOT/manifest.json" \
  --out-root "$PARTITIONED_BENCHMARK_ROOT"

echo "[stage] run curated deterministic gate benchmark -> $MAIN_DIR"
"$PYTHON" -m bench.run_expanded_benchmark \
  --data-root "$PARTITIONED_BENCHMARK_ROOT" \
  --smri-root "$PARTITION_ROOT/cohorts" \
  --cluster-root "$PARTITION_ROOT/cohorts" \
  --out-dir "$MAIN_DIR" \
  --harmonize "$HARMONIZE" \
  --use-evidence-partitions \
  --max-workers "$MAX_WORKERS" \
  --parallel-backend "$PARALLEL_BACKEND" \
  ${progress_arg:+"$progress_arg"}

echo "[stage] combine curated gate outputs -> $MAIN_DIR"
"$PYTHON" -m bench.combine_benchmark_results \
  --input "$MAIN_DIR/expanded_benchmark_results.json" \
  --out-dir "$MAIN_DIR"

if [[ "$RUN_SYNTHETIC" == "on" ]]; then
  mkdir -p "$SYNTHETIC_DIR"
  echo "[stage] run synthetic stress auxiliary -> $SYNTHETIC_DIR"
  "$PYTHON" -m bench.run_negatives_expansion \
    --root "$ROOT" \
    --out-dir "$SYNTHETIC_DIR"
fi

if [[ "$RUN_EXTERNAL" == "on" ]]; then
  mkdir -p "$NACC_DIR" "$CNP_DIR"
  echo "[stage] run NACC external auxiliary -> $NACC_DIR"
  "$PYTHON" -m bench.run_nacc_external \
    --out-dir "$NACC_DIR"

  echo "[stage] run CNP external auxiliary -> $CNP_DIR"
  "$PYTHON" -m bench.run_external_generic \
    --cohort data/prepared_data/external/ds000030.parquet \
    --claims data/external_benchmark/ds000030_claims.csv \
    --control-dx CONTROL \
    --cohort-name CNP \
    --out-dir "$CNP_DIR"
fi

echo "curated deterministic CONFIRM gate benchmark complete: $MAIN_DIR"
