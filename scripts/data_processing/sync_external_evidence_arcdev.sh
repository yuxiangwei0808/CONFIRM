#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SSH_HOST="${SSH_HOST:-${ARCDEV_HOST:-arcdev}}"
: "${REMOTE_ROOT:?Set REMOTE_ROOT to the remote preparation directory}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
REMOTE_RUN_DIR="$REMOTE_ROOT/runs/$RUN_ID"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-data/prepared_data/external_candidates/$RUN_ID}"
PYTHON="${PYTHON:-.venv/bin/python}"
PROMOTE="${PROMOTE:-0}"
BUILD_EVIDENCE="${BUILD_EVIDENCE:-1}"

mkdir -p "$LOCAL_RUN_DIR"
for directory in canonical quarantine audits manifests; do
  if ssh "$SSH_HOST" test -d "$REMOTE_RUN_DIR/$directory"; then
    mkdir -p "$LOCAL_RUN_DIR/$directory"
    rsync -a "$SSH_HOST:$REMOTE_RUN_DIR/$directory/" "$LOCAL_RUN_DIR/$directory/"
  fi
done

echo "synced external evidence run to $LOCAL_RUN_DIR"

if [[ "$PROMOTE" == "1" ]]; then
  export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  "$PYTHON" -m nbs_data.promote_external_evidence --run-dir "$LOCAL_RUN_DIR"
  if [[ "$BUILD_EVIDENCE" == "1" ]]; then
    "$PYTHON" -m confirm.evidence_partitions \
      --config configs/evidence_partitions.yml \
      --out-root data/prepared_data/evidence_partitions \
      --check-overlap
    "$PYTHON" -m nbs_data.prepare_external_evidence coverage \
      --config "${EXTERNAL_DATASET_CONFIG:-configs/external_datasets.local.yml}" \
      --datasets all \
      --out-root "$LOCAL_RUN_DIR" \
      --evidence-manifest data/prepared_data/evidence_partitions/manifest.json \
      --coverage-out-dir data/prepared_data/evidence_partitions/audits
  fi
fi
