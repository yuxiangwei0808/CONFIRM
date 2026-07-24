#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
PHASE="${PHASE:-package}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PROGRESS="${PROGRESS:-on}"
PACKAGE_DIR="${PACKAGE_DIR:-data/neuroclaimbench/v2.1}"
SOURCE_PACKAGE="${SOURCE_PACKAGE:-data/neuroclaimbench/v2.1-source}"
ALIGNMENT_DIR="${ALIGNMENT_DIR:-review-stage/neuroclaimbench-v2.1/alignment}"
ADJUDICATION_DIR="${ADJUDICATION_DIR:-review-stage/neuroclaimbench-v2.1/adjudication}"
CACHE_DIR="${CACHE_DIR:-data/neuroclaimbench/pubmed-cache-v2.1}"
REFERENCE_DIR="${REFERENCE_DIR:-review-stage/neuroclaimbench-v2.1/reference}"
DATA_ROOTS="${DATA_ROOTS:-data/prepared_data/evidence_partitions/benchmark_ready/cohorts data/prepared_data/evidence_partitions/cohorts review-stage/claim-search-safety-gpt55-r10-c10-v7/data/cohorts}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/neuroclaimbench-pycache}"

require_api() {
  if [[ "${ALLOW_API:-0}" != "1" ]]; then
    echo "PHASE=$PHASE requires ALLOW_API=1; no API/network call was started." >&2
    exit 2
  fi
}

progress_args=()
if [[ "$PROGRESS" == "off" ]]; then
  progress_args+=(--no-progress)
fi
data_args=()
for data_root in $DATA_ROOTS; do
  data_args+=(--data-root "$data_root")
done

case "$PHASE" in
  source)
    "$PYTHON" -u -m bench.run_neuroclaimbench_build \
      --out-dir "$SOURCE_PACKAGE"
    ;;
  alignment)
    require_api
    "$PYTHON" -u -m bench.run_neuroclaimbench_alignment \
      --phase all \
      --package-dir "$SOURCE_PACKAGE" \
      --out-dir "$ALIGNMENT_DIR" \
      --model "${MODEL:-google:gemini-3.5-flash}" \
      --schema-retries "${SCHEMA_RETRIES:-3}" \
      --max-workers "$MAX_WORKERS" \
      "${data_args[@]}" "${progress_args[@]}"
    ;;
  package)
    "$PYTHON" -u -m bench.run_neuroclaimbench_v21_build \
      --source-package "$SOURCE_PACKAGE" \
      --alignment-records "$ALIGNMENT_DIR/alignment_records.jsonl" \
      --out-dir "$PACKAGE_DIR" \
      "${data_args[@]}"
    ;;
  pubmed-plan|pubmed-fetch|pubmed-freeze|pubmed-audit)
    pubmed_phase="${PHASE#pubmed-}"
    if [[ "$pubmed_phase" == "plan" || "$pubmed_phase" == "fetch" ]]; then
      require_api
    fi
    cmd=(
      "$PYTHON" -u -m bench.run_neuroclaimbench_pubmed_cache
      --phase "$pubmed_phase"
      --package-dir "$PACKAGE_DIR"
      --cache-dir "$CACHE_DIR"
      --assessor-model "${ASSESSOR_1:-openai:gpt-5.5}"
      --assessor-model "${ASSESSOR_2:-google:gemini-3.5-flash}"
      --schema-retries "${SCHEMA_RETRIES:-3}"
      --max-records-per-query "${MAX_RECORDS_PER_QUERY:-5}"
      --max-evidence-records "${MAX_EVIDENCE_RECORDS:-12}"
      --max-workers "$MAX_WORKERS"
      --fetch-batch-size "${FETCH_BATCH_SIZE:-100}"
      --request-retries "${REQUEST_RETRIES:-4}"
      --retry-delay "${RETRY_DELAY:-1.0}"
      --pubmed-timeout "${PUBMED_TIMEOUT:-30}"
      "${progress_args[@]}"
    )
    if [[ -n "${NCBI_EMAIL:-}" ]]; then
      cmd+=(--pubmed-email "$NCBI_EMAIL")
    fi
    if [[ -n "${NCBI_API_KEY:-}" ]]; then
      cmd+=(--pubmed-api-key "$NCBI_API_KEY")
    fi
    "${cmd[@]}"
    ;;
  adjudication-pilot|adjudication-full|adjudication-pilot_audit|adjudication-finalize)
    adjudication_phase="${PHASE#adjudication-}"
    if [[ "$adjudication_phase" == "pilot" || "$adjudication_phase" == "full" ]]; then
      require_api
    fi
    cmd=(
      "$PYTHON" -u -m bench.run_neuroclaimbench_adjudication
      --phase "$adjudication_phase"
      --package-dir "$PACKAGE_DIR"
      --out-dir "$ADJUDICATION_DIR"
      --pubmed-cache-dir "$CACHE_DIR"
      --assessor-model "${ASSESSOR_1:-openai:gpt-5.5}"
      --assessor-model "${ASSESSOR_2:-google:gemini-3.5-flash}"
      --adjudicator-model "${ADJUDICATOR:-openrouter:anthropic/claude-opus-4.8}"
      --schema-retries "${SCHEMA_RETRIES:-3}"
      --max-workers "$MAX_WORKERS"
      --parallel-backend "${PARALLEL_BACKEND:-thread}"
      --llm-max-tokens "${LLM_MAX_TOKENS:-8192}"
      --pilot-seed "${PILOT_SEED:-20260723}"
      "${progress_args[@]}"
    )
    if [[ "$adjudication_phase" == "full" && -f "$ADJUDICATION_DIR/pilot_audit.json" ]]; then
      cmd+=(--accepted-pilot-audit "$ADJUDICATION_DIR/pilot_audit.json")
    fi
    "${cmd[@]}"
    ;;
  reference)
    "$PYTHON" -u -m bench.run_neuroclaimbench_reference_expansion \
      --package-dir "$PACKAGE_DIR" \
      --out-dir "$REFERENCE_DIR" \
      --no-observed-results
    ;;
  *)
    echo "Unknown PHASE=$PHASE" >&2
    echo "Expected source, alignment, package, pubmed-{plan,fetch,freeze,audit}," >&2
    echo "adjudication-{pilot,pilot_audit,full,finalize}, or reference." >&2
    exit 2
    ;;
esac

echo "NeuroClaimBench build phase complete: $PHASE"
