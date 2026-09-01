#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/autodl-tmp/TorchSpec}"
LIMIT="${2:-8}"
DEVICE="${3:-cuda:0}"
OUTPUT_DIR="${DCHORD_REPRO_OUTPUT:-/root/autodl-tmp/reports/dchord_torch_repro/schema_ar}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

for MODE in ar schema; do
  python "${REPO_DIR}/experiments/dchord/repro/torch_fourway_cached.py" \
    --mode "${MODE}" \
    --limit "${LIMIT}" \
    --warmup 1 \
    --device "${DEVICE}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/${MODE}_${LIMIT}.log"
done

echo "[PASS] Schema+AR outputs: ${OUTPUT_DIR}"
