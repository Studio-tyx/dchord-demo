#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/autodl-tmp/TorchSpec}"
DEVICE="${2:-cuda:0}"
OUTPUT_ROOT="${DCHORD_TRAINING_OUTPUT:-/root/autodl-tmp/reports/dchord_torch_repro/training}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export DCHORD_COMPAT_OUTPUT="${OUTPUT_ROOT}/compat"
export DCHORD_ONLINE_DATA_OUTPUT="${OUTPUT_ROOT}/data"

python "${REPO_DIR}/experiments/dchord/repro/training_compat.py" \
  --device "${DEVICE}" 2>&1 | tee "${LOG_DIR}/01_compat.log"
python "${REPO_DIR}/experiments/dchord/repro/prepare_online_training.py" \
  --limit 1000 2>&1 | tee "${LOG_DIR}/02_prepare_online_data.log"

echo "[PASS] TorchSpec training data: ${OUTPUT_ROOT}/data/train_1000.jsonl"
echo "[PASS] Initial DFlash checkpoint: ${DCHORD_INITIAL_DRAFT_OUTPUT:-/root/autodl-tmp/models/Qwen3.5-4B-DFlash-torchspec-dchord}"
