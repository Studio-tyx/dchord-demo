#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/autodl-tmp/TorchSpec}"
OUTPUT_DIR="${DCHORD_SELF_DISTILL_OUTPUT:-/root/autodl-tmp/reports/dchord_torch_repro/self_distill_1k}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export DCHORD_TORCHSPEC_REPO="${REPO_DIR}"
export DCHORD_SELF_DISTILL_OUTPUT="${OUTPUT_DIR}"
SCRIPT="${REPO_DIR}/experiments/dchord/repro/self_distill_1k.py"

python "${SCRIPT}" prepare 2>&1 | tee "${LOG_DIR}/01_prepare.log"
python "${SCRIPT}" generate --start 0 --end 500 --batch-size 8 \
  2>&1 | tee "${LOG_DIR}/02_generate_0000_0500.log"
python "${SCRIPT}" generate --start 500 --end 1000 --batch-size 8 \
  2>&1 | tee "${LOG_DIR}/03_generate_0500_1000.log"
python "${SCRIPT}" generate --start 0 --end 32 --batch-size 8 --tag repeat32 \
  2>&1 | tee "${LOG_DIR}/04_repeat32.log"
python "${SCRIPT}" finalize 2>&1 | tee "${LOG_DIR}/05_finalize.log"

echo "[PASS] Teacher data: ${OUTPUT_DIR}/dchord_teacher_1000.jsonl"
