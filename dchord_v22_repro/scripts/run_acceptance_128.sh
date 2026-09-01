#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/autodl-tmp/TorchSpec}"
CHECKPOINT="${2:-${DCHORD_K3_CHECKPOINT:-}}"
DEVICE="${3:-cuda:0}"
OUTPUT_DIR="${DCHORD_REPRO_OUTPUT:-/root/autodl-tmp/reports/dchord_torch_repro/acceptance_128}"
LOG_DIR="${OUTPUT_DIR}/logs"

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[FAIL] pass the external DChord-K3 pytorch_model.bin as argument 2" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export DCHORD_K3_CHECKPOINT="${CHECKPOINT}"

for MODE in dflash_b16 dchord_k3; do
  python "${REPO_DIR}/experiments/dchord/repro/torch_fourway_cached.py" \
    --mode "${MODE}" \
    --limit 128 \
    --warmup 1 \
    --device "${DEVICE}" \
    --output-dir "${OUTPUT_DIR}" \
    --export "${CHECKPOINT}" \
    2>&1 | tee "${LOG_DIR}/${MODE}_128.log"
done

python "${REPO_DIR}/experiments/dchord/repro/summarize_acceptance.py" \
  --result-dir "${OUTPUT_DIR}" \
  --target "${DCHORD_TARGET:-/root/autodl-tmp/models/Qwen3.5-4B}" \
  --output "${OUTPUT_DIR}/acceptance_summary.json" \
  2>&1 | tee "${LOG_DIR}/summarize_acceptance.log"

echo "[PASS] Acceptance summary: ${OUTPUT_DIR}/acceptance_summary.json"
