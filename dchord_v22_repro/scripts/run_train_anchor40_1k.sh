#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/root/autodl-tmp/TorchSpec}"
OUTPUT_ROOT="${DCHORD_TRAINING_OUTPUT:-/root/autodl-tmp/reports/dchord_torch_repro/training}"
CONFIG="${DCHORD_TRAIN_CONFIG:-${REPO_DIR}/experiments/dchord/repro/configs/dchord_q35_anchor40_1k.yaml}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

cd "${REPO_DIR}"
python -m torchspec.train_entry \
  --config "${CONFIG}" \
  model.target_model_path="${DCHORD_TARGET:-/root/autodl-tmp/models/Qwen3.5-4B}" \
  model.initial_draft_model_path="${DCHORD_INITIAL_DRAFT_OUTPUT:-/root/autodl-tmp/models/Qwen3.5-4B-DFlash-torchspec-dchord}" \
  model.draft_model_config="${REPO_DIR}/experiments/dchord/repro/assets/dflash_qwen35_torchspec_config.json" \
  dataset.train_data_path="${OUTPUT_ROOT}/data/train_1000.jsonl" \
  training.dchord_profiled_surface_path="${REPO_DIR}/experiments/dchord/repro/assets/profiled_surface_spec_v1.json" \
  training.dchord_export_path="${OUTPUT_ROOT}/dchord_k3_anchor40_export" \
  output_dir="${OUTPUT_ROOT}/dchord_k3_anchor40_run" \
  cache_dir="${DCHORD_TRAIN_CACHE:-/root/autodl-tmp/cache/dchord_torch_repro_anchor40}" \
  2>&1 | tee "${LOG_DIR}/03_train_anchor40_1k.log"

echo "[PASS] Exported checkpoint: ${OUTPUT_ROOT}/dchord_k3_anchor40_export/pytorch_model.bin"
