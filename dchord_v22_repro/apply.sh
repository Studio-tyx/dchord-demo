#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-/root/autodl-tmp/TorchSpec}"
EXPECTED_HEAD="43dec00d39a919309fb9b531b3fab66bdadbc397"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[FAIL] not a git repository: ${REPO_DIR}" >&2
  exit 2
fi

ACTUAL_HEAD="$(git -C "${REPO_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_HEAD}" != "${EXPECTED_HEAD}" ]]; then
  echo "[FAIL] unsupported TorchSpec revision" >&2
  echo "  expected: ${EXPECTED_HEAD}" >&2
  echo "  actual:   ${ACTUAL_HEAD}" >&2
  echo "Use the exact revision or regenerate the shared-file patch." >&2
  exit 3
fi

if ! git -C "${REPO_DIR}" diff --quiet -- \
  torchspec/models/draft/dflash.py \
  torchspec/config/train_config.py \
  torchspec/training/trainer_actor.py \
  torchspec/data/template.py \
  torchspec/training/dflash_trainer.py \
  torchspec/controller/loop.py \
  torchspec/inference/engine/vllm_engine.py; then
  echo "[FAIL] one or more shared TorchSpec integration files have local changes" >&2
  exit 4
fi

for relative in \
  torchspec/models/dchord.py \
  torchspec/training/dchord_trainer.py \
  experiments/dchord/repro/common.py \
  experiments/dchord/repro/cache_transaction.py \
  experiments/dchord/repro/torch_fourway_nocache.py \
  experiments/dchord/repro/torch_fourway_cached.py \
  experiments/dchord/repro/summarize_acceptance.py \
  experiments/dchord/repro/self_distill_1k.py \
  experiments/dchord/repro/prepare_online_training.py \
  experiments/dchord/repro/training_compat.py; do
  if [[ -e "${REPO_DIR}/${relative}" ]]; then
    echo "[FAIL] refusing to overwrite existing file: ${REPO_DIR}/${relative}" >&2
    exit 5
  fi
done

git -C "${REPO_DIR}" apply --check \
  "${PACKAGE_DIR}/patches/0001-dchord-torch-sparse-positions.patch"
git -C "${REPO_DIR}" apply --check \
  "${PACKAGE_DIR}/patches/0002-dchord-training-integration.patch"
git -C "${REPO_DIR}" apply \
  "${PACKAGE_DIR}/patches/0001-dchord-torch-sparse-positions.patch"
git -C "${REPO_DIR}" apply \
  "${PACKAGE_DIR}/patches/0002-dchord-training-integration.patch"
cp -R "${PACKAGE_DIR}/overlay/." "${REPO_DIR}/"

PYTHONPATH="${REPO_DIR}" python -m py_compile \
  "${REPO_DIR}/torchspec/models/dchord.py" \
  "${REPO_DIR}/torchspec/training/dchord_trainer.py" \
  "${REPO_DIR}/experiments/dchord/repro/common.py" \
  "${REPO_DIR}/experiments/dchord/repro/cache_transaction.py" \
  "${REPO_DIR}/experiments/dchord/repro/torch_fourway_nocache.py" \
  "${REPO_DIR}/experiments/dchord/repro/torch_fourway_cached.py" \
  "${REPO_DIR}/experiments/dchord/repro/summarize_acceptance.py" \
  "${REPO_DIR}/experiments/dchord/repro/self_distill_1k.py" \
  "${REPO_DIR}/experiments/dchord/repro/prepare_online_training.py" \
  "${REPO_DIR}/experiments/dchord/repro/training_compat.py"

echo "[PASS] DChord training and Torch inference code installed into ${REPO_DIR}"
echo "No custom serving-backend patch was installed."
