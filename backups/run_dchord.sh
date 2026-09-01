#!/bin/bash
# Launch dchord_5stage.py in a specific mode, fully detached.
export PATH=/root/autodl-tmp/conda/envs/torchspec/bin:$PATH
export LD_LIBRARY_PATH=/root/autodl-tmp/conda/envs/torchspec/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/root/autodl-tmp/TorchSpec_DChord:/root/autodl-tmp/dchord_v22_torch_repro_code_v2/overlay/experiments/dchord/repro:$PYTHONPATH
export DCHORD_K3_CHECKPOINT=/root/autodl-tmp/0822/dchord_k3_random_anchor_1k_export/pytorch_model.bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/0829/five_stage
MODE="$1"; LIMIT="${2:-4}"
LOG=/root/autodl-tmp/0829/backups/log/${MODE}_l${LIMIT}_$(date +%s).log
nohup python dchord_5stage.py --mode "$MODE" --limit "$LIMIT" > "$LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$LOG"
