#!/bin/bash
# Launcher for dchord_5stage.py on the AutoDL server.
# Usage: run_5stage.sh <mode> [strategy] [limit]
#   mode:     serial | parallel
#   strategy: dfs | bfs  (default: dfs, only used by parallel)
#   limit:    number of samples (default: 2)
set -e
MODE="$1"
STRATEGY="${2:-dfs}"
LIMIT="${3:-2}"
cd /root/autodl-tmp/0829/five_stage
export PATH=/root/autodl-tmp/conda/envs/torchspec/bin:$PATH
export LD_LIBRARY_PATH=/root/autodl-tmp/conda/envs/torchspec/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/root/autodl-tmp/TorchSpec_DChord:/root/autodl-tmp/dchord_v22_torch_repro_code_v2/overlay/experiments/dchord/repro:$PYTHONPATH
export DCHORD_K3_CHECKPOINT=/root/autodl-tmp/0822/dchord_k3_random_anchor_1k_export/pytorch_model.bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/autodl-tmp/reports/log
LOG="/root/autodl-tmp/reports/log/${MODE}_${STRATEGY}_$(date +%s).log"
nohup python -u dchord_5stage.py --limit "$LIMIT" --mode "$MODE" --strategy "$STRATEGY" --device cuda:0 > "$LOG" 2>&1 &
echo "PID=$!"
echo "LOG=$LOG"
