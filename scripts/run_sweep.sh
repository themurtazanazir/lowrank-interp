#!/bin/bash
# Run a sweep of experiments, auto-parallelized across available GPUs.
# Each GPU gets its own sequential queue so two models never share a GPU.
#
# Usage:
#   bash scripts/run_sweep.sh baseline                    # 4 baseline seeds
#   bash scripts/run_sweep.sh between relu 64             # between+relu at rank 64, 4 seeds
#   bash scripts/run_sweep.sh between relu "16 32 64 128" # rank sweep

set -e

PLACEMENT="${1:-}"
ACTIVATION="${2:-}"
RANKS="${3:-64}"
SEEDS="0 1 2 3"

# Auto-detect GPUs (falls back to 1 for CPU-only)
NUM_GPUS=$(python -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)
echo "Detected $NUM_GPUS GPU(s)"

# Build the list of jobs (seed + extra args)
JOBS=()
if [ "$PLACEMENT" = "baseline" ]; then
    for SEED in $SEEDS; do
        JOBS+=("$SEED|")
    done
else
    for RANK in $RANKS; do
        for SEED in $SEEDS; do
            JOBS+=("$SEED|--placement $PLACEMENT --activation $ACTIVATION --rank $RANK")
        done
    done
fi

# Run one GPU's queue: takes gpu_id and list of jobs
run_gpu_queue() {
    local gpu=$1
    shift
    for job in "$@"; do
        local seed="${job%%|*}"
        local extra="${job#*|}"
        echo "=== GPU $gpu | seed=$seed | $extra ==="
        CUDA_VISIBLE_DEVICES=$gpu python scripts/run_experiment.py --seed "$seed" $extra
    done
}

# Distribute jobs round-robin into per-GPU queues, then run each queue in parallel
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    GPU_JOBS=()
    for ((i=gpu; i<${#JOBS[@]}; i+=NUM_GPUS)); do
        GPU_JOBS+=("${JOBS[$i]}")
    done
    if [ ${#GPU_JOBS[@]} -gt 0 ]; then
        run_gpu_queue "$gpu" "${GPU_JOBS[@]}" &
    fi
done

wait
echo "=== All runs complete ==="
