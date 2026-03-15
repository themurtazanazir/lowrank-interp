#!/bin/bash
# Run a sweep of experiments, auto-parallelized across available GPUs.
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

JOB_IDX=0

run_job() {
    local seed=$1
    local gpu=$((JOB_IDX % NUM_GPUS))
    JOB_IDX=$((JOB_IDX + 1))
    shift
    echo "=== GPU $gpu | seed=$seed | $* ==="
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_experiment.py --seed "$seed" "$@"
}

if [ "$PLACEMENT" = "baseline" ]; then
    for SEED in $SEEDS; do
        run_job "$SEED" &
        if (( $(jobs -r | wc -l) >= NUM_GPUS )); then
            wait -n
        fi
    done
    wait
else
    for RANK in $RANKS; do
        for SEED in $SEEDS; do
            run_job "$SEED" --placement "$PLACEMENT" --activation "$ACTIVATION" --rank "$RANK" &
            if (( $(jobs -r | wc -l) >= NUM_GPUS )); then
                wait -n
            fi
        done
        wait  # finish all seeds for this rank before next
    done
fi

echo "=== All runs complete ==="
