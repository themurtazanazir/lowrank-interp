#!/bin/bash
# Run a sweep of experiments across 2 GPUs.
# Seeds are distributed round-robin across GPUs, 2 jobs run in parallel.
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
NUM_GPUS=2

run_job() {
    local seed=$1
    local gpu=$((seed % NUM_GPUS))
    shift
    echo "=== GPU $gpu | seed=$seed | $* ==="
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_experiment.py --seed "$seed" "$@"
}

if [ "$PLACEMENT" = "baseline" ]; then
    for SEED in $SEEDS; do
        run_job "$SEED" &
        # Wait if we've filled all GPUs
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
