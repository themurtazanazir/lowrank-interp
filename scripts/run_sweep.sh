#!/bin/bash
# Run a sweep of experiments.
# Usage:
#   bash scripts/run_sweep.sh baseline                    # 5 baseline seeds
#   bash scripts/run_sweep.sh between relu 64             # between+relu at rank 64, 5 seeds
#   bash scripts/run_sweep.sh between relu "16 32 64 128" # rank sweep

set -e

PLACEMENT="${1:-}"
ACTIVATION="${2:-}"
RANKS="${3:-64}"
SEEDS="0 1 2 3 4"

if [ "$PLACEMENT" = "baseline" ]; then
    for SEED in $SEEDS; do
        echo "=== baseline seed=$SEED ==="
        python scripts/run_experiment.py --seed "$SEED"
    done
else
    for RANK in $RANKS; do
        for SEED in $SEEDS; do
            echo "=== ${PLACEMENT}_${ACTIVATION}_r${RANK} seed=$SEED ==="
            python scripts/run_experiment.py \
                --placement "$PLACEMENT" \
                --activation "$ACTIVATION" \
                --rank "$RANK" \
                --seed "$SEED"
        done
    done
fi
