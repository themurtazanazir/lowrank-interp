#!/usr/bin/env python3
"""
Run a single training experiment.

Usage:
  python scripts/run_experiment.py                                    # baseline, seed 0
  python scripts/run_experiment.py --placement between --activation relu --rank 64 --seed 42
  python scripts/run_experiment.py --config configs/custom.json       # override from file
  python scripts/run_experiment.py --max_steps 100                    # quick smoke test
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--placement", choices=["between", "mlp"], default=None)
    parser.add_argument("--activation", choices=["linear", "relu"], default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--residual", action="store_true", help="Use residual (additive) bottleneck instead of replacement")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_steps", type=int, default=None, help="Limit steps for smoke tests")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # CLI overrides
    if args.placement:
        config["bottleneck"]["type"] = "bottleneck"
        config["bottleneck"]["placement"] = args.placement
    if args.activation:
        config["bottleneck"]["activation"] = args.activation
    if args.rank is not None:
        config["bottleneck"]["rank"] = args.rank
    if args.seed is not None:
        config["seed"] = args.seed
    if args.residual:
        config["bottleneck"]["residual"] = True

    # Validate
    bn = config["bottleneck"]
    if bn["type"] != "none":
        assert bn["placement"] in ("between", "mlp"), f"Invalid placement: {bn['placement']}"
        assert bn["activation"] in ("linear", "relu"), f"Invalid activation: {bn['activation']}"
        assert bn["rank"] is not None and bn["rank"] > 0, f"Invalid rank: {bn['rank']}"

    train(config, output_dir=args.output_dir, device=args.device, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
