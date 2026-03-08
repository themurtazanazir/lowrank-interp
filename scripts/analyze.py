#!/usr/bin/env python3
"""
Post-hoc analysis: load checkpoints, compute CKA/MMCS/PR, generate plots.

Usage:
  python scripts/analyze.py --group baseline          # all baseline seeds
  python scripts/analyze.py --group between_relu_r64  # all seeds for one config
  python scripts/analyze.py --compare                 # compare all groups
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Transformer
from src.data import get_dataloaders
from src.metrics import (
    extract_activations,
    extract_bottleneck_activations,
    participation_ratio,
    pairwise_cka,
    pairwise_mmcs,
)


def find_runs(output_dir, group):
    """Find all checkpoint dirs matching a group pattern."""
    pattern = os.path.join(output_dir, "checkpoints", f"{group}_s*")
    dirs = sorted(glob.glob(pattern))
    return dirs


def load_model(ckpt_dir, device):
    """Load model from checkpoint directory."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)
    model = Transformer(config).to(device)
    state = torch.load(os.path.join(ckpt_dir, "final.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, config


def analyze_group(group, output_dir="results", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = find_runs(output_dir, group)
    if not runs:
        print(f"No runs found for group '{group}'")
        return

    print(f"Found {len(runs)} runs for '{group}':")
    for r in runs:
        print(f"  {r}")

    # Load all models and extract activations
    models = []
    configs = []
    for r in runs:
        model, config = load_model(r, device)
        models.append(model)
        configs.append(config)

    # Use first model's config for dataloader
    _, val_dl = get_dataloaders(configs[0], max_val=configs[0].get("eval_samples", 1024))

    n_layers = configs[0]["model"]["layers"]
    layer_indices = list(range(n_layers))

    # Extract layer activations per seed
    all_layer_acts = []
    for model in models:
        acts = extract_activations(model, val_dl, layer_indices, device, max_batches=32)
        all_layer_acts.append(acts)

    # CKA per layer
    print("\n--- Linear CKA (pairwise across seeds) ---")
    results = {"cka": {}, "pr": {}}
    for layer_idx in layer_indices:
        layer_acts = [a[layer_idx].numpy() for a in all_layer_acts]
        mean_cka, std_cka, _ = pairwise_cka(layer_acts)
        pr_vals = [participation_ratio(a) for a in layer_acts]
        mean_pr = np.mean(pr_vals)

        results["cka"][f"layer_{layer_idx}"] = {"mean": float(mean_cka), "std": float(std_cka)}
        results["pr"][f"layer_{layer_idx}"] = float(mean_pr)
        print(f"  Layer {layer_idx}: CKA={mean_cka:.4f}±{std_cka:.4f}  PR={mean_pr:.1f}")

    # Bottleneck-specific metrics
    bn_type = configs[0]["bottleneck"]["type"]
    if bn_type != "none" and configs[0]["bottleneck"]["placement"] == "between":
        print("\n--- Bottleneck Metrics ---")
        all_bn_acts = []
        for model in models:
            bn_acts = extract_bottleneck_activations(model, val_dl, device, max_batches=32)
            all_bn_acts.append(bn_acts)

        n_bn = len(all_bn_acts[0])
        results["bottleneck_cka"] = {}
        results["bottleneck_mmcs"] = {}
        results["bottleneck_pr"] = {}

        for bn_idx in range(n_bn):
            bn_per_seed = [a[bn_idx].numpy() for a in all_bn_acts]
            mean_cka, std_cka, _ = pairwise_cka(bn_per_seed)
            pr_vals = [participation_ratio(a) for a in bn_per_seed]

            results["bottleneck_cka"][f"bn_{bn_idx}"] = {"mean": float(mean_cka), "std": float(std_cka)}
            results["bottleneck_pr"][f"bn_{bn_idx}"] = float(np.mean(pr_vals))

            print(f"  BN {bn_idx}: CKA={mean_cka:.4f}±{std_cka:.4f}  PR={np.mean(pr_vals):.1f}")

            if configs[0]["bottleneck"]["activation"] == "relu":
                mean_m, std_m, _ = pairwise_mmcs(bn_per_seed)
                results["bottleneck_mmcs"][f"bn_{bn_idx}"] = {"mean": float(mean_m), "std": float(std_m)}
                print(f"         MMCS={mean_m:.4f}±{std_m:.4f}")

    # Save results
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, f"{group}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {metrics_dir}/{group}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, help="Run group name (e.g., 'baseline', 'between_relu_r64')")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    analyze_group(args.group, args.output_dir, args.device)


if __name__ == "__main__":
    main()
