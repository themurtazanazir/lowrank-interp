"""
Metrics for measuring representational identifiability.

- Linear CKA: global representational similarity across seeds
- MMCS: feature-level identifiability (ReLU bottlenecks only)
- Participation Ratio: effective dimensionality
"""

import torch
import numpy as np

from src.model import Bottleneck, BottleneckMLP


def extract_activations(model, dataloader, layer_indices, device, max_batches=None):
    """Extract activations at specified layers for a dataset.

    Returns dict: {layer_idx: tensor of shape (N, d_model)}
    """
    model.eval()
    activations = {i: [] for i in layer_indices}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            activations[layer_idx].append(output.detach().mean(dim=1).cpu())
        return hook_fn

    for idx in layer_indices:
        h = model.blocks[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            model(x)

    for h in hooks:
        h.remove()

    return {k: torch.cat(v, dim=0) for k, v in activations.items()}


def _extract_with_hooks(model, dataloader, modules, device, max_batches=None):
    """Generic hook-based activation extraction.

    Args:
        modules: list of (name, nn.Module) pairs to hook
    Returns:
        dict: {name: tensor of shape (N, d)}
    """
    model.eval()
    activations = {name: [] for name, _ in modules}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name].append(output.detach().mean(dim=1).cpu())
        return hook_fn

    for name, module in modules:
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            model(x)

    for h in hooks:
        h.remove()

    return {k: torch.cat(v, dim=0) for k, v in activations.items()}


def extract_bottleneck_activations(model, dataloader, device, max_batches=None):
    """Extract post-activation bottleneck representations.

    Works for both between-block bottlenecks and MLP bottlenecks.
    Hooks into the full Bottleneck/BottleneckMLP module to get the intermediate
    representation after activation (not pre-activation).

    Returns list of tensors, each shape (N, rank).
    """
    model.eval()

    # Find all bottleneck modules
    bottleneck_modules = []
    for name, module in model.named_modules():
        if isinstance(module, Bottleneck):
            bottleneck_modules.append((name, module))
        elif isinstance(module, BottleneckMLP):
            bottleneck_modules.append((name, module))

    if not bottleneck_modules:
        return []

    # Hook into fc1/down to capture the intermediate representation
    activations = [[] for _ in bottleneck_modules]
    hooks = []

    for idx, (name, module) in enumerate(bottleneck_modules):
        if isinstance(module, Bottleneck):
            target = module.down
            act_type = module.activation
        else:
            target = module.fc1
            act_type = module.activation

        def make_hook(bn_idx, activation):
            def hook_fn(mod, input, output):
                act = output.detach().mean(dim=1)
                if activation == "relu":
                    act = torch.relu(act)
                elif activation != "linear":
                    act = torch.nn.functional.gelu(act)
                activations[bn_idx].append(act.cpu())
            return hook_fn

        h = target.register_forward_hook(make_hook(idx, act_type))
        hooks.append(h)

    with torch.no_grad():
        for j, (x, _) in enumerate(dataloader):
            if max_batches and j >= max_batches:
                break
            x = x.to(device)
            model(x)

    for h in hooks:
        h.remove()

    return [torch.cat(a, dim=0) for a in activations]


# --- Linear CKA ---

def linear_cka(X, Y):
    """Compute Linear CKA between two activation matrices.

    Uses the feature-space formulation: O(N * d^2) instead of O(N^2).
    X, Y: numpy arrays of shape (N, d1) and (N, d2)
    Returns: scalar in [0, 1]
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    YtX = Y.T @ X
    hsic_xy = np.sum(YtX ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)

    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


# --- Mean Max Cosine Similarity ---

def mmcs(X, Y):
    """Compute Mean Max Cosine Similarity between feature sets.

    X, Y: numpy arrays of shape (N, d) — each column is a feature's activation profile.
    Returns: scalar in [0, 1]
    """
    X_norm = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=0, keepdims=True) + 1e-10)

    sim = X_norm.T @ Y_norm

    max_sim_x = np.max(np.abs(sim), axis=1).mean()
    max_sim_y = np.max(np.abs(sim), axis=0).mean()

    return (max_sim_x + max_sim_y) / 2


# --- Participation Ratio ---

def participation_ratio(X):
    """Compute participation ratio of activation matrix.

    X: numpy array of shape (N, d)
    Returns: scalar — effective number of dimensions used
    """
    X = X - X.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    s2 = s ** 2
    return (s2.sum() ** 2) / (s2 ** 2).sum()


# --- Pairwise metrics across seeds ---

def pairwise_cka(activation_list):
    """Compute pairwise CKA for a list of activation matrices (one per seed).

    Returns: mean CKA, std CKA, full matrix
    """
    n = len(activation_list)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = linear_cka(activation_list[i], activation_list[j])
            matrix[i, j] = c
            matrix[j, i] = c
        matrix[i, i] = 1.0

    vals = [matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    return np.mean(vals), np.std(vals), matrix


def pairwise_mmcs(activation_list):
    """Compute pairwise MMCS for a list of activation matrices (one per seed)."""
    n = len(activation_list)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = mmcs(activation_list[i], activation_list[j])
            matrix[i, j] = m
            matrix[j, i] = m
        matrix[i, i] = 1.0

    vals = [matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    return np.mean(vals), np.std(vals), matrix
