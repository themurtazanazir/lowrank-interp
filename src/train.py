"""
Training loop for transformer experiments.

Handles: optimizer setup, LR schedule, checkpointing, logging.
"""

import json
import math
import os

import torch
from tqdm import tqdm

from src.model import Transformer
from src.data import get_dataloaders


def setup_optimizer(model, config):
    tc = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
        betas=(0.9, 0.95),
    )


def cosine_schedule(step, warmup_steps, total_steps, lr):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_dl, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def run_name(config):
    bn = config["bottleneck"]
    if bn["type"] == "none":
        name = "baseline"
    else:
        name = f"{bn['placement']}_{bn['activation']}_r{bn['rank']}"
    return f"{name}_s{config['seed']}"


def train(config, output_dir="results", device=None, max_steps=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    name = run_name(config)
    ckpt_dir = os.path.join(output_dir, "checkpoints", name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{name}] device={device}")

    model = Transformer(config).to(device)
    print(f"[{name}] params={model.count_parameters():,}")

    train_dl, val_dl = get_dataloaders(config)

    optimizer = setup_optimizer(model, config)
    tc = config["training"]
    total_steps = len(train_dl) * tc["epochs"]
    if max_steps:
        total_steps = min(total_steps, max_steps)

    step = 0
    log = []
    val_loss, val_ppl = float("nan"), float("nan")

    for epoch in range(tc["epochs"]):
        pbar = tqdm(train_dl, desc=f"[{name}] epoch {epoch}")
        for x, y in pbar:
            if max_steps and step >= max_steps:
                break

            x, y = x.to(device), y.to(device)

            lr = cosine_schedule(step, tc["warmup_steps"], total_steps, tc["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()

            if tc.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])

            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}")
            step += 1

        # End of epoch eval
        val_loss, val_ppl = evaluate(model, val_dl, device)
        print(f"[{name}] epoch {epoch} | val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
        log.append({"epoch": epoch, "val_loss": val_loss, "val_ppl": val_ppl, "step": step})

        if config.get("checkpoint_every_epoch"):
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

    # Final checkpoint
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))

    # Save training log
    with open(os.path.join(ckpt_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"[{name}] done. final val_ppl={val_ppl:.2f}")
    return model, log
