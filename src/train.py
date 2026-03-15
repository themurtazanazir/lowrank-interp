"""
Training loop for transformer experiments.

Handles: optimizer setup, LR schedule, checkpointing, logging.
"""

import json
import math
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.model import Transformer
from src.data import get_dataloaders

# Fixed prompts for sanity-checking generation quality across epochs/runs
SAMPLE_PROMPTS = [
    "Once upon a time",
    "The little dog",
    "She wanted to",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=64):
    """Generate from fixed prompts and print for sanity checking."""
    model.eval()
    print("  --- samples ---")
    for prompt in SAMPLE_PROMPTS:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"  | {text}")
    print("  ---------------")
    model.train()


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

    seed_everything(config["seed"])

    name = run_name(config)
    ckpt_dir = os.path.join(output_dir, "checkpoints", name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save config
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{name}] device={device}")

    model = Transformer(config).to(device)
    print(f"[{name}] params={model.count_parameters():,}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    data_cfg = config.get("data", {})
    train_dl, val_dl = get_dataloaders(
        config,
        max_train=data_cfg.get("max_train_examples"),
        max_val=data_cfg.get("max_val_examples"),
    )

    optimizer = setup_optimizer(model, config)
    tc = config["training"]
    total_steps = len(train_dl) * tc["epochs"]
    if max_steps:
        total_steps = min(total_steps, max_steps)

    log_every = tc.get("log_every_steps", 500)
    patience = tc.get("early_stop_patience", 0)  # 0 = disabled

    step = 0
    log = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(tc["epochs"]):
        pbar = tqdm(train_dl, desc=f"[{name}] epoch {epoch}")
        for x, y in pbar:
            if max_steps and step >= max_steps:
                break

            x, y = x.to(device), y.to(device)

            lr = cosine_schedule(step, tc["warmup_steps"], total_steps, tc["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()

            if tc.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])

            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}")
            step += 1

            if step % log_every == 0:
                print(f"[{name}] step {step} | train_loss={loss.item():.4f} lr={lr:.2e}")

        # End of epoch eval + samples
        val_loss, val_ppl = evaluate(model, val_dl, device)
        print(f"[{name}] epoch {epoch} | val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
        generate_samples(model, tokenizer, device)
        log.append({"epoch": epoch, "val_loss": val_loss, "val_ppl": val_ppl, "step": step})

        if config.get("checkpoint_every_epoch"):
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience and patience_counter >= patience:
                print(f"[{name}] early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Final checkpoint
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))

    # Save training log
    with open(os.path.join(ckpt_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

    val_ppl = math.exp(best_val_loss)
    print(f"[{name}] done. best val_ppl={val_ppl:.2f}")
    return model, log
