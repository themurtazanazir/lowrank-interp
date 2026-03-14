"""
TinyStories dataset loading and tokenization.

Uses GPT-2 tokenizer (vocab_size=50257) and HuggingFace datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class TinyStoriesDataset(Dataset):
    """Pre-tokenized TinyStories dataset."""

    def __init__(self, split, context_len, tokenizer=None, max_examples=None):
        self.context_len = context_len
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ds = load_dataset("roneneldan/TinyStories", split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))

        # Batch tokenize for speed
        texts = list(ds["text"])
        encoded = self.tokenizer(texts, add_special_tokens=False)["input_ids"]

        # Concatenate into one long sequence, then chunk
        all_tokens = []
        for ids in encoded:
            all_tokens.extend(ids)
            all_tokens.append(self.tokenizer.eos_token_id)

        # Chunk into context_len + 1 blocks (input + target)
        n_chunks = len(all_tokens) // (context_len + 1)
        all_tokens = all_tokens[: n_chunks * (context_len + 1)]
        self.chunks = torch.tensor(all_tokens).view(n_chunks, context_len + 1)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]  # input, target


def get_dataloaders(config, max_train=None, max_val=None):
    """Create train and validation dataloaders."""
    ctx = config["model"]["context_len"]
    bs = config["training"]["batch_size"]
    num_workers = config.get("data", {}).get("num_workers", 0)
    seed = config.get("seed", 0)

    train_ds = TinyStoriesDataset("train", ctx, max_examples=max_train)
    val_ds = TinyStoriesDataset("validation", ctx, max_examples=max_val)

    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=True, generator=g,
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_dl, val_dl
