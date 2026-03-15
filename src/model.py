"""
Decoder-only transformer with optional low-rank bottleneck injection.

Bottleneck variants:
  - "between": inserted between consecutive transformer blocks (constrains residual stream)
  - "mlp": replaces the MLP hidden layer within each block (constrains internal computation)

Activation variants (for bottleneck):
  - "linear": pure rank constraint, no nonlinearity in bottleneck (basis is arbitrary)
  - "relu": ReLU in bottleneck (breaks rotational symmetry, induces identifiability)

Between-block residual modes:
  - "residual": x = x + bottleneck(x)  — model can route around the bottleneck
  - "replacement": x = bottleneck(x)   — representation must pass through the bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Low-rank bottleneck: project d_model -> rank -> d_model."""

    def __init__(self, d_model, rank, activation="relu"):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.activation = activation

        # Initialized after _init_weights via _init_bottlenecks

    def forward(self, x):
        h = self.down(x)
        if self.activation == "relu":
            h = F.relu(h)
        return self.up(h)


class MLP(nn.Module):
    """Standard MLP: d_model -> 4*d_model -> d_model."""

    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class BottleneckMLP(nn.Module):
    """MLP with bottleneck replacing the hidden layer: d_model -> rank -> d_model.

    activation="linear" uses GELU (standard nonlinearity, only rank is constrained).
    activation="relu" uses ReLU (breaks rotational symmetry for identifiability).
    This ensures "linear" vs "relu" compares only the activation type, not rank+nonlinearity.
    """

    def __init__(self, d_model, rank, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, rank)
        self.fc2 = nn.Linear(rank, d_model)
        self.activation = activation

    def forward(self, x):
        h = self.fc1(x)
        if self.activation == "relu":
            h = F.relu(h)
        else:
            h = F.gelu(h)
        return self.fc2(h)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_class=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = mlp_class if mlp_class is not None else MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["model"]["d_model"]
        n_heads = config["model"]["n_heads"]
        n_layers = config["model"]["layers"]
        ctx = config["model"]["context_len"]
        vocab = config["model"]["vocab_size"]

        bn_cfg = config["bottleneck"]
        bn_type = bn_cfg["type"]
        bn_placement = bn_cfg.get("placement")
        bn_activation = bn_cfg.get("activation", "relu")
        bn_rank = bn_cfg.get("rank")
        self.bn_residual = bn_cfg.get("residual", False)

        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(ctx, d)

        # Build blocks with optional MLP replacement
        blocks = []
        for _ in range(n_layers):
            if bn_type != "none" and bn_placement == "mlp":
                mlp = BottleneckMLP(d, bn_rank, bn_activation)
            else:
                mlp = MLP(d)
            blocks.append(TransformerBlock(d, n_heads, mlp))
        self.blocks = nn.ModuleList(blocks)

        # Optional between-block bottlenecks
        self.between_bottlenecks = nn.ModuleList()
        if bn_type != "none" and bn_placement == "between":
            for _ in range(n_layers - 1):
                self.between_bottlenecks.append(Bottleneck(d, bn_rank, bn_activation))

        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._init_bottlenecks()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_bottlenecks(self):
        """Re-apply bottleneck-specific init after _init_weights."""
        for bn in self.between_bottlenecks:
            nn.init.kaiming_normal_(bn.down.weight)
            if self.bn_residual:
                # Residual mode: start bottleneck at zero so it doesn't disrupt early training
                nn.init.zeros_(bn.up.weight)
            # Replacement mode: keep the normal_(0.02) init from _init_weights

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.tok_emb(idx) + self.pos_emb(pos)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.between_bottlenecks):
                if self.bn_residual:
                    x = x + self.between_bottlenecks[i](x)
                else:
                    x = self.between_bottlenecks[i](x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Autoregressive generation. idx: (B, T) tensor of token indices."""
        for _ in range(max_new_tokens):
            # Crop to context length
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
