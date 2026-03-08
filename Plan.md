# Low-Rank Bottleneck Injection: Experimental Plan

## Core Thesis

Forcing intermediate representations through geometrically constrained bottlenecks during training yields representations that are more **identifiable** (consistent across random seeds) without catastrophic performance loss. If additionally a nonlinearity (ReLU) is introduced in the bottleneck, individual features become recoverable — not just the global geometry — connecting architectural constraints to classical identifiability results from ICA/NMF theory.

This experiment tests the foundational claim of interpretability-driven architecture design: that architectural constraints can make interpretability a property of the model class, rather than a post-hoc discovery.

## Motivation

### The Redundancy Puzzle

Multiple lines of evidence (LaCo, SLEB, ShortGPT, LayerShuffle) show that standard transformers learn highly redundant representations across layers. If adjacent layers compute similar things, are the learned features uniquely determined, or does redundancy reflect fundamental non-identifiability in how concepts are encoded? Bottleneck constraints may force the model to use its capacity more efficiently, reducing redundancy and improving identifiability.

### The SAE Reproducibility Problem

Paulo & Belrose (2025) showed that SAEs trained on identical data with different random seeds share only ~30% of features. This makes interpretability findings non-transferable across training runs. If the *architecture itself* produced more identifiable representations, downstream feature extraction (SAEs or otherwise) might inherit that identifiability.

## Experimental Setup

### Base Model

- **Architecture:** Decoder-only transformer (GPT-style)
- **Layers:** 6
- **d_model:** 256
- **Heads:** 4
- **Context length:** 256 tokens
- **Parameters:** ~5-8M (small enough for Colab free tier)

### Dataset

- **TinyStories** (Eldan & Li, 2023) via HuggingFace (`roneneldan/TinyStories`)
- BPE tokenized (GPT-2 tokenizer or custom small vocab)
- Semantic structure rich enough for interpretability questions (characters, actions, sentiment, narrative structure) while being small enough for fast iteration

### Training Configuration

- **Optimizer:** AdamW
- **Schedule:** Cosine learning rate decay with warmup
- **Epochs:** 5-10 (calibrate on baseline)
- **Seeds per configuration:** 5 (gives 10 unique seed pairs for cross-seed comparison)
- **Hardware:** Single GPU (Colab free T4 or local)
- **Estimated time per run:** 10-20 minutes

## Experimental Design

### The 2×2 Grid

We vary two factors:

**Factor 1 — Bottleneck Placement:**

- **(A) Between blocks:** Insert bottleneck between every consecutive pair of transformer blocks. This constrains the *residual stream* — the communication channel between layers.
  - Architecture: `Block_i output → W_down (256→r) → [activation] → W_up (r→256) → Block_{i+1} input`
  - This is additive to the residual stream (bottleneck output is added back, or replaces the stream — we start with replacement for cleaner constraint).

- **(B) Replace MLP hidden layer:** Replace the standard MLP expansion (`256→1024→256`) with a compressed version (`256→r→256`).
  - Architecture: Standard attention, but MLP becomes `W_down (256→r) → [activation] → W_up (r→256)`
  - This constrains what each layer can *compute internally*.

**Factor 2 — Bottleneck Nonlinearity:**

- **Linear:** Bottleneck is `W_down @ W_up` (pure rank constraint). The individual bottleneck dimensions have no privileged meaning — the basis is arbitrary up to rotation.

- **ReLU:** Bottleneck is `ReLU(W_down @ x) → W_up`. The nonlinearity breaks rotational symmetry, giving each dimension a privileged identity (it either fires or doesn't). Connects to ICA/NMF identifiability theory.

### Rank Sweep

For each cell in the 2×2 grid:

- **r ∈ {16, 32, 64, 128, 256}**
- r=256 is effectively the baseline (no compression) for type B
- For type A, r=256 means the bottleneck is full-rank (still adds parameters, but no compression)

### Full Run Count

| Phase | Configurations | Seeds | Total Runs |
|-------|---------------|-------|------------|
| Phase 1: Baseline (no bottleneck) | 1 | 5 | 5 |
| Phase 2: Quick signal check (r=64 only, all 4 grid cells) | 4 | 5 | 20 |
| Phase 3: Full rank sweep (most promising cell) | 5 ranks | 5 | 25 |
| Phase 4: Full rank sweep (second most promising cell) | 5 ranks | 5 | 25 |
| **Total** | | | **~75 runs** |

At 10-20 min per run, this is ~12-25 GPU hours — comfortably doable on Colab free over a few sessions.

## Metrics

### 1. Performance: Validation Perplexity

- Standard autoregressive language modeling loss on held-out TinyStories split
- Report perplexity for direct comparison
- **Purpose:** Quantify the performance cost of each constraint

### 2. Global Representational Similarity: Linear CKA

- **What it measures:** Whether two models (different seeds) encode the same similarity structure over inputs, invariant to rotation/scaling
- **How:** For a fixed eval set of N inputs, extract activations at the bottleneck layer. Build the (N×N) Gram matrix for each seed. CKA measures alignment of these Gram matrices.
- **Score:** 0 = unrelated, 1 = identical structure
- **Report:** Mean pairwise CKA across all 10 seed pairs (5 seeds → 10 pairs), at each layer
- **Applies to:** All 4 grid cells (linear and ReLU bottlenecks)

### 3. Feature-Level Identifiability: Mean Max Cosine Similarity (MMCS)

- **What it measures:** Whether individual features (bottleneck dimensions) can be matched 1-to-1 across seeds
- **How:** Each "feature" is represented by its activation profile across N eval inputs (a vector in R^N). For each feature in Seed A, find the feature in Seed B with maximum cosine similarity. Average these maxima.
- **Score:** 0 = no matching features, 1 = perfect 1-to-1 correspondence
- **Applies to:** ReLU bottleneck variants only (linear bottleneck dimensions are arbitrary up to rotation, making per-dimension MMCS meaningless)
- **For linear bottlenecks:** Optionally extract ICA directions first, then compute MMCS on those. This is secondary.

### 4. Effective Dimensionality: Participation Ratio

- **What it measures:** How many dimensions the model actually uses within the bottleneck
- **How:** Compute SVD of the (N×r) activation matrix, get singular values σ_i. Participation ratio = (Σ σ_i²)² / Σ σ_i⁴
- **Purpose:** Check if the model "spreads out" within the bottleneck or collapses to even lower rank. If a rank-64 bottleneck only uses ~20 effective dimensions, that tells us the true intrinsic dimensionality.
- **Applies to:** All 4 grid cells

### 5. Downstream Probing (Phase 4+)

- Train linear probes on bottleneck activations for known properties in TinyStories:
  - Character identity (which character is the sentence about?)
  - Sentiment (positive/negative story direction)
  - Action type (playing, eating, sleeping, etc.)
- **Purpose:** Do constrained features align with human-interpretable concepts more cleanly?
- **Applies to:** Most promising configurations from Phases 2-3

## Expected Results & Interpretation

### What Constitutes a Positive Result

1. **CKA increases as rank decreases** (for both linear and ReLU bottlenecks) — rank constraint alone improves global representational consistency
2. **MMCS is high for ReLU bottlenecks but not meaningfully measurable for linear** — the nonlinearity is what buys feature-level identifiability, not rank alone
3. **Performance degrades gracefully** — there exists a rank r* where CKA/MMCS is substantially higher than baseline while perplexity is within 10-20% of baseline
4. **A favorable Pareto frontier** exists in the (perplexity, identifiability) space

### What Would Be Really Exciting

- ReLU bottleneck **between blocks (type A)** achieves both high CKA *and* high MMCS at moderate ranks, with modest performance cost. This would mean: geometric constraint + nonlinearity together yield individually identifiable features in the residual stream.
- The features at the bottleneck are **inspectable** — individual dimensions clearly correspond to interpretable concepts (character, sentiment, narrative state) without any post-hoc analysis.
- Effective dimensionality at the bottleneck is *lower* than r, suggesting the model's true representational needs are compact — supporting the redundancy thesis.

### What Would Be a Negative (but Informative) Result

- CKA doesn't increase with bottleneck constraints → the model finds ways to encode equivalent information in non-comparable ways even under rank constraints (the constraint isn't strong enough)
- Performance collapses at any rank that improves identifiability → there's no favorable tradeoff (the constraint is too strong)
- MMCS is high for ReLU bottlenecks but the matched features aren't interpretable → identifiability ≠ interpretability (features are consistent but opaque)

### Key Comparison to Highlight

The **Linear vs. ReLU comparison at the same rank** is the most novel analysis:

- If CKA(linear) ≈ CKA(ReLU) but MMCS(ReLU) >> MMCS(linear): the rank constraint determines global geometry, but the nonlinearity determines feature-level identifiability. This is a clean, publishable finding with theoretical grounding in ICA.
- If CKA(ReLU) > CKA(linear): the nonlinearity helps even at the global level, suggesting ReLU-induced sparsity is a stronger regularizer than rank alone.

## Execution Timeline

### Phase 1: Baseline (Week 1)
- Implement tiny transformer
- Train 5 seeds, no bottleneck
- Compute CKA across all layers and seed pairs
- Compute participation ratio at each layer
- **Deliverable:** Baseline numbers — "how non-identifiable is a standard tiny transformer?"

### Phase 2: Signal Check (Week 1-2)
- Implement both bottleneck types (A and B) with both nonlinearities (linear and ReLU)
- Train all 4 configurations at r=64, 5 seeds each
- Compute CKA, MMCS (ReLU only), participation ratio, perplexity
- **Deliverable:** 2×2 results table — "is there signal? which cell is most promising?"

### Phase 3: Rank Sweep (Week 2-3)
- Full sweep of r ∈ {16, 32, 64, 128, 256} for the top 1-2 configurations
- Plot Pareto frontiers: perplexity vs. CKA, perplexity vs. MMCS
- **Deliverable:** Tradeoff curves showing the sweet spot

### Phase 4: Feature Analysis (Week 3-4)
- Qualitative inspection of bottleneck features for most promising configuration
- Linear probing for TinyStories concepts
- Optionally: train SAEs on constrained vs. unconstrained models, measure SAE cross-seed reproducibility
- **Deliverable:** Evidence (or not) that identifiable features are also interpretable

## Future Extensions (Beyond This Experiment)

- **Scale up:** Does the finding hold for GPT-2 scale? For vision transformers?
- **Bottleneck placement ablation:** Every layer vs. specific layers. Where does the constraint matter most?
- **Formal theory:** Can we prove identifiability guarantees for ReLU bottleneck representations under mild assumptions? (Connects to ICA theory, nonlinear ICA, identifiable VAEs)
- **SAE connection:** Train SAEs on bottleneck representations. Does architectural identifiability make SAE features more reproducible?
- **Comparison to saliency-guided training:** Does the architectural constraint achieve what Feizi et al.'s training intervention does, but more directly?

## References

- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited" — Linear CKA
- Paulo & Belrose (2025). "Sparse Autoencoders Trained on the Same Data Learn Different Features" — SAE reproducibility, MMCS
- Eldan & Li (2023). "TinyStories" — Dataset
- Fel et al. (2025). "Archetypal SAE" — Geometric anchoring for identifiability
- Zhu et al. (2025). "AbsTopK" — Bidirectional SAE features
- Ismail, Bravo & Feizi (2021). "Saliency Guided Training" — Training interventions for interpretability
- Mueller, Geiger, Wiegreffe et al. (2025). "MIB" — Mechanistic interpretability benchmark
- Hyvärinen & Pajunen (1999). "Nonlinear ICA" — Theoretical grounding for identifiability via nonlinearity
