# Low-Rank Bottleneck Injection

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one of the honking great ideas -- let's do more of those!
```

---

## Memory for Future Selves

This file is a persistent memory shared across sessions and agents. Save important decisions, conventions, discoveries, and lessons learned here so that future runs can build on past work rather than rediscovering from scratch.

## Repo Structure

```
src/model.py      — transformer + bottleneck variants (Bottleneck, BottleneckMLP, Transformer)
src/data.py       — TinyStories loading via HuggingFace, GPT-2 tokenizer
src/train.py      — training loop, cosine LR schedule, checkpointing
src/metrics.py    — CKA, MMCS, participation ratio, activation extraction
configs/           — JSON experiment configs (default.json is the base)
scripts/           — CLI entrypoints (run_experiment.py, analyze.py, run_sweep.sh)
results/           — checkpoints + activations (gitignored), metrics (tracked)
figures/           — generated plots (tracked)
```

## Conventions

- **Config format:** JSON. CLI args override config values.
- **Run naming:** `{placement}_{activation}_r{rank}_s{seed}` or `baseline_s{seed}`
- **Syncing:** Git-based. Checkpoints/activations are gitignored (regenerable). Metrics JSON and figures are committed.
- **All code in `src/`, all entrypoints in `scripts/`.** Notebooks are thin wrappers.
- **Seeds:** 0-4 (5 seeds per config, giving 10 pairwise comparisons)

## Decisions Log
