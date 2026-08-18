# OutreachLM V4 Architecture

OutreachLM V4 is a character-level transformer designed to keep the successful
recovery-training path while changing internal modeling components.

## Core configuration

- Vocabulary: 490 characters
- Context length: 256
- Embedding dimension: 256
- Transformer layers: 4
- Attention heads: 8
- Head dimension: 32

## Architectural changes

- Rotary position encoding (RoPE) in attention
- RMSNorm in place of LayerNorm
- QKV attention with Q/K normalization and causal SDPA
- SwiGLU feed-forward network
- Tied language-model head (output projection tied to token embeddings)

## Training recipe

- Shifted next-character prediction objective
- Balanced cross-entropy
- Label smoothing: 0.05
- Recovery loss weight: 2.0
- AdamW
- Learning rate: 5e-4
- Warmup: 250 steps
- Cosine decay
- Batch size: 8
- Gradient clipping: 1.0
- Evaluation interval: 250 steps
- Step checkpoint interval: 500 steps
- Default target: 4500 steps
- Final artifact selection: highest validation `free_match` checkpoint (rollout-aware)
- Conservative early stop: triggered only after sustained free-match degradation
- Optional post-error recovery objective (disabled by default): trains on model-generated
  trajectories after a forced wrong token near the divergence boundary

## Run commands

Train:

```bash
python -m outreachlm.train_v4
```

Explicit steps:

```bash
python -m outreachlm.train_v4 --steps 4500
```

Post-error recovery objective experiment:

```bash
python -m outreachlm.train_v4 --steps 4500 --post-error-loss-weight 1.0
```

Generate:

```bash
python -m outreachlm.v4_generate --prompt "OutreachLM is"
```
