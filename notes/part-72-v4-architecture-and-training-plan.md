# OutreachLM — Part 72: V4 Architecture and Training Plan

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

Implemented V4 components in source:

- [v4_model.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/v4_model.py)
- [train_v4.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train_v4.py)
- [v4_generate.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/v4_generate.py)
- [v4-architecture.md](C:/Users/josep/OneDrive/Desktop/OutreachLM/docs/v4-architecture.md)

V4 configuration:

- vocab: 490
- context: 256
- embedding: 256
- layers: 4
- heads: 8
- head dim: 32
- RoPE
- RMSNorm
- QKV attention + Q/K normalization + causal SDPA
- SwiGLU FFN
- tied LM head

Training recipe in [train_v4.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train_v4.py):

- shifted next-character objective
- balanced CE + label smoothing 0.05
- recovery loss weight 2.0
- AdamW
- lr 5e-4
- warmup 250
- cosine decay
- batch size 8
- gradient clipping 1.0
- checkpoint every 500 steps
- default steps 4500

Outputs:

- `experiments/v4-training/v4_config.json`
- `experiments/v4-training/tokenizer.json`
- `experiments/v4-training/v4-checkpoint-step-00500.pt` ...
- `experiments/v4-training/v4-final.pt`