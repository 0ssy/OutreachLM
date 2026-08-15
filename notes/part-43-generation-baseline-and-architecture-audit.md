# OutreachLM — Generation Baseline and Architecture Audit (Current Local State)

## Generation baseline at 10k steps
Observed quality profile:

- Character prediction: working
- Word formation: emerging
- Local syntax: emerging
- Sentence continuation: weak
- Long-range dependency handling: weak
- Semantic consistency: very weak
- Generalization: limited
- Stability: limited

Representative outputs:

```text
OutreachLM is ther, and as toping, the cames the cards flactions this the
The purpose of a transformer is alize and on and whing to all to sign the decommaning, blow
Machine learning allows.
And with a cludest of schore infrom arration.
He chan at w
A computer system cans, that and anying comployer state and ind stared in pulassi
```

Interpretation:
the model has learned short-context character statistics, but not robust semantic language modeling yet.

---

## Architecture audit snapshot (no code changes)
Audited files:
- [model.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/model.py)
- [transformer_block.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/transformer_block.py)
- [attention.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/attention.py)
- [feed_forward.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/feed_forward.py)
- [tokenizer.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/tokenizer.py)
- [datasets.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/datasets.py)
- [corpus.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/corpus.py)
- [train.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py)
- [generate.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/generate.py)

### Model structure
- Transformer blocks: `num_layers=1` default in [OutreachModel](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/model.py:49)
- Attention heads: `num_heads=4` default in [OutreachModel](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/model.py:49)
- Positional representation: learned positional embedding ([PositionalEmbedding](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/model.py:28))
- LayerNorm placement: pre-norm inside block (`norm -> sublayer -> residual`) in [TransformerBlock](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/transformer_block.py:8), plus final norm in [OutreachModel](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/model.py:103)
- Feed-forward: `Linear(D,4D) -> GELU -> Linear(4D,D)` in [FeedForward](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/feed_forward.py:5)
- Residual connections: present on attention and feed-forward branches in [TransformerBlock](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/transformer_block.py:48)

### Attention implementation
- Q/K/V + output linear projections per head-grouped attention in [CausalSelfAttention](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/attention.py:6)
- Causal mask: lower-triangular (`torch.tril`) applied before softmax in [CausalSelfAttention](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/attention.py:130)
- Causal masking logic appears correct for autoregressive training/generation.

### Tokenization + data path
- Tokenizer: character-level with special tokens `<PAD>`, `<UNK>` in [CharacterTokenizer](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/tokenizer.py:1)
- Dataset windows: next-token shifted fixed-length windows in [LanguageModelDataset](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/datasets.py:5)
- Training sampling: direct random windows from token ID tensor in [get_random_batch](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py:219)
- Corpus ingestion: recursive `*.txt` loader with UTF-8 and `errors="ignore"` in [Corpus](/C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/corpus.py:4)

### Parameter count (current default training shape)
For `vocab_size=490`, `context_length=32`, `embedding_dim=64`, `num_layers=1`, `num_heads=4`:
- total parameters: **115,370**

### Generation/training config consistency
- [generate.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/generate.py) now reconstructs tokenizer from the same train split logic (`CORPUS_PATH + VALIDATION_SPLIT`) to match vocabulary.
- Generation still depends on current constants/imported helpers, not an explicit saved run-config artifact.

### Checkpoint reproducibility status
- A dedicated checkpoint module exists in [checkpoint.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/checkpoint.py), including config build/load/validate helpers.
- Current [train.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py) still contains local `save_checkpoint/load_checkpoint` functions and does not yet enforce config compatibility via `validate_config(...)`.
- Therefore, full run-config compatibility enforcement is not fully integrated yet.
