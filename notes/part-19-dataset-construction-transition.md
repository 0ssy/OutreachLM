# OutreachLM — Dataset Construction Transition Notes

Yes. **This is working.** Your current result tells us something important:

- FineWeb was downloaded into your corpus.
- You have ~1000+ text files.
- Your tokenizer can encode the corpus into IDs.
- Your decoder reconstructs readable text.
- BPE is producing multi-character tokens such as IDs in the 400s.
- The decoded text is imperfect, but that is expected at this stage because the corpus itself contains noisy/web-extracted text.
  We are now **past tokenizer construction**.

## Where we are in OutreachLM
The progression is:

1. ~~Character vocabulary~~
2. ~~Pair counting~~
3. ~~BPE merge learning~~
4. ~~Merge ranks~~
5. ~~BPE encoding~~
6. ~~Token → ID~~
7. ~~ID → token decoding~~
8. ~~Corpus loader~~
9. ~~FineWeb ingestion~~
10. ~~Corpus statistics~~
11. **Real corpus tokenization ← WE ARE HERE**
12. **Dataset/chunk construction**
13. Train/validation split
14. Batch generation
15. Embeddings
16. Positional information
17. Self-attention
18. Multi-head attention
19. Transformer block
20. Causal masking
21. GPT-style model
22. Loss
23. Backpropagation
24. Optimizer
25. Training loop
26. Checkpointing
27. Text generation
28. Sampling
29. Evaluation
30. Scaling the model
So **do not go back to BPE**. We are moving from the tokenizer into the **actual training data pipeline**.

---

# NEXT: Turn token IDs into training sequences
Right now your tokenizer gives us something conceptually like:

```
document
   ↓
text
   ↓
BPE tokenizer
   ↓
token IDs
   ↓
[1, 2, 3, 432, 7, 446, ...]
```

But a transformer cannot simply consume an arbitrarily long document.

We need to turn the enormous stream of IDs into fixed-length training examples.

For example, suppose:

```
token IDs:

[10, 25, 83, 91, 7, 44, 62, 18, 99, 4, 12, 77, ...]
```

and our context length is:

```
8
```

We create:

```
[10, 25, 83, 91, 7, 44, 62, 18]
[25, 83, 91, 7, 44, 62, 18, 99]
[83, 91, 7, 44, 62, 18, 99, 4]
...
```

But for language-model training, we actually need **input and target sequences**.

For example:

```
tokens:
[10, 25, 83, 91, 7, 44, 62, 18, 99]
```

becomes:

```
INPUT:
[10, 25, 83, 91, 7, 44, 62, 18]

TARGET:
[25, 83, 91, 7, 44, 62, 18, 99]
```

Notice what happened.

The model is learning:

```
10 → 25
25 → 83
83 → 91
91 → 7
...
```

This is the fundamental training objective behind our language model.

### The important idea
The model doesn't receive:

> "Predict the next word."
> It receives **token IDs** and learns to predict the **next token ID**.

So our pipeline becomes:

```
FineWeb
   ↓
documents
   ↓
BPE tokenizer
   ↓
token IDs
   ↓
continuous token stream
   ↓
fixed-length sequences
   ↓
X / Y pairs
   ↓
Transformer
```

This is the next component we're building.

---

# Your next implementation
Create:

```
outreachlm/
    __init__.py
    main.py
    version.py
    corpus_loader.py
    dataset.py       ← NEW
```

`dataset.py` will eventually handle:

```
tokens → training examples
```

For now, **don't touch the transformer yet**.

We're going to build the dataset pipeline properly first, test it against your 1000+ documents, and inspect the resulting sequences.

### The first thing we'll implement
A simple sequence builder:

```
class LanguageModelDataset:
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):
        x = self.token_ids[index:index + self.context_length]
        y = self.token_ids[index + 1:index + self.context_length + 1]

        return x, y
```

Don't worry about memorizing that yet. **I'll teach exactly why every line exists before we build the production version.**

### Our immediate milestone
We want to get output like:

```
==================================================
DATASET TEST
==================================================

Total tokens: 184732
Context length: 128
Training examples: 184604

Example 1
Input IDs:
[...]

Target IDs:
[...]

Input:
"The quick brown fox..."

Target:
"he quick brown fox..."
```

Once that works, **then we move to batching**.

And after batching, we finally have the data entering the territory where the neural network itself begins.

---

## BPE notes — continuation
Add these to the notes you already have:

### 12. BPE is now integrated with the real corpus
Our tokenizer is no longer being tested only on:

```
low
lower
lowest
```

It has now processed real web-derived text.

The pipeline is:

```
Raw document
    ↓
preprocessing
    ↓
BPE encoding
    ↓
token IDs
```

### 13. Token IDs are the interface between tokenizer and model
The neural network does **not** understand:

```
"However it is not simply..."
```

directly.

It receives:

```
[1, 2, 3, 432, 7, ...]
```

The vocabulary provides the mapping:

```
token ↔ integer ID
```

This means the tokenizer is effectively the **boundary between human-readable text and numerical model input**.

### 14. Decoding verifies reversibility
We successfully performed:

```
text
 ↓
tokens
 ↓
IDs
 ↓
tokens
 ↓
text
```

and obtained readable reconstructed text.

That confirms the tokenizer's basic encode/decode pipeline is functioning.

### 15. Real corpora expose tokenizer behavior
The FineWeb output contains things such as:

```
unusual spacing
HTML-derived artifacts
encoding artifacts
URLs
punctuation
noisy text
```

Therefore, seeing imperfect-looking decoded text **does not automatically mean the tokenizer is broken**.

We distinguish:

```
Tokenizer error
```

from:

```
Corpus noise
```

That distinction becomes important as OutreachLM grows.

### 16. Current architectural boundary
We have finished the first major subsystem:

```
                 OUTREACHLM

        ┌───────────────────────┐
        │      CORPUS           │
        │ FineWeb / documents   │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │      TOKENIZER        │
        │         BPE           │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │       TOKEN IDs       │
        └───────────┬───────────┘
                    ↓
              ** WE ARE HERE **
                    ↓
        ┌───────────────────────┐
        │   TRAINING DATASET    │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │       BATCHES         │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │     TRANSFORMER       │
        └───────────────────────┘
```

**Next lesson: Dataset construction — token stream → `(X, Y)` training pairs.**
