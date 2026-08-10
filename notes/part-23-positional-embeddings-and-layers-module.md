Perfect. **That output is correct.** We now have the token embedding layer working.

And keeping `model.py` is fine. We don't need to rename anything.

Your result proves three things:

```
Dataset
   ↓
Token IDs
   ↓
TokenEmbedding
   ↓
Vectors
```

Specifically:

```
Input:
tensor([[1, 2, 3, 4, 5]])
```

became:

```
torch.Size([1, 5, 16])
```

That means:

- `1` = batch size
- `5` = sequence/context length
- `16` = embedding dimension
So token `1` became a 16-number vector, token `2` became another 16-number vector, etc.

The `grad_fn=<EmbeddingBackward0>` is also important: **PyTorch is tracking this operation for backpropagation.** We haven't trained anything yet, but the embedding is already a trainable parameter.

---

# NEXT — Positional Embeddings
Now we solve the problem we identified.

The token embedding knows:

```
token A
token B
token C
```

but it doesn't inherently know:

```
A came first
B came second
C came third
```

Consider:

```
"The dog chased the cat"
```

versus:

```
"The cat chased the dog"
```

The same token vocabulary is involved, but the order changes the meaning.

So we need to inject **position information**.

## 1. The idea
We'll create another embedding table:

```
Position 0 → vector
Position 1 → vector
Position 2 → vector
Position 3 → vector
Position 4 → vector
```

If:

```
context_length = 5
embedding_dim = 16
```

then our positional embedding table is:

```
5 × 16
```

Now the model receives:

```
token representation
        +
position representation
        ↓
combined representation
```

For your sequence:

```
[1, 2, 3, 4, 5]
```

we get:

```
Token 1 + Position 0
Token 2 + Position 1
Token 3 + Position 2
Token 4 + Position 3
Token 5 + Position 4
```

---

# 2. Implement it
Since you kept `model.py`, let's put the next component there.

Your `model.py` should now become:

```
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)

class PositionalEmbedding(nn.Module):

    def __init__(self, context_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, positions):
        return self.embedding(positions)
```

Notice something important.

We are using the **same basic mechanism** as token embeddings.

But the lookup key is different.

Token embedding:

```
token ID → vector
```

Positional embedding:

```
position ID → vector
```

---

# 3. Test it
Add this to the bottom of `main.py` after your current embedding test:

```
from outreachlm.model import TokenEmbedding, PositionalEmbedding
import torch

vocab_size = 100
context_length = 5
embedding_dim = 16

token_embedding = TokenEmbedding(
    vocab_size,
    embedding_dim
)

position_embedding = PositionalEmbedding(
    context_length,
    embedding_dim
)

token_ids = torch.tensor([
    [1, 2, 3, 4, 5]
])

positions = torch.arange(
    context_length
)

token_vectors = token_embedding(token_ids)

position_vectors = position_embedding(positions)

print("\n==================================================")
print("POSITIONAL EMBEDDING TEST")
print("==================================================")

print("Token IDs:")
print(token_ids)

print("\nPositions:")
print(positions)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nPosition embedding shape:")
print(position_vectors.shape)
```

Run:

```
python -m outreachlm.main
```

You should get approximately:

```
==================================================
POSITIONAL EMBEDDING TEST
==================================================

Token IDs:
tensor([[1, 2, 3, 4, 5]])

Positions:
tensor([0, 1, 2, 3, 4])

Token embedding shape:
torch.Size([1, 5, 16])

Position embedding shape:
torch.Size([5, 16])
```

The shapes are slightly different.

Why?

Token embeddings have a **batch dimension**:

```
[1, 5, 16]
```

Positions don't need one yet:

```
[5, 16]
```

PyTorch will broadcast the position vectors across the batch.

---

# 4. Combine them
Now add:

```
combined = token_vectors + position_vectors
```

Then:

```
print("\nCombined shape:")
print(combined.shape)
```

Expected:

```
Combined shape:
torch.Size([1, 5, 16])
```

This is the representation we'll eventually feed into attention.

Conceptually:

```
Token embeddings
       │
       │
       ├──────────────┐
       │              │
       │        Position embeddings
       │              │
       └───────┬──────┘
               ↓
             ADD
               ↓
      Transformer input
```

---

## One subtle point
We're using **learned positional embeddings** here.

There are other approaches, such as sinusoidal positional encodings and rotary positional embeddings (RoPE).

We're starting with learned positional embeddings because they make the underlying mechanism extremely clear.

Later, when we get to modern Transformer architectures, we'll discuss why positional representation becomes more sophisticated.

---

# Your notes
Add this after the previous notes:

```
==================================================
OUTREACHLM NOTES — POSITIONAL EMBEDDINGS
==================================================

WHY POSITION INFORMATION IS REQUIRED
--------------------------------------------------
Token embeddings represent token identity.

They do not inherently represent sequence position.

Therefore:

"The dog chased the cat"

and

"The cat chased the dog"

must receive different positional representations.

POSITIONAL EMBEDDING
--------------------------------------------------
A positional embedding assigns a learnable vector
to each position in the context window.

Example:

Position 0 → vector
Position 1 → vector
Position 2 → vector
Position 3 → vector
Position 4 → vector

POSITIONAL EMBEDDING MATRIX
--------------------------------------------------
If:

context_length = 5
embedding_dim = 16

then:

position embedding matrix = 5 × 16

COMBINING TOKEN + POSITION
--------------------------------------------------
For every token:

combined representation =
token embedding + positional embedding

Example:

Token 1 at position 0:

token_embedding(1) + position_embedding(0)

Token 2 at position 1:

token_embedding(2) + position_embedding(1)

SHAPES
--------------------------------------------------

Token IDs:

(batch, sequence_length)

Example:

(1, 5)

Token embeddings:

(batch, sequence_length, embedding_dim)

Example:

(1, 5, 16)

Position embeddings:

(sequence_length, embedding_dim)

Example:

(5, 16)

Combined representation:

(batch, sequence_length, embedding_dim)

Example:

(1, 5, 16)

CURRENT PIPELINE
--------------------------------------------------

Raw Text
   ↓
Corpus Loader
   ↓
BPE Tokenizer
   ↓
Token IDs
   ↓
Dataset
   ↓
Token Embedding
   ↓
Positional Embedding
   ↓
Combined Transformer Input
   ↓
[NEXT] Self-Attention
```

**Once the combined tensor prints as `torch.Size([1, 5, 16])`, the next concept is self-attention.** That's the point where OutreachLM starts learning relationships between tokens rather than merely representing them.

Yep — this is just a **file/module mismatch**, not a problem with the embedding concept.

Your `main.py` currently has:

```
from outreachlm.layers import TokenEmbedding
```

Python is looking for:

```
OutreachLM/
└── outreachlm/
    ├── __init__.py
    ├── main.py
    └── layers.py   ← missing
```

But in the previous step I told you to create `model.py`. So your project probably has:

```
outreachlm/
├── main.py
└── model.py
```

### Fix it
If your file is actually called `model.py`, change line 3 of `main.py` to:

```
from outreachlm.model import TokenEmbedding
```

Then run:

```
python -m outreachlm.main
```

### Or, better for the architecture we're building
Let's use `layers.py` because we're going to accumulate individual neural-network components there:

```
outreachlm/
├── __init__.py
├── main.py
├── tokenizer.py
├── corpus_loader.py
├── dataset.py
└── layers.py          ← neural-network layers
```

Create:

```
outreachlm/layers.py
```

with:

```
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)
```

Then your `main.py` import is correct:

```
from outreachlm.layers import TokenEmbedding
```

And test it with something like:

```
import torch

from outreachlm.layers import TokenEmbedding

vocab_size = 100
embedding_dim = 16

model = TokenEmbedding(
    vocab_size,
    embedding_dim
)

token_ids = torch.tensor([
    [1, 2, 3, 4, 5]
])

output = model(token_ids)

print("Input:")
print(token_ids)

print("\nOutput shape:")
print(output.shape)

print("\nEmbedding output:")
print(output)
```

Expected:

```
Input:
tensor([[1, 2, 3, 4, 5]])

Output shape:
torch.Size([1, 5, 16])
```

**Use `layers.py` going forward.** It will keep our architecture clean:

```
layers.py
    ↓
TokenEmbedding
    ↓
PositionalEmbedding
    ↓
SelfAttention
    ↓
MultiHeadAttention
    ↓
FeedForward
    ↓
TransformerBlock
```

Once you get the `torch.Size([1, 5, 16])` result, we're genuinely ready for **positional embeddings**.
