more notes 
Exactly. From here on, **we're not just adding code—we're learning what we're building while we build it.** I’ll explain the mathematical/ML idea first, then the implementation, then give you a test, and only then move to the next component.

We currently have:

```
Token IDs
   ↓
Token Embedding
   ↓
Causal Self-Attention
   ↓
Residual + LayerNorm
   ↓
Feed-Forward Network
   ↓
Residual + LayerNorm
   ↓
Transformer Block
```

The next thing we need is **positional information**.

---

# Step 4 — Teach the model where tokens are
There's an important problem with our current model.

Suppose we have:

```
[1, 2, 3, 4, 5]
```

The embedding layer knows what token `1` represents.

But it doesn't inherently know that this particular `1` is:

```
position 0
```

Compare:

```
[1, 2, 3, 4, 5]
```

with:

```
[5, 4, 3, 2, 1]
```

The same token IDs occur in both sequences.

Without positional information, the model has a problem distinguishing **token identity** from **token position**.

This matters enormously for sequence learning.

---

## Think about language
Consider:

```
"The dog chased the cat."
```

versus:

```
"The cat chased the dog."
```

The words are almost identical.

But their arrangement changes the meaning.

Therefore the model needs two pieces of information:

```
What is this token?
        +
Where is this token?
```

We already have the first:

```
Token embedding
```

Now we need the second:

```
Positional embedding
```

---

# 1. The basic idea
For every position, we'll have another vector.

Our current embedding dimension is:

```
16
```

and our context length is:

```
5
```

So we'll create:

```
5 × 16
```

positional representations.

Conceptually:

```
Position 0 → [16 numbers]
Position 1 → [16 numbers]
Position 2 → [16 numbers]
Position 3 → [16 numbers]
Position 4 → [16 numbers]
```

Then we combine token and position information:

```
final_embedding = token_embedding + positional_embedding
```

So:

```
Token representation
        +
Position representation
        ↓
Position-aware representation
```

---

# 2. Why addition?
This is an important design decision.

Suppose:

```
token_embedding = [a₁, a₂, ..., a₁₆]

position_embedding = [b₁, b₂, ..., b₁₆]
```

We calculate:

```
[a₁+b₁, a₂+b₂, ..., a₁₆+b₁₆]
```

The dimensionality remains 16.

That's useful because everything downstream still expects:

```
[B, T, 16]
```

rather than suddenly becoming:

```
[B, T, 32]
```

---

# 3. What type should we use?
There are two major approaches:

### Learned positional embeddings
The model learns a vector for each position.

For example:

```
position 0 → learned vector
position 1 → learned vector
position 2 → learned vector
...
```

This is simple and a good choice for our first implementation.

Later, we can investigate alternatives such as:

- sinusoidal positional encoding
- RoPE
- ALiBi
- relative positional representations
But **don't jump ahead yet**.

We're building the fundamental machinery ourselves first.

---

# 4. Create `positional_embedding.py`
Create:

```
id="6bce8a"
outreachlm/positional_embedding.py
```

Put this inside:

```
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        context_length,
        embedding_dim
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, x):

        # x shape:
        # (batch_size, sequence_length, embedding_dim)

        batch_size, sequence_length, embedding_dim = x.shape

        # Position IDs:
        # [0, 1, 2, ..., sequence_length - 1]

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Convert positions into learned vectors

        position_vectors = self.position_embedding(
            positions
        )

        # position_vectors shape:
        # (sequence_length, embedding_dim)

        # Add a batch dimension so broadcasting works

        position_vectors = position_vectors.unsqueeze(0)

        # Shape:
        # (1, sequence_length, embedding_dim)

        return x + position_vectors
```

---

# 5. Let's understand every part
This:

```
self.position_embedding = nn.Embedding(
    context_length,
    embedding_dim
)
```

creates a learnable table.

With:

```
context_length = 5
embedding_dim = 16
```

the table is effectively:

```
5 × 16
```

Something conceptually like:

```
position 0 → [ ...16 values... ]
position 1 → [ ...16 values... ]
position 2 → [ ...16 values... ]
position 3 → [ ...16 values... ]
position 4 → [ ...16 values... ]
```

These values start randomly.

**Training will eventually modify them.**

That's why they're called **learned positional embeddings**.

---

# 6. Creating positions
This:

```
positions = torch.arange(
    sequence_length,
    device=x.device
)
```

with:

```
sequence_length = 5
```

produces:

```
tensor([0, 1, 2, 3, 4])
```

Those are not token IDs.

They're **position IDs**.

That's an important distinction:

```
Token IDs:
[1, 2, 3, 4, 5]

Position IDs:
[0, 1, 2, 3, 4]
```

---

# 7. Position vectors
We then do:

```
position_vectors = self.position_embedding(
    positions
)
```

which gives:

```
[5, 16]
```

Then:

```
position_vectors = position_vectors.unsqueeze(0)
```

changes:

```
[5, 16]
```

into:

```
[1, 5, 16]
```

Now it can be added to:

```
x = [batch_size, 5, 16]
```

through PyTorch broadcasting.

---

# 8. Important concept: broadcasting
Suppose:

```
x
shape = [1, 5, 16]

position_vectors
shape = [1, 5, 16]
```

Then:

```
x + position_vectors
```

is straightforward.

But even with:

```
x                [B, 5, 16]
position_vectors [1, 5, 16]
```

PyTorch can broadcast the positional vectors across every batch element.

So if:

```
B = 32
```

we don't need to create 32 separate positional tables.

One table can be reused:

```
[1, 5, 16]
      ↓ broadcasting
[32, 5, 16]
```

---

# 9. Test it before integrating it
Modify `main.py` temporarily:

```
import torch

from outreachlm.model import TokenEmbedding
from outreachlm.positional_embedding import PositionalEmbedding

def main():

    print("=" * 50)
    print("POSITIONAL EMBEDDING TEST")
    print("=" * 50)

    vocab_size = 20
    context_length = 5
    embedding_dim = 16

    token_embedding = TokenEmbedding(
        vocab_size,
        embedding_dim
    )

    positional_embedding = PositionalEmbedding(
        context_length,
        embedding_dim
    )

    input_ids = torch.tensor([
        [1, 2, 3, 4, 5]
    ])

    embeddings = token_embedding(
        input_ids
    )

    output = positional_embedding(
        embeddings
    )

    print("\nInput IDs:")
    print(input_ids)

    print("\nToken embedding shape:")
    print(embeddings.shape)

    print("\nPosition-aware embedding shape:")
    print(output.shape)

    print("\nPosition embeddings:")
    print(
        positional_embedding.position_embedding.weight
    )

    assert output.shape == (
        1,
        context_length,
        embedding_dim
    )

    print("\n✓ Positional embeddings added.")
    print("✓ Sequence shape preserved.")
    print("✓ Position vectors are learnable.")

if __name__ == "__main__":
    main()
```

Run:

```
python -m outreachlm.main
```

You should see:

```
Token embedding shape:
torch.Size([1, 5, 16])

Position-aware embedding shape:
torch.Size([1, 5, 16])

✓ Positional embeddings added.
✓ Sequence shape preserved.
✓ Position vectors are learnable.
```

---

# 10. The experiment I want you to understand
Before we move on, we're going to test **why positional embeddings actually matter**.

We'll take the same token:

```
token = 1
```

and put it at two different positions:

```
[1, 2, 3, 4, 5]
 ^
 position 0
```

and:

```
[2, 3, 4, 5, 1]
             ^
             position 4
```

The token embedding for `1` should be the same.

But its **position-aware representation should be different**.

That's exactly what we want.

The model should learn:

```
"what am I?"
+
"where am I?"
```

rather than only:

```
"what am I?"
```

---

### Your task now
Create `positional_embedding.py`, run the test, and paste the output.

**Then we'll perform that position experiment before integrating it into the Transformer block.**

That way you're learning *why* the architecture works, rather than just accumulating files.
