# OutreachLM — Current Notes

## 1. Project direction
**OutreachLM** is being built from the ground up as a machine-learning system, not necessarily only as a language model.

The architecture should therefore remain general enough to support:

- Language modeling
- Sequence prediction
- Classification
- Regression
- Decision-making
- Other learned representations
- Potentially more experimental learning systems
The goal is to **understand and build the underlying machinery**, rather than simply import a finished Transformer.

---

# 2. Current project structure
Current important files:

```
OutreachLM/
│
├── outreachlm/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── attention.py
│   └── ...
│
└── ...
```

You kept the original model implementation as:

```
outreachlm/model.py
```

rather than creating the previously referenced `layers.py`.

That caused the earlier error:

```
ModuleNotFoundError: No module named 'outreachlm.layers'
```

So imports in `main.py` need to match the actual project structure.

---

# 3. Dataset
The first test dataset is deliberately tiny.

```
Total tokens: 20
Context length: 5
Dataset length: 15
```

The task is next-token prediction.

### Example 0

```
Input : [0, 1, 2, 3, 4]
Target: [1, 2, 3, 4, 5]
```

### Example 1

```
Input : [1, 2, 3, 4, 5]
Target: [2, 3, 4, 5, 6]
```

### Example 2

```
Input : [2, 3, 4, 5, 6]
Target: [3, 4, 5, 6, 7]
```

So the model is currently learning:

```
x_t → x_(t+1)
```

with a context window of 5 tokens.

This is intentionally simple so that each component can be inspected before introducing complexity.

---

# 4. Token embeddings
The embedding dimension is:

```
embedding_dim = 16
```

Given:

```
Input:
tensor([[1, 2, 3, 4, 5]])
```

the input shape is:

```
[1, 5]
```

where:

```
1 = batch size
5 = sequence/context length
```

The embedding layer transforms this into:

```
[1, 5, 16]
```

So:

```
Token IDs
   ↓
[batch, sequence]
   ↓
Embedding
   ↓
[batch, sequence, embedding_dim]
```

Specifically:

```
[1, 5]
    ↓
[1, 5, 16]
```

Each token now has a learned 16-dimensional representation.

The numerical values in the embedding output are initially essentially arbitrary learned parameters. They become meaningful through training.

---

# 5. Self-attention
The next major component is:

```
CausalSelfAttention
```

implemented in:

```
outreachlm/attention.py
```

The class currently contains four learned linear transformations:

```
self.query
self.key
self.value
self.output
```

Each maps:

```
16 → 16
```

---

# 6. Q, K and V
Given:

```
x.shape = [B, T, D]
```

the attention layer creates:

```
Q = query(x)
K = key(x)
V = value(x)
```

with shapes:

```
Q = [B, T, D]
K = [B, T, D]
V = [B, T, D]
```

For the current test:

```
B = 1
T = 5
D = 16
```

therefore:

```
Q = [1, 5, 16]
K = [1, 5, 16]
V = [1, 5, 16]
```

---

# 7. Attention scores
The attention score calculation is:

```
scores = torch.matmul(
    Q,
    K.transpose(-2, -1)
)
```

Conceptually:

```
Q @ Kᵀ
```

Shape:

```
[B, T, D] @ [B, D, T]
```

produces:

```
[B, T, T]
```

Therefore:

```
[1, 5, 16]
      ×
[1, 16, 5]
      =
[1, 5, 5]
```

The resulting 5×5 matrix tells us how strongly each position relates to every other position **before masking**.

---

# 8. Scaling
The scores are divided by:

```
math.sqrt(embedding_dim)
```

So:

```
scores = scores / √D
```

With:

```
D = 16
```

the scale factor is:

```
√16 = 4
```

This prevents the dot products from becoming excessively large and making softmax too extreme.

---

# 9. Causal masking
The attention mechanism was upgraded to genuinely autoregressive attention.

The important code is:

```
casual_mask = torch.tril(
    torch.ones(
        sequence_length,
        sequence_length,
        device=x.device
    )
)
```

This produces:

```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

The intended rule is:

> Position `t` can attend only to positions `≤ t`.
Therefore:

```
Token 0 → token 0

Token 1 → tokens 0,1

Token 2 → tokens 0,1,2

Token 3 → tokens 0,1,2,3

Token 4 → tokens 0,1,2,3,4
```

Future tokens are forbidden.

---

# 10. Masking implementation
Future positions are replaced with negative infinity:

```
scores = scores.masked_fill(
    casual_mask == 0,
    float("-inf")
)
```

This is important because the next operation is softmax.

For example:

```
[-∞, -∞, 2.0, 1.0, 0.5]
```

after softmax gives:

```
[0, 0, probability, probability, probability]
```

Thus future tokens receive exactly zero attention probability.

---

# 11. Softmax
The masked scores are converted into probabilities:

```
attention_weights = torch.softmax(
    scores,
    dim=-1
)
```

The last dimension corresponds to the tokens being attended to.

The resulting matrix from the test was approximately:

```
[
 [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],

 [0.4825, 0.5175, 0.0000, 0.0000, 0.0000],

 [0.4958, 0.2544, 0.2498, 0.0000, 0.0000],

 [0.2294, 0.2729, 0.2396, 0.2581, 0.0000],

 [0.2141, 0.1730, 0.1934, 0.1799, 0.2397]
]
```

Notice the structure.

For position 0:

```
[1, 0, 0, 0, 0]
```

For position 1:

```
[0.4825, 0.5175, 0, 0, 0]
```

For position 2:

```
[0.4958, 0.2544, 0.2498, 0, 0]
```

And so on.

Every row sums approximately to:

```
1
```

---

# 12. Attention output
The weighted values are calculated with:

```
attention_output = torch.matmul(
    attention_weights,
    V
)
```

Shapes:

```
attention_weights = [B, T, T]

V = [B, T, D]

result = [B, T, D]
```

Therefore:

```
[1, 5, 5]
     ×
[1, 5, 16]
     =
[1, 5, 16]
```

The attention mechanism therefore **preserves the sequence representation shape**.

---

# 13. Output projection
Finally:

```
output = self.output(
    attention_output
)
```

The projection maps:

```
16 → 16
```

Therefore the final attention output remains:

```
[1, 5, 16]
```

---

# 14. Current verification
The test successfully confirmed:

```
Input IDs       : [1, 5]

Embeddings      : [1, 5, 16]

Attention output: [1, 5, 16]
```

More importantly, the causal test produced:

```
Future-token attention values:

[
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0]
]
```

Therefore:

```
✓ Causal mask is working.
✓ No token is attending to a future token.
✓ Attention preserves the embedding shape.
```

This is a major milestone.

---

# 15. Current architecture
At this point the pipeline is approximately:

```
                 INPUT
                   │
                   ▼
             Token IDs
             [B, T]
                   │
                   ▼
            TokenEmbedding
                   │
                   ▼
             [B, T, D]
                   │
                   ▼
          CausalSelfAttention
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
          Q        K        V
          │        │        │
          └─── QKᵀ ─────────┘
                │
                ▼
             Scaling
                │
                ▼
          Causal Mask
                │
                ▼
             Softmax
                │
                ▼
       Attention Weights
                │
                ▼
           Weights × V
                │
                ▼
          Output Projection
                │
                ▼
             [B,T,D]
```

---

# 16. Important terminology correction
Your code currently calls it:

```
casual_mask
```

The correct term is:

```
causal_mask
```

because **causal** means the model respects the direction of cause/effect in the sequence by preventing information from the future from influencing the present.

You should eventually rename:

```
casual_mask
```

to:

```
causal_mask
```

It doesn't affect functionality, but it matters for clean engineering and terminology.

---

# 17. What has NOT been built yet
The current system is **not yet a complete language model**.

We currently have:

```
✓ Dataset
✓ Token IDs
✓ Token embeddings
✓ Q/K/V projections
✓ Scaled dot-product attention
✓ Causal masking
✓ Softmax attention
✓ Attention output projection
```

Still needed for the basic Transformer-style learning system:

```
[Next]

Position information
        ↓
Attention + residual connection
        ↓
Layer normalization
        ↓
Feed-forward network
        ↓
Second residual connection
        ↓
Layer normalization
        ↓
Transformer block
        ↓
Language/model output head
        ↓
Logits
        ↓
Loss
        ↓
Backpropagation
        ↓
Optimizer
        ↓
Training loop
        ↓
Evaluation
```

---

# 18. Important direction for OutreachLM
We should **not prematurely lock the project into "language model" architecture**.

The deeper objective is to understand the learning machine itself.

A better abstraction is:

```
             OUTREACHLM
                 │
        ┌────────┴────────┐
        │                 │
   Representation     Prediction
        │                 │
        └────────┬────────┘
                 │
              Learning
                 │
        ┌────────┼────────┐
        │        │        │
     Language  Vision   Signals
        │        │        │
        └────────┼────────┘
                 │
             Decision
                 │
             Action
```

That leaves room later for systems that aren't simply next-token predictors.

And when we reach the point where you're testing whether an assumption in the architecture can be violated, we'll distinguish between **mathematical/physical constraints, engineering assumptions, and merely conventional ML design choices**. We shouldn't treat every convention as a law of nature.

---

# 19. Current checkpoint
**OutreachLM checkpoint: Causal Attention**

```
Dataset                    ✓
Token embedding            ✓
Q/K/V projections          ✓
Attention scores           ✓
Scaling                    ✓
Causal mask                ✓
Softmax                    ✓
Weighted values            ✓
Output projection          ✓
Shape verification         ✓
Future-information test    ✓
```

### Next engineering step
**Add positional information.**

Right now, the embedding alone tells the model **what token** it is seeing, but not inherently **where that token occurs in the sequence**.

For example, the model needs a way to distinguish:

```
[1, 2, 3, 4, 5]
```

from:

```
[5, 4, 3, 2, 1]
```

when the token representations themselves are otherwise identical.

The next component should therefore be:

```
Token Embedding + Positional Information
                    ↓
              Causal Attention
```

Then we can move toward the complete Transformer block rather than jumping directly into training.
