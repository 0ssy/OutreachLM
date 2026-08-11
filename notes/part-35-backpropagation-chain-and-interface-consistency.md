# OUTREACHLM — BACKPROPAGATION + GRADIENT DESCENT
We’re continuing from **gradient descent/backpropagation**, and this time we will finish the entire chain instead of jumping around.

## Where we are
Your current OutreachLM has already established:

- token embeddings
- positional embeddings
- Q/K/V projections
- multi-head causal self-attention
- softmax attention
- residual connections
- layer normalization
- feed-forward network
- output projection
- cross-entropy training
- PyTorch autograd
- a successful gradient sanity check
Your gradient check also told us something important:

> PyTorch's autograd gradient exactly matched the analytical gradient.
The numerical check discrepancy is a finite-difference precision issue, **not evidence that your analytical backpropagation is wrong**.

And the tuple error:

```
TypeError: layer_norm(): argument 'input' (position 1) must be Tensor, not tuple
```

is an **architecture interface bug**, not a gradient-learning failure. Model components must agree on whether a transformer block returns a Tensor or `(Tensor, attention_weights)`.

---

# The path we're going to complete
We are going to learn and implement this as **one continuous system**:

```
FORWARD PASS
    ↓
LOSS
    ↓
GRADIENT DESCENT
    ↓
MANUAL BACKPROPAGATION
    ↓
LINEAR
    ↓
MLP
    ↓
LAYER NORM
    ↓
ATTENTION
    ↓
Q / K / V
    ↓
SOFTMAX
    ↓
EMBEDDINGS
    ↓
FULL TRANSFORMER
```

The key idea:

> **Backpropagation is just the chain rule applied backward through the exact operations used in the forward pass.**
Gradient descent then uses those gradients to change the parameters.

---

# 1. Gradient descent
Suppose our model has a parameter:

[
w
]

and the loss is:

[
L(w)
]

The gradient tells us:

[
\frac{\partial L}{\partial w}
]

That answers:

> "If I increase `w`, which direction does the loss move?"
Gradient descent changes the parameter in the **opposite direction**:

# [
w_{\text{new}}
w-\eta\frac{\partial L}{\partial w}
]

where (\eta) is the learning rate.

For example:

```
w = 3
gradient = -16
learning_rate = 0.001
```

Then:

```
w_new = 3 - 0.001(-16)
      = 3.016
```

The negative gradient caused `w` to increase.

---

# 2. Why this is the same thing your Transformer does
Your training loop is conceptually:

```
logits = model(inputs)

loss = loss_function(logits, targets)

loss.backward()

optimizer.step()
```

Those four lines represent an enormous mathematical process.

```
model(inputs)
      │
      ▼
forward pass
      │
      ▼
logits
      │
      ▼
loss
      │
      ▼
backward()
      │
      ▼
∂L/∂parameter
      │
      ▼
optimizer.step()
      │
      ▼
new parameters
```

---

# 3. Manual backpropagation through a linear layer
Consider:

[
y=wx+b
]

Suppose:

```
x = 2
w = 3
b = 1
```

Then:

[
y=3(2)+1=7
]

Suppose:

[
L=(y-10)^2
]

Then:

[
L=(7-10)^2=9
]

We first calculate:

# [
\frac{\partial L}{\partial y}
2(y-10)
]

so:

# [
\frac{\partial L}{\partial y}
-6
]

Now:

[
y=wx+b
]

therefore:

[
\frac{\partial y}{\partial w}=x
]

and:

[
\frac{\partial y}{\partial b}=1
]

Thus:

# [
\frac{\partial L}{\partial w}
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial w}
]

[
=-6(2)
]

[
=-12
]

and:

[
\frac{\partial L}{\partial b}=-6
]

---

# 4. Matrix linear layers
In OutreachLM we normally have:

[
Y=XW^T+b
]

For a linear layer:

```
linear = nn.Linear(input_dim, output_dim)
```

important gradients are:

[
\frac{\partial L}{\partial W}
]

[
\frac{\partial L}{\partial b}
]

and:

[
\frac{\partial L}{\partial X}
]

Backward flow:

```
dL/dY
  │
  ├──────────────► dL/dW
  │
  ├──────────────► dL/db
  │
  ▼
dL/dX
```

---

# 5. MLP backpropagation
Feed-forward network:

```
X
│
▼
Linear 1
│
▼
activation
│
▼
Linear 2
│
▼
output
```

For example:

[
H=XW_1^T+b_1
]

then:

[
A=\operatorname{GELU}(H)
]

then:

[
Y=AW_2^T+b_2
]

Backward:

```
dL/dY
   │
   ▼
Linear 2
   │
   ▼
dL/dA
   │
   ▼
GELU
   │
   ▼
dL/dH
   │
   ▼
Linear 1
   │
   ▼
dL/dX
```

---

# 6. LayerNorm
LayerNorm in the Transformer:

```
self.norm1
self.norm2
```

Forward:

[
\mu=\frac1D\sum_i x_i
]

[
\sigma^2=
\frac1D\sum_i(x_i-\mu)^2
]

# [
\hat{x}
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
]

[
y=\gamma\hat{x}+\beta
]

Backward must flow through:

```
y
↓
γ and β
↓
normalized x
↓
variance
↓
mean
↓
original x
```

---

# 7. Attention
Forward:

[
Q=XW_Q
]

[
K=XW_K
]

[
V=XW_V
]

Then:

[
S=\frac{QK^T}{\sqrt{d_k}}
]

Then causal masking:

[
S_{ij}=-\infty
]

Then:

[
A=\operatorname{softmax}(S)
]

Then:

[
O=AV
]

Then output projection:

[
Y=OW_O
]

Backward reverses the chain.

---

# 8. The crucial thing about softmax
Softmax:

[
p_i=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
]

Outputs are **coupled**.

Jacobian:

# [
\frac{\partial p_i}{\partial z_j}
p_i(\delta_{ij}-p_j)
]

---

# 9. Embeddings
Embedding matrix:

[
E\in\mathbb{R}^{V\times D}
]

Forward selects rows for token IDs.

Repeated tokens accumulate gradient contributions.

---

# 10. Full Transformer backward pass
Backward chain:

```
LOSS
 ↓
CROSS ENTROPY
 ↓
LM HEAD
 ↓
TRANSFORMER
 ↓
MLP
 ↓
RESIDUAL
 ↓
LAYER NORM
 ↓
ATTENTION
 ↓
OUTPUT PROJECTION
 ↓
AV
 ↓
SOFTMAX
 ↓
QKᵀ
 ↓
Q / K / V
 ↓
WQ / WK / WV
 ↓
INPUT REPRESENTATION
 ↓
TOKEN + POSITION EMBEDDINGS
```

---

# Interface consistency fix summary
Keep a clean API:

```
attention
    ↓
(output, weights)

transformer block
    ↓
(output, weights)

model
    ↓
(logits, attention information)
```

And always unpack explicitly in loops:

```
x, attention_weights = block(x)
```

not:

```
x = block(x)
```

---

# Current scale guidance
One transformer block is appropriate right now:

```
1 block
1 head group
16 dimensions
20-token vocabulary
CPU
```

Scale later, after mechanism mastery.

---

# Educational lab roadmap
Target structure:

```
outreachlm/
│
├── attention.py
├── feed_forward.py
├── transformer_block.py
├── model.py
│
├── manual_linear.py
├── manual_mlp.py
├── manual_layernorm.py
├── manual_softmax.py
├── manual_attention.py
├── manual_embedding.py
│
├── gradient_checks.py
│
└── transition_experiment.py
```

For each component:

```
FORWARD
   ↓
ANALYTICAL BACKWARD
   ↓
PYTORCH AUTOGRAD
   ↓
NUMERICAL CHECK
   ↓
COMPARE
```
