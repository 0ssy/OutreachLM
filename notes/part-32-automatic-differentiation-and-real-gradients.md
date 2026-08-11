# OutreachLM — Lesson 10: Automatic Differentiation & Real Gradients
We've covered **what backpropagation is**. Now we're going one level deeper: **how PyTorch actually computes it**.

This is important because from here onward, you should be able to inspect OutreachLM's learning process rather than treating PyTorch as a black box.

---

## 1. The computational graph
When you write:

```
x = torch.tensor(2.0, requires_grad=True)

y = x * 3

loss = y ** 2
```

PyTorch doesn't just calculate numbers.

It also records the operations:

```
x
│
├── × 3
│
▼
y
│
├── squared
│
▼
loss
```

This is called the **computation graph**.

PyTorch uses this graph to work backward from `loss`.

---

## 2. `requires_grad=True`
This tells PyTorch:

> Track operations involving this tensor because I may need its gradient.
Example:

```
x = torch.tensor(
    2.0,
    requires_grad=True
)
```

Now:

```
y = x * 3
```

and:

```
loss = y ** 2
```

PyTorch remembers the operations.

---

## 3. Calling `.backward()`
Now:

```
loss.backward()
```

PyTorch calculates:

[
\frac{\partial loss}{\partial x}
]

Let's calculate it ourselves.

We have:

[
y=3x
]

and:

[
loss=y^2
]

Therefore:

[
loss=(3x)^2
]

[
loss=9x^2
]

So:

[
\frac{dloss}{dx}=18x
]

At:

[
x=2
]

we get:

[
\frac{dloss}{dx}=36
]

Therefore:

```
print(x.grad)
```

gives approximately:

```
tensor(36.)
```

That's backpropagation happening automatically.

---

# 4. The critical distinction: `.grad`
After:

```
loss.backward()
```

the gradient is stored in:

```
x.grad
```

For a model parameter:

```
model.some_layer.weight.grad
```

contains the gradient of the loss with respect to that weight.

Mathematically:

[
\frac{\partial L}{\partial W}
]

---

# 5. Let's connect this directly to OutreachLM
Your model contains parameters such as:

```
TokenEmbedding
      ↓
PositionEmbedding
      ↓
Attention Q
Attention K
Attention V
      ↓
Attention output
      ↓
Feed Forward
      ↓
Output Head
```

Each contains learnable parameters.

After:

```
loss.backward()
```

we can inspect them.

For example:

```
print(
    model.token_embedding.embedding.weight.grad
)
```

You should get a tensor shaped:

```
[20, 16]
```

because your embedding matrix is:

```
20 tokens × 16 dimensions
```

---

# 6. Not every parameter receives the same gradient
This is important.

Suppose your current input is:

```
[0, 1, 2, 3, 4]
```

The model uses the embeddings for those tokens.

Therefore, those embeddings participate directly in the computation.

Other token embeddings may receive little or no gradient for that particular operation, depending on the architecture and computation.

So we can inspect:

```
grad = model.token_embedding.embedding.weight.grad

for token_id in range(20):

    magnitude = grad[token_id].norm().item()

    print(
        f"Token {token_id}: "
        f"gradient magnitude = {magnitude:.6f}"
    )
```

Now we're not merely looking at the embedding.

We're looking at **how training is trying to change it**.

---

# 7. Gradient magnitude
A gradient is a vector/tensor, not just one number.

We often summarize it using its norm.

For a vector:

[
|g|
]

This gives us a measure of gradient magnitude.

Conceptually:

```
small gradient
     │
     ▼
parameter barely changes

large gradient
     │
     ▼
parameter changes more strongly
```

This becomes extremely important later when we study:

- exploding gradients
- vanishing gradients
- gradient clipping
- deep transformer networks

---

# 8. The complete training step
Now let's break down the actual code you have been using.

```
optimizer.zero_grad()
```

### Step 1 — Clear old gradients
PyTorch accumulates gradients.

We don't want yesterday's gradient mixed with today's gradient.

---
Then:

```
logits = model(input_ids)
```

### Step 2 — Forward pass
The model processes the input.

```
input
 ↓
embedding
 ↓
position
 ↓
attention
 ↓
feed-forward
 ↓
output head
 ↓
logits
```

---
Then:

```
loss = criterion(
    logits,
    targets
)
```

### Step 3 — Calculate loss
Now we know how wrong the prediction was.

---
Then:

```
loss.backward()
```

### Step 4 — Backpropagation
PyTorch walks backward through the computation graph.

```
LOSS
 ↓
OUTPUT HEAD
 ↓
TRANSFORMER
 ↓
ATTENTION
 ↓
EMBEDDINGS
```

and calculates:

[
\frac{\partial L}{\partial \theta}
]

for every trainable parameter (\theta).

---
Then:

```
optimizer.step()
```

### Step 5 — Update parameters
The optimizer uses those gradients.

Conceptually:

# [
\theta_{new}

## \theta_{old}
\eta\nabla_\theta L
]

Adam modifies this basic idea with running estimates of gradient moments, but the underlying principle remains gradient-based optimization.

---

# 9. What `loss.backward()` does NOT do
This is worth remembering.

It does **not**:

```
❌ generate knowledge
❌ modify weights directly
❌ choose the learning rate
❌ perform the optimizer update
```

It calculates gradients.

The optimizer performs the update.

So:

```
loss.backward()
```

means:

> Calculate how the parameters should change.
While:

```
optimizer.step()
```

means:

> Actually change the parameters according to the optimizer's update rule.

---

# 10. Why the optimizer needs gradients
Imagine a parameter:

```
W = 0.8
```

and the gradient:

```
dL/dW = -0.4
```

With a simplified learning rate:

```
η = 0.1
```

the update is:

# [
W_{new}
0.8-(0.1)(-0.4)
]

[
W_{new}=0.84
]

So the parameter increases.

If instead:

```
dL/dW = +0.4
```

then:

[
W_{new}=0.8-(0.1)(0.4)
]

[
W_{new}=0.76
]

So the parameter decreases.

**The sign of the gradient matters.**

---

# 11. Why this matters for OutreachLM
You previously saw:

```
Training Loss:
3.39
↓
1.57
↓
0.28
↓
0.059
↓
0.002
```

You now know what was happening underneath.

Something approximately like:

```
Random parameters
       ↓
Forward pass
       ↓
Wrong prediction
       ↓
Large loss
       ↓
Backward pass
       ↓
Large gradients
       ↓
Parameter updates
       ↓
Better prediction
       ↓
Smaller loss
       ↓
Repeat
```

Eventually:

```
loss ≈ 0.002
```

because the model has adjusted its parameters to fit those training examples.

---

# 12. But here is the problem we discovered
Your transition experiment produced:

```
0 → 1 ✓
1 → 2 ✓
...
17 → 18 ✓
18 → 19 ✗
```

Training accuracy:

```
100%
```

Held-out transition:

```
18 → 19
```

failed.

This is one of the most important results we've obtained so far.

The model had learned:

> "Produce the correct outputs for the examples I was trained on."
It had **not demonstrated that it had learned the abstract operation**:

[
f(x)=x+1
]

That's the difference between:

### Memorization
and

### Generalization

---

# 13. Why this happened
Your model's output head is essentially learning a mapping between discrete token IDs.

It can learn:

```
0 → 1
1 → 2
2 → 3
...
17 → 18
```

without necessarily discovering the mathematical rule:

[
x+1
]

The fact that:

```
18 → 3
```

was produced demonstrates that the network did not possess a reliable mechanism for extrapolating beyond the training distribution.

This is exactly why we don't want to spend the rest of the project doing endless experiments on:

```
0 → 1
1 → 2
...
```

We've learned what that experiment was designed to teach us.

**Now we move upward.**

---

# 14. A useful gradient experiment
We should now make one small engineering tool—not another meaningless learning experiment.

Create:

```
outreachlm/
    gradient_analysis.py
```

Its job will be to answer:

> **What is actually happening to OutreachLM's gradients during one training step?**
The experiment should report things like:

```
============================================================
GRADIENT ANALYSIS
============================================================

Parameter                              Shape       Grad Norm
------------------------------------------------------------
token_embedding.embedding.weight       [20,16]      ...
position_embedding.embedding.weight    [5,16]       ...
attention.query.weight                  [16,16]      ...
attention.key.weight                    [16,16]      ...
attention.value.weight                  [16,16]      ...
attention.output.weight                [16,16]      ...
feed_forward...                         ...          ...
output_head.weight                      [20,16]      ...

Total parameters: ...
Parameters with gradients: ...

Maximum gradient: ...
Minimum non-zero gradient: ...
```

That will let you **see backpropagation inside your actual model**.

After that, the next major lesson is:

# Gradient Flow
We'll examine how the gradient travels through:

```
Output Head
     ↓
Transformer
     ↓
Feed Forward
     ↓
Attention
     ↓
Embedding
```

and then we'll study why deeper neural networks can suffer from **vanishing and exploding gradients**.

That is the bridge between the small OutreachLM you've built and the engineering problems that appear when we eventually turn it into a real multi-layer language model.
