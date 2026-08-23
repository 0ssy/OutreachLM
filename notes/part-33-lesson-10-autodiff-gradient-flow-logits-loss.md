# OutreachLM — Lesson 10: Automatic Differentiation & Real Gradients

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.
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

---

# LESSON 11 — GRADIENT FLOW

## 1. What is gradient flow?
A neural network is a chain of mathematical operations.

For OutreachLM:

```
Input tokens
     ↓
Token embedding
     ↓
Position embedding
     ↓
Transformer block
     ↓
Attention
     ↓
Feed-forward
     ↓
Output head
     ↓
Logits
     ↓
Loss
```

The forward pass travels downward.

Backpropagation travels backward:

```
Loss
 ↑
Output head
 ↑
Transformer
 ↑
Attention / Feed-forward
 ↑
Embeddings
```

The gradient tells each parameter:

> "If you change me slightly, how does the loss change?"
Mathematically:

[
\frac{\partial L}{\partial \theta}
]

where:

- (L) = loss
- (\theta) = model parameter

---

# 2. The chain rule
Backpropagation is fundamentally the **chain rule of calculus**.

Suppose:

[
x \rightarrow y \rightarrow z
]

and:

[
y=f(x)
]

[
z=g(y)
]

Then:

# [
\frac{dz}{dx}
\frac{dz}{dy}
\frac{dy}{dx}
]

A neural network is essentially a huge composition of functions.

Therefore:

# [
\frac{\partial L}{\partial W_1}
\frac{\partial L}{\partial W_n}
\frac{\partial W_n}{\partial W_{n-1}}
\cdots
\frac{\partial W_2}{\partial W_1}
]

The exact expression becomes complicated, but the principle is simple:

> **Gradients are propagated backward through every operation.**

---

# 3. Why gradient magnitude matters
Imagine a network with several layers.

Suppose every layer multiplies the gradient by approximately:

[
0.5
]

After ten layers:

# [
0.5^{10}
0.0009765625
]

The gradient has become tiny.

The early layers barely receive a learning signal.

This is the basic idea behind **vanishing gradients**.

Now suppose every layer multiplies the gradient by:

[
2
]

After ten layers:

[
2^{10}=1024
]

The gradient becomes huge.

That's the basic idea behind **exploding gradients**.

---

# 4. Vanishing gradients
A vanishing gradient means:

[
\left|\nabla_\theta L\right| \approx 0
]

The parameter technically has a gradient, but it is so small that the parameter barely changes.

Training can therefore look like:

```
Layer 12: learns
Layer 11: learns
Layer 10: learns
...
Layer 2: barely learns
Layer 1: almost frozen
```

This becomes especially dangerous in deep networks.

---

# 5. Exploding gradients
The opposite is:

[
\left|\nabla_\theta L\right| \gg 1
]

Large gradients can cause enormous parameter updates.

Instead of:

```
weight
1.20 → 1.19 → 1.18 → ...
```

you could get:

```
weight
1.20 → -40 → 9000 → NaN
```

Training becomes unstable.

---

# 6. Why Transformers use LayerNorm
Your Transformer already contains:

```
self.norm1 = nn.LayerNorm(embedding_dim)
self.norm2 = nn.LayerNorm(embedding_dim)
```

Layer normalization helps keep activations numerically well-behaved.

Your block is:

```
x
│
▼
LayerNorm
│
▼
Attention
│
▼
Residual addition
│
▼
LayerNorm
│
▼
Feed Forward
│
▼
Residual addition
```

This architecture is not accidental.

Normalization and residual connections are major parts of why deep Transformers can be trained effectively.

---

# 7. Residual connections and gradients
Your attention block contains:

```
x = x + attention_output
```

Mathematically:

[
y=x+F(x)
]

Differentiate:

# [
\frac{dy}{dx}
1+\frac{dF(x)}{dx}
]

That `1` is extremely important.

Even if the gradient through (F(x)) becomes small, the residual path provides a direct route.

Conceptually:

```
             ┌───────────────┐
             │               │
x ───────────┼──────────────► + ───► y
│            │               ▲
│            ▼               │
└──────► F(x) ───────────────┘
```

The gradient has a shortcut.

This is one reason residual networks are so powerful.

---

# 8. OutreachLM's current architecture
Your current model is effectively:

```
                 TOKEN IDs
                    │
                    ▼
             Token Embedding
                20 × 16
                    │
                    +
                    │
            Position Embedding
                 5 × 16
                    │
                    ▼
             Transformer Block
              ┌─────────────┐
              │ LayerNorm   │
              │     ↓       │
              │ Attention   │
              │     ↓       │
              │ Residual    │
              │     ↓       │
              │ LayerNorm   │
              │     ↓       │
              │ FeedForward │
              │     ↓       │
              │ Residual    │
              └─────────────┘
                    │
                    ▼
               Output Head
                  16 → 20
                    │
                    ▼
                  LOGITS
                    │
                    ▼
              Cross Entropy
                    │
                    ▼
                  LOSS
```

Then:

```
LOSS
 ↓
Output Head
 ↓
Transformer
 ↓
Attention
 ↓
Embeddings
```

That's your gradient flow.

---

# 9. Gradient analysis
Our next engineering tool should therefore inspect:

```
parameter
gradient
gradient norm
```

For example:

```
for name, parameter in model.named_parameters():

    if parameter.grad is None:
        continue

    gradient_norm = parameter.grad.norm().item()

    print(
        f"{name:50s}"
        f"{gradient_norm:.8f}"
    )
```

This gives us an actual measurement rather than guessing.

---

# 10. What we are looking for
A healthy tiny model might produce something like:

```
============================================================
GRADIENT FLOW
============================================================

Parameter                                      Grad Norm
------------------------------------------------------------
token_embedding.embedding.weight               0.0312
position_embedding.embedding.weight            0.0148
transformer.attention.query.weight             0.0271
transformer.attention.key.weight               0.0234
transformer.attention.value.weight             0.0412
transformer.attention.output.weight            0.0387
transformer.feed_forward...                    0.0521
transformer.norm1...                            0.0189
transformer.norm2...                            0.0223
output_head.weight                              0.1842
output_head.bias                                0.0901
```

The exact numbers are not important.

What matters is the **pattern**.

We want to understand whether gradients are:

- present
- extremely small
- extremely large
- concentrated in particular components
- disappearing as depth increases

---

# 11. Gradient clipping
One technique used when gradients become too large is **gradient clipping**.

PyTorch provides:

```
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

The training sequence becomes:

```
optimizer.zero_grad()

logits = model(input_ids)

loss = criterion(
    logits,
    targets
)

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()
```

Notice the order.

**Clip after backward, before optimizer.step().**

Why?

Because the gradients don't exist until:

```
loss.backward()
```

---

# 12. Gradient clipping does not solve everything
Clipping is not a magic solution.

If the model constantly produces terrible gradients, clipping only hides part of the problem.

We first want to understand:

```
WHY are the gradients large?
```

before automatically clipping them.

This is why we're studying gradient flow first.

---

# LESSON 12 — LOGITS, SOFTMAX AND LOSS
We now connect another piece that you've already been using without fully unpacking it.

Your output head produces:

```
[batch, sequence, vocabulary]
```

For your current model:

```
[1, 5, 20]
```

That means:

```
1 example
5 positions
20 possible next tokens
```

For one position, the output is:

```
20 numbers
```

These are **logits**.

---

## 13. Logits are not probabilities
Suppose the output head produces:

```
[-2.1, 0.5, 4.7, 1.2, ...]
```

Those numbers can be positive or negative.

They don't need to add up to 1.

They are raw scores.

---

# 14. Softmax
Softmax converts logits into probabilities:

# [
P_i
\frac{e^{z_i}}
{\sum_j e^{z_j}}
]

For example:

```
Logits:

token 0 → 1.2
token 1 → 0.4
token 2 → 4.8
token 3 → -0.5
```

After softmax, token 2 receives the largest probability.

Conceptually:

```
logits
  ↓
softmax
  ↓
probability distribution
```

---

# 15. Cross entropy
If the correct answer is token 2:

[
L=-\log P(\text{token 2})
]

If:

[
P(2)=0.99
]

then:

[
L\approx0.010
]

Very small loss.

If:

[
P(2)=0.01
]

then:

[
L\approx4.605
]

Large loss.

This explains your training curves.

When OutreachLM became highly confident about the training examples, the loss dropped dramatically.

---

# 16. Important PyTorch detail
When you use:

```
nn.CrossEntropyLoss()
```

you normally give it **raw logits**.

You should NOT manually do:

```
softmax(logits)
```

before passing them into `CrossEntropyLoss`.

PyTorch's cross-entropy implementation internally handles the appropriate log-softmax operation.

So:

```
loss = criterion(
    logits,
    targets
)
```

is correct.

---

# 17. The complete learning loop
At this point you should be able to mentally execute:

```
optimizer.zero_grad()

logits = model(input_ids)

loss = criterion(
    logits,
    targets
)

loss.backward()

optimizer.step()
```

as:

```
CLEAR OLD GRADIENTS
        ↓
FORWARD PASS
        ↓
GENERATE LOGITS
        ↓
COMPARE WITH TARGET
        ↓
CALCULATE LOSS
        ↓
BACKPROPAGATE
        ↓
CALCULATE GRADIENTS
        ↓
OPTIONALLY CLIP
        ↓
OPTIMIZER UPDATES WEIGHTS
        ↓
REPEAT
```

That is the fundamental training engine.

---

# 18. The bigger curriculum
We are **not done with the Transformer yet**.

Our path from here is:

```
                         OUTREACHLM
                             │
             ┌───────────────┴───────────────┐
             │                               │
        ARCHITECTURE                       TRAINING
             │                               │
             ▼                               ▼
      Gradient Flow                    Backpropagation
             │                               │
             ▼                               ▼
   Vanishing / Exploding              Optimization
       Gradients                            │
             │                              ▼
             ▼                         Learning Rates
      Residual Paths                       │
             │                              ▼
             ▼                         Gradient Clipping
      Deeper Networks                       │
             │                              │
             └──────────────┬───────────────┘
                            ▼
                     BETTER TRANSFORMER
                            │
                            ▼
                       TOKENIZATION
                            │
                            ▼
                        TEXT DATA
                            │
                            ▼
                   LANGUAGE MODELING
                            │
                            ▼
                  TRAINING INFRASTRUCTURE
                            │
                            ▼
                     EVALUATION
                            │
                            ▼
                       GENERATION
                            │
                            ▼
                    OUTREACHLM v1
```

And eventually we'll replace the toy:

```
0 1 2 3 4 5 ...
```

with actual text:

```
The quick brown fox ...
```

But **not yet**.

We first need to make sure you understand the machinery that makes the model learn.

---

# Your permanent notes
Keep these five statements:

### 1. Forward propagation

> The model transforms input into predictions.

### 2. Loss

> Loss measures how wrong the prediction is.

### 3. Backpropagation

> Backpropagation calculates how each parameter contributed to the loss.

### 4. Gradient

> A gradient tells us the direction and magnitude in which changing a parameter affects the loss.

### 5. Optimizer

> The optimizer uses gradients to change the parameters.
In one line:

[
\boxed{
\text{Input}
\rightarrow
\text{Prediction}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Parameter Update}
}
]

That is the core learning loop of OutreachLM.

**Next in the curriculum: gradient-flow implementation and then vanishing/exploding gradients.**