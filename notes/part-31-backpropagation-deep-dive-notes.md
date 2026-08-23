## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.
more notes update the other's if needed 
You're right. I skipped an important part of the curriculum.

Before we continue modifying OutreachLM, you need to understand **backpropagation**, because it is the mechanism that actually makes the model learn. The experiments we've been running are only meaningful if you understand what is happening between the forward pass and the weight update.

# OutreachLM Notes — Backpropagation

## 1. The big picture
Training a neural network is essentially this loop:

```
INPUT
  │
  ▼
FORWARD PASS
  │
  ▼
PREDICTION
  │
  ▼
LOSS
  │
  ▼
BACKPROPAGATION
  │
  ▼
GRADIENTS
  │
  ▼
PARAMETER UPDATE
  │
  ▼
REPEAT
```

In code:

```
logits = model(input_ids)

loss = loss_function(
    logits,
    targets
)

loss.backward()

optimizer.step()

optimizer.zero_grad()
```

Those five operations are the core of neural-network training.

---

# 2. What is a parameter?
A parameter is a number inside the model that the model is allowed to learn.

For example, your token embedding contains:

```
20 tokens × 16 dimensions
```

so you have:

```
320 learned numbers
```

just in the token embedding.

Your linear layers contain additional weights and biases.

The model therefore looks conceptually like:

```
Parameters:

W1
W2
W3
...
Wn
```

The goal of training is to find values for these parameters that make the model's predictions better.

---

# 3. What is the forward pass?
The forward pass is when information moves **through** the network.

For OutreachLM:

```
token IDs
   │
   ▼
token embedding
   │
   +
   │
position embedding
   │
   ▼
self-attention
   │
   ▼
feed-forward network
   │
   ▼
transformer blocks
   │
   ▼
hidden representation
   │
   ▼
output head
   │
   ▼
logits
```

Suppose the input is:

```
[0, 1, 2, 3, 4]
```

The target is:

```
[1, 2, 3, 4, 5]
```

The model might produce:

```
Prediction:

[0, 7, 3, 4, 12]
```

The forward pass simply produced that prediction.

**Nothing has been learned yet during that pass.**

---

# 4. What is the loss?
The loss measures how wrong the prediction was.

For language models, we normally use **cross-entropy loss**.

Conceptually:

```
Correct prediction
       ↓
    low loss

Wrong prediction
       ↓
   high loss
```

For example:

```
Target:     5
Model says: 5 with probability 0.99

→ very small loss
```

versus:

```
Target:     5
Model says: 12 with probability 0.99

→ large loss
```

The loss gives us a single quantity representing how badly the model performed.

---

# 5. But loss alone doesn't tell us how to improve
Suppose:

```
Loss = 2.5
```

That's useful, but we need more information.

We need to know:

> Which parameters caused the loss, and in which direction should each parameter move?
That's what the **gradient** tells us.

---

# 6. What is a gradient?
A gradient tells us how sensitive the loss is to a parameter.

Mathematically:

[
\frac{\partial L}{\partial w}
]

means:

> How much does the loss (L) change if parameter (w) changes?
Suppose:

```
weight = 0.50
```

and:

```
∂L/∂w = +2.0
```

The positive gradient means increasing that weight would increase the loss locally.

So we want to move it in the opposite direction.

If:

```
∂L/∂w = -2.0
```

then increasing the weight would decrease the loss locally.

The gradient therefore gives us a **direction for learning**.

---

# 7. Gradient descent
The fundamental update is:

# [
w_{\text{new}}

## w_{\text{old}}
\eta
\frac{\partial L}{\partial w}
]

where:

- (w) = parameter
- (L) = loss
- (\eta) = learning rate
- (\frac{\partial L}{\partial w}) = gradient
For example:

```
weight = 0.50
gradient = 2.0
learning rate = 0.01
```

Then:

```
new weight
=
0.50 - (0.01 × 2.0)

= 0.48
```

The parameter moved in the direction that should reduce the loss.

---

# 8. Why is it called "backpropagation"?
Because the error information travels **backward through the network**.

Forward:

```
Input
  ↓
Embedding
  ↓
Attention
  ↓
Feed Forward
  ↓
Output
  ↓
Prediction
  ↓
Loss
```

Backward:

```
Loss
  ↓
Output Head
  ↓
Transformer
  ↓
Feed Forward
  ↓
Attention
  ↓
Embeddings
```

The forward pass answers:

> What prediction did we make?
Backpropagation answers:

> How should every parameter change because of that prediction?

---

# 9. The chain rule
This is the mathematical foundation of backpropagation.

Suppose:

[
x \rightarrow y \rightarrow L
]

where:

[
y=f(x)
]

and:

[
L=g(y)
]

Then:

# [
\frac{dL}{dx}
\frac{dL}{dy}
\frac{dy}{dx}
]

This is the **chain rule**.

Neural networks are essentially enormous compositions of functions.

For example:

```
x
↓
Layer 1
↓
Layer 2
↓
Layer 3
↓
Layer 4
↓
Loss
```

Backpropagation repeatedly applies the chain rule to determine how the final loss depends on every parameter.

---

# 10. A tiny example
Imagine:

[
y = wx
]

and:

[
L=(y-t)^2
]

Suppose:

```
x = 2
w = 3
target = 10
```

Forward pass:

[
y=3(2)=6
]

Loss:

[
L=(6-10)^2
]

[
L=16
]

Now we need:

[
\frac{dL}{dw}
]

Using the chain rule:

# [
\frac{dL}{dw}
\frac{dL}{dy}
\frac{dy}{dw}
]

First:

[
L=(y-t)^2
]

therefore:

[
\frac{dL}{dy}=2(y-t)
]

At (y=6):

[
\frac{dL}{dy}=2(6-10)
]

[
=-8
]

And:

[
y=wx
]

so:

[
\frac{dy}{dw}=x
]

and (x=2).

Therefore:

[
\frac{dL}{dw}=(-8)(2)
]

[
=-16
]

So the gradient is:

```
-16
```

The optimizer can now update the weight.

---

# 11. How this relates directly to OutreachLM
Your model contains a huge chain of operations.

For one token:

```
token ID
   ↓
embedding lookup
   ↓
token vector
   ↓
+ positional vector
   ↓
LayerNorm
   ↓
Q/K/V projections
   ↓
attention scores
   ↓
scaling
   ↓
causal mask
   ↓
softmax
   ↓
weighted values
   ↓
output projection
   ↓
residual
   ↓
LayerNorm
   ↓
feed-forward
   ↓
residual
   ↓
final LayerNorm
   ↓
output head
   ↓
logits
   ↓
cross entropy
   ↓
LOSS
```

Backpropagation goes through this entire computational graph in reverse.

---

# 12. What happens inside PyTorch?
This line:

```
loss.backward()
```

is extremely important.

PyTorch has been tracking the mathematical operations used to produce the loss.

When you call:

```
loss.backward()
```

PyTorch calculates gradients for the parameters.

For example:

```
model.token_embedding.embedding.weight.grad
```

contains gradients for the token embedding.

Likewise:

```
model.output_head.weight.grad
```

contains gradients for the output head.

And attention parameters have gradients too:

```
model.transformer_blocks[0].attention.query.weight.grad
```

etc.

---

# 13. The optimizer
After backpropagation:

```
optimizer.step()
```

updates the parameters.

For basic stochastic gradient descent:

```
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)
```

the basic idea is:

```
parameter
    ↓
gradient
    ↓
parameter - learning_rate × gradient
    ↓
new parameter
```

We're currently using Adam-style optimization in our experiments, which is more sophisticated than plain SGD.

But the fundamental idea remains:

> **Use gradients to determine how parameters should change.**

---

# 14. Why do we zero the gradients?
This line:

```
optimizer.zero_grad()
```

is also important.

PyTorch normally **accumulates gradients**.

Imagine:

```
Step 1:
gradient = 2

Step 2:
gradient = 3
```

Without clearing:

```
stored gradient = 5
```

Usually we want each training step to use the current batch's gradients.

So:

```
optimizer.zero_grad()
```

clears the previous gradients.

The normal training sequence is:

```
optimizer.zero_grad()

logits = model(inputs)

loss = criterion(
    logits,
    targets
)

loss.backward()

optimizer.step()
```

Think:

```
CLEAR
  ↓
FORWARD
  ↓
LOSS
  ↓
BACKWARD
  ↓
UPDATE
```

---

# 15. The complete learning loop
Your current OutreachLM training loop should conceptually look like:

```
for step in range(training_steps):

    optimizer.zero_grad()

    logits = model(input_ids)

    loss = criterion(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )

    loss.backward()

    optimizer.step()
```

Every iteration does:

```
1. Start fresh
2. Make prediction
3. Calculate error
4. Calculate gradients
5. Change parameters
```

Then repeat hundreds or millions of times.

---

# 16. Why your training loss fell
You saw something like:

```
Step   1 | Loss: 3.393236
Step  10 | Loss: 1.578165
Step  20 | Loss: 0.287996
Step  30 | Loss: 0.059402
...
Step 100 | Loss: 0.002143
```

That means the optimization process found parameter values that make the **training examples increasingly easy to predict**.

But remember our earlier experiment:

```
Training Accuracy: 100%
```

while the held-out transition:

```
18 -> 19
```

failed.

This is a crucial lesson.

### Low training loss ≠ intelligence.
It can mean:

```
"The model memorized what I showed it."
```

rather than:

```
"The model learned the underlying rule."
```

That distinction will become extremely important when we train OutreachLM on real language.

---

# 17. Backpropagation through attention
This is particularly important for what you've built.

Your attention computes:

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

Then:

[
A=\operatorname{softmax}(S)
]

Then:

[
Z=AV
]

Then:

[
O=ZW_O
]

During backpropagation, the loss sends gradient information backward through:

```
Loss
 ↓
Output projection
 ↓
Z
 ↓
A and V
 ↓
softmax
 ↓
attention scores
 ↓
Q and K
 ↓
WQ / WK / WV
 ↓
input representation
```

So your attention heads don't merely produce numbers.

They contain **learnable parameters whose values are changed by gradients**.

---

# 18. What happens to the embeddings?
This is particularly important given the embedding analysis we just performed.

Initially:

```
Token 0 → random vector
Token 1 → random vector
Token 2 → random vector
...
```

During training, gradients reach the embedding matrix.

Therefore:

```
embedding[token]
       ↑
       │
    gradient
       │
      loss
```

The embedding vectors gradually change.

Eventually tokens that are useful in similar contexts can develop useful geometric relationships.

This is why your embedding-distance analysis was worth doing.

But the distances you saw **should not be interpreted as semantic understanding yet**, because the training data was far too artificial.

---

# 19. Three different things you must keep separate
This will save you a lot of confusion later.

### Forward propagation
Answers:

> What does the model predict?

### Backpropagation
Answers:

> How does the loss depend on the model's parameters?

### Optimization
Answers:

> Given those gradients, how should we change the parameters?
Together:

```
FORWARD
   ↓
PREDICTION
   ↓
LOSS
   ↓
BACKPROPAGATION
   ↓
GRADIENTS
   ↓
OPTIMIZER
   ↓
UPDATED PARAMETERS
```

---

# 20. The most important mental model
Don't think:

> "The model sees the answer and learns it."
Think:

> **The model makes a prediction, measures its error, calculates how every parameter contributed to that error, and nudges those parameters so that similar predictions become slightly better next time.**
Repeated millions/billions of times, those tiny updates can produce surprisingly powerful behavior.

---

# 21. What we need to learn next
Our curriculum should now proceed in this order:

```
✓ Forward propagation
✓ Embeddings
✓ Positional embeddings
✓ Self-attention
✓ Causal masking
✓ Transformer block
✓ Loss
✓ Backpropagation
✓ Gradients
✓ Gradient descent / optimization

        ↓ NEXT

AUTODIFF IN PYTORCH
        ↓
Inspecting real gradients
        ↓
Gradient flow
        ↓
Vanishing/exploding gradients
        ↓
Residual connections
        ↓
Layer normalization
        ↓
Transformer stack
        ↓
Tokenizer
        ↓
Real text dataset
        ↓
Language-model training
        ↓
Evaluation
        ↓
Generation
```

**That is the path I'm going to keep us on.** No more disconnected toy experiments unless an experiment directly teaches one of these concepts or validates a component of OutreachLM.

And before we scale OutreachLM further, you should be able to look at:

```
loss.backward()
```

and understand that it isn't a magical "make the AI learn" command.

It is PyTorch executing the **chain rule across the entire computation graph** to calculate the gradients that tell the optimizer how to change the model.
