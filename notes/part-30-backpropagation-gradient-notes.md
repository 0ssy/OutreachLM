more notes 
Next. Now we cross the line from **calculating error** to **calculating how every parameter should change**.

# Step 1 — Backpropagation
We currently have:

```
input
  ↓
model
  ↓
logits
  ↓
loss
```

The loss is just a number.

For example:

```
Loss = 3.14
```

But that doesn't tell us *which parameters are responsible* for that error.

Backpropagation answers:

> **If the loss is high, which direction should each parameter move to reduce it?**
PyTorch does this with:

```
loss.backward()
```

---

## 2. Your first gradient experiment
Keep your current `main.py`, but after calculating the loss add:

```
loss.backward()
```

Then inspect a few parameters:

```
print()
print("Gradient inspection")
print("=" * 60)

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        print()
        print("Parameter:", name)
        print("Shape:", parameter.shape)
        print("Gradient mean:", parameter.grad.mean().item())
        print("Gradient absolute mean:", parameter.grad.abs().mean().item())
```

So the important section is:

```
logits = model(input_ids)

loss_function = nn.CrossEntropyLoss()

loss = loss_function(
    logits.transpose(1, 2),
    target_ids
)

print("Loss:", loss.item())

loss.backward()

print()
print("Gradient inspection")
print("=" * 60)

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        print()
        print("Parameter:", name)
        print("Shape:", parameter.shape)
        print("Gradient mean:", parameter.grad.mean().item())
        print(
            "Gradient absolute mean:",
            parameter.grad.abs().mean().item()
        )
```

Run:

```
python -m outreachlm.main
```

---

# 3. What you're about to see
You'll see parameters such as:

```
output_head.weight
output_head.bias
transformer.attention.query.weight
transformer.attention.key.weight
transformer.attention.value.weight
...
```

and each should have gradients.

Something approximately like:

```
Parameter: output_head.weight
Shape: torch.Size([20, 16])
Gradient mean: ...
Gradient absolute mean: ...

Parameter: output_head.bias
Shape: torch.Size([20])
Gradient mean: ...
Gradient absolute mean: ...

Parameter: transformer.attention.query.weight
Shape: torch.Size([16, 16])
Gradient mean: ...
Gradient absolute mean: ...
```

The exact values don't matter yet.

What matters is that gradients exist.

---

# 4. What is a gradient?
Suppose one parameter is:

```
w = 0.50
```

and its gradient is:

```
gradient = +0.8
```

The gradient tells us approximately:

> Increasing this parameter increases the loss.
Therefore, to reduce the loss, we'd generally want to move it in the opposite direction.

If:

```
gradient > 0
```

we move:

```
w ↓
```

If:

```
gradient < 0
```

we move:

```
w ↑
```

This is the fundamental idea behind gradient descent.

Mathematically:

[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
]

where:

- (w) = parameter
- (L) = loss
- (\frac{\partial L}{\partial w}) = gradient
- (\eta) = learning rate
We're not implementing that manually yet.

---

# 5. What `loss.backward()` actually does
Your model is a computational graph:

```
Input
  │
  ▼
Embedding
  │
  ▼
Position
  │
  ▼
Attention
  │
  ▼
Feed Forward
  │
  ▼
Output Head
  │
  ▼
Logits
  │
  ▼
Loss
```

When we execute:

```
loss.backward()
```

PyTorch traverses that graph **backwards**.

Conceptually:

```
LOSS
 ↑
 │
Output Head
 ↑
 │
Transformer
 ↑
 │
Attention
 ↑
 │
Embeddings
```

It applies the chain rule to calculate derivatives.

So the error signal propagates backward through the entire model.

That's why it's called **backpropagation**.

---

# 6. Important distinction
Backpropagation **does not change the model parameters**.

This is important.

After:

```
loss.backward()
```

we have:

```
parameters
     │
     ▼
gradients
```

but the parameters themselves haven't been updated yet.

We currently have:

```
Forward pass
     ↓
Logits
     ↓
Loss
     ↓
Backward pass
     ↓
Gradients
```

We're still missing:

```
Gradients
     ↓
Optimizer
     ↓
Parameter update
```

---

# 7. One more experiment
Before we add the optimizer, let's prove that gradients actually exist throughout the network.

Use:

```
for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            name,
            "→",
            "gradient exists" if parameter.grad is not None
            else "NO GRADIENT"
        )
```

You want to see essentially:

```
embedding.embedding.weight → gradient exists
position_embedding.embedding.weight → gradient exists
transformer.attention.query.weight → gradient exists
transformer.attention.query.bias → gradient exists
transformer.attention.key.weight → gradient exists
...
output_head.weight → gradient exists
output_head.bias → gradient exists
```

That is our next verification.

---

# Where we are now
Our learning pipeline is:

```
                FORWARD
                   │
                   ▼
                Input
                   │
                   ▼
               Embeddings
                   │
                   ▼
              Transformer
                   │
                   ▼
                 Logits
                   │
                   ▼
                  Loss
                   │
                   │
              BACKWARD
                   │
                   ▼
               Gradients
```

And the next missing component is:

```
                Gradients
                    │
                    ▼
                 Optimizer
                    │
                    ▼
             Update parameters
                    │
                    ▼
              Forward again
                    │
                    ▼
              New, hopefully
               smaller loss
```

That creates the actual learning loop:

```
        ┌──────────────────────┐
        │                      │
        ▼                      │
     Forward                  │
        │                     │
        ▼                     │
       Loss                   │
        │                     │
        ▼                     │
     Backward                 │
        │                     │
        ▼                     │
    Gradients                 │
        │                     │
        ▼                     │
     Optimizer ───────────────┘
```

**Next: we introduce the optimizer and perform OutreachLM's first actual parameter update.**

And we'll verify it experimentally by recording the parameter **before and after** the update, rather than simply trusting that `optimizer.step()` worked.
