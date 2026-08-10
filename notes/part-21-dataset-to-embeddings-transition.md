# OutreachLM — Dataset to Embeddings Transition Notes

Exactly. **Now we are past the tokenizer and corpus-loading stage.** Your latest output means we have successfully reached the point where raw token IDs are being turned into **training examples for the neural network**.

# NEXT: Training data → input/target pairs
You just got:

```
Total tokens: 20
Context length: 5
Dataset length: 15

Example 0
Input : [0, 1, 2, 3, 4]
Target: [1, 2, 3, 4, 5]
```

This is a very important transition.

We are no longer primarily building the tokenizer.

We are now building the **language-model training pipeline**.

---

## 1. What does the dataset actually contain?
Suppose our tokenized corpus is:

```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
```

and we choose:

```
context_length = 5
```

We take **5 tokens as the input**, and ask the model to predict what comes next at every position.

So:

```
Input:
[0, 1, 2, 3, 4]

Target:
[1, 2, 3, 4, 5]
```

Notice something important:

The target is the input shifted **one position to the left**.

```
INPUT
  0  1  2  3  4
  ↓  ↓  ↓  ↓  ↓
TARGET
  1  2  3  4  5
```

That is the fundamental training objective of a causal language model.

---

# 2. Why don't we just use one target?
Because we want the model to learn prediction at **every position**.

Consider:

```
Input:  [0, 1, 2, 3, 4]
Target: [1, 2, 3, 4, 5]
```

This actually represents five prediction problems:

```
0 → 1
0,1 → 2
0,1,2 → 3
0,1,2,3 → 4
0,1,2,3,4 → 5
```

The transformer will eventually process the entire sequence simultaneously, but a **causal attention mask** will make sure each position can only use information from the past.

That distinction becomes extremely important when we build the transformer.

---

## 3. The sliding-window dataset
Your dataset is essentially doing this:

```
Tokens:

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
│────────────│
    input
  │────────────│
     target
```

Then the window moves one token:

```
Input  0 1 2 3 4
Target   1 2 3 4 5
```

Then:

```
Input  1 2 3 4 5
Target   2 3 4 5 6
```

Then:

```
Input  2 3 4 5 6
Target   3 4 5 6 7
```

Your output confirms exactly that:

```
Example 0
Input : [0, 1, 2, 3, 4]
Target: [1, 2, 3, 4, 5]

Example 1
Input : [1, 2, 3, 4, 5]
Target: [2, 3, 4, 5, 6]

Example 2
Input : [2, 3, 4, 5, 6]
Target: [3, 4, 5, 6, 7]
```

So **this component is working.**

---

# 4. What happens after this?
Now we have:

```
RAW TEXT
   ↓
CORPUS
   ↓
TOKENIZER
   ↓
TOKEN IDs
   ↓
TRAINING DATASET   ← YOU ARE HERE
   ↓
BATCHES
   ↓
EMBEDDINGS
   ↓
POSITIONAL INFORMATION
   ↓
TRANSFORMER
   ↓
LOGITS
   ↓
LOSS
   ↓
BACKPROPAGATION
   ↓
OPTIMIZER
   ↓
LEARNING
```

This is the next major phase of OutreachLM.

---

# 5. Next concept: batching
Right now one dataset example is:

```
x = [0, 1, 2, 3, 4]
y = [1, 2, 3, 4, 5]
```

But the neural network shouldn't normally train on one sequence at a time.

We create a **batch**.

For example:

```
X

[
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7]
]
```

and:

```
Y

[
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8]
]
```

Now the model receives a **batch of sequences**.

Conceptually:

```
              CONTEXT
                 ↓

Sequence 1 → [0 1 2 3 4]
Sequence 2 → [1 2 3 4 5]     → TRANSFORMER
Sequence 3 → [2 3 4 5 6]
Sequence 4 → [3 4 5 6 7]
                              ↓
                         PREDICTIONS
```

The batch dimension is what allows us to make training computationally efficient.

---

# 6. The dimensions we are heading toward
This is worth understanding now because these dimensions will appear everywhere in the transformer.

Eventually our input tensor will look like:

```
[B, T]
```

where:

```
B = batch size
T = context length
```

For example:

```
[4, 5]
```

means:

```
4 sequences
5 tokens per sequence
```

After the embedding layer, it becomes:

```
[B, T, D]
```

where:

```
B = batch size
T = sequence length
D = embedding dimension
```

For example:

```
[4, 5, 128]
```

means:

> 4 sequences, each containing 5 tokens, with every token represented by a 128-dimensional vector.
> That is the first major transformation from **discrete token IDs → continuous neural representations**.

---

# 7. Why embeddings are necessary
Our tokenizer gives us:

```
"the" → 432
"cat" → 817
"sat" → 91
```

But the neural network cannot meaningfully understand that:

```
817 > 432
```

or that:

```
91 is closer to 817
```

Token IDs are merely **addresses**.

So we create an embedding matrix:

```
Embedding Matrix

        dimension
       0    1    2    3   ... 127
     ┌─────────────────────────────
ID 0 │ 0.12 -0.31 0.44 ...
ID 1 │ 0.82  0.11 0.09 ...
ID 2 │-0.17  0.52 0.71 ...
...
ID N │ 0.33 -0.28 0.16 ...
```

If:

```
vocab_size = 1000
embedding_dim = 128
```

then:

```
Embedding matrix = [1000, 128]
```

Token:

```
432
```

simply retrieves row:

```
Embedding[432]
```

and produces a 128-dimensional vector.

This is where **OutreachLM starts becoming a neural model rather than a symbolic text-processing system.**

---

# Your next implementation
Before we touch the transformer, we should build the **batching + embedding stage**.

The progression should be:

```
Dataset
   ↓
DataLoader / batch sampler
   ↓
X, Y
   ↓
TokenEmbedding
   ↓
[B, T, D]
```

Then we inspect the tensors and make sure the dimensions are correct.

**Do not build attention yet.** We first make sure the model can successfully turn your token IDs into vectors.

### Your next task
Implement:

```
outreachlm/
├── tokenizer/
├── corpus_loader.py
├── dataset.py        ← already here
├── embeddings.py     ← NEXT
└── main.py
```

The next lesson will be **Token Embeddings: how `[1, 432, 7, ...]` becomes `[batch, sequence, embedding_dim]`**, followed immediately by implementing it and testing the tensor shapes.

### BPE notes status
Your BPE notes are complete through:

- character/base vocabulary
- pair counting
- merge selection
- merge ranks
- repeated merging
- vocabulary growth
- encoding
- decoding
- boundaries
- unknown-token handling
- tokenizer testing
- corpus statistics
- scaling the tokenizer to the FineWeb corpus
We've moved on.

**Current OutreachLM position:**

> **Tokenizer ✅ → Corpus pipeline ✅ → Dataset/windowing ✅ → Embeddings ⬅️ NEXT → Transformer**
