# Next: Build the training dataset

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.
We're at the exact point where **tokenized corpus → model-ready examples**.

We already have:

```
FineWeb documents
      ↓
BPE
      ↓
token IDs
```

Now we need:

```
token IDs
      ↓
fixed-length sequences
      ↓
(input, target)
```

## 1. The concept: next-token prediction
Suppose the tokenizer produces:

```
[The, quick, brown, fox, jumps]
```

and those correspond to:

```
[41, 82, 193, 17, 56]
```

The model's job is not to reproduce the same sequence.

It learns:

```
Input:  [41, 82, 193, 17]
Target: [82, 193, 17, 56]
```

So each position asks:

```
Given everything before me, what token comes next?
```

That gives us:

PositionInputTarget0Thequick1quickbrown2brownfox3foxjumpsThis **one-token shift** is fundamental to GPT-style language-model training.

---

## 2. Why use a context length?
We cannot normally feed an entire FineWeb document into the model.

Imagine one document contains:

```
12,000 tokens
```

Our first model might only support:

```
128 tokens
```

So we create windows.

For:

```
tokens =
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

with:

```
context_length = 5
```

we get:

```
Example 1

X = [0, 1, 2, 3, 4]
Y = [1, 2, 3, 4, 5]

Example 2

X = [1, 2, 3, 4, 5]
Y = [2, 3, 4, 5, 6]

Example 3

X = [2, 3, 4, 5, 6]
Y = [3, 4, 5, 6, 7]
```

Notice something important:

**the windows overlap.**

We're not throwing away the information between them.

---

# 3. Implement it
Create:

```
outreachlm/dataset.py
```

Put this in it:

```
class LanguageModelDataset:

    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):
        x = self.token_ids[
            index:index + self.context_length
        ]

        y = self.token_ids[
            index + 1:index + self.context_length + 1
        ]

        return x, y
```

Now let's understand it rather than blindly copying it.

---

## 4. `__init__`

```
def __init__(self, token_ids, context_length):
```

The dataset receives two things.

### `token_ids`
Our enormous numerical corpus:

```
[
    1, 2, 3, 432, 7, 446,
    7, 444, 11, 2, ...
]
```

### `context_length`
How many tokens the model sees at once.

For our first experiments:

```
context_length = 128
```

Later this becomes a model architecture decision.

---

# 5. `__len__`

```
def __len__(self):
    return len(self.token_ids) - self.context_length
```

Suppose:

```
total tokens = 1000
context length = 128
```

Then:

```
1000 - 128 = 872
```

possible starting positions.

Why not 1000?

Because the final example needs a target token **one position beyond its input**.

For example, the final input might be:

```
[872 ... 999]
```

and its target would need:

```
[873 ... 1000]
```

But token `1000` doesn't exist.

---

# 6. `__getitem__`
This is where the important part happens.

```
x = self.token_ids[
    index:index + self.context_length
]
```

This extracts the input.

Then:

```
y = self.token_ids[
    index + 1:index + self.context_length + 1
]
```

extracts the same region shifted one position forward.

So:

```
token stream:

A B C D E F G
```

becomes:

```
X:

A B C D E

Y:

B C D E F
```

The model therefore learns:

```
A → B
B → C
C → D
D → E
E → F
```

---

# 7. Test it before touching PyTorch
Let's make a tiny test in `main.py`.

```
from outreachlm.dataset import LanguageModelDataset

tokens = list(range(20))

dataset = LanguageModelDataset(
    tokens,
    context_length=5
)

print("=" * 50)
print("DATASET TEST")
print("=" * 50)

print("Total tokens:", len(tokens))
print("Context length:", dataset.context_length)
print("Dataset length:", len(dataset))

for i in range(3):
    x, y = dataset[i]

    print()
    print("Example", i)
    print("Input :", x)
    print("Target:", y)
```

Run:

```
python -m outreachlm.main
```

You should get approximately:

```
==================================================
DATASET TEST
==================================================
Total tokens: 20
Context length: 5
Dataset length: 15

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

If you get this, we've successfully built the **first form of the actual language-model training dataset**.

---

# 8. One important distinction
Right now we're using:

```
one giant token stream
```

Eventually we need to think carefully about **document boundaries**.

We don't necessarily want:

```
END OF DOCUMENT A
        ↓
FIRST TOKEN OF DOCUMENT B
```

to become a training relationship.

That's why we'll eventually introduce special tokens such as:

```
<BOS>
<EOS>
```

and potentially document packing.

**But don't implement that yet.**

First we prove the fundamental dataset mechanism works.

---

# Notes — OutreachLM

### 17. Language-model dataset
The tokenizer produces a sequence of integer token IDs.

The dataset converts that sequence into fixed-length training examples.

```
token IDs
    ↓
context windows
    ↓
input X + target Y
```

### 18. Next-token prediction
The target sequence is the input sequence shifted one position to the left.

```
X = [A, B, C, D]
Y = [B, C, D, E]
```

Therefore the model learns:

> Given previous tokens, predict the next token.

### 19. Context length
`context_length` determines how many tokens the model processes in one training example.

Example:

```
context_length = 128
```

means the model receives up to 128 input tokens at a time.

### 20. Overlapping windows
Training examples normally overlap:

```
[A B C D]
[B C D E]
[C D E F]
```

This allows every position in the token stream to participate in next-token prediction.

### 21. Current OutreachLM pipeline

```
RAW TEXT
   ↓
CORPUS
   ↓
BPE TOKENIZER
   ↓
TOKEN IDs
   ↓
LANGUAGE MODEL DATASET   ← CURRENT
   ↓
BATCHES
   ↓
EMBEDDINGS
   ↓
TRANSFORMER
   ↓
LOGITS
   ↓
LOSS
   ↓
BACKPROPAGATION
   ↓
UPDATED WEIGHTS
```

**Next: batching.** That's where individual `(X, Y)` examples become tensors that can be processed together by the model.