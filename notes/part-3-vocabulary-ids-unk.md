# OutreachLM Notes — Part 3: Vocabulary, IDs & `<UNK>`

## 21. Tokens must become numbers
Our Version 2 tokenizer produced strings:

```
"I love TRS."
        ↓
["I", "love", "TRS", "."]
```

But neural networks operate on numerical data.

So we introduced a **vocabulary** that maps tokens to integer IDs:

```
I       → 1
love    → 2
TRS     → 3
.       → 4
```

The numbers are **IDs, not values**.

`TRS → 3` does not mean TRS is somehow "greater" than `love → 2`.

The IDs are simply addresses that allow the model to refer to vocabulary entries.

---

## 22. Vocabulary
A vocabulary is the collection of tokens our tokenizer knows.

Our first vocabulary is built from the training text.

For example:

```
I love TRS.
TRS loves TerraNode.
```

produces:

```
<UNK>     → 0
I         → 1
love      → 2
TRS       → 3
.         → 4
loves     → 5
TerraNode → 6
```

The IDs are assigned as tokens are first encountered.

If `TRS` appears again, it **doesn't receive another ID**.

```
TRS → 3
TRS → 3
TRS → 3
```

This gives us a stable mapping within this tokenizer.

---

## 23. Encoding
We introduced the concept of **encoding**:

> Convert text into token IDs.
> Our pipeline is now:

```
Text
 ↓
Tokenize
 ↓
Tokens
 ↓
Vocabulary lookup
 ↓
IDs
```

Example:

```
"I love TRS."
```

becomes:

```
["I", "love", "TRS", "."]
```

and then:

```
[1, 2, 3, 4]
```

---

## 24. The Out-of-Vocabulary problem
Our vocabulary was trained on:

```
I love TRS.
TRS loves TerraNode.
```

Now we encounter:

```
I love elephants.
```

The tokenizer recognizes:

```
I
love
.
```

but `elephants` isn't in the vocabulary.

We therefore need a fallback.

---

## 25. The `<UNK>` token
We reserve:

```
<UNK> → 0
```

`<UNK>` means:

> **Unknown token**
> If the vocabulary doesn't contain a token, we use ID `0`.

Therefore:

```
elephants → <UNK> → 0
```

Our sentence becomes:

```
"I love elephants."
        ↓
["I", "love", "elephants", "."]
        ↓
[1, 2, 0, 4]
```

The tokenizer doesn't crash.

Every token can produce an ID.

---

## 26. Why `<UNK>` is useful
For our current **word-level tokenizer**, `<UNK>` is the simplest robust solution.

Without it:

```
unknown word
     ↓
error
     ↓
pipeline stops
```

With it:

```
unknown word
     ↓
<UNK>
     ↓
0
     ↓
pipeline continues
```

This gives our tokenizer a fixed vocabulary.

---

## 27. The major weakness of `<UNK>`
The problem is information loss.

Different unknown words become identical:

```
elephant   → 0
elephants  → 0
zebra      → 0
OutreachLM → 0
```

Once they become:

```
0
```

the tokenizer has lost the distinction between them.

This is the fundamental weakness we are going to encounter with word-level tokenization.

---

## 28. Why not dynamically add words?
Our current model architecture assumes a fixed vocabulary.

Eventually we'll have an embedding matrix such as:

```
Vocabulary size × embedding dimension
```

If the vocabulary has 10,000 tokens, the model has 10,000 corresponding embedding vectors.

Adding a new token during inference would require adding another embedding representation.

Therefore, **our current architecture cannot simply keep expanding the vocabulary during inference.**

A system could be specifically designed to resize its embeddings, but that isn't the architecture we're building.

---

## 29. Current tokenizer architecture
We now have:

```
                    RAW TEXT
                       │
                       ▼
                 ┌───────────┐
                 │ Tokenizer │
                 └─────┬─────┘
                       │
                       ▼
             ["I", "love", "TRS", "."]
                       │
                       ▼
                 ┌───────────┐
                 │ Vocabulary│
                 └─────┬─────┘
                       │
                       ▼
                  [1, 2, 3, 4]
                       │
                       ▼
                  Neural Model
```

Unknown token:

```
elephants
    │
    ▼
 <UNK>
    │
    ▼
    0
```

---

## 30. The problem that comes next
We've now reached an important limitation:

> **What if we could represent an unknown word without throwing the entire word away?**
> Instead of:

```
elephants → <UNK>
```

we could potentially represent it using reusable pieces:

```
elephant + s
```

And those pieces could have their own IDs.

That is the motivation for **subword tokenization**.

But there's an important distinction:

```
Current system:
word → ID

Future system:
subword → ID
```

And eventually:

```
BPE:
characters/bytes
      ↓
learned merges
      ↓
subword vocabulary
      ↓
token IDs
```

We have **not built that yet**.

---

## OutreachLM roadmap

```
[V1] Space splitting                 ✓
       ↓
[V2] Punctuation                    ✓
       ↓
[V3] Vocabulary + IDs + <UNK>       ✓
       ↓
[V4] Understand OOV limitations      ← NEXT
       ↓
[V5] Subword representation
       ↓
[V6] BPE from scratch
       ↓
[V7] Train tokenizer
       ↓
[V8] Encode / decode
       ↓
[V9] Embeddings
       ↓
[V10] Language model
```

### Current state
You have now built a tokenizer that can take:

```
"I love TRS."
```

and turn it into:

```
[1, 2, 3, 4]
```

That is a real, functioning **word-level tokenizer**.

The next step isn't immediately writing BPE. First we'll understand **exactly what information `<UNK>` destroys and why subwords solve that problem**.
