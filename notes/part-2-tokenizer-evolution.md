# OutreachLM Notes — Part 2: Tokenizer Evolution

## 12. Version 1 — Space-based tokenizer

Our first tokenizer made one simple assumption:

> **Spaces define tokens.**

Implementation:

```
class Tokenizer:
    def tokenize(self, text):
        return text.split()
```

Example:

```
I love TRS
```

becomes:

```
["I", "love", "TRS"]
```

### What Version 1 taught us

It proved the basic tokenizer pipeline:

```
Text
 ↓
Split
 ↓
Tokens
```

But it also exposed a major weakness: **words and punctuation are glued together.**

---

## 13. The problems with Version 1

### Punctuation becomes part of the word

```
"I love TRS."
```

becomes:

```
["I", "love", "TRS."]
```

Now:

```
TRS
TRS.
TRS!
TRS,
```

can all become separate vocabulary entries.

That's unnecessary duplication.

### Hyphenated text becomes one token

```
"I-love-TRS"
```

becomes:

```
["I-love-TRS"]
```

The tokenizer has no understanding that these are separate pieces.

### Unknown words aren't solved

```
"elephants"
```

becomes:

```
["elephants"]
```

The tokenizer doesn't know whether the word is in a vocabulary because **we haven't built a vocabulary yet**.

This distinction is important:

> Version 1 can split text, but it cannot yet represent language using a fixed vocabulary.

---

## 14. Version 2 — Punctuation-aware tokenizer

We decided to fix **punctuation first**.

Why?

### 1. It reduces vocabulary duplication

Without punctuation splitting:

```
TRS
TRS.
TRS!
TRS,
```

could become four different tokens.

With punctuation splitting:

```
TRS
.
!
,
```

the word `TRS` remains reusable.

### 2. It prevents false OOV problems

`TRS.` isn't a new word.

It's:

```
TRS + .
```

Separating them means the vocabulary doesn't have to learn `TRS.` as a completely new unit.

### 3. It creates a cleaner foundation for subword tokenization

Later, when we implement BPE, we don't want punctuation attached to arbitrary words.

---

## 15. Our Version 2 implementation

We changed:

```
def tokenize(self, text):
    return text.split()
```

to:

```
import re

def tokenize(self, text):
    return re.findall(r"\w+|[.,!?;:]", text)
```

The important part is the regular expression:

```
\w+ | [.,!?;:]
```

It means:

> Find a sequence of word characters **OR** one of our defined punctuation characters.

---

## 16. What `\w+` means

```
\w+
```

means:

> Match one or more word characters.

For example:

```
Hello
TRS
TerraNode
123
```

can be matched as individual word-like units.

The `+` means **one or more**.

Without `+`, we'd be matching individual characters instead.

---

## 17. What `[.,!?;:]` means

The brackets define a character set.

So:

```
[.,!?;:]
```

means:

> Match one character that is `.`, `,`, `!`, `?`, `;`, or `:`.

Therefore:

```
Hello, Chatty!
```

becomes:

```
["Hello", ",", "Chatty", "!"]
```

---

## 18. Version 2 examples

### Example 1

```
I love TRS.
```

↓

```
["I", "love", "TRS", "."]
```

### Example 2

```
Hello, Chatty!
```

↓

```
["Hello", ",", "Chatty", "!"]
```

### Example 3

```
TRS: TerraNode
```

↓

```
["TRS", ":", "TerraNode"]
```

---

## 19. What Version 2 does NOT solve

We deliberately haven't solved unknown words yet.

For example:

```
elephants
```

still becomes:

```
["elephants"]
```

That's okay.

We're building the tokenizer incrementally.

Current architecture:

```
                 VERSION 1
                     │
                     ▼
                Split text
                     │
                     ▼
                 VERSION 2
                     │
                     ▼
             Separate punctuation
                     │
                     ▼
                   Tokens
```

We haven't reached:

```
Tokens
   ↓
Vocabulary
   ↓
IDs
```

yet.

---

## 20. Important engineering principle

We're not trying to build the final tokenizer immediately.

We're deliberately building:

```
V1 → identify weakness
V2 → fix weakness
V3 → identify next weakness
V4 → fix it
...
```

This lets us understand **why modern tokenizers have the architecture they do**, rather than simply copying an implementation.

---

## Current OutreachLM tokenizer roadmap

```
[V1] Space splitting
        ↓
[V2] Punctuation handling        ← WE ARE HERE
        ↓
[V3] Vocabulary + token IDs
        ↓
[V4] Unknown/OOV handling
        ↓
[V5] Subword representation
        ↓
[V6] BPE from scratch
        ↓
[V7] Training tokenizer
        ↓
[V8] Encoding / decoding
        ↓
[V9] Connect tokenizer to embeddings
        ↓
[V10] Begin building the language model
```

**We are currently at V2.**

Nothing else needs to be implemented yet.
