## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.
## Part 16 — Finish the BPE Merging Phase + Notes
Before we move on, **yes: this is the point where we should close out merging properly and document it.**

You now have the core BPE learning loop:

```
Corpus
  ↓
Base units
  ↓
Initial vocabulary
  ↓
Count adjacent pairs
  ↓
Select best pair
  ↓
Create merged token
  ↓
Add token to vocabulary
  ↓
Record merge rank
  ↓
Apply merge to corpus
  ↓
Repeat
  ↓
Stop at target vocabulary size
```

### Your final merging notes

# OutreachLM — BPE Merging Notes

### 1. Why BPE?
A word-level tokenizer cannot represent every possible word efficiently.

For example:

```
elephant
elephants
elephantine
```

Instead of requiring every complete word in the vocabulary, BPE learns reusable pieces.

```
elephants
→ elephant + s
```

The model can therefore represent previously unseen words using known pieces.

---

### 2. Base vocabulary
We don't start with 100,000 manually chosen tokens.

We start with atomic units.

For a byte-based system:

```
256 possible byte values
```

For our initial learning experiments, we're using character units.

Example:

```
low
→ l o w
```

---

### 3. Pair counting
BPE examines **adjacent units**.

Given:

```
low
lower
lowest
```

we obtain:

```
('l', 'o') → 3
('o', 'w') → 3
('w', 'e') → 2
('e', 'r') → 1
('e', 's') → 1
('s', 't') → 1
```

Pairs are counted **within each word/token sequence**, not across word boundaries.

---

### 4. Selecting the best pair
The learner chooses the pair with the highest frequency.

For the first iteration:

```
('l', 'o') → 3
('o', 'w') → 3
```

There is a tie, so our implementation needs a deterministic tie-breaking rule.

The selected pair becomes a merge rule.

---

### 5. Merging
If:

```
('l', 'o')
```

wins, we create:

```
lo
```

So:

```
l o w
```

becomes:

```
lo w
```

The next iteration can discover:

```
('lo', 'w')
```

and create:

```
low
```

This is why BPE is **iterative**.

---

### 6. Repeated merging
Our learner repeatedly performs:

```
count
→ select
→ merge
→ recount
→ select
→ merge
```

For the example corpus, we obtained:

```
('l', 'o')     → 'lo'
('lo', 'w')    → 'low'
('low', 'e')   → 'lowe'
('lowe', 'r') → 'lower'
('lowe', 's') → 'lowes'
```

Notice how tokens become progressively larger.

---

### 7. Merge ranks
We record the order in which merges were learned.

```
{
    ('l', 'o'): 0,
    ('lo', 'w'): 1,
    ('low', 'e'): 2,
    ('lowe', 'r'): 3,
    ('lowe', 's'): 4
}
```

The rank tells us **which merge has priority**.

Lower rank means the merge was learned earlier.

---

### 8. Merge tokens
We separately record what each pair produces:

```
{
    ('l', 'o'): 'lo',
    ('lo', 'w'): 'low',
    ('low', 'e'): 'lowe',
    ('lowe', 'r'): 'lower',
    ('lowe', 's'): 'lowes'
}
```

So:

```
merge rank
    ↓
which rule has priority?

merge token
    ↓
what does the rule produce?
```

---

### 9. Vocabulary growth
Every newly learned token gets an ID.

For example:

```
<UNK> → 0
l     → 1
o     → 2
w     → 3
e     → 4
r     → 5
s     → 6
t     → 7

lo    → 8
low   → 9
lowe  → 10
lower → 11
```

The vocabulary therefore **grows as BPE learns**.

We don't manually create 100,000 tokens.

---

### 10. Vocabulary target
Our initial target is:

```
100,000 tokens
```

but this is only a **starter target**.

Eventually we'll determine the appropriate vocabulary size experimentally based on:

- corpus size
- languages
- token compression
- model size
- training efficiency
- memory requirements
- sequence length

---

### 11. Stopping conditions
The learner should stop when:

```
vocabulary size >= target vocabulary size
```

or when:

```
there are no mergeable pairs left
```

This prevents an infinite learning loop.

---

### 12. Important architecture distinction
We now have three separate pieces:

```
VOCABULARY
"I" → 1
"lo" → 8
"low" → 9

MERGE RANKS
('l','o') → 0
('lo','w') → 1

MERGE TOKENS
('l','o') → 'lo'
('lo','w') → 'low'
```

They have different jobs and should remain separate.

---

# What we have actually built
At this point, OutreachLM has moved beyond a simple whitespace tokenizer.

We have:

```
                    OutreachLM Tokenizer
                            │
              ┌─────────────┴─────────────┐
              │                           │
          Tokenization                BPE Learning
              │                           │
        regex splitting              pair counting
              │                           │
          base units                best pair
              │                           │
          vocabulary               repeated merging
              │                           │
             IDs                    merge rules
```

And that's the important milestone.

**The BPE learning/merging subsystem is now conceptually complete.**

The next subsystem is **not another merge algorithm**.

The next question is:

> **How does a tokenizer that has already learned these merges use them on NEW text?**
> For example, if training learned:

```
l + o → lo
lo + w → low
```

and tomorrow the tokenizer receives:

```
lowest
```

we need to apply the learned rules correctly.

That's the beginning of the **BPE encoding/inference phase**.

We'll tackle that next.
