# OutreachLM Notes — Part 5: BPE Pair Counting & First Merge

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## 31. BPE starts from atomic units
Before BPE begins, text is represented using the initial vocabulary's smallest units.

For example:

```
low
lower
lowest
```

starts as:

```
l o w
l o w e r
l o w e s t
```

At this point, `low`, `lower`, and `lowest` are **not necessarily vocabulary tokens**. BPE has to discover useful combinations.

---

## 32. Respect sequence boundaries
BPE counts adjacent pairs **inside each word/sequence**.

For example:

```
low lower
```

must **not** create a pair between:

```
w + l
```

because those belong to different words.

So:

```
low   → l o w
lower → l o w e r
```

are processed separately.

The general rule is:

> **Never create a BPE merge across a sequence boundary.**

---

## 33. Counting adjacent pairs
Given:

```
low
lower
lowest
```

we count every adjacent pair.

PairFrequency`(l, o)`3`(o, w)`3`(w, e)`2`(e, r)`1`(e, s)`1`(s, t)`1The frequency tells us how often a pair occurs in the training corpus.

---

## 34. Selecting the winning pair
Our current rule is:

### Rule 1
The pair with the **highest frequency** wins.

### Rule 2
If two or more pairs have the same frequency, use a **deterministic tie-breaker**.

Our chosen tie-breaker:

> **The pair encountered earliest wins.**
> Therefore:

```
(l, o) → 3
(o, w) → 3
```

results in:

```
(l, o)
```

winning because it was encountered first.

This makes the tokenizer deterministic.

---

## 35. Performing the merge
The winning pair:

```
l + o
```

becomes a new token:

```
lo
```

The corpus changes from:

```
l o w
l o w e r
l o w e s t
```

to:

```
lo w
lo w e r
lo w e s t
```

`lo` is now a learned vocabulary unit.

---

## 36. BPE is iterative
We don't count pairs once and stop.

The process repeats:

```
Count pairs
     ↓
Find best pair
     ↓
Merge pair
     ↓
Update corpus
     ↓
Count pairs again
     ↓
Find next pair
     ↓
Merge
     ↓
Repeat
```

Conceptually:

```
l + o → lo
lo + w → low
low + e → lowe
...
```

The actual sequence depends entirely on the statistics of the training corpus.

---

## 37. The vocabulary grows
Initially:

```
Base vocabulary
```

Then every successful merge creates another vocabulary entry:

```
l + o → lo
```

Vocabulary:

```
base units + lo
```

Next:

```
lo + w → low
```

Vocabulary:

```
base units + lo + low
```

Eventually the vocabulary approaches our target capacity:

```
100,000 tokens
```

So the important distinction is:

> **100,000 is the target size, not the starting size.**

---

## 38. The problem we now have
Suppose we train BPE and learn:

```
l + o → lo
```

Later, we receive new text:

```
lower
```

How does the tokenizer know that it should combine:

```
l + o
```

into:

```
lo
```

?

The tokenizer needs to **remember every merge it learned during training**.

That's what the **merge table** is for.

---

## 39. Current BPE architecture
We now have:

```
Training corpus
       ↓
Base units
       ↓
Initial vocabulary
       ↓
Count adjacent pairs
       ↓
Select best pair
       ↓
Merge
       ↓
New vocabulary token
       ↓
Record merge
       ↓
Repeat
```

The missing component is:

```
                ┌──────────────┐
                │  Merge Table │
                └──────┬───────┘
                       │
          remembers learned rules
                       │
                       ▼
                 New text can
               reproduce merges
```

---

## Key lesson
A BPE tokenizer has **two different things to remember**:

### Vocabulary
What tokens exist:

```
l
o
lo
w
low
...
```

### Merge rules
How tokens were created:

```
l + o → lo
lo + w → low
```

They are related, but they are **not the same thing**.

---

### Current OutreachLM progress

```
[V1] Space tokenizer                 ✓
[V2] Punctuation handling            ✓
[V3] Vocabulary + IDs + <UNK>        ✓
[V4] Understand OOV                  ✓
[V5] Base vocabulary design          ✓
[V6] Pair counting                   ✓
[V7] Merge selection                 ✓
[V8] Merge table                     ← NEXT
[V9] BPE training loop
[V10] BPE encoding
...
```

**Next: Merge Table.**