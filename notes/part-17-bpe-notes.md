# OutreachLM — BPE Notes

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## 1. Why we moved beyond a word-level tokenizer
Our Version 1 tokenizer used spaces to define tokens:

```
"I love TRS"
→ ["I", "love", "TRS"]
```

This caused problems:

```
"TRS."       → "TRS."
"I-love-TRS" → "I-love-TRS"
```

We first separated punctuation using regex:

```
re.findall(r"\w+|[.,!?,:;]", text)
```

So punctuation became independent tokens.

---

## 2. `<UNK>` token
We added:

```
self.vocab = {
    "<UNK>": 0
}
```

An unknown token is mapped to `<UNK>` rather than dynamically changing the vocabulary during inference.

The vocabulary is fixed after training.

---

## 3. Why we chose BPE
A pure word-level vocabulary has poor coverage.

For example:

```
elephant
elephants
elephant's
```

could become separate words.

BPE allows the tokenizer to learn reusable pieces:

```
elephant
elephants
```

could share pieces such as:

```
elephant + s
```

Instead of requiring every complete word to exist in the vocabulary.

---

# 4. BPE starts with atomic units
We decided **not** to begin with 100,000 tokens.

Instead:

```
small base vocabulary
        ↓
BPE merges
        ↓
larger vocabulary
```

Our experiment started with characters:

```
l o w e r s t
```

Punctuation was also an atomic unit:

```
!
.
```

The eventual vocabulary size can be controlled by:

```
num_merges
```

or:

```
target_vocab_size
```

---

# 5. BPE corpus representation
We introduced a prepared corpus so BPE doesn't operate directly on raw strings.

For:

```
lowest!
```

the representation becomes:

```
[
    ["l", "o", "w", "e", "s", "t"],
    ["!"]
]
```

The separate sequences are important.

They prevent BPE from learning an invalid cross-boundary pair:

```
("t", "!")
```

So BPE can merge:

```
l + o → lo
lo + w → low
```

but cannot merge:

```
t + !
```

---

# 6. Pair counting
BPE examines adjacent units and counts how often each pair occurs.

Our example produced:

```
('l', 'o') → 3
('o', 'w') → 3
('w', 'e') → 2
('e', 'r') → 1
('e', 's') → 1
('s', 't') → 1
```

The learner uses frequency to determine which pair should be merged.

We discussed probability, but decided **not to introduce it yet** because we are implementing the core BPE algorithm first.

---

# 7. Selecting the best pair
Our method:

```
select_best_pair(pair_counts)
```

finds the pair with the highest frequency.

Conceptually:

```
pair counts
     ↓
highest frequency
     ↓
best pair
```

When frequencies tie, our current implementation deterministically chooses the first pair encountered.

---

# 8. Creating a new token
Once the best pair is selected:

```
new_token = best_pair[0] + best_pair[1]
```

For example:

```
('l', 'o')
```

becomes:

```
lo
```

Then:

```
lo + w
```

becomes:

```
low
```

---

# 9. Repeated merging
BPE doesn't perform only one merge.

It repeatedly:

```
COUNT PAIRS
    ↓
SELECT BEST
    ↓
CREATE TOKEN
    ↓
MERGE
    ↓
REPEAT
```

until either:

```
num_merges
```

is reached, or:

```
target_vocab_size
```

is reached, or there are no more mergeable pairs.

---

# 10. Merge ranks
Every learned merge receives a rank:

```
('l', 'o')     → rank 0
('lo', 'w')    → rank 1
('low', 'e')   → rank 2
('lowe', 'r')  → rank 3
```

The rank represents the order in which the merge was learned.

We store this in:

```
self.merge_ranks
```

---

# 11. Merge tokens
We also store what each pair becomes:

```
self.merge_tokens
```

Example:

```
('l', 'o') → 'lo'
('lo', 'w') → 'low'
('low', 'e') → 'lowe'
('lowe', 'r') → 'lower'
```

This gives the tokenizer enough information to reproduce the learned merges later.

---

# 12. Applying learned merges
We implemented:

```
apply_merges()
```

It repeatedly finds the highest-priority applicable merge and applies it until no applicable merge remains.

For example:

```
l o w e s t
```

becomes:

```
lo w e s t
```

then:

```
low e s t
```

then:

```
lowe s t
```

and so on according to the learned merge rules.

---

# 13. Vocabulary growth
Every newly created BPE token is added to the vocabulary.

Our successful experiment produced:

```
<UNK> → 0
l → 1
o → 2
w → 3
e → 4
r → 5
s → 6
t → 7
! → 8
. → 9
lo → 10
low → 11
lowe → 12
lower → 13
lowes → 14
```

Therefore:

```
Vocabulary size: 15
```

The important principle is:

> **BPE grows the vocabulary through learned merges rather than us manually creating thousands of vocabulary entries.**

---

# 14. Boundary preservation
This was our latest important architectural fix.

Our final corpus looked like:

```
[
    [["low"]],
    [["lower"]],
    [["lowes", "t"], ["!"]],
    [["lower"], ["."]]
]
```

Notice:

```
["lowes", "t"]
["!"]
```

are separate sequences.

Therefore BPE cannot accidentally learn:

```
("t", "!")
```

---

# 15. Final encoding test
Our working test was:

```
Input:
lowest!
```

BPE produced:

```
['lowes', 't', '!']
```

Those pieces became:

```
[14, 7, 8]
```

Then decoding returned:

```
lowest!
```

So we successfully verified the round trip:

```
TEXT
 ↓
BPE PIECES
 ↓
TOKEN IDs
 ↓
TEXT
```

with:

```
lowest!
→ ['lowes', 't', '!']
→ [14, 7, 8]
→ lowest!
```

---

# 16. Current BPE architecture
At the end of this stage, OutreachLM has:

```
RAW TEXT
   ↓
REGEX TOKENIZER
   ↓
PUNCTUATION SEPARATION
   ↓
CORPUS PREPARATION
   ↓
ATOMIC CHARACTER SEQUENCES
   ↓
PAIR COUNTING
   ↓
BEST-PAIR SELECTION
   ↓
MERGE
   ↓
VOCABULARY GROWTH
   ↓
MERGE RANKS + MERGE TOKENS
   ↓
REPEAT
   ↓
LEARNED BPE TOKENIZER
   ↓
ENCODE → IDs
   ↓
DECODE → TEXT
```

### BPE stage: **COMPLETE for our current implementation.** ✅
We have a working experimental BPE tokenizer.

The next stage is **not another BPE lesson**. We need to move from our tiny manually supplied corpus to a **real corpus/training-data pipeline**, then train the tokenizer on substantially more text before we build the neural-network portion of OutreachLM.