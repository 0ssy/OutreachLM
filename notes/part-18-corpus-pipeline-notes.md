# OutreachLM — Corpus Pipeline Notes

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## 17. Moving beyond the toy corpus
Our BPE implementation worked on a tiny manually created corpus:

```
low
lower
lowest!
lower.
```

But manually writing training text isn't scalable.

We therefore separated:

```
CORPUS = DATA
TOKENIZER = SYSTEM THAT LEARNS FROM THE DATA
```

The tokenizer should not contain the training data itself.

Instead:

```
text files
   ↓
corpus loader
   ↓
Tokenizer
```

---

## 18. Local corpus directory
We created:

```
OutreachLM/
│
├── corpus/
│   └── text/
│       ├── example1.txt
│       └── example2.txt
│
└── outreachlm/
    ├── main.py
    └── version.py
```

This lets us add documents without changing the tokenizer code.

---

## 19. Corpus loader
We introduced:

```
def load_corpus(directory):
    texts = []

    directory = Path(directory)

    for file_path in sorted(directory.rglob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        texts.append(text)

    return texts
```

The loader:

- searches for `.txt` files
- searches recursively through subdirectories
- reads files as UTF-8
- returns the documents as a list
- uses `sorted()` to make loading deterministic
  Therefore:

```
corpus/text/
       ↓
load_corpus()
       ↓
texts[]
```

---

## 20. Why deterministic ordering matters
We used:

```
sorted(directory.rglob("*.txt"))
```

rather than relying on filesystem ordering.

This means the same corpus produces the same document ordering between runs.

That becomes important when we eventually care about reproducible tokenizer/model training.

---

## 21. Corpus statistics
We added basic measurements:

```
total_characters = sum(len(text) for text in texts)
```

and:

```
Documents: 2
Characters: 181
```

This was our first real corpus measurement.

Instead of saying:

> "We have some training data."
> we can now say:

> "The tokenizer is currently working with 2 documents containing 181 characters."

---

# 22. Exact document deduplication
We identified a problem with large corpora:

```
document A
document A
document A
document B
```

Repeated documents artificially increase the frequency of their character/token pairs.

Since BPE chooses merges based on pair frequency, duplicated data can influence the learned vocabulary.

We therefore added exact deduplication.

```
def deduplicate_corpus(texts):
    unique_texts = []
    seen = set()

    for text in texts:
        if not text.strip():
            continue

        if text in seen:
            continue

        seen.add(text)
        unique_texts.append(text)

    return unique_texts
```

The pipeline is now:

```
.txt files
    ↓
load_corpus()
    ↓
remove empty documents
    ↓
remove exact duplicates
    ↓
clean corpus
    ↓
prepare_corpus()
    ↓
BPE
```

---

## 23. What our deduplication does
It removes **exact duplicates**.

For example:

```
Document 1:
The cat sat on the mat.

Document 2:
The cat sat on the mat.
```

becomes:

```
Document 1:
The cat sat on the mat.
```

But these remain separate:

```
The cat sat on the mat.

A cat sat on the mat.
```

because they are not identical strings.

We deliberately have **not** implemented semantic or near-duplicate detection yet.

---

# 24. Why we preserve whitespace
During our encoding/decoding test, we discovered:

```
The<UNK>quick<UNK>brown...
```

The problem was that our tokenizer had discarded spaces.

We changed the tokenizer so whitespace remains representable.

Our successful test now produces:

```
Input IDs:
[36, 4, 5, 6, 37, ...]
```

and:

```
Decoded text:
The quick brown fox jumps over the lazy dog.
The fox is fast and the dog is lazy.
```

Therefore:

```
TEXT
 ↓
BPE
 ↓
IDs
 ↓
DECODE
 ↓
SAME TEXT
```

works for our current corpus.

This established an important requirement:

> **The tokenizer must preserve enough information for decoding to reconstruct the original text.**

---

# 25. Current OutreachLM data architecture
We now have:

```
                    CORPUS
                      │
                      ▼
               .txt documents
                      │
                      ▼
                load_corpus()
                      │
                      ▼
              exact deduplication
                      │
                      ▼
             corpus statistics
                      │
                      ▼
              prepare_corpus()
                      │
                      ▼
                 BPE learner
                      │
             ┌────────┴────────┐
             ▼                 ▼
       vocabulary         merge rules
             │                 │
             └────────┬────────┘
                      ▼
                  encoding
                      │
                      ▼
                     IDs
                      │
                      ▼
                  decoding
                      │
                      ▼
                  original text
```

---

# 26. Current status

### BPE
**Working. ✅**

We have implemented:

- atomic units
- pair counting
- best-pair selection
- repeated merging
- merge ranks
- merge tokens
- vocabulary growth
- BPE application
- encoding
- decoding
- boundary preservation
- whitespace preservation

### Corpus system
**Working at prototype level. ✅**

We have:

- `.txt` loading
- recursive directory scanning
- deterministic file ordering
- document statistics
- empty-document removal
- exact duplicate removal

---

# Next: the real corpus
Now we're ready to stop inventing tiny sentences.

There are already large public corpora we can use. For example, Hugging Face hosts **FineWeb**, a large cleaned/deduplicated English web corpus, and **FineWeb-Edu**, an educationally filtered corpus. ([Hugging Face](https://huggingface.co/datasets?utm_source=chatgpt.com))

We **do not** want to download terabytes onto your laptop.

For our first real experiment, we'll take a **small, manageable subset** and convert it into the same local `.txt` format our current loader already understands.

That means our architecture doesn't change:

```
PUBLIC DATASET
      ↓
small local subset
      ↓
corpus/text/
      ↓
our loader
      ↓
our BPE
```

This is the important part: **we're not replacing the tokenizer with someone else's tokenizer.** We're using public text as training data for the tokenizer *we built*.