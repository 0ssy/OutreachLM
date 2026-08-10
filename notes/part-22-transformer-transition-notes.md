## NEXT — Build the Training Dataset Pipeline
You have now crossed an important boundary.

Your tokenizer is working, your corpus loader is working, and your dataset test confirms that token IDs can be converted into **training examples**.

The output you just showed:

```
Total tokens: 20
Context length: 5
Dataset length: 15

Example 0
Input : [0, 1, 2, 3, 4]
Target: [1, 2, 3, 4, 5]
```

is the basic mechanism behind **next-token prediction**.

### 1. What the model is actually learning
Given:

```
[0, 1, 2, 3, 4]
```

the model must predict:

```
[1, 2, 3, 4, 5]
```

So at each position:

```
0 → 1
0,1 → 2
0,1,2 → 3
0,1,2,3 → 4
0,1,2,3,4 → 5
```

This is the fundamental learning objective we're building OutreachLM around.

genui{"functions_sequences_graphs":{"type_id":"GRAPHABLE_FUNCTION","content":"y=x+1"}}

The important distinction is that **the model isn't trained to memorize the entire target sequence as one answer**. During training, every position provides a prediction target.

---

# 2. The next component: the model
So far we have:

```
RAW TEXT
   ↓
CORPUS LOADER
   ↓
BPE TOKENIZER
   ↓
TOKEN IDs
   ↓
TRAINING DATASET
```

Now we add:

```
TRAINING DATASET
       ↓
   TRANSFORMER
       ↓
 PREDICT NEXT TOKEN
```

This is where OutreachLM becomes an actual language model rather than a tokenizer/data-processing system.

And **we are not jumping straight into a giant Transformer**.

We're going to build the smallest useful Transformer ourselves.

---

# 3. First component: token embeddings
The model receives IDs such as:

```
[1, 2, 3, 4, 5]
```

But neural networks don't understand the semantic meaning of ID `1`.

An ID is simply an index.

So we create an **embedding table**.

For example:

```
token 1 → [ 0.12, -0.43,  0.77, ...]
token 2 → [-0.51,  0.18,  0.04, ...]
token 3 → [ 0.31,  0.92, -0.27, ...]
```

If our vocabulary has:

```
500 tokens
```

and our embedding dimension is:

```
128
```

we create:

```
500 × 128
```

parameters for the embedding matrix.

The token ID selects a row.

Mathematically:

```
E ∈ R^(vocab_size × embedding_dim)
```

and:

```
embedding = E[token_id]
```

---

# 4. Why embeddings exist
Suppose:

```
"cat"
"dog"
"car"
```

Initially their vectors are random.

During training, however, the model adjusts them.

Eventually we want relationships to emerge such as:

```
cat ─── dog
 │
 │
animal
```

and:

```
car ─── truck
```

The model isn't explicitly told:

> "cats and dogs are animals."
The representation is shaped by the training objective.

This is our first glimpse of **representation learning**.

---

# 5. But there's a problem
Suppose the input is:

```
the dog chased the cat
```

The embeddings tell us what tokens are present.

But they don't tell us **where the tokens occur**.

Compare:

```
dog bites man
```

with:

```
man bites dog
```

Same words.

Different meaning.

Therefore we need **positional information**.

---

# 6. Token embeddings + positional embeddings
For every position we create another vector.

For example:

```
position 0 → P0
position 1 → P1
position 2 → P2
position 3 → P3
position 4 → P4
```

Then:

```
input representation
=
token embedding
+
position embedding
```

So the Transformer receives something like:

```
Token:
the  → vector

Position:
0    → vector

               ↓

Combined representation
```

For a sequence:

```
[the, dog, chased, the, cat]
```

we therefore have:

```
X =
[
 token(the)    + position(0)
 token(dog)    + position(1)
 token(chased) + position(2)
 token(the)    + position(3)
 token(cat)    + position(4)
]
```

---

# 7. Then comes the important part
The Transformer needs to determine:

> Which previous tokens should each token pay attention to?
That's **self-attention**.

For example:

```
The cat sat on the mat because it was tired.
                                      ↑
```

When processing:

```
it
```

the model needs to determine what `it` refers to.

Attention allows the representation of `it` to incorporate information from relevant previous tokens.

This is the next major subsystem we will implement.

---

# 8. Our architecture from here
We're going to build:

```
                 OutreachLM
                     │
             ┌───────┴───────┐
             │               │
        Token Embedding   Position Embedding
             │               │
             └───────┬───────┘
                     ↓
                Transformer
                   Block
                     │
             ┌───────┴───────┐
             │               │
        Self-Attention       │
             │               │
             └───────┬───────┘
                     ↓
              Feed Forward
                     ↓
                 Output
                     ↓
              Vocabulary logits
                     ↓
              Next-token prediction
```

Eventually:

```
logits
  ↓
softmax
  ↓
probability for every token
  ↓
select/sample next token
```

---

# 9. What we're implementing next
**Next code step: `model.py` — token embeddings.**

We will create the first neural component and test it independently before adding attention.

The progression will be:

```
1. TokenEmbedding
2. PositionalEmbedding
3. Self-Attention
4. Causal Mask
5. Multi-Head Attention
6. Feed-Forward Network
7. Transformer Block
8. Language Model Head
9. Loss
10. Backpropagation
11. Training loop
12. Checkpointing
13. Text generation
```

We are **not** going back to BPE or corpus preparation. Those layers are already working.

### Notes — up to this point

```
OUTREACHLM NOTES
==================================================

1. BPE TOKENIZATION
--------------------------------------------------
- Built a working BPE tokenizer.
- Started with character-level vocabulary.
- Learned merge pairs from corpus statistics.
- Stored merge ranks.
- Applied merges repeatedly during encoding.
- Converted BPE pieces → token IDs.
- Implemented token ID → text decoding.
- Verified round-trip encoding/decoding.

2. CORPUS
--------------------------------------------------
- Built corpus loader.
- Loaded local text files.
- Downloaded/processed FineWeb documents.
- Successfully saved approximately 1000 documents.
- Corpus directory contains 1003 .txt files.
- Dataset can be read by the tokenizer.

3. TOKENIZATION OF REAL DATA
--------------------------------------------------
- Successfully tokenized real corpus text.
- Large token sequences are produced.
- Decoding successfully reconstructs readable text.
- Therefore tokenizer + vocabulary + corpus pipeline works.

4. DATASET
--------------------------------------------------
- Converted token stream into supervised examples.
- Uses a context window.
- Example:

  Input:
  [0, 1, 2, 3, 4]

  Target:
  [1, 2, 3, 4, 5]

- This implements next-token prediction training data.

5. CURRENT ARCHITECTURE
--------------------------------------------------

Raw Text
   ↓
Corpus Loader
   ↓
BPE Tokenizer
   ↓
Token IDs
   ↓
Training Dataset
   ↓
[CURRENT STEP]
Transformer Language Model
   ↓
Predicted Next Token

6. CURRENTLY NOT BUILT
--------------------------------------------------
- Embedding layer
- Positional representation
- Self-attention
- Causal attention mask
- Multi-head attention
- Feed-forward network
- Transformer blocks
- Output projection
- Softmax
- Cross-entropy loss
- Backpropagation training
- Optimizer
- Checkpointing
- Generation/sampling

7. CURRENT MILESTONE
--------------------------------------------------
Tokenizer:             WORKING
Corpus loader:         WORKING
Corpus storage:        WORKING
Dataset construction:  WORKING
Transformer:           NEXT
Training:              NOT STARTED
Generation:            NOT STARTED
```

**Next: build `model.py` and implement the token embedding layer from scratch.**
