# OutreachLM Notes — Part 1: Foundations

## 1. What is an LLM?

An LLM (Large Language Model) is a neural network that learns patterns in language from large amounts of text.

Its fundamental task is:

> **Given some text, predict what comes next.**
> Examples:

```text
The capital of Kenya is ...
```

↓

```text
Nairobi
```

or

```text
I would like to ...
```

↓

The model predicts the most likely continuation.

It does **not** understand language like humans do. It learns statistical patterns from examples.

---

## 2. Computers do not understand text
Computers only work with numbers.

Text must be transformed before a model can use it.

Pipeline:

```text
Text
   ↓
Tokenizer
   ↓
Token IDs (integers)
   ↓
Neural Network
```

Example:

```text
Hello
```

↓

```text
384
```

The model learns from the ID, not the word itself.

---

## 3. What is a tokenizer?
A tokenizer converts text into tokens and assigns every token an integer ID.

Example:

Vocabulary

```text
Hello → 1
world → 2
```

Sentence

```text
Hello world
```

↓

```text
1 2
```

A tokenizer does **not** know grammar, meaning, or context.

Its only job is to convert text into IDs.

---

## 4. What is a token?
A token is:

> **The smallest unit chosen by the tokenizer to receive one ID.**
> A token can be:

- a word
- part of a word
- a character
- punctuation
- even a single byte
Examples:

```text
Hello
```

↓

One token

or

```text
Hel
lo
```

↓

Two tokens

The tokenizer decides.

---

## 5. Vocabulary
The vocabulary is the complete list of tokens the tokenizer knows.

Example:

```text
I
love
TRS
Terra
Node
```

Every vocabulary entry has an integer ID.

Example:

```text
I → 1

love → 2

TRS → 3
```

The vocabulary size is fixed after training.

---

## 6. Why not one token per word?

Because language is infinite.

Examples:

- elephants
- OutreachLM
- TerraNode
- misspellings
- new programming languages
A word-level vocabulary would constantly grow.

---

## 7. The Out-of-Vocabulary (OOV) problem
Suppose the vocabulary is

```text
I
love
cats
```

Now we receive

```text
I love elephants
```

The tokenizer has never seen:

```text
elephants
```

Possible solutions:

- `<UNK>` (unknown token)
- split into smaller pieces
- dynamic vocabulary
- characters
Modern LLMs avoid losing information by using subword tokenization.

---

## 8. Tokenization with subwords

Instead of

```text
elephants
```

↓

```text
<UNK>
```

we tokenize

```text
elephant
+s
```

or

```text
ele
phant
s
```

Every piece already exists in the vocabulary.

This allows any word to be represented.

---

## 9. Compression, not grammar

This is one of the biggest ideas we've covered.

Humans tokenize using meaning.

Modern tokenizers tokenize using **statistics**.

The tokenizer asks:

> Which merges reduce the total number of tokens across the dataset?
> It is solving a compression problem.

Meaning comes later, inside the neural network.

---

## 10. Byte Pair Encoding (BPE)

BPE begins with very small symbols (characters or bytes).

Example:

```text
l o w

l o w e r

l o w e s t
```

Count adjacent pairs.

```text
l o = 3

o w = 3

w e = 2
```

Merge the most common pair.

```text
l + o

↓

lo
```

Repeat.

Over many iterations:

```text
l

↓

lo

↓

low

↓

lower
```

The vocabulary grows automatically from data.

---

## 11. Important realization

The tokenizer is **not** the intelligence.

It is simply preparing data.

The neural network will later learn:

- meaning
- grammar
- reasoning
- patterns
The tokenizer only converts text into a language the computer can process.

---

## Mental Model

Think of the system like a factory:

```text
Raw Text
     │
     ▼
Tokenizer
     │
Token IDs
     │
     ▼
Embeddings
     │
Vectors
     │
     ▼
Transformer
     │
Prediction
```

Today we've only studied the **first machine in the factory**: the tokenizer.

---
