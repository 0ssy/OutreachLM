from pathlib import Path

from outreachlm.version import Tokenizer


# ============================================================
# CORPUS LOADER
# ============================================================

def load_corpus(directory):
    texts = []

    directory = Path(directory)

    for file_path in directory.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        texts.append(text)

    return texts


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


# ============================================================
# CREATE TOKENIZER
# ============================================================

tokenizer = Tokenizer()


# ============================================================
# LOAD TRAINING CORPUS
# ============================================================

texts = load_corpus("corpus/fineweb")
texts = deduplicate_corpus(texts)


print("=" * 50)
print("TRAINING CORPUS")
print("=" * 50)

print("Documents:", len(texts))

for i, text in enumerate(texts, start=1):
    print()
    print(f"Document {i}:")
    print(text)


total_characters = sum(len(text) for text in texts)

print()
print("=" * 50)
print("CORPUS STATISTICS")
print("=" * 50)

print("Documents:", len(texts))
print("Characters:", total_characters)


# ============================================================
# PREPARE CORPUS FOR BPE
# ============================================================

corpus = tokenizer.prepare_corpus(texts)


print()
print("=" * 50)
print("PREPARED CORPUS")
print("=" * 50)

for text in corpus:
    print(text)


# ============================================================
# LEARN BPE MERGES
# ============================================================

corpus = tokenizer.learn_merges(
    corpus,
    num_merges=20
)


# ============================================================
# VOCABULARY
# ============================================================

print()
print("=" * 50)
print("VOCABULARY")
print("=" * 50)

print("Vocabulary size:", len(tokenizer.vocab))

for token, token_id in tokenizer.vocab.items():
    print(f"{token} → {token_id}")


# ============================================================
# MERGE RANKS
# ============================================================

print()
print("=" * 50)
print("MERGE RANKS")
print("=" * 50)

for pair, rank in tokenizer.merge_ranks.items():
    print(f"{pair} → rank {rank}")


# ============================================================
# MERGE TOKENS
# ============================================================

print()
print("=" * 50)
print("MERGE TOKENS")
print("=" * 50)

for pair, token in tokenizer.merge_tokens.items():
    print(f"{pair} → {token}")


# ============================================================
# FINAL CORPUS
# ============================================================

print()
print("=" * 50)
print("FINAL CORPUS")
print("=" * 50)

for text in corpus:
    print(text)


# ============================================================
# BPE ENCODING TEST
# ============================================================

test_text = texts[0]

print()
print("=" * 50)
print("BPE ENCODING TEST")
print("=" * 50)

print("Input:", test_text)

pieces = tokenizer.apply_bpe(test_text)

print("BPE pieces:", pieces)


# ============================================================
# TOKEN → ID
# ============================================================

print()
print("=" * 50)
print("TOKEN → ID")
print("=" * 50)

ids = []

for piece in pieces:

    token_id = tokenizer.vocab.get(
        piece,
        tokenizer.vocab["<UNK>"]
    )

    ids.append(token_id)

    print(f"{piece} → {token_id}")


print()
print("Encoded IDs:", ids)


# ============================================================
# DECODE TEST
# ============================================================

print()
print("=" * 50)
print("DECODE TEST")
print("=" * 50)

decoded_tokens = []

for token_id in ids:

    for token, vocabulary_id in tokenizer.vocab.items():

        if vocabulary_id == token_id:
            decoded_tokens.append(token)
            break


decoded_text = "".join(decoded_tokens)

print("Input IDs:", ids)
print("Decoded text:", decoded_text)