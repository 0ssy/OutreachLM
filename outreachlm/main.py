from outreachlm.version import Tokenizer


# ============================================================
# CREATE TOKENIZER
# ============================================================

tokenizer = Tokenizer()


# ============================================================
# TRAINING TEXT
# ============================================================

texts = [
    "low",
    "lower",
    "lowest!",
    "lower."
]


# ============================================================
# PREPARE CORPUS FOR BPE
# ============================================================

corpus = tokenizer.prepare_corpus(texts)

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
    num_merges=5
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

test_text = "lowest!"

print()
print("=" * 50)
print("BPE ENCODING TEST")
print("=" * 50)

print("Input:", test_text)

pieces = tokenizer.apply_bpe(test_text)

print("BPE pieces:", pieces)


# ============================================================
# TOKEN ID TEST
# ============================================================

print()
print("=" * 50)
print("TOKEN ID TEST")
print("=" * 50)

ids = []

for piece in pieces:
    token_id = tokenizer.vocab.get(
        piece,
        tokenizer.vocab["<UNK>"]
    )

    ids.append(token_id)

print("Input:", test_text)
print("BPE pieces:", pieces)
print("Encoded IDs:", ids)


# ============================================================
# TOKEN → ID
# ============================================================

print()
print("=" * 50)
print("TOKEN → ID")
print("=" * 50)

for piece in pieces:
    token_id = tokenizer.vocab.get(
        piece,
        tokenizer.vocab["<UNK>"]
    )

    print(f"{piece} → {token_id}")


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