from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

corpus = [
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
    ["l", "o", "w", "e", "s", "t"]
]

result = tokenizer.learn_merges(
    corpus,
    target_vocab_size=12
)

print("Vocabulary size:", len(tokenizer.vocab))

print("\nVocabulary:")

for token, token_id in tokenizer.vocab.items():
    print(token, "→", token_id)

print("\nMerge ranks:")
print(tokenizer.merge_ranks)

print("\nFinal corpus:")
for text in result:
    print(text)