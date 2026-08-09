from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

corpus = [
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
    ["l", "o", "w", "e", "s", "t"]
]

result = tokenizer.learn_merges(
    corpus,
    num_merges=5
)

print("\nVocabulary:")

for token, token_id in tokenizer.vocab.items():
    print(token, "→", token_id)