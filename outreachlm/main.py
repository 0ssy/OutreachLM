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

print("Final corpus:")

for text in result:
    print(text)

print("\nMerge ranks:")
print(tokenizer.merge_ranks)

print("\nMerge tokens:")
print(tokenizer.merge_tokens)