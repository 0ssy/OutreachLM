from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

corpus = [
    ["l", "o", "w"],
    ["l", "o", "w", "e", "r"],
    ["l", "o", "w", "e", "s", "t"]
]

counts = tokenizer.count_pairs(corpus)

print("Pair counts:")

for pair, count in counts.items():
    print(pair, "→", count)