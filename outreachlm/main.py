from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

tokenizer.merge_ranks = {
    ("a", "b"): 2,
    ("b", "c"): 5,
    ("c", "d"): 1
}

tokens = ["a", "b", "c", "d"]

print("Tokens:", tokens)
print("Merge ranks:", tokenizer.merge_ranks)

for pair, rank in tokenizer.merge_ranks.items():
    print("PAIR:", pair, "RANK:", rank)

best = tokenizer.find_best_merge(tokens)

print("Best merge:", best)