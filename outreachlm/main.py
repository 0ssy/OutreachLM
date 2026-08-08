from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

tokenizer.merge_ranks = {
    ("a", "b"): 2,
    ("b", "c"): 5,
    ("c", "d"): 1
}

tokens = ["a", "b", "c", "d"]

best = tokenizer.find_best_merge(tokens)

print("Tokens:", tokens)
print("Merge ranks:", tokenizer.merge_ranks)
print("Best merge:", best)