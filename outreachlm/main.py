from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

pair_counts = {
    ("c", "d"): 5,
    ("a", "b"): 5,
    ("e", "f"): 2
}

best_pair = tokenizer.select_best_pair(pair_counts)

print("Best pair:", best_pair)