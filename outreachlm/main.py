from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

tokenizer.merge_ranks = {
    ("a", "b"): 0,
    ("ab", "c"): 1
}

tokenizer.merge_tokens = {
    ("a", "b"): "ab",
    ("ab", "c"): "abc"
}

tokens = ["a", "b", "c"]

print("Original:", tokens)

result = tokenizer.apply_merges(tokens)

print("Final:", result)

tokens = ["a", "b", "a", "b"]

result = tokenizer.apply_merges(tokens)

print("Repeated pair:", result)