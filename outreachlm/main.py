from outreachlm.version import Tokenizer
tokenizer = Tokenizer()

tokens = ["a", "b", "c", "d"]
pair = ("c","d")
new_token = "cd"

result = tokenizer.merge_pair(
    tokens,
    pair,
    new_token
)

print("Original:", tokens)
print("Pair:", pair)
print("New token:", new_token)
print("Result:", result)