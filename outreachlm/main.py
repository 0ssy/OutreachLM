from outreachlm.version import Tokenizer


tokenizer = Tokenizer()

training_data = [
    "I love TRS.",
    "TRS loves TerraNode.",
]

tokenizer.build_vocab(training_data)

print(tokenizer.vocab)

print(tokenizer.encode("I love TRS."))
print(tokenizer.encode("I love elephants."))