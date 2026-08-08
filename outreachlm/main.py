from outreachlm.version import Tokenizer

tokenizer = Tokenizer()
tokens = tokenizer.tokenize("I love TRS")
print(tokens)
tokens = tokenizer.tokenize("Hello Chatty")
print(tokens)