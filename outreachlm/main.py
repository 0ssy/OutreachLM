from outreachlm.version import Tokenizer

tokenizer = Tokenizer()

print(tokenizer.tokenize("I love TRS"))
print(tokenizer.tokenize("Hello Chatty"))
print(tokenizer.tokenize("TRS: TerraNode"))
