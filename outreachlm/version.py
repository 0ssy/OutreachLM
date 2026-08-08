import re

class Tokenizer:
    """"
    Version 3
    splits texts into spaces
    Builds a vocabulary of unique tokens
    converts tokens to IDs and vice versa
    uses <UNK> for unkown tokens
    """

    def __init__(self):
        self.vocab ={
            "<UNK>": 0
        }

    def tokenize(self, text):
        return  re.findall(r"\w+|[.,!?,:;]", text)

    def build_vocab(self,texts):
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, text):
        tokens = self.tokenize(text)

        return [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]