import re

class Tokenizer:
    """"
    Version 1
    splits texts into spaces
    """
    def tokenize(self, text):
        return  re.findall(r"\w+|[.,!?,:;]", text)