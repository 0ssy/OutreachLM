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
        # Initialize the vocabulary with a special token for unknown words
        self.vocab ={
            "<UNK>": 0
        }
        # Initialize the merge ranks dictionary
        self.merge_ranks = {}
        self.merge_tokens ={}

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

        ids = []

        for token in tokens:
            token_id = self.vocab.get(token, self.vocab["<UNK>"])
            ids.append(token_id)

        return ids
        # Find the best merge pair
    def find_best_merge(self, tokens):  #decison maker which tells us the hihest priority pair to merge based on the ranks
        best_pair = None
        best_rank = None

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])

            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]

                print("found rank", rank)
                print("current best rank", best_rank)

                if best_rank is None or rank < best_rank:
                    print("New best")
                    best_pair = pair
                    best_rank = rank

        print("Final best pair", best_pair)
        print("Final rank", best_rank)

        return best_pair
#a repeated merge function that merges the best pair until no more pairs can be merged
    def merge_pair(self, tokens, pair, new_token):
        result = []
        i = 0

        while i < len(tokens):
            if(
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
                

        return result
    #a repeated merge function that merges the best pair until no more pairs can be merged
    def apply_merges(self, tokens):
        while True:
            pair = self.find_best_merge(tokens)
            if pair is None:
                break
            new_token = self.merge_tokens[pair]
            tokens = self.merge_pair(tokens, pair, new_token)
        return tokens
    #a function that counts the number of times each pair of tokens appears in the corpus
    def count_pairs(self, corpus):
        pair_counts = {}

        for text in corpus:
            for i in range(len(text) - 1):
                pair = (text[i], text[i + 1])
                if pair not in pair_counts:
                    pair_counts[pair] = 0
                pair_counts[pair] += 1
        return pair_counts