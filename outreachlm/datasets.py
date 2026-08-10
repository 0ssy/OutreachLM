class LanguageModelDataset:
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
    def __len__(self):
        return len(self.token_ids) - self.context_length
    def __getitem__(self, index):
        x = self.token_ids[index:index + self.context_length]
        y = self.token_ids[index + 1:index + self.context_length + 1]
        return x, y