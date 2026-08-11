class CharacterTokenizer:

    def __init__(self, text):
        characters = sorted(set(text))

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        self.tokens = [
            self.pad_token,
            self.unk_token,
            *characters
        ]

        self.token_to_id = {
            token: index
            for index, token in enumerate(self.tokens)
        }

        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text):

        return [
            self.token_to_id.get(
                character,
                self.token_to_id[self.unk_token]
            )
            for character in text
        ]

    def decode(self, token_ids):

        output = []

        for token_id in token_ids:

            token = self.id_to_token[token_id]

            if token in (
                self.pad_token,
                self.unk_token
            ):
                continue

            output.append(token)

        return "".join(output)