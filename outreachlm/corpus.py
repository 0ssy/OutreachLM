from pathlib import Path


class TextCorpus:

    def __init__(self, text_path):
        self.text_path = Path(text_path)

    def load(self):
        if not self.text_path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {self.text_path}"
            )

        text = self.text_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            raise ValueError(
                "Corpus is empty."
            )

        return text