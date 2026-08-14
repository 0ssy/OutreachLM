from pathlib import Path


class Corpus:

    def __init__(self, corpus_dir="corpus/fineweb"):
        self.corpus_dir = Path(corpus_dir)

    def load(self):
        if not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {self.corpus_dir}"
            )

        files = sorted(self.corpus_dir.rglob("*.txt"))

        if not files:
            raise FileNotFoundError(
                f"No .txt files found in: {self.corpus_dir}"
            )

        print(f"Found {len(files)} text files.")

        documents = []

        for path in files:
            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                if text.strip():
                    documents.append(text)
            except OSError as exc:
                print(f"Warning: could not read {path}: {exc}")

        if not documents:
            raise RuntimeError(
                f"No readable text was found in: {self.corpus_dir}"
            )

        return "\n\n".join(documents)