from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")


@dataclass(frozen=True)
class StupidTokenizer:
    token_to_id: dict[str, int]

    @property
    def id_to_token(self) -> list[str]:
        tokens = [""] * len(self.token_to_id)
        for token, token_id in self.token_to_id.items():
            tokens[token_id] = token
        return tokens

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for token in text.strip().split():
            ids.append(self.token_to_id.get(token, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        id_to_token = self.id_to_token
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(id_to_token):
                token = "<UNK>"
            else:
                token = id_to_token[token_id]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return " ".join(pieces)

    def save(self, path: str | Path) -> None:
        payload = {
            "token_to_id": self.token_to_id,
            "special_tokens": list(SPECIAL_TOKENS),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StupidTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        token_to_id = payload["token_to_id"]
        for token in SPECIAL_TOKENS:
            if token not in token_to_id:
                raise ValueError(f"Tokenizer file missing required special token: {token}")
        return cls(token_to_id=token_to_id)


def build_stupid_tokenizer_from_lines(lines: list[str]) -> StupidTokenizer:
    token_to_id: dict[str, int] = {}
    for line in lines:
        for token in line.strip().split():
            if token and token not in token_to_id:
                token_to_id[token] = len(token_to_id)
    for token in SPECIAL_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    return StupidTokenizer(token_to_id=token_to_id)


def build_stupid_tokenizer_from_file(path: str | Path) -> StupidTokenizer:
    corpus = Path(path).read_text(encoding="utf-8").splitlines()
    return build_stupid_tokenizer_from_lines(corpus)
