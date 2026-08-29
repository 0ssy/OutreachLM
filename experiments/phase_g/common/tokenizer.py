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
        output = [""] * len(self.token_to_id)
        for token, token_id in self.token_to_id.items():
            output[token_id] = token
        return output

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        output: list[int] = []
        if add_bos:
            output.append(self.bos_id)
        for token in text.strip().split():
            output.append(self.token_to_id.get(token, self.unk_id))
        if add_eos:
            output.append(self.eos_id)
        return output

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        id_to_token = self.id_to_token
        words: list[str] = []
        for token_id in token_ids:
            token = id_to_token[token_id] if 0 <= token_id < len(id_to_token) else "<UNK>"
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"token_to_id": self.token_to_id, "special_tokens": list(SPECIAL_TOKENS)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "StupidTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        token_to_id = payload["token_to_id"]
        for special in SPECIAL_TOKENS:
            if special not in token_to_id:
                raise ValueError(f"Missing required special token: {special}")
        return cls(token_to_id=token_to_id)


def build_stupid_tokenizer_from_lines(lines: list[str]) -> StupidTokenizer:
    token_to_id: dict[str, int] = {}
    for line in lines:
        for token in line.strip().split():
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
    for special in SPECIAL_TOKENS:
        token_to_id.setdefault(special, len(token_to_id))
    return StupidTokenizer(token_to_id=token_to_id)
