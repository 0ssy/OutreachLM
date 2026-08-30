from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MultiTierContextHierarchy:
    local_window: list[str] = field(default_factory=list)
    sentence_memory: list[list[str]] = field(default_factory=list)
    document_memory: list[str] = field(default_factory=list)
    long_term_memory: dict[str, str] = field(default_factory=dict)

    def ingest_tokens(self, tokens: list[str], *, max_local: int = 256) -> None:
        self.local_window.extend(tokens)
        if len(self.local_window) > max_local:
            self.local_window = self.local_window[-max_local:]
        self.sentence_memory.append(tokens)
        self.document_memory.extend(tokens)
        for idx, token in enumerate(tokens):
            if token.startswith("KEY_") and idx + 1 < len(tokens):
                self.long_term_memory[token] = tokens[idx + 1]

    def recall(self, key: str) -> str | None:
        return self.long_term_memory.get(key)

