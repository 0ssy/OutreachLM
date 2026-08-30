from __future__ import annotations

from dataclasses import dataclass, field
import re


TRANSFER_PATTERN = re.compile(r"(\w+)\s+gave\s+(\w+)\s+the\s+(\w+)", re.IGNORECASE)
PRONOUN_TRANSFER_PATTERN = re.compile(r"(\w+)\s+gave\s+it\s+to\s+(\w+)", re.IGNORECASE)


@dataclass
class RelationshipTracker:
    ownership: dict[str, str] = field(default_factory=dict)
    transfers: list[tuple[str, str, str]] = field(default_factory=list)
    errors: int = 0

    def apply_sentence(self, sentence: str) -> None:
        text = sentence.strip().rstrip(".")
        direct = TRANSFER_PATTERN.search(text)
        if direct:
            source = direct.group(1)
            target = direct.group(2)
            item = direct.group(3)
            self.ownership[item] = target
            self.transfers.append((item, source, target))
            return
        pronoun = PRONOUN_TRANSFER_PATTERN.search(text)
        if pronoun and self.transfers:
            source = pronoun.group(1)
            target = pronoun.group(2)
            item = self.transfers[-1][0]
            self.ownership[item] = target
            self.transfers.append((item, source, target))
            return
        self.errors += 1

    def current_owner(self, item: str) -> str | None:
        return self.ownership.get(item)

