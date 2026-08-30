from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SemanticTuple:
    actor: str
    action: str
    obj: str
    time: str
    confidence: float


ACTIVE_PATTERN = re.compile(r"(\w+)\s+is\s+(\w+ing)\s+(\w+)", re.IGNORECASE)
PASSIVE_PATTERN = re.compile(r"(\w+)\s+is\s+currently\s+being\s+(\w+ed)\s+by\s+(\w+)", re.IGNORECASE)


CANONICAL_ACTION = {
    "building": "build",
    "built": "build",
    "developing": "build",
    "developed": "build",
    "testing": "test",
    "tested": "test",
}


def _canonicalize_action(action: str) -> str:
    lowered = action.lower()
    if lowered in CANONICAL_ACTION:
        return CANONICAL_ACTION[lowered]
    if lowered.endswith("ing"):
        stem = lowered[:-3]
        if stem == "develop":
            return "build"
        return stem
    if lowered.endswith("ed"):
        stem = lowered[:-2]
        if stem == "develop":
            return "build"
        return stem
    if lowered == "develop":
        return "build"
    return lowered


def extract_tuple(text: str) -> SemanticTuple | None:
    cleaned = text.strip().rstrip(".")
    active = ACTIVE_PATTERN.search(cleaned)
    if active:
        return SemanticTuple(
            actor=active.group(1).lower(),
            action=_canonicalize_action(active.group(2)),
            obj=active.group(3).lower(),
            time="present",
            confidence=0.94,
        )
    passive = PASSIVE_PATTERN.search(cleaned)
    if passive:
        return SemanticTuple(
            actor=passive.group(3).lower(),
            action=_canonicalize_action(passive.group(2)),
            obj=passive.group(1).lower(),
            time="present",
            confidence=0.91,
        )
    return None


def tuple_similarity(left: SemanticTuple, right: SemanticTuple) -> float:
    score = 0.0
    score += 0.4 if left.actor == right.actor else 0.0
    score += 0.3 if left.action == right.action else 0.0
    score += 0.2 if left.obj == right.obj else 0.0
    score += 0.1 if left.time == right.time else 0.0
    return score
