from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RelationGraph:
    edges: dict[str, set[str]] = field(default_factory=dict)

    def add(self, left: str, right: str) -> None:
        self.edges.setdefault(left, set()).add(right)

    def has_direct(self, left: str, right: str) -> bool:
        return right in self.edges.get(left, set())

    def infer_transitive(self, left: str, right: str) -> bool:
        visited: set[str] = set()
        stack = [left]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for nxt in self.edges.get(current, set()):
                if nxt == right:
                    return True
                stack.append(nxt)
        return False

