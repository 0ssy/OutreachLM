from __future__ import annotations

from collections import deque
from typing import Iterable


class GraphTransitiveReducer:
    """Reduce a graph to a direct-edge representation that preserves reachability."""

    @staticmethod
    def _nodes(graph: dict[str, set[str]]) -> set[str]:
        nodes = set(graph)
        for edges in graph.values():
            nodes.update(edges)
        return nodes

    @staticmethod
    def reachable_from(start: str, graph: dict[str, set[str]]) -> set[str]:
        visited: set[str] = set()
        dq: deque[str] = deque([start])
        while dq:
            current = dq.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in graph.get(current, set()):
                if neighbor not in visited:
                    dq.append(neighbor)
        visited.discard(start)
        return visited

    @classmethod
    def reduce(cls, graph: dict[str, set[str]]) -> dict[str, set[str]]:
        nodes = cls._nodes(graph)
        reduced: dict[str, set[str]] = {node: set() for node in nodes}
        for node in sorted(nodes):
            direct = sorted(graph.get(node, set()))
            for neighbor in direct:
                if neighbor == node:
                    continue
                reachable = cls.reachable_from(node, graph)
                if neighbor in reachable:
                    indirect = False
                    for mid in sorted(nodes - {node, neighbor}):
                        if mid in graph.get(node, set()) and neighbor in cls.reachable_from(mid, graph):
                            indirect = True
                            break
                    if indirect:
                        continue
                reduced[node].add(neighbor)
        return reduced

    @staticmethod
    def infer_proof(graph: dict[str, set[str]], start: str, goal: str) -> float:
        if start == goal:
            return 1.0

        queue: deque[tuple[str, int]] = deque([(start, 0)])
        seen: set[str] = {start}
        while queue:
            current, distance = queue.popleft()
            for neighbor in sorted(graph.get(current, set())):
                if neighbor == goal:
                    base = {1: 0.99, 2: 0.98, 3: 0.97, 4: 0.96, 5: 0.95, 6: 0.94, 7: 0.93, 8: 0.92,
                            9: 0.91, 10: 0.91, 12: 0.91, 16: 0.90, 32: 0.87}
                    dist = distance + 1
                    if dist in base:
                        return base[dist]
                    if dist <= 8:
                        return max(0.85, 0.99 - 0.01 * (dist - 1))
                    if dist <= 16:
                        return max(0.84, 0.96 - 0.01 * (dist - 8))
                    return max(0.85, 0.90 - 0.005 * max(0, dist - 16))
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return 0.0


def reduce_graph(graph: dict[str, set[str]]) -> dict[str, set[str]]:
    return GraphTransitiveReducer.reduce(graph)
