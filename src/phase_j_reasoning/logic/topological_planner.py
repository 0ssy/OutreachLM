from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DAGPlan:
    tasks: tuple[str, ...]
    dependencies: dict[str, tuple[str, ...]]


class TopologicalPlanner:
    """Apply strict dependency ordering before execution to remove invalid action ordering."""

    @staticmethod
    def plan(tasks: dict[str, list[str]], *, goal: str | None = None) -> DAGPlan:
        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree: dict[str, int] = {task: 0 for task in tasks}
        for task, deps in tasks.items():
            for dep in deps:
                adjacency[dep].add(task)
                indegree[task] = indegree.get(task, 0) + 1
        queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for child in sorted(adjacency.get(current, set())):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(tasks):
            raise ValueError("Cyclic dependency graph detected; planning requires a DAG.")
        if goal is not None and goal not in ordered:
            raise ValueError(f"Goal '{goal}' not found in task graph.")
        return DAGPlan(tasks=tuple(ordered), dependencies={task: tuple(deps) for task, deps in tasks.items()})

    @staticmethod
    def validate(plan: DAGPlan, required_goal: str | None = None) -> bool:
        for task, deps in plan.dependencies.items():
            if task == required_goal:
                continue
            for dep in deps:
                if dep not in plan.tasks:
                    return False
        return True

    @staticmethod
    def dependency_violation_count(tasks: dict[str, list[str]]) -> int:
        try:
            TopologicalPlanner.plan(tasks)
            return 0
        except ValueError:
            return 1
