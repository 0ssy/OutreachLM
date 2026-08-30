from __future__ import annotations


OPENERS = "([{"
CLOSERS = ")]}"
MATCH = {")": "(", "]": "[", "}": "{"}


def validate_closure(text: str, *, max_depth: int) -> tuple[bool, int, int]:
    stack: list[str] = []
    max_seen_depth = 0
    failures = 0
    for ch in text:
        if ch in OPENERS:
            stack.append(ch)
            max_seen_depth = max(max_seen_depth, len(stack))
            if len(stack) > max_depth:
                failures += 1
        elif ch in CLOSERS:
            if not stack or stack[-1] != MATCH[ch]:
                failures += 1
            else:
                stack.pop()
    if stack:
        failures += len(stack)
    return failures == 0, max_seen_depth, failures


def classify_closure(text: str, *, max_depth: int, expected_valid: bool) -> tuple[bool, int, int]:
    is_valid, depth, failures = validate_closure(text, max_depth=max_depth)
    classification_correct = is_valid == expected_valid
    return classification_correct, depth, failures
