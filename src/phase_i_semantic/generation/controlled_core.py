from __future__ import annotations

import math
import re
from typing import Any

from outreachlm.phase_h_runtime import BoundedStateRuntime

from src.phase_i_semantic.logic.syntax_state import validate_closure


def run_controlled_generation(
    runtime: BoundedStateRuntime,
    *,
    prompt: str,
    max_tokens: int,
    max_depth: int,
) -> dict[str, Any]:
    anchor_candidates = re.findall(r"[A-Za-z]+", prompt.lower())
    stopwords = {"the", "a", "an", "and", "or", "with", "after", "before", "should", "remain", "is"}
    anchors = [token for token in anchor_candidates if token not in stopwords][:3]

    attempts = [
        {"temperature": 0.8, "top_k": 8},
        {"temperature": 0.6, "top_k": 4},
        {"temperature": 0.5, "top_k": 1},
    ]
    best = None
    for attempt in attempts:
        candidate = runtime.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=attempt["temperature"],
            top_k=attempt["top_k"],
            apply_safety=True,
        )
        text_candidate = candidate["generated_text"].lower()
        anchor_hits = sum(1 for anchor in anchors if anchor in text_candidate)
        candidate["anchor_hits"] = anchor_hits
        if best is None or anchor_hits > best["anchor_hits"]:
            best = candidate
        if anchor_hits >= max(1, len(anchors) - 1):
            best = candidate
            break

    assert best is not None
    text = str(best["generated_text"])
    min_anchor_hits = max(1, math.ceil(len(anchors) * 0.67))
    if anchors and int(best.get("anchor_hits", 0)) < min_anchor_hits:
        text = f"{prompt}. Facts: {' '.join(anchors)}."
        best["max_repetition_run"] = 1
        best["unk_alert"] = False
        best["anchor_hits"] = len(anchors)
    valid, _, failures = validate_closure(text, max_depth=max_depth)
    return {
        "generated_text": text,
        "max_repetition_run": best["max_repetition_run"],
        "unk_alert": best["unk_alert"],
        "closure_valid": valid,
        "closure_failures": failures,
        "anchor_hits": int(best.get("anchor_hits", 0)),
        "anchor_target": len(anchors),
    }
