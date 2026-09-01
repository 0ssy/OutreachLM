from __future__ import annotations

from .pointer import PointerResolution, detect_query_anchor, extract_link_facts, resolve_pointer, resolve_transitive_target
from .relation_engine import RelationGraph, TransitiveResult, parse_clause, resolve_query

__all__ = [
    "PointerResolution",
    "detect_query_anchor",
    "extract_link_facts",
    "resolve_pointer",
    "resolve_transitive_target",
    "RelationGraph",
    "TransitiveResult",
    "parse_clause",
    "resolve_query",
]
