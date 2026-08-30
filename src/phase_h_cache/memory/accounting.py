from __future__ import annotations

from collections.abc import Mapping
import sys
from typing import Any

import psutil


def process_memory_bytes() -> dict[str, int]:
    info = psutil.Process().memory_info()
    return {"rss_bytes": int(info.rss), "vms_bytes": int(info.vms)}


def deep_sizeof_bytes(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += deep_sizeof_bytes(key, seen)
            size += deep_sizeof_bytes(value, seen)
        return size
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_sizeof_bytes(item, seen)
        return size
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            size += deep_sizeof_bytes(key, seen)
            size += deep_sizeof_bytes(value, seen)
        return size
    return size

