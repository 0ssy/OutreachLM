from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_phase_i_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "config.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase I config must parse to an object.")
    return payload

