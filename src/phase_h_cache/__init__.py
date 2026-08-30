from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PhaseHConfig:
    raw: dict[str, Any]

    @classmethod
    def load_default(cls) -> "PhaseHConfig":
        config_path = Path(__file__).resolve().parent / "config.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Phase H config must parse to a mapping.")
        return cls(raw=payload)

