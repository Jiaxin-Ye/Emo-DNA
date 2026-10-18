from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    updated = deepcopy(config)
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated


