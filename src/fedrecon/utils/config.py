from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_by_path(config: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = config
    for part in dotted_path.split("."):
        cur = cur[part]
    return cur


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    cur: Any = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value
