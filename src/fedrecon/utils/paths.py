from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: str | Path | None = None) -> Path:
    if start is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start).resolve()

    if start_path.is_file():
        start_path = start_path.parent

    for path in [start_path, *start_path.parents]:
        if (path / "requirements.txt").exists() and (path / "src").exists():
            return path

    return Path.cwd().resolve()


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def make_run_dir(output_dir: str | Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir) / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config(config: dict[str, Any], run_dir: str | Path) -> None:
    path = Path(run_dir) / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
