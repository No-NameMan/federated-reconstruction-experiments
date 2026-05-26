from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

from fedrecon.simulation.trainer import run_training
from fedrecon.utils.config import load_config, set_by_path


def resolve_sweep_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidate_from_cwd = Path.cwd() / path
    if candidate_from_cwd.exists():
        return candidate_from_cwd
    return base_dir / path


def slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def iter_sweep_configs(sweep_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    sweep = sweep_config["sweep"]

    if "parameter" in sweep:
        parameter = sweep["parameter"]
        values = sweep["values"]

        runs = []
        for value in values:
            suffix = f"{parameter.replace('.', '_')}_{slug(value)}"
            runs.append((suffix, {parameter: value}))
        return runs

    if "parameters" in sweep:
        params = sweep["parameters"]
        names = [p["name"] for p in params]
        value_lists = [p["values"] for p in params]

        runs = []
        for values in itertools.product(*value_lists):
            overrides = dict(zip(names, values))
            suffix = "__".join(
                f"{name.replace('.', '_')}_{slug(value)}"
                for name, value in overrides.items()
            )
            runs.append((suffix, overrides))
        return runs

    raise ValueError("Sweep config must contain either 'parameter' or 'parameters'.")


def run_sweep(sweep_config_path: str | Path) -> None:
    sweep_config_path = Path(sweep_config_path).resolve()
    sweep_config = load_config(sweep_config_path)

    base_config_path = resolve_sweep_path(
        sweep_config["base_config"],
        base_dir=sweep_config_path.parent,
    )
    base_config = load_config(base_config_path)

    sweep_name = sweep_config["sweep"]["name"]
    seeds = sweep_config["sweep"].get("seeds", [base_config["experiment"]["seed"]])
    run_specs = iter_sweep_configs(sweep_config)

    print(f"Sweep: {sweep_name}")
    print(f"Base config: {base_config_path}")
    print(f"Runs per seed: {len(run_specs)}")
    print(f"Seeds: {seeds}")

    for seed in seeds:
        for suffix, overrides in run_specs:
            config = copy.deepcopy(base_config)

            for path, value in overrides.items():
                set_by_path(config, path, value)

            config["experiment"]["seed"] = int(seed)
            config["experiment"]["name"] = f"{sweep_name}__{suffix}__seed_{seed}"

            print("=" * 80)
            print(f"Running: {config['experiment']['name']}")
            print(f"Overrides: {overrides}")
            print("=" * 80)

            run_training(config)
