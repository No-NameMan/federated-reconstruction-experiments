from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from fedrecon.utils.config import get_by_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_sweep_metric(
    *,
    runs_glob: str,
    output: str | Path,
    label_param: str,
    metric: str,
) -> None:
    run_dirs = sorted(glob.glob(runs_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs matched: {runs_glob}")

    plt.figure()

    for run_dir in run_dirs:
        run_dir_path = Path(run_dir)
        metrics_path = run_dir_path / "metrics_per_round.csv"
        config_path = run_dir_path / "config.yaml"

        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        metrics = metrics.dropna(subset=[metric])
        metrics = metrics[metrics[metric] != ""]

        if metrics.empty:
            continue

        metrics[metric] = metrics[metric].astype(float)

        config = load_yaml(config_path)
        label_value = get_by_path(config, label_param)
        label = f"{label_param}={label_value}"

        plt.plot(metrics["round"], metrics[metric], marker="o", label=label)

    plt.xlabel("Round")
    plt.ylabel(metric)
    plt.title(f"{metric} by {label_param}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label_param", type=str, required=True)
    parser.add_argument("--metric", type=str, default="val_rmse")
    args = parser.parse_args(argv)

    plot_sweep_metric(
        runs_glob=args.runs_glob,
        output=args.output,
        label_param=args.label_param,
        metric=args.metric,
    )
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
