from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.utils.config import get_by_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label_param", type=str, required=True)
    parser.add_argument("--metric", type=str, default="val_rmse")
    args = parser.parse_args()

    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs matched: {args.runs_glob}")

    plt.figure()

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics_per_round.csv"
        config_path = run_dir / "config.yaml"

        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        metrics = metrics.dropna(subset=[args.metric])
        metrics = metrics[metrics[args.metric] != ""]

        if metrics.empty:
            continue

        metrics[args.metric] = metrics[args.metric].astype(float)

        config = load_yaml(config_path)
        label_value = get_by_path(config, args.label_param)
        label = f"{args.label_param}={label_value}"

        plt.plot(metrics["round"], metrics[args.metric], marker="o", label=label)

    plt.xlabel("Round")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} by {args.label_param}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
