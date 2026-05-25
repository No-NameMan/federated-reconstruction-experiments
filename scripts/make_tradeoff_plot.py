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
    parser.add_argument("--x_metric", type=str, default="total_bytes")
    parser.add_argument("--y_metric", type=str, default="val_rmse")
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
        metrics = metrics.dropna(subset=[args.x_metric, args.y_metric])
        metrics = metrics[
            (metrics[args.x_metric] != "") & (metrics[args.y_metric] != "")
        ]

        if metrics.empty:
            continue

        metrics[args.x_metric] = metrics[args.x_metric].astype(float)
        metrics[args.y_metric] = metrics[args.y_metric].astype(float)

        config = load_yaml(config_path)
        label_value = get_by_path(config, args.label_param)
        seed = config["experiment"]["seed"]
        label = f"{args.label_param}={label_value}, seed={seed}"

        plt.plot(
            metrics[args.x_metric] / (1024 * 1024),
            metrics[args.y_metric],
            marker="o",
            label=label,
            alpha=0.75,
        )

    plt.xlabel("Total transmitted data, MB")
    plt.ylabel(args.y_metric)
    plt.title(f"{args.y_metric} vs communication")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
