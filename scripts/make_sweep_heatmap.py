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


def _metric_value(metrics: pd.DataFrame, metric: str, mode: str) -> float:
    valid = metrics.dropna(subset=[metric])
    valid = valid[valid[metric] != ""]
    if valid.empty:
        return float("nan")

    valid[metric] = valid[metric].astype(float)

    if mode == "final":
        return float(valid.iloc[-1][metric])

    if mode == "best":
        return float(valid[metric].min())

    raise ValueError("mode must be either 'final' or 'best'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--x_param", type=str, required=True)
    parser.add_argument("--y_param", type=str, required=True)
    parser.add_argument("--metric", type=str, default="val_rmse")
    parser.add_argument("--mode", type=str, default="final", choices=["final", "best"])
    args = parser.parse_args()

    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs matched: {args.runs_glob}")

    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics_per_round.csv"
        config_path = run_dir / "config.yaml"

        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        config = load_yaml(config_path)

        rows.append(
            {
                args.x_param: get_by_path(config, args.x_param),
                args.y_param: get_by_path(config, args.y_param),
                args.metric: _metric_value(metrics, args.metric, args.mode),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid runs found.")

    table = df.pivot_table(
        index=args.y_param,
        columns=args.x_param,
        values=args.metric,
        aggfunc="mean",
    )

    table = table.sort_index(axis=0).sort_index(axis=1)

    plt.figure()
    image = plt.imshow(table.values, aspect="auto")
    plt.colorbar(image, label=f"{args.mode} {args.metric}")

    plt.xticks(range(len(table.columns)), [str(x) for x in table.columns])
    plt.yticks(range(len(table.index)), [str(y) for y in table.index])

    plt.xlabel(args.x_param)
    plt.ylabel(args.y_param)
    plt.title(f"{args.mode} {args.metric}")

    for y in range(table.shape[0]):
        for x in range(table.shape[1]):
            value = table.values[y, x]
            plt.text(x, y, f"{value:.4f}", ha="center", va="center")

    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print(table)
    print(f"Saved heatmap to: {output}")


if __name__ == "__main__":
    main()
