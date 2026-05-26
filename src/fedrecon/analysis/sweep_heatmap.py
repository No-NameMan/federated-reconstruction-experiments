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


def metric_value(metrics: pd.DataFrame, metric: str, mode: str) -> float:
    valid = metrics.dropna(subset=[metric]).copy()
    valid = valid[valid[metric] != ""]
    if valid.empty:
        return float("nan")

    valid[metric] = valid[metric].astype(float)

    if mode == "final":
        return float(valid.iloc[-1][metric])

    if mode == "best":
        return float(valid[metric].min())

    raise ValueError("mode must be either 'final' or 'best'.")


def plot_sweep_heatmap(
    *,
    runs_glob: str,
    output: str | Path,
    x_param: str,
    y_param: str,
    metric: str,
    mode: str,
) -> pd.DataFrame:
    run_dirs = sorted(glob.glob(runs_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs matched: {runs_glob}")

    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_dir_path = Path(run_dir)
        metrics_path = run_dir_path / "metrics_per_round.csv"
        config_path = run_dir_path / "config.yaml"

        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        config = load_yaml(config_path)

        rows.append(
            {
                x_param: get_by_path(config, x_param),
                y_param: get_by_path(config, y_param),
                metric: metric_value(metrics, metric, mode),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid runs found.")

    table = df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
    table = table.sort_index(axis=0).sort_index(axis=1)

    plt.figure()
    image = plt.imshow(table.values, aspect="auto")
    plt.colorbar(image, label=f"{mode} {metric}")

    plt.xticks(range(len(table.columns)), [str(x) for x in table.columns])
    plt.yticks(range(len(table.index)), [str(y) for y in table.index])

    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"{mode} {metric}")

    for y in range(table.shape[0]):
        for x in range(table.shape[1]):
            value = table.values[y, x]
            plt.text(x, y, f"{value:.4f}", ha="center", va="center")

    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    return table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--x_param", type=str, required=True)
    parser.add_argument("--y_param", type=str, required=True)
    parser.add_argument("--metric", type=str, default="val_rmse")
    parser.add_argument("--mode", type=str, default="final", choices=["final", "best"])
    args = parser.parse_args(argv)

    table = plot_sweep_heatmap(
        runs_glob=args.runs_glob,
        output=args.output,
        x_param=args.x_param,
        y_param=args.y_param,
        metric=args.metric,
        mode=args.mode,
    )
    print(table)
    print(f"Saved heatmap to: {args.output}")


if __name__ == "__main__":
    main()
