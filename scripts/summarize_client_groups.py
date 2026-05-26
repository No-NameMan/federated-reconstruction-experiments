from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any

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
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label_params", type=str, nargs="*", default=[])
    args = parser.parse_args()

    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs matched: {args.runs_glob}")

    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / f"{args.split}_client_metrics.csv"
        config_path = run_dir / "config.yaml"

        if not metrics_path.exists() or not config_path.exists():
            continue

        df = pd.read_csv(metrics_path)
        config = load_yaml(config_path)

        bins = [0, 50, 100, 200, 500, float("inf")]
        labels = ["<=50", "51-100", "101-200", "201-500", ">500"]

        df["rating_count_group"] = pd.cut(
            df["num_examples_total"],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        )

        grouped = (
            df.groupby("rating_count_group", observed=True)
            .agg(
                n_clients=("client_id", "count"),
                mean_num_examples=("num_examples_total", "mean"),
                mean_rmse=("rmse", "mean"),
                median_rmse=("rmse", "median"),
                mean_mae=("mae", "mean"),
                mean_accuracy=("accuracy", "mean"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            out = {
                "run_dir": run_dir.name,
                "split": args.split,
                "rating_count_group": str(row["rating_count_group"]),
                "n_clients": int(row["n_clients"]),
                "mean_num_examples": float(row["mean_num_examples"]),
                "mean_rmse": float(row["mean_rmse"]),
                "median_rmse": float(row["median_rmse"]),
                "mean_mae": float(row["mean_mae"]),
                "mean_accuracy": float(row["mean_accuracy"]),
            }

            for param in args.label_params:
                out[param] = get_by_path(config, param)

            rows.append(out)

    summary = pd.DataFrame(rows)
    sort_cols = args.label_params + ["rating_count_group"]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved client group summary to: {output}")


if __name__ == "__main__":
    main()
