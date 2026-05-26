from __future__ import annotations

import argparse
import glob
import math
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


def summarize_group(group: pd.DataFrame) -> dict[str, float | int]:
    total_query = float(group["num_query_examples"].sum())

    if total_query > 0:
        pooled_mse = (
            (group["num_query_examples"] * (group["rmse"] ** 2)).sum()
            / total_query
        )
        pooled_rmse = math.sqrt(float(pooled_mse))

        weighted_mae = float(
            (group["num_query_examples"] * group["mae"]).sum() / total_query
        )
        weighted_accuracy = float(
            (group["num_query_examples"] * group["accuracy"]).sum() / total_query
        )
    else:
        pooled_rmse = float("nan")
        weighted_mae = float("nan")
        weighted_accuracy = float("nan")

    return {
        "n_clients": int(group["client_id"].count()),
        "total_query_examples": int(group["num_query_examples"].sum()),
        "mean_num_examples": float(group["num_examples_total"].mean()),
        "mean_rmse": float(group["rmse"].mean()),
        "median_rmse": float(group["rmse"].median()),
        "pooled_rmse": pooled_rmse,
        "mean_mae": float(group["mae"].mean()),
        "weighted_mae": weighted_mae,
        "mean_accuracy": float(group["accuracy"].mean()),
        "weighted_accuracy": weighted_accuracy,
    }


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

        for group_name, group in df.groupby("rating_count_group", observed=True):
            out = {
                "run_dir": run_dir.name,
                "split": args.split,
                "rating_count_group": str(group_name),
                **summarize_group(group),
            }

            for param in args.label_params:
                out[param] = get_by_path(config, param)

            rows.append(out)

    summary = pd.DataFrame(rows)

    group_order = {
        "<=50": 0,
        "51-100": 1,
        "101-200": 2,
        "201-500": 3,
        ">500": 4,
    }
    summary["_group_order"] = summary["rating_count_group"].map(group_order)

    sort_cols = args.label_params + ["_group_order"]
    summary = summary.sort_values(sort_cols).drop(columns=["_group_order"]).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved client group summary to: {output}")


if __name__ == "__main__":
    main()