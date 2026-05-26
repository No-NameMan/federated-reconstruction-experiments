from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.utils.config import get_by_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_final_metrics(
    runs_glob: str,
    output: str | Path,
    label_params: list[str],
    aggregate_output: str | Path | None = None,
) -> pd.DataFrame:
    run_dirs = sorted(glob.glob(runs_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs matched: {runs_glob}")

    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "final_metrics.json"
        config_path = run_dir / "config.yaml"

        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = load_json(metrics_path)
        config = load_yaml(config_path)

        row: dict[str, Any] = {
            "run_dir": run_dir.name,
            "seed": config["experiment"]["seed"],
        }

        for split in ["val", "test"]:
            split_metrics = metrics.get(split, {})
            for metric_name in [
                "rmse",
                "mae",
                "accuracy",
                "num_clients",
                "num_examples",
            ]:
                if metric_name in split_metrics:
                    row[f"{split}_{metric_name}"] = split_metrics[metric_name]

        for key in ["rounds", "epochs", "total_bytes"]:
            if key in metrics:
                row[key] = metrics[key]

        if "total_bytes" in row:
            row["total_mb"] = row["total_bytes"] / (1024 * 1024)

        for param in label_params:
            row[param] = get_by_path(config, param)

        rows.append(row)

    df = pd.DataFrame(rows)
    sort_cols = label_params + ["seed"] if label_params else ["seed"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    if aggregate_output is not None:
        metric_cols = [
            "val_rmse",
            "val_mae",
            "val_accuracy",
            "test_rmse",
            "test_mae",
            "test_accuracy",
            "total_mb",
        ]
        existing_metric_cols = [c for c in metric_cols if c in df.columns]

        agg = (
            df.groupby(label_params, dropna=False)[existing_metric_cols]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )

        agg.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in agg.columns.to_flat_index()
        ]

        aggregate_output = Path(aggregate_output)
        aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(aggregate_output, index=False)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--aggregate_output", type=str, default=None)
    parser.add_argument("--label_params", type=str, nargs="*", default=[])
    args = parser.parse_args()

    df = summarize_final_metrics(
        runs_glob=args.runs_glob,
        output=args.output,
        label_params=args.label_params,
        aggregate_output=args.aggregate_output,
    )

    print(df.to_string(index=False))
    print(f"\nSaved final metrics summary to: {args.output}")
    if args.aggregate_output:
        print(f"Saved aggregate summary to: {args.aggregate_output}")


if __name__ == "__main__":
    main()
