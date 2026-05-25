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


def _valid_metric_rows(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = metrics.dropna(subset=[metric]).copy()
    valid = valid[valid[metric] != ""]
    if valid.empty:
        return valid
    valid[metric] = valid[metric].astype(float)
    return valid


def _metric_at_or_before_round(valid: pd.DataFrame, metric: str, round_value: int) -> float:
    subset = valid[valid["round"].astype(int) <= round_value]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[-1][metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--label_params",
        type=str,
        nargs="+",
        default=["reconstruction.steps"],
    )
    parser.add_argument("--metric", type=str, default="val_rmse")
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="*",
        default=[0, 5, 10, 15, 20, 25, 30],
        help="Round checkpoints to include in the summary.",
    )
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

        valid = _valid_metric_rows(metrics, args.metric)
        if valid.empty:
            continue

        final = valid.iloc[-1]
        best = valid.loc[valid[args.metric].idxmin()]
        first = valid.iloc[0]

        row: dict[str, Any] = {
            "run_dir": run_dir.name,
            "first_round": int(first["round"]),
            "final_round": int(final["round"]),
            f"first_{args.metric}": float(first[args.metric]),
            f"final_{args.metric}": float(final[args.metric]),
            f"best_{args.metric}": float(best[args.metric]),
            f"best_{args.metric}_round": int(best["round"]),
            f"delta_{args.metric}_first_to_final": float(first[args.metric]) - float(final[args.metric]),
        }

        for round_value in args.rounds:
            row[f"{args.metric}@{round_value}"] = _metric_at_or_before_round(
                valid,
                args.metric,
                round_value,
            )

        for col in ["val_mae", "val_accuracy", "train_loss", "total_bytes"]:
            if col in final.index and final[col] != "":
                row[f"final_{col}"] = float(final[col])

        for param in args.label_params:
            row[param] = get_by_path(config, param)

        rows.append(row)

    summary = pd.DataFrame(rows)

    if args.label_params:
        summary = summary.sort_values(args.label_params).reset_index(drop=True)
    else:
        summary = summary.sort_values(f"final_{args.metric}").reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {output}")


if __name__ == "__main__":
    main()