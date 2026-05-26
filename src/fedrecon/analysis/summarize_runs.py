from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from fedrecon.utils.config import get_by_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def valid_metric_rows(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = metrics.dropna(subset=[metric]).copy()
    valid = valid[valid[metric] != ""]
    if valid.empty:
        return valid
    valid[metric] = valid[metric].astype(float)
    return valid


def metric_at_or_before_round(valid: pd.DataFrame, metric: str, round_value: int) -> float:
    subset = valid[valid["round"].astype(int) <= round_value]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[-1][metric])


def build_runs_summary(
    *,
    runs_glob: str,
    label_params: list[str],
    metric: str,
    rounds: list[int],
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

        valid = valid_metric_rows(metrics, metric)
        if valid.empty:
            continue

        final = valid.iloc[-1]
        best = valid.loc[valid[metric].idxmin()]
        first = valid.iloc[0]

        row: dict[str, Any] = {
            "run_dir": run_dir_path.name,
            "first_round": int(first["round"]),
            "final_round": int(final["round"]),
            f"first_{metric}": float(first[metric]),
            f"final_{metric}": float(final[metric]),
            f"best_{metric}": float(best[metric]),
            f"best_{metric}_round": int(best["round"]),
            f"delta_{metric}_first_to_final": float(first[metric]) - float(final[metric]),
        }

        for round_value in rounds:
            row[f"{metric}@{round_value}"] = metric_at_or_before_round(
                valid,
                metric,
                round_value,
            )

        for col in ["val_mae", "val_accuracy", "train_loss", "total_bytes"]:
            if col in final.index and final[col] != "":
                row[f"final_{col}"] = float(final[col])

        for param in label_params:
            row[param] = get_by_path(config, param)

        rows.append(row)

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    if label_params:
        summary = summary.sort_values(label_params).reset_index(drop=True)
    else:
        summary = summary.sort_values(f"final_{metric}").reset_index(drop=True)

    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label_params", type=str, nargs="+", default=["reconstruction.steps"])
    parser.add_argument("--metric", type=str, default="val_rmse")
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="*",
        default=[0, 5, 10, 15, 20, 25, 30],
        help="Round checkpoints to include in the summary.",
    )
    args = parser.parse_args(argv)

    summary = build_runs_summary(
        runs_glob=args.runs_glob,
        label_params=args.label_params,
        metric=args.metric,
        rounds=args.rounds,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {output}")


if __name__ == "__main__":
    main()
