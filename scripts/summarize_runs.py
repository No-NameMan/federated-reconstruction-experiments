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
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--label_params",
        type=str,
        nargs="+",
        default=["reconstruction.steps"],
    )
    parser.add_argument("--metric", type=str, default="val_rmse")
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

        valid = metrics.dropna(subset=[args.metric])
        valid = valid[valid[args.metric] != ""]

        if valid.empty:
            continue

        valid[args.metric] = valid[args.metric].astype(float)

        final = valid.iloc[-1]
        best = valid.loc[valid[args.metric].idxmin()]

        row: dict[str, Any] = {
            "run_dir": run_dir.name,
            "final_round": int(final["round"]),
            f"final_{args.metric}": float(final[args.metric]),
            f"best_{args.metric}": float(best[args.metric]),
            f"best_{args.metric}_round": int(best["round"]),
        }

        for col in ["val_mae", "val_accuracy", "train_loss", "total_bytes"]:
            if col in final.index and final[col] != "":
                row[f"final_{col}"] = float(final[col])

        for param in args.label_params:
            row[param] = get_by_path(config, param)

        rows.append(row)

    summary = pd.DataFrame(rows)

    sort_cols = args.label_params if args.label_params else [f"final_{args.metric}"]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {output}")


if __name__ == "__main__":
    main()