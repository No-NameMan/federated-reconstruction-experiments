from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_final_metrics(run_dir: str | Path, label: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "final_metrics.json"
    config_path = run_dir / "config.yaml"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing final_metrics.json: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    method = label
    if method is None:
        method = metrics.get("baseline", None)
    if method is None:
        method = run_dir.name

    row: dict[str, Any] = {
        "method": method,
        "run_dir": run_dir.name,
    }

    for split in ["val", "test"]:
        split_metrics = metrics.get(split, {})
        for metric_name in ["rmse", "mae", "accuracy", "num_clients", "num_examples"]:
            if metric_name in split_metrics:
                row[f"{split}_{metric_name}"] = split_metrics[metric_name]

    if "rounds" in metrics:
        row["rounds"] = metrics["rounds"]
    if "epochs" in metrics:
        row["epochs"] = metrics["epochs"]
    if "total_bytes" in metrics:
        row["total_bytes"] = metrics["total_bytes"]
        row["total_mb"] = metrics["total_bytes"] / (1024 * 1024)

    if config_path.exists():
        row["has_config"] = True
    else:
        row["has_config"] = False

    return row


def compare_final_metrics(run_specs: list[str], output: str | Path) -> pd.DataFrame:
    rows = []

    for spec in run_specs:
        if "::" in spec:
            run_dir, label = spec.split("::", maxsplit=1)
        else:
            run_dir, label = spec, None

        rows.append(load_final_metrics(run_dir, label))

    df = pd.DataFrame(rows)

    preferred_cols = [
        "method",
        "test_rmse",
        "test_mae",
        "test_accuracy",
        "val_rmse",
        "val_mae",
        "val_accuracy",
        "rounds",
        "epochs",
        "total_mb",
        "test_num_clients",
        "test_num_examples",
        "run_dir",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    return df
