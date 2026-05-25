from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_metrics_file(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_all_runs(results_dir: str | Path) -> pd.DataFrame:
    rows = []
    for metrics_path in Path(results_dir).glob("*/metrics_per_round.csv"):
        df = pd.read_csv(metrics_path)
        df["run_dir"] = str(metrics_path.parent)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
