from __future__ import annotations
import pandas as pd


def final_row(metrics: pd.DataFrame) -> pd.Series:
    valid = metrics.dropna(subset=["val_rmse"])
    valid = valid[valid["val_rmse"] != ""]
    if valid.empty:
        return metrics.iloc[-1]
    return valid.iloc[-1]
