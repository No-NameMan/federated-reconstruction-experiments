from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_rmse_vs_rounds(metrics: pd.DataFrame, output_path: str | Path) -> None:
    df = metrics.dropna(subset=["val_rmse"]).copy()
    df = df[df["val_rmse"] != ""]
    df["val_rmse"] = df["val_rmse"].astype(float)

    plt.figure()
    plt.plot(df["round"], df["val_rmse"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Validation RMSE")
    plt.title("FEDRECON validation RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
