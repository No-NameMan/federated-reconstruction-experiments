from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_group_metric(
    *,
    input_csv: str | Path,
    output: str | Path,
    x_col: str,
    group_col: str,
    y_col: str,
    title: str | None = None,
) -> None:
    df = pd.read_csv(input_csv)

    plt.figure()

    for group_value, group_df in df.groupby(group_col, sort=False):
        group_df = group_df.sort_values(x_col)
        plt.plot(
            group_df[x_col],
            group_df[y_col],
            marker="o",
            label=str(group_value),
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"{y_col} by {group_col}")
    plt.grid(True)
    plt.legend(title=group_col)
    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
