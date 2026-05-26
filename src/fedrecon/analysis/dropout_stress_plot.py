from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dropout_stress(
    *,
    input_csv: str | Path,
    output: str | Path,
    metric_col: str = "test_rmse_mean",
) -> None:
    df = pd.read_csv(input_csv)

    x_col = "federated.client_dropout_prob"
    group_col = "federated.clients_per_round"

    plt.figure()

    for clients_per_round, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        yerr_col = metric_col.replace("_mean", "_std")
        yerr = group_df[yerr_col] if yerr_col in group_df.columns else None

        plt.errorbar(
            group_df[x_col],
            group_df[metric_col],
            yerr=yerr,
            marker="o",
            capsize=4,
            label=f"clients_per_round={clients_per_round}",
        )

    plt.xlabel("Client dropout probability")
    plt.ylabel(metric_col)
    plt.title("Dropout stress-test")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()
