from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="compression.method")
    parser.add_argument("--x_col", type=str, default="final_total_bytes")
    parser.add_argument("--y_col", type=str, default="final_val_rmse")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)

    grouped = (
        df.groupby(args.label_col)
        .agg(
            x_mean=(args.x_col, "mean"),
            x_std=(args.x_col, "std"),
            y_mean=(args.y_col, "mean"),
            y_std=(args.y_col, "std"),
            n=(args.y_col, "size"),
        )
        .reset_index()
    )

    grouped["x_mean_mb"] = grouped["x_mean"] / (1024 * 1024)
    grouped["x_std_mb"] = grouped["x_std"].fillna(0.0) / (1024 * 1024)
    grouped["y_std"] = grouped["y_std"].fillna(0.0)

    # Sort by communication cost, descending, so the visual order is intuitive.
    grouped = grouped.sort_values("x_mean_mb", ascending=False)

    plt.figure()

    for _, row in grouped.iterrows():
        plt.errorbar(
            row["x_mean_mb"],
            row["y_mean"],
            xerr=row["x_std_mb"],
            yerr=row["y_std"],
            marker="o",
            capsize=4,
            label=str(row[args.label_col]),
        )
        plt.text(
            row["x_mean_mb"],
            row["y_mean"],
            f"  {row[args.label_col]}",
            va="center",
        )

    plt.xlabel("Total transmitted data, MB")
    plt.ylabel(args.y_col)
    plt.title(f"{args.y_col} vs communication")
    plt.grid(True)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print(grouped.to_string(index=False))
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
