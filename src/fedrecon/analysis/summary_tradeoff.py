from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def build_summary_tradeoff_table(
    *,
    summary: str | Path,
    label_col: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(summary)

    grouped = (
        df.groupby(label_col)
        .agg(
            x_mean=(x_col, "mean"),
            x_std=(x_col, "std"),
            y_mean=(y_col, "mean"),
            y_std=(y_col, "std"),
            n=(y_col, "size"),
        )
        .reset_index()
    )

    grouped["x_mean_mb"] = grouped["x_mean"] / (1024 * 1024)
    grouped["x_std_mb"] = grouped["x_std"].fillna(0.0) / (1024 * 1024)
    grouped["y_std"] = grouped["y_std"].fillna(0.0)
    grouped = grouped.sort_values("x_mean_mb", ascending=False)
    return grouped


def plot_summary_tradeoff(
    *,
    summary: str | Path,
    output: str | Path,
    label_col: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    grouped = build_summary_tradeoff_table(
        summary=summary,
        label_col=label_col,
        x_col=x_col,
        y_col=y_col,
    )

    plt.figure()

    for _, row in grouped.iterrows():
        plt.errorbar(
            row["x_mean_mb"],
            row["y_mean"],
            xerr=row["x_std_mb"],
            yerr=row["y_std"],
            marker="o",
            capsize=4,
            label=str(row[label_col]),
        )
        plt.text(row["x_mean_mb"], row["y_mean"], f"  {row[label_col]}", va="center")

    plt.xlabel("Total transmitted data, MB")
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs communication")
    plt.grid(True)
    plt.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    return grouped


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="compression.method")
    parser.add_argument("--x_col", type=str, default="final_total_bytes")
    parser.add_argument("--y_col", type=str, default="final_val_rmse")
    args = parser.parse_args(argv)

    grouped = plot_summary_tradeoff(
        summary=args.summary,
        output=args.output,
        label_col=args.label_col,
        x_col=args.x_col,
        y_col=args.y_col,
    )

    print(grouped.to_string(index=False))
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
