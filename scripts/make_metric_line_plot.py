from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--x_col", type=str, required=True)
    parser.add_argument("--y_col", type=str, required=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv).sort_values(args.x_col)

    plt.figure()
    plt.plot(df[args.x_col], df[args.y_col], marker="o")
    plt.xlabel(args.x_col)
    plt.ylabel(args.y_col)
    plt.title(args.title or f"{args.y_col} vs {args.x_col}")
    plt.grid(True)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
