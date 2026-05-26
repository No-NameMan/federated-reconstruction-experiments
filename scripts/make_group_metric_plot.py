from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.analysis.group_metric_plot import plot_group_metric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--x_col", type=str, required=True)
    parser.add_argument("--group_col", type=str, required=True)
    parser.add_argument("--y_col", type=str, required=True)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    plot_group_metric(
        input_csv=args.input_csv,
        output=args.output,
        x_col=args.x_col,
        group_col=args.group_col,
        y_col=args.y_col,
        title=args.title,
    )

    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
