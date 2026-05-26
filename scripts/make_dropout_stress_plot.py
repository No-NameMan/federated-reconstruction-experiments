from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.analysis.dropout_stress_plot import plot_dropout_stress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--metric_col", type=str, default="test_rmse_mean")
    args = parser.parse_args()

    plot_dropout_stress(
        input_csv=args.input_csv,
        output=args.output,
        metric_col=args.metric_col,
    )

    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
