from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.analysis.load_results import load_metrics_file
from fedrecon.analysis.plots import plot_rmse_vs_rounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    metrics = load_metrics_file(args.metrics)
    plot_rmse_vs_rounds(metrics, args.output)


if __name__ == "__main__":
    main()
