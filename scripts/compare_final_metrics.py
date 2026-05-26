from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.analysis.compare_final_metrics import compare_final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Run dirs. Optional label syntax: path::label",
    )
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = compare_final_metrics(args.runs, args.output)
    print(df.to_string(index=False))
    print(f"\nSaved comparison to: {args.output}")


if __name__ == "__main__":
    main()
