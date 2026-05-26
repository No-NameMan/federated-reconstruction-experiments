from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.analysis.summarize_final_metrics import summarize_final_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_glob", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--aggregate_output", type=str, default=None)
    parser.add_argument("--label_params", type=str, nargs="*", default=[])
    args = parser.parse_args()

    df = summarize_final_metrics(
        runs_glob=args.runs_glob,
        output=args.output,
        label_params=args.label_params,
        aggregate_output=args.aggregate_output,
    )

    print(df.to_string(index=False))
    print(f"\nSaved final metrics summary to: {args.output}")
    if args.aggregate_output:
        print(f"Saved aggregate summary to: {args.aggregate_output}")


if __name__ == "__main__":
    main()
