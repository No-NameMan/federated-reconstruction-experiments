from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.analysis.export_final_artifacts import export_final_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="results/final")
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    copied = export_final_artifacts(
        project_root=args.project_root,
        output_dir=args.output_dir,
        overwrite=not args.no_overwrite,
    )

    for src, dst, exists in copied:
        status = "copied" if exists else "missing"
        print(f"[{status}] {src} -> {dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()
