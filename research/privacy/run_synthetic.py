"""Run the archived independent verification without data or coursework code."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--controls", action="store_true", help="Also repeat the 320 assumption controls")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # The historical scripts write beside themselves. Copy only their inputs
    # into the output directory so the distributed source and reference stay intact.
    math = output / "independent_math"
    shutil.copytree(source / "independent_math", math, ignore=shutil.ignore_patterns("__pycache__"))
    scripts = [(math / name, []) for name in ("verify_math.py", "batch_observability.py")]
    if args.controls:
        scripts.append((source / "assumption_controls/run_controls.py",
                        ["--output", str(output / "controls")]))
    for script, options in scripts:
        with (output / f"{script.stem}.log").open("wb") as log:
            run = subprocess.run([sys.executable, "-B", str(script), *options], cwd=output,
                                 stdout=log, stderr=subprocess.STDOUT, check=False)
        if run.returncode:
            raise SystemExit(f"Failed: {script.name}. See {log.name}")

    summary = json.loads((math / "summary.json").read_text(encoding="utf-8"))
    batch = json.loads((math / "batch_observability.json").read_text(encoding="utf-8"))
    full = summary["full_batch"]
    hidden = batch["hidden_order_counterexample"]
    assert summary["status"] == batch["status"] == "PASS"
    assert all(x["cases"] == x["relative_ratings_exact"] == 240 for x in full.values())
    assert hidden["max_message_difference"] == 0 and hidden["ratings_a"] != hidden["ratings_b"]
    result = {
        "status": "PASS",
        "full_batch_cases": sum(x["cases"] for x in full.values()),
        "exact_relative_cases": sum(x["relative_ratings_exact"] for x in full.values()),
        "minibatch_profile_cases": sum(x["cases"] for x in summary["minibatch"].values()),
        "hidden_order_max_message_difference": hidden["max_message_difference"],
    }
    if args.controls:
        controls = json.loads((output / "controls/results.json").read_text(encoding="utf-8"))
        result["assumption_control_rows"] = len(controls["rows"])
        assert result["assumption_control_rows"] == 320
    payload = json.dumps(result, indent=2)
    (output / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
