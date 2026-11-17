"""Repeat a coursework training run or the historical 30-client diagnostic."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REVISION = "02293ae3dd917ea0f45ceafc3f30facc4a7af982"
RATINGS_SHA256 = "506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f"
CHECKPOINT_SHA256 = "49e6dbbf8fbb82f90f1079064c523df6a58bbe4b96721b978796ce5d4400366f"


def check_input(path, expected):
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}\nExpected {expected}\nFound    {actual}")


def train(repo, data_dir, output, reference, expected):
    import yaml
    from fedrecon.simulation.trainer import run_training

    config_path = HERE / "configs/reference_100r.yaml" if reference else repo / "configs/debug.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment"]["output_dir"] = str(output / "training")
    config["data"]["data_dir"] = str(data_dir)
    if not reference:
        config["experiment"]["name"] = "seminar_smoke"
        config["federated"].update(rounds=2, clients_per_round=3)
        config["evaluation"].update(max_eval_clients=5, max_final_eval_clients=5)
        config["logging"]["checkpoint_every"] = 2
    train_dir = run_training(config)
    metrics = json.loads((train_dir / "final_metrics.json").read_text(encoding="utf-8"))
    rounds = 100 if reference else 2
    assert metrics["rounds"] == rounds
    assert (train_dir / "final_model.pt").is_file()
    assert (train_dir / f"checkpoint_round_{rounds}.pt").is_file()
    assert metrics["test"]["num_clients"] == (604 if reference else 5)
    assert all(math.isfinite(metrics["test"][k]) for k in ("rmse", "mae", "accuracy"))
    saved = expected["reference_rerun"]["test"] if reference else expected["smoke"]["test"]
    test = metrics["test"]
    return {
        "training_dir": train_dir.relative_to(output).as_posix(),
        "rounds": rounds,
        "test": test,
        "abs_difference_from_reference": {k: abs(test[k] - saved[k]) for k in ("rmse", "mae", "accuracy")},
    }


def pilot(data_dir, checkpoint, expected):
    import numpy as np
    import torch
    from verify_math import Message, Metadata, Public, recover_relative
    from fedrecon.algorithms.client_update import compute_client_delta
    from fedrecon.algorithms.reconstruction import reconstruct_local_params
    from fedrecon.data.movielens import load_movielens_1m_clients
    from fedrecon.data.splits import random_support_query_split
    from fedrecon.models.matrix_factorization import GlobalMatrixFactorization

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    data = load_movielens_1m_clients(data_dir)
    model = GlobalMatrixFactorization(data.num_items, 32)
    model.load_global_state(saved["model_state"])
    state = model.clone_global_state()
    public = Public(state["item_embeddings.weight"].double().numpy(),
                    state["item_bias.weight"].flatten().double().numpy(),
                    float(state["global_bias"]))
    diagnostics = []
    for client_id in sorted(data.train_clients)[:30]:
        split = random_support_query_split(data.train_clients[client_id], .5, 43)
        torch.manual_seed(123000 + client_id)
        local = reconstruct_local_params(model, split.support, 5, .1, True, .05)
        delta, count, _ = compute_client_delta(model, local, split.query, 10, .2)
        message = Message(delta["item_embeddings.weight"].double().numpy(),
                          delta["item_bias.weight"].flatten().double().numpy(),
                          float(delta["global_bias"]))
        inferred = recover_relative(public, message, Metadata(count, 10, .2))
        # Private values below are used only to score the decoder output.
        truth_by_id = dict(zip(split.query.item_ids.tolist(), split.query.ratings.tolist()))
        ids = inferred["ids"].tolist()
        assert set(ids) == set(truth_by_id)
        truth = np.array([truth_by_id[i] for i in ids])
        target = local.user_embedding.double().numpy()
        p_error = float(np.linalg.norm(inferred["p"] - target) / np.linalg.norm(target))
        y_error = float(np.max(np.abs(inferred["continuous"] - (truth - float(local.user_bias)))))
        assert p_error < 1e-5 and y_error < 2e-5
        exact_relative = bool(np.array_equal(inferred["normalized"], truth - truth.min()))
        candidates = inferred["candidates"]
        assert any(np.array_equal(candidate, truth) for candidate in candidates)
        diagnostics.append((count, exact_relative, len(candidates) == 1, p_error, y_error))
    stats = dict(clients=len(diagnostics), query_pairs=sum(x[0] for x in diagnostics),
                 exact_items_clients=len(diagnostics),
                 exact_relative_clients=sum(x[1] for x in diagnostics),
                 unique_absolute_clients=sum(x[2] for x in diagnostics),
                 max_p_relative_l2=max(x[3] for x in diagnostics),
                 max_shifted_rating_error=max(x[4] for x in diagnostics))
    for key in ("clients", "query_pairs", "exact_items_clients", "exact_relative_clients", "unique_absolute_clients"):
        assert stats[key] == expected["pilot"][key], (key, stats[key])
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "reference", "pilot"))
    parser.add_argument("--repo", type=Path, default=HERE.parents[1],
                        help="Checkout with the unmodified coursework simulator (default: project root)")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing ml-1m/ratings.dat")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--checkpoint", type=Path, help="Trusted original May final_model.pt (pilot only)")
    args = parser.parse_args()
    if (args.mode == "pilot") != (args.checkpoint is not None):
        parser.error("--checkpoint is required for pilot and unused in training modes")
    repo, data_dir, output = (path.resolve() for path in (args.repo, args.data_dir, args.output))
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    # Documentation and research code can advance independently of the simulator.
    # These modes reproduce the historical training and require its input code.
    scope = ["src", "configs/debug.yaml"]
    changed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", REVISION, "--", *scope],
        check=False,
    )
    untracked = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard", "--", *scope]
    )
    if changed.returncode or untracked:
        raise SystemExit(
            f"Use a checkout whose simulator and debug config match {REVISION}. "
            "See the instructions for a detached worktree."
        )
    check_input(data_dir / "ml-1m/ratings.dat", RATINGS_SHA256)
    if args.checkpoint is not None:
        check_input(args.checkpoint, CHECKPOINT_SHA256)
    expected = json.loads((HERE / "reference/observed.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)

    sys.dont_write_bytecode = True
    sys.path[:0] = [str(repo / "src"), str(HERE / "independent_math")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    if args.mode == "pilot":
        record = {"pilot": pilot(data_dir, args.checkpoint, expected)}
    else:
        record = train(repo, data_dir, output, args.mode == "reference", expected)
    record.update(status="PASS", mode=args.mode, git_revision=revision,
                  coursework_baseline_revision=REVISION,
                  ratings_sha256=RATINGS_SHA256, numpy=np.__version__, torch=torch.__version__,
                  python=sys.version.split()[0], device="cpu")
    if args.mode == "pilot":
        record["checkpoint_sha256"] = CHECKPOINT_SHA256
    payload = json.dumps(record, indent=2)
    (output / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
