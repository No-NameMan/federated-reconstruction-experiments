from __future__ import annotations
from pathlib import Path
import torch
import json
import time
from tqdm import trange
from fedrecon.algorithms.fedrecon import run_fedrecon_round
from fedrecon.data.movielens import load_movielens_1m_clients
from fedrecon.data.samplers import ClientSampler, apply_dropout
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization
from fedrecon.simulation.evaluator import (
    evaluate_reconstruction,
    evaluate_reconstruction_detailed,
)
from fedrecon.simulation.logging import CSVLogger
from fedrecon.utils.device import get_device
from fedrecon.utils.paths import (
    find_project_root,
    make_run_dir,
    resolve_project_path,
    save_config,
)
from fedrecon.utils.seed import seed_everything


def _parse_optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "all", "full"}:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def run_training(config: dict) -> Path:
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    start_time = time.perf_counter()

    project_root = find_project_root()
    output_dir = resolve_project_path(config["experiment"]["output_dir"], project_root)
    data_dir = resolve_project_path(config["data"]["data_dir"], project_root)

    run_dir = make_run_dir(output_dir, config["experiment"]["name"])
    save_config(config, run_dir)

    device = get_device(prefer_cuda=True)
    print(f"Using device: {device}")

    data_cfg = config["data"]
    data = load_movielens_1m_clients(
        data_dir=data_dir,
        min_ratings_per_user=int(data_cfg["min_ratings_per_user"]),
        user_split=data_cfg["user_split"],
        seed=int(data_cfg["split_seed"]),
    )

    model_cfg = config["model"]
    model = GlobalMatrixFactorization(
        num_items=data.num_items,
        embedding_dim=int(model_cfg["embedding_dim"]),
        use_item_bias=bool(model_cfg["use_item_bias"]),
        use_global_bias=bool(model_cfg.get("use_global_bias", True)),
        init_std=float(model_cfg["init_std"]),
    ).to(device)

    sampler = ClientSampler(
        client_ids=sorted(data.train_clients.keys()),
        clients_per_round=int(config["federated"]["clients_per_round"]),
        seed=seed,
    )

    logger = CSVLogger(
        run_dir / "metrics_per_round.csv",
        fieldnames=[
            "round",
            "train_loss",
            "val_rmse",
            "val_mae",
            "val_accuracy",
            "round_bytes",
            "total_bytes",
            "num_clients",
            "num_examples",
        ],
    )

    eval_reconstruction_steps = int(
        config["evaluation"].get(
            "reconstruction_steps",
            config["reconstruction"]["steps"],
        )
    )

    eval_reconstruction_lr = float(
        config["evaluation"].get(
            "reconstruction_lr",
            config["reconstruction"]["lr"],
        )
    )

    reconstruction_batch_size = _parse_optional_int(
        config["reconstruction"].get("batch_size", None)
    )

    client_update_batch_size = _parse_optional_int(
        config["client_update"].get("batch_size", None)
    )

    eval_reconstruction_batch_size = _parse_optional_int(
        config["evaluation"].get(
            "reconstruction_batch_size",
            config["reconstruction"].get("batch_size", None),
        )
    )

    total_bytes = 0

    if bool(config["evaluation"].get("eval_at_start", True)):
        val_metrics = evaluate_reconstruction(
            model=model,
            clients=data.val_clients.values(),
            support_fraction=float(data_cfg["support_fraction"]),
            split_seed=int(data_cfg["split_seed"]) + 10_000,
            reconstruction_steps=eval_reconstruction_steps,
            reconstruction_lr=eval_reconstruction_lr,
            use_user_bias=bool(model_cfg["use_user_bias"]),
            init_std=float(model_cfg["init_std"]),
            max_clients=int(config["evaluation"]["max_eval_clients"]),
            reconstruction_batch_size=eval_reconstruction_batch_size,
        )

        logger.log(
            {
                "round": 0,
                "train_loss": "",
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_accuracy": val_metrics["accuracy"],
                "round_bytes": 0,
                "total_bytes": total_bytes,
                "num_clients": 0,
                "num_examples": 0,
            }
        )

    rounds = int(config["federated"]["rounds"])

    for round_idx in trange(1, rounds + 1, desc="FEDRECON rounds"):
        selected_ids = sampler.sample()
        selected_ids = apply_dropout(
            selected_ids,
            dropout_prob=float(config["federated"]["client_dropout_prob"]),
            seed=seed + round_idx,
        )
        if len(selected_ids) == 0:
            continue

        compression_method = None
        if bool(config["compression"]["enabled"]):
            compression_method = str(config["compression"]["method"])

        result = run_fedrecon_round(
            model=model,
            selected_clients=[data.train_clients[cid] for cid in selected_ids],
            support_fraction=float(data_cfg["support_fraction"]),
            split_seed=int(data_cfg["split_seed"]) + round_idx,
            reconstruction_steps=int(config["reconstruction"]["steps"]),
            reconstruction_lr=float(config["reconstruction"]["lr"]),
            client_update_steps=int(config["client_update"]["steps"]),
            client_update_lr=float(config["client_update"]["lr"]),
            server_lr=float(config["server"]["lr"]),
            use_user_bias=bool(model_cfg["use_user_bias"]),
            init_std=float(model_cfg["init_std"]),
            compression_method=compression_method,
            reconstruction_batch_size=reconstruction_batch_size,
            client_update_batch_size=client_update_batch_size,
        )
        total_bytes += result.transmitted_bytes

        val_metrics = {"rmse": "", "mae": "", "accuracy": ""}
        if round_idx % int(config["evaluation"]["eval_every"]) == 0:
            val_metrics = evaluate_reconstruction(
                model=model,
                clients=data.val_clients.values(),
                support_fraction=float(data_cfg["support_fraction"]),
                split_seed=int(data_cfg["split_seed"]) + 10_000 + round_idx,
                reconstruction_steps=eval_reconstruction_steps,
                reconstruction_lr=eval_reconstruction_lr,
                use_user_bias=bool(model_cfg["use_user_bias"]),
                init_std=float(model_cfg["init_std"]),
                max_clients=int(config["evaluation"]["max_eval_clients"]),
                reconstruction_batch_size=eval_reconstruction_batch_size,
            )

        logger.log(
            {
                "round": round_idx,
                "train_loss": result.mean_client_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_accuracy": val_metrics["accuracy"],
                "round_bytes": result.transmitted_bytes,
                "total_bytes": total_bytes,
                "num_clients": result.num_clients,
                "num_examples": result.num_examples,
            }
        )

        if round_idx % int(config["logging"]["checkpoint_every"]) == 0:
            torch.save(
                {
                    "round": round_idx,
                    "model_state": model.state_dict(),
                    "config": config,
                },
                run_dir / f"checkpoint_round_{round_idx}.pt",
            )

    torch.save(
        {"round": rounds, "model_state": model.state_dict(), "config": config},
        run_dir / "final_model.pt",
    )

    final_eval_max_clients_raw = config["evaluation"].get(
        "max_final_eval_clients", None
    )

    if (
        final_eval_max_clients_raw is None
        or str(final_eval_max_clients_raw).lower() == "all"
    ):
        max_final_eval_clients = None
    else:
        max_final_eval_clients = int(final_eval_max_clients_raw)

    final_val_metrics, val_client_metrics = evaluate_reconstruction_detailed(
        model=model,
        clients=data.val_clients.values(),
        support_fraction=float(data_cfg["support_fraction"]),
        split_seed=int(data_cfg["split_seed"]) + 20_000,
        reconstruction_steps=eval_reconstruction_steps,
        reconstruction_lr=eval_reconstruction_lr,
        use_user_bias=bool(model_cfg["use_user_bias"]),
        init_std=float(model_cfg["init_std"]),
        max_clients=max_final_eval_clients,
        reconstruction_batch_size=eval_reconstruction_batch_size,
    )

    final_test_metrics, test_client_metrics = evaluate_reconstruction_detailed(
        model=model,
        clients=data.test_clients.values(),
        support_fraction=float(data_cfg["support_fraction"]),
        split_seed=int(data_cfg["split_seed"]) + 30_000,
        reconstruction_steps=eval_reconstruction_steps,
        reconstruction_lr=eval_reconstruction_lr,
        use_user_bias=bool(model_cfg["use_user_bias"]),
        init_std=float(model_cfg["init_std"]),
        max_clients=max_final_eval_clients,
        reconstruction_batch_size=eval_reconstruction_batch_size,
    )

    val_client_metrics.to_csv(run_dir / "val_client_metrics.csv", index=False)
    test_client_metrics.to_csv(run_dir / "test_client_metrics.csv", index=False)

    elapsed_seconds = time.perf_counter() - start_time

    final_metrics = {
        "val": final_val_metrics,
        "test": final_test_metrics,
        "total_bytes": total_bytes,
        "rounds": rounds,
        "elapsed_seconds": elapsed_seconds,
        "run_dir": str(run_dir),
    }

    with (run_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    print("Final validation metrics:", final_val_metrics)
    print("Final test metrics:", final_test_metrics)

    print(f"Run saved to: {run_dir}")
    return run_dir
