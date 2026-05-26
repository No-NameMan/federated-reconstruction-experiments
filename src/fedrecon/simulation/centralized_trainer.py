from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

import time

from fedrecon.baselines.centralized_mf import (
    build_train_tensors,
    get_centralized_optimizer,
    make_global_eval_model,
    parse_max_final_eval_clients,
)
from fedrecon.data.movielens import load_movielens_1m_clients
from fedrecon.models.centralized_matrix_factorization import CentralizedMatrixFactorization
from fedrecon.simulation.evaluator import evaluate_reconstruction_detailed
from fedrecon.simulation.logging import CSVLogger
from fedrecon.utils.device import get_device
from fedrecon.utils.paths import find_project_root, make_run_dir, resolve_project_path, save_config
from fedrecon.utils.seed import seed_everything


def _evaluate_centralized_reconeval(
    *,
    central_model: CentralizedMatrixFactorization,
    data: Any,
    clients: Any,
    config: dict[str, Any],
    split_seed: int,
    max_clients: int | None,
    device: torch.device,
) -> tuple[dict[str, float], Any]:
    data_cfg = config["data"]
    model_cfg = config["model"]

    eval_model = make_global_eval_model(
        central_model,
        num_items=data.num_items,
        embedding_dim=int(model_cfg["embedding_dim"]),
        use_item_bias=bool(model_cfg["use_item_bias"]),
        use_global_bias=bool(model_cfg.get("use_global_bias", True)),
        init_std=float(model_cfg["init_std"]),
        device=device,
    )

    return evaluate_reconstruction_detailed(
        model=eval_model,
        clients=clients,
        support_fraction=float(data_cfg["support_fraction"]),
        split_seed=split_seed,
        reconstruction_steps=int(config["evaluation"]["reconstruction_steps"]),
        reconstruction_lr=float(config["evaluation"]["reconstruction_lr"]),
        use_user_bias=bool(model_cfg["use_user_bias"]),
        init_std=float(model_cfg["init_std"]),
        max_clients=max_clients,
    )


def run_centralized_training(config: dict[str, Any]) -> Path:
    """Train centralized MF and evaluate item/global parameters via RECONEVAL."""
    seed_everything(int(config["experiment"]["seed"]))

    start_time = time.perf_counter()

    project_root = find_project_root()
    output_dir = resolve_project_path(config["experiment"]["output_dir"], project_root)
    data_dir = resolve_project_path(config["data"]["data_dir"], project_root)

    run_dir = make_run_dir(output_dir, config["experiment"]["name"])
    save_config(config, run_dir)

    device = get_device(prefer_cuda=True)
    print(f"Using device: {device}")

    data_cfg = config["data"]
    model_cfg = config["model"]

    data = load_movielens_1m_clients(
        data_dir=data_dir,
        min_ratings_per_user=int(data_cfg["min_ratings_per_user"]),
        user_split=data_cfg["user_split"],
        seed=int(data_cfg["split_seed"]),
    )

    user_rows, item_ids, ratings, client_to_row = build_train_tensors(data.train_clients)

    dataset = TensorDataset(user_rows, item_ids, ratings)
    loader = DataLoader(
        dataset,
        batch_size=int(config["centralized"]["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    model = CentralizedMatrixFactorization(
        num_train_users=len(client_to_row),
        num_items=data.num_items,
        embedding_dim=int(model_cfg["embedding_dim"]),
        use_user_bias=bool(model_cfg["use_user_bias"]),
        use_item_bias=bool(model_cfg["use_item_bias"]),
        use_global_bias=bool(model_cfg.get("use_global_bias", True)),
        init_std=float(model_cfg["init_std"]),
    ).to(device)

    optimizer = get_centralized_optimizer(config, model)

    logger = CSVLogger(
        run_dir / "metrics_per_epoch.csv",
        fieldnames=["epoch", "train_loss", "val_rmse", "val_mae", "val_accuracy"],
    )

    eval_every = int(config["evaluation"]["eval_every"])
    epochs = int(config["centralized"]["epochs"])

    for epoch in trange(1, epochs + 1, desc="Centralized MF epochs"):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch_user, batch_item, batch_rating in loader:
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_rating = batch_rating.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_user, batch_item)
            loss = torch.mean((pred - batch_rating) ** 2)
            loss.backward()
            optimizer.step()

            batch_size = int(batch_rating.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size

        train_loss = total_loss / max(1, total_examples)

        val_metrics: dict[str, Any] = {"rmse": "", "mae": "", "accuracy": ""}
        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics, _ = _evaluate_centralized_reconeval(
                central_model=model,
                data=data,
                clients=data.val_clients.values(),
                config=config,
                split_seed=int(data_cfg["split_seed"]) + 10_000 + epoch,
                max_clients=int(config["evaluation"]["max_eval_clients"]),
                device=device,
            )

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

    max_final_eval_clients = parse_max_final_eval_clients(config)

    final_val_metrics, val_client_metrics = _evaluate_centralized_reconeval(
        central_model=model,
        data=data,
        clients=data.val_clients.values(),
        config=config,
        split_seed=int(data_cfg["split_seed"]) + 20_000,
        max_clients=max_final_eval_clients,
        device=device,
    )

    final_test_metrics, test_client_metrics = _evaluate_centralized_reconeval(
        central_model=model,
        data=data,
        clients=data.test_clients.values(),
        config=config,
        split_seed=int(data_cfg["split_seed"]) + 30_000,
        max_clients=max_final_eval_clients,
        device=device,
    )

    val_client_metrics.to_csv(run_dir / "val_client_metrics.csv", index=False)
    test_client_metrics.to_csv(run_dir / "test_client_metrics.csv", index=False)

    elapsed_seconds = time.perf_counter() - start_time

    final_metrics = {
        "val": final_val_metrics,
        "test": final_test_metrics,
        "epochs": epochs,
        "elapsed_seconds": elapsed_seconds,
        "run_dir": str(run_dir),
        "baseline": "centralized_mf_reconeval",
    }

    with (run_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "client_to_row": client_to_row,
        },
        run_dir / "final_model.pt",
    )

    print("Final validation metrics:", final_val_metrics)
    print("Final test metrics:", final_test_metrics)
    print(f"Run saved to: {run_dir}")

    return run_dir
