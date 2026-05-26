from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fedrecon.data.movielens import load_movielens_1m_clients
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization
from fedrecon.simulation.evaluator import evaluate_reconstruction_detailed
from fedrecon.simulation.logging import CSVLogger
from fedrecon.utils.config import load_config
from fedrecon.utils.device import get_device
from fedrecon.utils.paths import (
    find_project_root,
    make_run_dir,
    resolve_project_path,
    save_config,
)
from fedrecon.utils.seed import seed_everything


class CentralizedMatrixFactorization(nn.Module):
    def __init__(
        self,
        *,
        num_train_users: int,
        num_items: int,
        embedding_dim: int,
        use_user_bias: bool,
        use_item_bias: bool,
        use_global_bias: bool,
        init_std: float,
    ) -> None:
        super().__init__()

        self.use_user_bias = use_user_bias
        self.use_item_bias = use_item_bias
        self.use_global_bias = use_global_bias

        self.user_embeddings = nn.Embedding(num_train_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=init_std)

        if use_user_bias:
            self.user_bias = nn.Embedding(num_train_users, 1)
            nn.init.zeros_(self.user_bias.weight)
        else:
            self.user_bias = None

        if use_item_bias:
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.item_bias.weight)
        else:
            self.item_bias = None

        if use_global_bias:
            self.global_bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("global_bias", torch.tensor(0.0))

    def forward(
        self, user_idx: torch.LongTensor, item_ids: torch.LongTensor
    ) -> torch.Tensor:
        user_emb = self.user_embeddings(user_idx)
        item_emb = self.item_embeddings(item_ids)

        pred = (user_emb * item_emb).sum(dim=-1) + self.global_bias

        if self.user_bias is not None:
            pred = pred + self.user_bias(user_idx).squeeze(-1)

        if self.item_bias is not None:
            pred = pred + self.item_bias(item_ids).squeeze(-1)

        return pred


def build_train_tensors(
    train_clients: dict[int, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    train_client_ids = sorted(train_clients.keys())
    client_to_row = {client_id: row for row, client_id in enumerate(train_client_ids)}

    user_rows = []
    item_ids = []
    ratings = []

    for client_id in train_client_ids:
        client = train_clients[client_id]
        row = client_to_row[client_id]

        user_rows.append(
            torch.full(
                size=(len(client),),
                fill_value=row,
                dtype=torch.long,
            )
        )
        item_ids.append(client.item_ids)
        ratings.append(client.ratings)

    return (
        torch.cat(user_rows),
        torch.cat(item_ids),
        torch.cat(ratings),
        client_to_row,
    )


def make_global_eval_model(
    central_model: CentralizedMatrixFactorization,
    *,
    num_items: int,
    embedding_dim: int,
    use_item_bias: bool,
    use_global_bias: bool,
    init_std: float,
    device: torch.device,
) -> GlobalMatrixFactorization:
    eval_model = GlobalMatrixFactorization(
        num_items=num_items,
        embedding_dim=embedding_dim,
        use_item_bias=use_item_bias,
        use_global_bias=use_global_bias,
        init_std=init_std,
    ).to(device)

    with torch.no_grad():
        eval_model.item_embeddings.weight.copy_(central_model.item_embeddings.weight)

        if (
            use_item_bias
            and central_model.item_bias is not None
            and eval_model.item_bias is not None
        ):
            eval_model.item_bias.weight.copy_(central_model.item_bias.weight)

        eval_model.global_bias.copy_(central_model.global_bias)

    return eval_model


def get_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    name = str(config["centralized"]["optimizer"]).lower()
    lr = float(config["centralized"]["lr"])
    weight_decay = float(config["centralized"].get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer: {name}")


def parse_max_final_eval_clients(config: dict) -> int | None:
    raw = config["evaluation"].get("max_final_eval_clients", None)
    if raw is None or str(raw).lower() == "all":
        return None
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))

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

    user_rows, item_ids, ratings, client_to_row = build_train_tensors(
        data.train_clients
    )

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

    optimizer = get_optimizer(config, model)

    logger = CSVLogger(
        run_dir / "metrics_per_epoch.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "val_rmse",
            "val_mae",
            "val_accuracy",
        ],
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

        val_metrics = {"rmse": "", "mae": "", "accuracy": ""}
        if epoch % eval_every == 0 or epoch == epochs:
            eval_model = make_global_eval_model(
                model,
                num_items=data.num_items,
                embedding_dim=int(model_cfg["embedding_dim"]),
                use_item_bias=bool(model_cfg["use_item_bias"]),
                use_global_bias=bool(model_cfg.get("use_global_bias", True)),
                init_std=float(model_cfg["init_std"]),
                device=device,
            )

            val_metrics, _ = evaluate_reconstruction_detailed(
                model=eval_model,
                clients=data.val_clients.values(),
                support_fraction=float(data_cfg["support_fraction"]),
                split_seed=int(data_cfg["split_seed"]) + 10_000 + epoch,
                reconstruction_steps=int(config["evaluation"]["reconstruction_steps"]),
                reconstruction_lr=float(config["evaluation"]["reconstruction_lr"]),
                use_user_bias=bool(model_cfg["use_user_bias"]),
                init_std=float(model_cfg["init_std"]),
                max_clients=int(config["evaluation"]["max_eval_clients"]),
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

    eval_model = make_global_eval_model(
        model,
        num_items=data.num_items,
        embedding_dim=int(model_cfg["embedding_dim"]),
        use_item_bias=bool(model_cfg["use_item_bias"]),
        use_global_bias=bool(model_cfg.get("use_global_bias", True)),
        init_std=float(model_cfg["init_std"]),
        device=device,
    )

    max_final_eval_clients = parse_max_final_eval_clients(config)

    final_val_metrics, val_client_metrics = evaluate_reconstruction_detailed(
        model=eval_model,
        clients=data.val_clients.values(),
        support_fraction=float(data_cfg["support_fraction"]),
        split_seed=int(data_cfg["split_seed"]) + 20_000,
        reconstruction_steps=int(config["evaluation"]["reconstruction_steps"]),
        reconstruction_lr=float(config["evaluation"]["reconstruction_lr"]),
        use_user_bias=bool(model_cfg["use_user_bias"]),
        init_std=float(model_cfg["init_std"]),
        max_clients=max_final_eval_clients,
    )

    final_test_metrics, test_client_metrics = evaluate_reconstruction_detailed(
        model=eval_model,
        clients=data.test_clients.values(),
        support_fraction=float(data_cfg["support_fraction"]),
        split_seed=int(data_cfg["split_seed"]) + 30_000,
        reconstruction_steps=int(config["evaluation"]["reconstruction_steps"]),
        reconstruction_lr=float(config["evaluation"]["reconstruction_lr"]),
        use_user_bias=bool(model_cfg["use_user_bias"]),
        init_std=float(model_cfg["init_std"]),
        max_clients=max_final_eval_clients,
    )

    val_client_metrics.to_csv(run_dir / "val_client_metrics.csv", index=False)
    test_client_metrics.to_csv(run_dir / "test_client_metrics.csv", index=False)

    final_metrics = {
        "val": final_val_metrics,
        "test": final_test_metrics,
        "epochs": epochs,
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


if __name__ == "__main__":
    main()
