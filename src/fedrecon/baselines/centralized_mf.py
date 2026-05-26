from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fedrecon.data.client_dataset import ClientDataset
from fedrecon.models.centralized_matrix_factorization import CentralizedMatrixFactorization
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization


def build_train_tensors(
    train_clients: dict[int, ClientDataset],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """Flatten client-partitioned ratings into tensors for centralized training."""
    train_client_ids = sorted(train_clients.keys())
    client_to_row = {client_id: row for row, client_id in enumerate(train_client_ids)}

    user_rows: list[torch.Tensor] = []
    item_ids: list[torch.Tensor] = []
    ratings: list[torch.Tensor] = []

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
    """Copy centralized global/item parameters into a FEDRECON eval model."""
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


def get_centralized_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
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


def parse_max_final_eval_clients(config: dict[str, Any]) -> int | None:
    raw = config["evaluation"].get("max_final_eval_clients", None)
    if raw is None or str(raw).lower() == "all":
        return None
    return int(raw)
