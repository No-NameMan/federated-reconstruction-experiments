from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch

from fedrecon.algorithms.reconstruction import reconstruct_local_params
from fedrecon.data.client_dataset import ClientDataset
from fedrecon.data.splits import random_support_query_split
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization
from fedrecon.simulation.metrics import mae, rmse, rounded_accuracy


@torch.no_grad()
def _evaluate_query(
    model: GlobalMatrixFactorization,
    local,
    query: ClientDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    query = query.to(device)
    pred = model.predict(query.item_ids, local)
    return pred.detach(), query.ratings.detach()


def evaluate_reconstruction_detailed(
    model: GlobalMatrixFactorization,
    clients: Iterable[ClientDataset],
    *,
    support_fraction: float,
    split_seed: int,
    reconstruction_steps: int,
    reconstruction_lr: float,
    use_user_bias: bool,
    init_std: float,
    max_clients: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    rows: list[dict] = []

    for idx, client in enumerate(clients):
        if max_clients is not None and idx >= max_clients:
            break

        split = random_support_query_split(
            client,
            support_fraction=support_fraction,
            seed=split_seed,
        )

        local = reconstruct_local_params(
            model=model,
            support=split.support,
            steps=reconstruction_steps,
            lr=reconstruction_lr,
            use_user_bias=use_user_bias,
            init_std=init_std,
        )

        pred, target = _evaluate_query(model, local, split.query)

        pred_cpu = pred.cpu()
        target_cpu = target.cpu()

        preds.append(pred_cpu)
        targets.append(target_cpu)

        rows.append(
            {
                "client_id": int(client.client_id),
                "num_examples_total": int(len(client)),
                "num_support_examples": int(len(split.support)),
                "num_query_examples": int(len(split.query)),
                "rmse": rmse(pred_cpu, target_cpu),
                "mae": mae(pred_cpu, target_cpu),
                "accuracy": rounded_accuracy(pred_cpu, target_cpu),
            }
        )

    if not preds:
        empty_metrics = {
            "rmse": float("nan"),
            "mae": float("nan"),
            "accuracy": float("nan"),
            "num_clients": 0,
            "num_examples": 0,
        }
        return empty_metrics, pd.DataFrame(rows)

    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)

    metrics = {
        "rmse": rmse(pred_all, target_all),
        "mae": mae(pred_all, target_all),
        "accuracy": rounded_accuracy(pred_all, target_all),
        "num_clients": len(rows),
        "num_examples": int(target_all.numel()),
    }

    return metrics, pd.DataFrame(rows)


def evaluate_reconstruction(
    model: GlobalMatrixFactorization,
    clients: Iterable[ClientDataset],
    *,
    support_fraction: float,
    split_seed: int,
    reconstruction_steps: int,
    reconstruction_lr: float,
    use_user_bias: bool,
    init_std: float,
    max_clients: int | None = None,
) -> dict[str, float]:
    metrics, _ = evaluate_reconstruction_detailed(
        model=model,
        clients=clients,
        support_fraction=support_fraction,
        split_seed=split_seed,
        reconstruction_steps=reconstruction_steps,
        reconstruction_lr=reconstruction_lr,
        use_user_bias=use_user_bias,
        init_std=init_std,
        max_clients=max_clients,
    )
    return metrics
