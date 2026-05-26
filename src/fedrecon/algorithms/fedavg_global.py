from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from fedrecon.algorithms.aggregation import (
    apply_delta_to_state,
    weighted_average_deltas,
)
from fedrecon.algorithms.compression import compress_delta, estimate_delta_bytes
from fedrecon.algorithms.reconstruction import initialize_local_user_params
from fedrecon.data.client_dataset import ClientDataset
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization


@dataclass
class FedAvgGlobalRoundResult:
    mean_client_loss: float
    num_examples: int
    num_clients: int
    transmitted_bytes: int


def compute_fedavg_client_delta(
    global_model: GlobalMatrixFactorization,
    client: ClientDataset,
    *,
    local_steps: int,
    local_lr: float,
    use_user_bias: bool,
    init_std: float,
) -> tuple[dict[str, torch.Tensor], int, float]:
    device = next(global_model.parameters()).device
    client = client.to(device)

    client_model = copy.deepcopy(global_model).to(device)
    before = client_model.clone_global_state()

    local = initialize_local_user_params(
        embedding_dim=client_model.embedding_dim,
        device=device,
        use_user_bias=use_user_bias,
        init_std=init_std,
    )

    params = list(client_model.parameters()) + [local.user_embedding]
    if local.user_bias is not None:
        params.append(local.user_bias)

    optimizer = torch.optim.SGD(params, lr=local_lr)

    last_loss = 0.0

    for _ in range(local_steps):
        optimizer.zero_grad(set_to_none=True)
        pred = client_model.predict(client.item_ids, local)
        loss = torch.mean((pred - client.ratings) ** 2)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    after = client_model.clone_global_state()
    delta = {key: after[key] - before[key] for key in before.keys()}

    return delta, len(client), last_loss


def run_fedavg_global_round(
    model: GlobalMatrixFactorization,
    selected_clients: list[ClientDataset],
    *,
    local_steps: int,
    local_lr: float,
    server_lr: float,
    use_user_bias: bool,
    init_std: float,
    compression_method: str | None = None,
) -> FedAvgGlobalRoundResult:
    deltas: list[dict[str, torch.Tensor]] = []
    weights: list[int] = []
    losses: list[float] = []
    transmitted_bytes = 0

    for client in selected_clients:
        delta, num_examples, client_loss = compute_fedavg_client_delta(
            global_model=model,
            client=client,
            local_steps=local_steps,
            local_lr=local_lr,
            use_user_bias=use_user_bias,
            init_std=init_std,
        )

        delta = compress_delta(delta, method=compression_method)
        transmitted_bytes += estimate_delta_bytes(delta, method=compression_method)

        deltas.append(delta)
        weights.append(num_examples)
        losses.append(client_loss)

    avg_delta = weighted_average_deltas(deltas, weights)
    new_state = apply_delta_to_state(
        model.clone_global_state(),
        avg_delta,
        server_lr=server_lr,
    )
    model.load_global_state(new_state)

    return FedAvgGlobalRoundResult(
        mean_client_loss=float(sum(losses) / max(1, len(losses))),
        num_examples=int(sum(weights)),
        num_clients=len(selected_clients),
        transmitted_bytes=transmitted_bytes,
    )
