from __future__ import annotations
from dataclasses import dataclass
from fedrecon.algorithms.aggregation import (
    apply_delta_to_state,
    weighted_average_deltas,
)
from fedrecon.algorithms.client_update import compute_client_delta
from fedrecon.algorithms.compression import compress_delta, estimate_delta_bytes
from fedrecon.algorithms.reconstruction import reconstruct_local_params
from fedrecon.data.client_dataset import ClientDataset
from fedrecon.data.splits import random_support_query_split
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization


@dataclass
class FedReconRoundResult:
    mean_client_loss: float
    num_examples: int
    num_clients: int
    transmitted_bytes: int


def run_fedrecon_round(
    model: GlobalMatrixFactorization,
    selected_clients: list[ClientDataset],
    *,
    support_fraction: float,
    split_seed: int,
    reconstruction_steps: int,
    reconstruction_lr: float,
    client_update_steps: int,
    client_update_lr: float,
    server_lr: float,
    use_user_bias: bool,
    init_std: float,
    compression_method: str | None = None,
    reconstruction_batch_size: int | None = None,
    client_update_batch_size: int | None = None,
) -> FedReconRoundResult:
    deltas = []
    weights = []
    losses = []
    transmitted_bytes = 0

    for client in selected_clients:
        split = random_support_query_split(
            client, support_fraction=support_fraction, seed=split_seed
        )
        local = reconstruct_local_params(
            model=model,
            support=split.support,
            steps=reconstruction_steps,
            lr=reconstruction_lr,
            use_user_bias=use_user_bias,
            init_std=init_std,
            batch_size=reconstruction_batch_size,
        )
        delta, num_query_examples, client_loss = compute_client_delta(
            global_model=model,
            local=local,
            query=split.query,
            steps=client_update_steps,
            lr=client_update_lr,
            batch_size=client_update_batch_size,
        )
        delta = compress_delta(delta, method=compression_method)
        transmitted_bytes += estimate_delta_bytes(delta, method=compression_method)
        deltas.append(delta)
        weights.append(num_query_examples)
        losses.append(client_loss)

    avg_delta = weighted_average_deltas(deltas, weights)
    new_state = apply_delta_to_state(
        model.clone_global_state(), avg_delta, server_lr=server_lr
    )
    model.load_global_state(new_state)

    return FedReconRoundResult(
        mean_client_loss=float(sum(losses) / max(1, len(losses))),
        num_examples=int(sum(weights)),
        num_clients=len(selected_clients),
        transmitted_bytes=transmitted_bytes,
    )
