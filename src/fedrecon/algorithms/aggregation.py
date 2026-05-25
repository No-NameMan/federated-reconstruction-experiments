from __future__ import annotations
from collections.abc import Mapping
import torch

Delta = dict[str, torch.Tensor]


def weighted_average_deltas(deltas: list[Delta], weights: list[int | float]) -> Delta:
    if len(deltas) == 0:
        raise ValueError("Cannot aggregate an empty list of deltas.")
    if len(deltas) != len(weights):
        raise ValueError("deltas and weights must have the same length.")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Total aggregation weight must be positive.")

    result: Delta = {}
    for key in deltas[0].keys():
        acc = torch.zeros_like(deltas[0][key])
        for delta, weight in zip(deltas, weights):
            acc = acc + delta[key] * (float(weight) / total_weight)
        result[key] = acc
    return result


def apply_delta_to_state(
    state: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    server_lr: float,
) -> dict[str, torch.Tensor]:
    return {key: value + server_lr * delta[key] for key, value in state.items()}
