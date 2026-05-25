from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ClientSampler:
    client_ids: Sequence[int]
    clients_per_round: int
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample(self) -> list[int]:
        if self.clients_per_round > len(self.client_ids):
            raise ValueError("clients_per_round cannot exceed number of clients.")
        return self._rng.sample(list(self.client_ids), self.clients_per_round)


def apply_dropout(client_ids: list[int], dropout_prob: float, seed: int | None = None) -> list[int]:
    if dropout_prob <= 0:
        return client_ids
    rng = random.Random(seed)
    return [cid for cid in client_ids if rng.random() >= dropout_prob]
