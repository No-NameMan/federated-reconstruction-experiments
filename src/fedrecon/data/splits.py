from __future__ import annotations
from dataclasses import dataclass
import torch
from fedrecon.data.client_dataset import ClientDataset


@dataclass(frozen=True)
class SupportQuerySplit:
    support: ClientDataset
    query: ClientDataset


def random_support_query_split(
    client: ClientDataset,
    support_fraction: float,
    seed: int | None = None,
) -> SupportQuerySplit:
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support_fraction must be in (0, 1).")
    n = len(client)
    if n < 2:
        raise ValueError("Cannot split a client with fewer than 2 examples.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed + int(client.client_id))

    perm = torch.randperm(n, generator=generator)
    support_size = max(1, min(n - 1, int(round(n * support_fraction))))
    support_idx = perm[:support_size]
    query_idx = perm[support_size:]

    def subset(indices: torch.LongTensor) -> ClientDataset:
        return ClientDataset(
            client_id=client.client_id,
            item_ids=client.item_ids[indices],
            ratings=client.ratings[indices],
            timestamps=None if client.timestamps is None else client.timestamps[indices],
        )

    return SupportQuerySplit(support=subset(support_idx), query=subset(query_idx))
