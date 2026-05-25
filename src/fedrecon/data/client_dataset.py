from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ClientDataset:
    client_id: int
    item_ids: torch.LongTensor
    ratings: torch.FloatTensor
    timestamps: torch.LongTensor | None = None

    def __len__(self) -> int:
        return int(self.ratings.numel())

    def to(self, device: torch.device) -> "ClientDataset":
        return ClientDataset(
            client_id=self.client_id,
            item_ids=self.item_ids.to(device),
            ratings=self.ratings.to(device),
            timestamps=None if self.timestamps is None else self.timestamps.to(device),
        )
