from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class LocalUserParams:
    user_embedding: torch.Tensor
    user_bias: torch.Tensor | None = None


class GlobalMatrixFactorization(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        use_item_bias: bool = True,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_item_bias = use_item_bias

        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=init_std)

        if use_item_bias:
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.item_bias.weight)
        else:
            self.item_bias = None

        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def predict(self, item_ids: torch.LongTensor, local: LocalUserParams) -> torch.Tensor:
        item_emb = self.item_embeddings(item_ids)
        user_emb = local.user_embedding
        if user_emb.dim() == 1:
            user_emb = user_emb.unsqueeze(0).expand_as(item_emb)
        scores = (item_emb * user_emb).sum(dim=-1) + self.global_bias
        if self.item_bias is not None:
            scores = scores + self.item_bias(item_ids).squeeze(-1)
        if local.user_bias is not None:
            scores = scores + local.user_bias
        return scores

    def clone_global_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_global_state(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=True)
