from __future__ import annotations

import torch
from torch import nn


class CentralizedMatrixFactorization(nn.Module):
    """Centralized matrix factorization model for train users.

    This model owns user-specific parameters for the centralized training users.
    For RECONEVAL on held-out users, only the learned global/item parameters are
    copied into ``GlobalMatrixFactorization`` and user parameters are reconstructed.
    """

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
        self,
        user_idx: torch.LongTensor,
        item_ids: torch.LongTensor,
    ) -> torch.Tensor:
        user_emb = self.user_embeddings(user_idx)
        item_emb = self.item_embeddings(item_ids)

        pred = (user_emb * item_emb).sum(dim=-1) + self.global_bias

        if self.user_bias is not None:
            pred = pred + self.user_bias(user_idx).squeeze(-1)

        if self.item_bias is not None:
            pred = pred + self.item_bias(item_ids).squeeze(-1)

        return pred
