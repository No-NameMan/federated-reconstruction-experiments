from __future__ import annotations
import torch
from fedrecon.data.client_dataset import ClientDataset
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization, LocalUserParams


def initialize_local_user_params(
    embedding_dim: int,
    device: torch.device,
    use_user_bias: bool = True,
    init_std: float = 0.05,
) -> LocalUserParams:
    user_embedding = torch.empty(embedding_dim, device=device).normal_(mean=0.0, std=init_std)
    user_embedding.requires_grad_(True)
    user_bias = torch.zeros((), device=device, requires_grad=True) if use_user_bias else None
    return LocalUserParams(user_embedding=user_embedding, user_bias=user_bias)


def reconstruct_local_params(
    model: GlobalMatrixFactorization,
    support: ClientDataset,
    steps: int,
    lr: float,
    use_user_bias: bool,
    init_std: float,
) -> LocalUserParams:
    device = next(model.parameters()).device
    support = support.to(device)
    local = initialize_local_user_params(model.embedding_dim, device, use_user_bias, init_std)

    params = [local.user_embedding]
    if local.user_bias is not None:
        params.append(local.user_bias)

    for p in model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.SGD(params, lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model.predict(support.item_ids, local)
        loss = torch.mean((pred - support.ratings) ** 2)
        loss.backward()
        optimizer.step()

    detached = LocalUserParams(
        user_embedding=local.user_embedding.detach(),
        user_bias=None if local.user_bias is None else local.user_bias.detach(),
    )

    for p in model.parameters():
        p.requires_grad_(True)

    return detached
