from __future__ import annotations
import copy
import torch
from fedrecon.data.client_dataset import ClientDataset
from fedrecon.models.matrix_factorization import GlobalMatrixFactorization, LocalUserParams


def compute_client_delta(
    global_model: GlobalMatrixFactorization,
    local: LocalUserParams,
    query: ClientDataset,
    steps: int,
    lr: float,
) -> tuple[dict[str, torch.Tensor], int, float]:
    device = next(global_model.parameters()).device
    query = query.to(device)
    client_model = copy.deepcopy(global_model).to(device)
    before = client_model.clone_global_state()

    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)
    last_loss = 0.0

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = client_model.predict(query.item_ids, local)
        loss = torch.mean((pred - query.ratings) ** 2)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    after = client_model.clone_global_state()
    delta = {key: after[key] - before[key] for key in before.keys()}
    return delta, len(query), last_loss
