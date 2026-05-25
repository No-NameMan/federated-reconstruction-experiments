from __future__ import annotations
import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(mse(pred, target)).detach().cpu())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).detach().cpu())


def rounded_accuracy(pred: torch.Tensor, target: torch.Tensor, min_rating: float = 1.0, max_rating: float = 5.0) -> float:
    rounded = torch.round(pred).clamp(min_rating, max_rating)
    return float((rounded == target).float().mean().detach().cpu())
