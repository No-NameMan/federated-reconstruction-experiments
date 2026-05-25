from __future__ import annotations
import torch


def compress_delta(delta: dict[str, torch.Tensor], method: str | None = None) -> dict[str, torch.Tensor]:
    if method is None or method == "none":
        return delta
    if method == "fp16":
        return {k: v.half().float() for k, v in delta.items()}
    if method == "sign":
        return {k: torch.sign(v) * torch.mean(torch.abs(v)) for k, v in delta.items()}
    if method in {"int8", "int4"}:
        levels = 255 if method == "int8" else 15
        return {k: _uniform_symmetric_quantize_dequantize(v, levels=levels) for k, v in delta.items()}
    raise ValueError(f"Unknown compression method: {method}")


def _uniform_symmetric_quantize_dequantize(x: torch.Tensor, levels: int) -> torch.Tensor:
    max_abs = torch.max(torch.abs(x))
    if max_abs == 0:
        return torch.zeros_like(x)
    qmax = levels // 2
    scale = max_abs / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def estimate_delta_bytes(delta: dict[str, torch.Tensor], method: str | None = None) -> int:
    numel = sum(v.numel() for v in delta.values())
    if method is None or method == "none":
        return int(numel * 4)
    if method == "fp16":
        return int(numel * 2)
    if method == "int8":
        return int(numel)
    if method == "int4":
        return int((numel + 1) // 2)
    if method == "sign":
        return int((numel + 7) // 8 + 4 * len(delta))
    raise ValueError(f"Unknown compression method: {method}")
