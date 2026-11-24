"""Masked probability path, its training loss, and the reveal schedule.

The path runs from ``x_0 = M^L`` at ``t = 0`` to clean DNA at ``t = 1``. Each
base is revealed independently with probability ``t`` as per
    q_t(x | z) = prod_i [ (1 - t) 1{x_i = M} + t 1{x_i = z_i} ]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .alphabet import MASK_ID, NUM_BASES

__all__ = [
    "MASK_ID",
    "as_ids",
    "sample_t",
    "sample_x_t",
    "masked_cross_entropy",
    "time_grid",
    "reveal_probability",
]

TIME_EPS = 1e-8


def sample_t(
    batch_size: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    eps: float = TIME_EPS,
) -> Tensor:
    t = torch.rand(batch_size, device=device, generator=generator)
    return t * (1.0 - eps)


def as_ids(x: Tensor) -> Tensor:
    return x if x.dtype == torch.long else x.long()


def _broadcast_t(t: Tensor | float, x_1: Tensor) -> Tensor:
    """Normalize t to shape(B, 1)"""
    if not isinstance(t, Tensor):
        t = torch.as_tensor(t, device=x_1.device, dtype=torch.float32)
    t = t.to(device=x_1.device)
    if t.ndim == 0:
        t = t.expand(x_1.shape[0])
    if t.shape != (x_1.shape[0],):
        raise ValueError(f"expected t of shape ({x_1.shape[0]},) or a scalar, got {tuple(t.shape)}")
    return t.unsqueeze(1)


def sample_x_t(
    x_1: Tensor,
    t: Tensor | float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``x_t ~ q_t(. | x_1)``: keep each base with probability ``t``.

    Args:
        x_1: (B, L) clean token ids
        t: (B,)

    Returns:
        (B, L) ids
    """
    x_1 = as_ids(x_1)
    t = _broadcast_t(t, x_1)
    noise = torch.rand(x_1.shape, device=x_1.device, generator=generator)
    keep = noise < t
    return x_1.masked_fill(~keep, MASK_ID)


def masked_cross_entropy(logits: Tensor, x_1: Tensor, x_t: Tensor) -> Tensor:
    """calculate cross entropy loss between logits and x_1, masked by x_t."""
    if logits.shape[-1] != NUM_BASES:
        raise ValueError(f"expected {NUM_BASES} logits per position, got {logits.shape[-1]}")
    if logits.shape[:2] != x_1.shape or x_1.shape != x_t.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)}, x_1 {tuple(x_1.shape)}, "
            f"x_t {tuple(x_t.shape)}"
        )
    x_1 = as_ids(x_1)
    per_position = F.cross_entropy(
        logits.transpose(1, 2).float(), x_1, reduction="none"
    )  # (B, L)
    masked = (x_t == MASK_ID).to(per_position.dtype)  # (B, L)
    per_sample = (per_position * masked).sum(dim=1) / masked.sum(dim=1).clamp(min=1.0)
    return per_sample.mean()


def time_grid(
    num_steps: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Uniform grid ``t_k = k / K`` for ``k = 0 .. K``, shape ``(K + 1,)``."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")
    return torch.arange(num_steps + 1, device=device, dtype=torch.float32) / num_steps


def reveal_probability(t: Tensor | float, t_next: Tensor | float) -> Tensor:
    """calculate per site reveal prob"""
    t = torch.as_tensor(t, dtype=torch.float32)
    t_next = torch.as_tensor(t_next, dtype=torch.float32)
    remaining = (1.0 - t).clamp(min=TIME_EPS)
    return ((t_next - t) / remaining).clamp(0.0, 1.0)
