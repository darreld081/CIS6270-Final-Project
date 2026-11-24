"""Unguided sampler for the masked discrete flow
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .alphabet import MASK_ID, NUM_BASES, SEQ_LENGTH
from .path import reveal_probability, time_grid

__all__ = ["sample"]


@torch.no_grad()
def _sample_batch(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    x = torch.full((batch_size, seq_length), MASK_ID, dtype=torch.long, device=device)
    grid = time_grid(num_steps, device=device)

    for k in range(num_steps):
        t_k = grid[k]
        d_k = reveal_probability(grid[k], grid[k + 1]).to(device)

        # One denoiser call per step; every draw below uses these predictions.
        probs = torch.softmax(model(x, t_k), dim=-1)  # (B, L, NUM_BASES)
        proposal = torch.multinomial(
            probs.reshape(-1, NUM_BASES), num_samples=1, generator=generator
        ).reshape(batch_size, seq_length)

        noise = torch.rand(x.shape, device=device, generator=generator)
        reveal = (noise < d_k) & (x == MASK_ID)
        x = torch.where(reveal, proposal, x)

    return x


def sample(
    model: nn.Module,
    num_samples: int,
    seq_length: int = SEQ_LENGTH,
    num_steps: int = 128,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate sequences from a trained denoiser.

    Args:
        model: denoiser taking ``(x, t)`` and returning ``(B, L, NUM_BASES)``.
        num_samples: how many sequences to produce.
        num_steps: denoiser-call budget ``K``.
        batch_size: sequences per forward pass; defaults to all at once.
        device: defaults to the model's device.
        seed: makes sampling reproducible.

    Returns:
        ``int8[num_samples, seq_length]`` token ids, nucleotides only.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")

    device = torch.device(device) if device is not None else next(model.parameters()).device
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    batch_size = batch_size or num_samples

    was_training = model.training
    model.eval()
    try:
        chunks: list[Tensor] = []
        remaining = num_samples
        while remaining > 0:
            size = min(batch_size, remaining)
            chunks.append(_sample_batch(model, size, seq_length, num_steps, device, generator))
            remaining -= size
        x = torch.cat(chunks, dim=0)
    finally:
        model.train(was_training)

    if (x == MASK_ID).any():
        raise RuntimeError("sampling finished with masked positions remaining")

    return x.to(torch.int8).cpu().numpy()
