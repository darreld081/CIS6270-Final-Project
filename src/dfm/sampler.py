"""Unguided sampler for the masked discrete flow (project description 4.4).

Starting from ``x_0 = M^L``, each step makes **one** denoiser call, draws a
replacement for every position from the predicted categorical, and commits it
only at positions selected to reveal. The per-site reveal probability is
``d_k = (t_{k+1} - t_k) / (1 - t_k)``, which equals 1 on the last step, so no
mask can survive to the end.

``rho`` and ``gamma`` are accepted but must be zero here. They are the hooks for
the Stage 2 motif-jump controller — the joint reveal rate and the motif tilt —
and this loop is shaped so that controller slots in without a rewrite.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .alphabet import MASK_ID, NUM_BASES, SEQ_LENGTH
from .path import reveal_probability, time_grid

__all__ = ["sample", "SamplingTrace"]


class SamplingTrace(dict):
    """Per-step diagnostics: masked fraction, reveal probability, time."""


def _resolve_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _make_generator(
    seed: int | None,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Generator | None:
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g
    return generator


@torch.no_grad()
def _sample_batch(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    num_steps: int,
    device: torch.device,
    generator: torch.Generator | None,
    trace: SamplingTrace | None,
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

        if trace is not None:
            trace["t"].append(float(t_k))
            trace["reveal_probability"].append(float(d_k))
            trace["masked_fraction"].append(float((x == MASK_ID).float().mean()))

    return x


def sample(
    model: nn.Module,
    num_samples: int,
    seq_length: int = SEQ_LENGTH,
    num_steps: int = 128,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    rho: float = 0.0,
    gamma: float = 0.0,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, SamplingTrace]:
    """Generate sequences from a trained denoiser.

    Args:
        model: denoiser taking ``(x, t)`` and returning ``(B, L, NUM_BASES)``.
        num_samples: how many sequences to produce.
        num_steps: denoiser-call budget ``K``.
        batch_size: sequences per forward pass; defaults to all at once.
        seed: convenience, builds a generator on the sampling device.
        rho: joint-reveal rate. Stage 2 only; must be 0 here.
        gamma: motif tilt strength. Stage 2 only; must be 0 here.
        return_trace: also return per-step diagnostics.

    Returns:
        ``int8[num_samples, seq_length]`` token ids, nucleotides only.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")
    if rho != 0.0 or gamma != 0.0:
        raise NotImplementedError(
            "rho and gamma are Stage 2 (motif guidance) parameters; "
            f"this sampler is unguided and requires rho=gamma=0, got {rho=}, {gamma=}"
        )

    device = _resolve_device(model, device)
    generator = _make_generator(seed, generator, device)
    batch_size = batch_size or num_samples

    trace = (
        SamplingTrace(t=[], reveal_probability=[], masked_fraction=[])
        if return_trace
        else None
    )

    was_training = model.training
    model.eval()
    try:
        chunks: list[Tensor] = []
        remaining = num_samples
        while remaining > 0:
            size = min(batch_size, remaining)
            # Only the first chunk records a trace; later chunks would overwrite it.
            chunk_trace = trace if (trace is not None and not chunks) else None
            chunks.append(
                _sample_batch(
                    model, size, seq_length, num_steps, device, generator, chunk_trace
                )
            )
            remaining -= size
        x = torch.cat(chunks, dim=0)
    finally:
        model.train(was_training)

    if (x == MASK_ID).any():
        raise RuntimeError("sampling finished with masked positions remaining")

    out = x.to(torch.int8).cpu().numpy()
    return (out, trace) if return_trace else out
