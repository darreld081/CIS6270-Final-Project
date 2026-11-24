"""Time-conditioned residual dilated CNN denoiser.

Adapted from the ``CNNModel`` released with MOG-DFM,
with two changes for masked discrete flow matching:

* the token embedding has **5** entries (A/C/G/T/M) rather than 4, so the mask
  token gets a learned representation;
* the output head emits **4** logits per position, one per nucleotide — the
  model never predicts a mask.

The stack is translation-equivariant: there is no positional embedding, so a
motif's learned treatment does not depend on where it sits. The dilation
schedule 1, 1, 4, 16, 64 with kernel 9 gives a receptive field of roughly 680
positions, comfortably covering the full 200 bp sequence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .alphabet import NUM_BASES, VOCAB_SIZE
from .path import as_ids

__all__ = ["GaussianFourierProjection", "CNNDenoiser"]

DEFAULT_DILATIONS: tuple[int, ...] = (1, 1, 4, 16, 64)


class GaussianFourierProjection(nn.Module):
    """Random Fourier features for continuous time ``t`` in [0, 1].

    The frequencies are drawn once and never trained, but they are registered
    as a *persistent* buffer: they are part of the function the checkpoint
    represents, and redrawing them on load would silently change every
    prediction.
    """

    W: Tensor

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even for sin/cos pairs, got {embed_dim}")
        self.embed_dim = embed_dim
        self.register_buffer("W", torch.randn(embed_dim // 2) * scale, persistent=True)

    def forward(self, t: Tensor) -> Tensor:
        """``t``: ``(B,)`` -> ``(B, embed_dim)``."""
        projected = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class CNNDenoiser(nn.Module):
    """Predicts the clean base at every position of a partially masked sequence.

    Args:
        embed_dim: token and time embedding width.
        hidden_dim: channel width of the convolutional trunk.
        kernel_size: must be odd so that padding preserves sequence length.
        dilations: one residual block per entry.
        time_scale: standard deviation of the Fourier frequencies.
        center_logits: subtract the per-position mean logit. Softmax and
            cross-entropy are both shift-invariant, so this changes nothing
            numerically; it is kept for parity with the released model and only
            has to stay consistent between training and sampling.

    Shapes:
        input ``x``: ``(B, L)`` token ids including ``MASK_ID``;
        input ``t``: ``(B,)`` or scalar, in [0, 1];
        output: ``(B, L, NUM_BASES)`` logits.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        kernel_size: int = 9,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        time_scale: float = 30.0,
        center_logits: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd to preserve length, got {kernel_size}")
        if not dilations:
            raise ValueError("dilations must not be empty")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.center_logits = center_logits

        # 5 entries: the mask is a token the model must be able to read.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim, scale=time_scale),
            nn.Linear(embed_dim, embed_dim),
        )

        self.stem = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        # padding = dilation * (kernel_size - 1) / 2 keeps every block length-preserving.
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    dilation=d,
                    padding=d * (kernel_size - 1) // 2,
                )
                for d in self.dilations
            ]
        )
        self.time_proj = nn.ModuleList(
            [nn.Linear(embed_dim, hidden_dim) for _ in self.dilations]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(1, hidden_dim) for _ in self.dilations])

        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, NUM_BASES, kernel_size=1),
        )

    def _time_vector(self, t: Tensor | float, batch_size: int, device: torch.device) -> Tensor:
        if not isinstance(t, Tensor):
            t = torch.as_tensor(t, device=device, dtype=torch.float32)
        t = t.to(device=device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.shape != (batch_size,):
            raise ValueError(f"expected t of shape ({batch_size},) or a scalar, got {tuple(t.shape)}")
        return t

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected x of shape (B, L), got {tuple(x.shape)}")
        x = as_ids(x)
        t = self._time_vector(t, x.shape[0], x.device)

        h = self.token_embedding(x).permute(0, 2, 1)  # (B, E, L)
        time_emb = F.silu(self.time_embed(t))  # (B, E); silu == swish
        h = F.silu(self.stem(h))  # (B, H, L)

        for conv, proj, norm in zip(self.blocks, self.time_proj, self.norms):
            # FiLM-style additive time conditioning, broadcast along length.
            conditioned = norm(h + proj(time_emb)[:, :, None])
            h = h + F.silu(conv(conditioned))

        logits = self.head(h).permute(0, 2, 1)  # (B, L, NUM_BASES)
        if self.center_logits:
            logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits
