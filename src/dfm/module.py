"""LightningModule wrapping the denoiser, the mask path, and the Eq. (7) loss.

Hyperparameters arrive as plain keyword arguments and are recorded into the
checkpoint by ``save_hyperparameters()``, so a checkpoint is self-describing
for the section A.4 reproducibility record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from .alphabet import MASK_ID
from .model import DEFAULT_DILATIONS, CNNDenoiser, count_parameters
from .path import as_ids, masked_cross_entropy, sample_t, sample_x_t

__all__ = ["MaskedDFM"]


class MaskedDFM(pl.LightningModule):
    """Masked discrete flow matching on 200 bp DNA.

    Validation uses a *fixed* masking pattern per batch index, reseeded
    identically every epoch. With freshly drawn times and masks each epoch the
    validation loss carries enough sampling noise to make `save_top_k=1`
    partly a lottery; holding the pattern fixed means successive epochs are
    compared on the same problem.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        kernel_size: int = 9,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        time_scale: float = 30.0,
        center_logits: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        warmup_frac: float = 0.1,
        min_lr_scale: float = 0.1,
        val_seed: int = 1234,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CNNDenoiser(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dilations=dilations,
            time_scale=time_scale,
            center_logits=center_logits,
        )

    # --- plumbing -----------------------------------------------------------

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        return self.model(x, t)

    @property
    def num_parameters(self) -> int:
        return count_parameters(self.model)

    @staticmethod
    def _unpack(batch) -> Tensor:
        """Accept a bare tensor or a one-element tuple from the DataLoader."""
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        return as_ids(batch)

    def _val_generator(self, batch_idx: int, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.hparams.val_seed * 100_003 + batch_idx)
        return generator

    # --- steps --------------------------------------------------------------

    def _shared_step(
        self, x_1: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        t = sample_t(x_1.shape[0], device=x_1.device, generator=generator)
        x_t = sample_x_t(x_1, t, generator=generator)
        logits = self.model(x_t, t)
        loss = masked_cross_entropy(logits, x_1, x_t)
        return loss, logits, x_t

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x_1 = self._unpack(batch)
        loss, _, _ = self._shared_step(x_1)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=x_1.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        x_1 = self._unpack(batch)
        generator = self._val_generator(batch_idx, x_1.device)
        loss, logits, x_t = self._shared_step(x_1, generator=generator)

        masked = x_t == MASK_ID
        correct = (logits.argmax(dim=-1) == x_1) & masked
        # Accuracy over masked positions only; visible positions are given.
        accuracy = correct.sum() / masked.sum().clamp(min=1)

        self.log(
            "val_loss", loss, prog_bar=True, on_epoch=True, batch_size=x_1.shape[0]
        )
        self.log(
            "val_acc", accuracy, prog_bar=True, on_epoch=True, batch_size=x_1.shape[0]
        )
        return loss

    # --- optimization -------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(self.hparams.warmup_frac * total_steps))
        min_scale = self.hparams.min_lr_scale

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup from min_scale to 1.
                return min_scale + (1.0 - min_scale) * (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_scale + (1.0 - min_scale) * cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda),
                "interval": "step",
                "frequency": 1,
            },
        }
