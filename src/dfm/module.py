"""LightningModule wrapping the denoiser, the mask path, and the Eq. (7) loss."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

from .model import DEFAULT_DILATIONS, CNNDenoiser
from .path import as_ids, masked_cross_entropy, sample_t, sample_x_t

__all__ = ["MaskedDFM"]


class MaskedDFM(pl.LightningModule):
    """Masked discrete flow matching on 200 bp DNA."""

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

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        return self.model(x, t)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x_1 = as_ids(batch)
        t = sample_t(x_1.shape[0], device=x_1.device)
        x_t = sample_x_t(x_1, t)
        loss = masked_cross_entropy(self.model(x_t, t), x_1, x_t)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=x_1.shape[0],
        )
        return loss

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
