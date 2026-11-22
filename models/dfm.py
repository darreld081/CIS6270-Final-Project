import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers import BertConfig

import pdb

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier features for continuous time t in [0, 1].
    Produces 2 * embed_dim features: [sin(W t), cos(W t)].
    """

    W: torch.Tensor
    
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even."
        self.embed_dim = embed_dim
        # Fixed random frequencies
        self.register_buffer(
            "W",
            torch.randn(embed_dim // 2) * scale,
            persistent=False,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) continuous in [0, 1]
        returns: (B, embed_dim)
        """
        # Ensure float
        t = t.float().unsqueeze(-1)  # (B, 1)
        angles = t * self.W          # (B, embed_dim//2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TimeEmbedding(nn.Module):
    """
    t -> time embedding in model hidden_dim using Fourier features + MLP.
    """
    def __init__(self, hidden_dim: int, fourier_dim: int = 64):
        super().__init__()
        assert fourier_dim % 2 == 0, "fourier_dim must be even for sine/cosine pairs."

        self.fourier = GaussianFourierProjection(fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) in [0, 1]
        returns: (B, hidden_dim)
        """
        ft = self.fourier(t)        # (B, fourier_dim)
        return self.mlp(ft)         # (B, hidden_dim)

# -------------------------
# DiT building blocks
# -------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DiTBlock1D(nn.Module):
    """
    1D DiT block with AdaLayerNorm-style time conditioning.
    x:      (B, L, D)
    t_emb:  (B, T) global time embedding
    mask:   (B, L) bool or 0/1, True/1 for padding (for MultiheadAttention)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        time_dim: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)

        # time-conditioned scale & shift for both norms
        self.time_proj1 = nn.Linear(time_dim, 2 * hidden_dim)  # scale1, shift1
        self.time_proj2 = nn.Linear(time_dim, 2 * hidden_dim)  # scale2, shift2

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # (B, L, D)
        )
        self.mlp = MLP(hidden_dim, mlp_ratio=mlp_ratio, dropout=resid_dropout)

    def forward(self, x, t_emb, key_padding_mask=None):
        B, L, D = x.shape

        # ----- Self-attention branch -----
        # Adaptive LayerNorm + FiLM from time embedding
        scale1, shift1 = self.time_proj1(t_emb).chunk(2, dim=-1)  # (B, D), (B, D)
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)   # (B, L, D)

        attn_out, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,  # True for pads
            need_weights=False,
        )
        x = x + attn_out

        # ----- MLP branch -----
        scale2, shift2 = self.time_proj2(t_emb).chunk(2, dim=-1)
        h2 = self.norm2(x)
        h2 = h2 * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)

        mlp_out = self.mlp(h2)
        x = x + mlp_out

        return x


class DNAModel(nn.Module):
    """
    Base model for discrete flow matching over DNABERT-2 tokens.
    Input:
        x_t: (B, L) token ids at time t
        t:   (B,) continuous in [0, 1]
        attention_mask: (B, L) with 1 for valid tokens, 0 for pad
    Output:
        logits over x_1: (B, L, V)
    """
    def __init__(
        self,
        cfg,
        n_layers: int = 4,
        n_heads: int = 8,
        time_dim: int = 512,
        time_fourier_dim: int = 64,
        dna_model_name: str = "zhihan1996/DNABERT-2-117M",
        num_unfrozen_layers: int = 3,
        freeze_dna_encoder: bool = True,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.cfg = cfg

        dnabert_config = BertConfig.from_pretrained(dna_model_name)
        self.dnabert = AutoModel.from_pretrained(dna_model_name, trust_remote_code=True, config=dnabert_config)

        if freeze_dna_encoder:
            for name, param in self.dnabert.named_parameters():
                param.requires_grad = False

            encoder_layers = self.dnabert.encoder.layer  # ModuleList of length 12
            for layer in encoder_layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        # Time embedding
        self.time_embed = TimeEmbedding(
            hidden_dim=time_dim,
            fourier_dim=time_fourier_dim,
        )

        self.blocks = nn.ModuleList([
            DiTBlock1D(
                hidden_dim=self.cfg.model.hidden_dim,
                num_heads=n_heads,
                time_dim=time_dim,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(self.cfg.model.hidden_dim, eps=1e-6)
        self.output_proj = nn.Linear(self.cfg.model.hidden_dim, self.cfg.model.vocab_size)

    def forward(
        self,
        x_t: torch.Tensor,           # (B, L) token ids
        t: torch.Tensor,             # (B,) continuous in [0, 1]
        attention_mask: torch.Tensor | None = None,  # (B, L) or None
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, L, V)
        """
        if attention_mask is None:
            attention_mask = (x_t != 0).long()  # naive: treat id=0 as pad

        h = self.dnabert(x_t)[0] # (B, L, hidden_dim)

        # Time embedding
        t_emb = self.time_embed(t)          # (B, time_dim)

        # Transformer head (use key padding mask: True for pads)
        key_padding_mask = (attention_mask == 0)  # (B, L) bool

        for blk in self.blocks:
            h = blk(h, t_emb, key_padding_mask=key_padding_mask)

        # Final norm + projection to vocab logits
        h = self.final_norm(h)            # (B, L, hidden_dim)
        logits = self.output_proj(h)      # (B, L, V)

        return logits


class DFM(pl.LightningModule):
    def __init__(self, model, path, loss_fn, cfg):
        super().__init__()

        self.model = model
        self.path = path
        self.loss_fn = loss_fn
        self.cfg = cfg

        self._total_steps = None
    
    def forward(self, x_t, t):
        return self.model(x_t, t)

    def general_step(self, x_1):
        x_0 = torch.randint_like(x_1, high=self.cfg.model.vocab_size)
        t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - 1e-8)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        logits = self.model(x_t=path_sample.x_t, t=path_sample.t)
        loss = self.loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

        return loss, logits, t
    
    def training_step(self, batch, batch_idx):
        x_1, label = batch
        B = x_1.shape[0]

        loss, logits, t = self.general_step(x_1)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_1, label = batch
        B = x_1.shape[0]

        loss, logits, t = self.general_step(x_1)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)

        prediction = logits.argmax(-1)
        accuracy = (prediction == x_1).float().mean()
        self.log("val_acc", accuracy, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.training.lr),
            betas=(self.cfg.optim.beta1, self.cfg.optim.beta2),
            eps=float(self.cfg.optim.eps),
            weight_decay=self.cfg.optim.weight_decay,
            fused=self.cfg.optim.fused,
        )

        warmup_ratio = getattr(self.cfg.optim, "warmup_ratio", 0.1)
        min_scale = 0.1

        def lr_lambda(global_step: int):
            # until on_train_start runs we just return 1.0
            if self._total_steps is None or self._total_steps == 0:
                return 1.0

            total_steps = self._total_steps
            warmup_steps = max(1, int(warmup_ratio * total_steps))

            if global_step < warmup_steps:
                # linear warmup: 0.1 -> 1.0
                alpha = (global_step + 1) / warmup_steps
                return 0.1 + 0.9 * alpha
            else:
                # cosine from 1.0 down to min_scale
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))   # 1 -> 0
                return min_scale + (1.0 - min_scale) * cosine

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",   # <- per-step
                "frequency": 1,
            },
        }
    
    def on_train_start(self):
        # how many optimizer steps we will take in this fit
        self._total_steps = self.trainer.estimated_stepping_batches