"""Fréchet Biological Distance (FBD).

Following Stark et al. (Dirichlet Flow Matching, 2024): embed real and
generated sequences with a trained sequence classifier and report the Fréchet
distance between the two embedding distributions, as FID does for images.

There is no pretrained oracle for this dataset, so the embedder is a small CNN
trained to tell the four DHS cell lines apart. Its penultimate layer has to
encode cell-type-specific regulatory grammar to do that, which is what makes
the distance more informative than k-mer statistics.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .alphabet import NUM_BASES
from .data import CELL_TAGS
from .metrics import frechet_distance

__all__ = [
    "EMBED_DIM",
    "CellTypeCNN",
    "train_classifier",
    "save_classifier",
    "load_classifier",
    "embed",
    "fbd",
]

EMBED_DIM = 128


class CellTypeCNN(nn.Module):
    """Token ids ``(B, L)`` -> cell-type logits; ``embed`` gives the features."""

    def __init__(self, num_classes: int = len(CELL_TAGS), channels: int = 128,
                 embed_dim: int = EMBED_DIM, dropout: float = 0.2):
        super().__init__()
        layers, c_in = [], NUM_BASES
        for _ in range(3):
            layers += [nn.Conv1d(c_in, channels, 9, padding=4), nn.BatchNorm1d(channels),
                       nn.ReLU(), nn.MaxPool1d(2)]
            c_in = channels
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(channels, embed_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = F.one_hot(x.long(), NUM_BASES).float().transpose(1, 2)  # (B, 4, L)
        return self.proj(self.conv(h).mean(dim=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(x)))


def _revcomp_ids(x: torch.Tensor) -> torch.Tensor:
    # A=0 C=1 G=2 T=3, so the complement is 3 - id.
    return (NUM_BASES - 1 - x).flip(dims=[1])


@torch.no_grad()
def _accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str,
              batch_size: int = 1024) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).long().to(device)
        correct += (model(xb).argmax(1).cpu().numpy() == y[i:i + batch_size]).sum()
    return float(correct / len(x))


def train_classifier(
    labeled_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> tuple[CellTypeCNN, dict]:
    """Train the embedder, keeping the epoch with the best val accuracy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_tr, y_tr = labeled_splits["train"]
    x_va, y_va = labeled_splits["val"]
    x_te, y_te = labeled_splits["test"]

    model = CellTypeCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state, stale = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_tr))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb = torch.from_numpy(x_tr[idx]).long().to(device)
            yb = torch.from_numpy(y_tr[idx]).to(device)
            # DHS are strand-agnostic: flip a random half of the batch.
            flip = torch.rand(len(idx), device=device) < 0.5
            xb = torch.where(flip[:, None], _revcomp_ids(xb), xb)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        val_acc = _accuracy(model, x_va, y_va, device)
        if verbose:
            print(f"epoch {epoch + 1:2d}  loss {loss.item():.3f}  val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc, best_state, stale = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    metrics = {"val_acc": best_acc, "test_acc": _accuracy(model, x_te, y_te, device),
               "chance": float(np.bincount(y_te).max() / len(y_te))}
    return model, metrics


def save_classifier(model: CellTypeCNN, path: Path | str, metrics: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics or {}}, path)


def load_classifier(path: Path | str, device: str = "cpu") -> tuple[CellTypeCNN, dict]:
    blob = torch.load(path, map_location=device)
    model = CellTypeCNN().to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("metrics", {})


@torch.no_grad()
def embed(model: CellTypeCNN, sequences: np.ndarray, batch_size: int = 512,
          device: str = "cpu") -> np.ndarray:
    """Penultimate-layer features, shape ``(N, EMBED_DIM)``."""
    model.eval()
    sequences = np.asarray(sequences)
    out = []
    for i in range(0, len(sequences), batch_size):
        xb = torch.from_numpy(sequences[i:i + batch_size]).long().to(device)
        out.append(model.embed(xb).float().cpu().numpy())
    return np.concatenate(out)


def fbd(model: CellTypeCNN, generated: np.ndarray, reference: np.ndarray,
        device: str = "cpu") -> float:
    """Fréchet Biological Distance between two sets of token-id sequences."""
    return frechet_distance(embed(model, generated, device=device),
                            embed(model, reference, device=device))
