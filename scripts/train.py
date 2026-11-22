"""Train the masked-DFM denoiser on the K562 splits.

    python scripts/train.py                  # full run, settings below
    python scripts/train.py --smoke          # 50 CPU steps, proves the pipeline
    python scripts/train.py --name my_run    # label the run directory

Hyperparameters live here as constants. Edit them in place; every run copies
them into run.json and into the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from dfm.data import load_split
from dfm.module import MaskedDFM

# --- hyperparameters --------------------------------------------------------
SEED = 0

# model
EMBED_DIM = 256
HIDDEN_DIM = 256
KERNEL_SIZE = 9
DILATIONS = (1, 1, 4, 16, 64)
TIME_SCALE = 30.0
CENTER_LOGITS = True

# optimization
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 0.01
BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8
EPOCHS = 200
WARMUP_FRAC = 0.1
MIN_LR_SCALE = 0.1
GRAD_CLIP = 1.0

# runtime
NUM_WORKERS = 2
ACCELERATOR = "auto"

# smoke-test overrides (--smoke). VAL_CHECK_INTERVAL must be below MAX_STEPS, or
# the run stops before validation ever fires and there is no val_loss to select
# a checkpoint on.
SMOKE = dict(
    EMBED_DIM=64,
    HIDDEN_DIM=64,
    BATCH_SIZE=32,
    EPOCHS=1,
    MAX_STEPS=50,
    NUM_WORKERS=0,
    VAL_CHECK_INTERVAL=25,
)
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"


def make_loader(split: str, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    # A Tensor is a valid map-style Dataset, so batches arrive as (B, L) ints
    # with no TensorDataset tuple wrapper.
    sequences = torch.from_numpy(load_split(PROCESSED_DIR, split)).long()
    return DataLoader(
        sequences,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=shuffle and len(sequences) > batch_size,
        persistent_workers=workers > 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run")
    parser.add_argument("--name", default=None, help="run directory name")
    parser.add_argument("--epochs", type=int, default=None, help="override EPOCHS")
    args = parser.parse_args()

    embed_dim, hidden_dim = EMBED_DIM, HIDDEN_DIM
    batch_size, epochs, workers = BATCH_SIZE, EPOCHS, NUM_WORKERS
    max_steps, accelerator = -1, ACCELERATOR
    val_check_interval = 1.0  # once per epoch

    if args.smoke:
        embed_dim, hidden_dim = SMOKE["EMBED_DIM"], SMOKE["HIDDEN_DIM"]
        batch_size, epochs = SMOKE["BATCH_SIZE"], SMOKE["EPOCHS"]
        max_steps, workers, accelerator = SMOKE["MAX_STEPS"], SMOKE["NUM_WORKERS"], "cpu"
        val_check_interval = SMOKE["VAL_CHECK_INTERVAL"]
    if args.epochs is not None:
        epochs = args.epochs

    run_name = args.name or ("smoke" if args.smoke else "base")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(SEED, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    precision = "bf16-mixed" if torch.cuda.is_available() else "32-true"

    train_loader = make_loader("train", batch_size, workers, shuffle=True)
    val_loader = make_loader("val", batch_size, workers, shuffle=False)

    module = MaskedDFM(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        kernel_size=KERNEL_SIZE,
        dilations=DILATIONS,
        time_scale=TIME_SCALE,
        center_logits=CENTER_LOGITS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
        eps=EPS,
        warmup_frac=WARMUP_FRAC,
        min_lr_scale=MIN_LR_SCALE,
    )

    checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_epochs=epochs,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        gradient_clip_val=GRAD_CLIP,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    record = {
        "run_name": run_name,
        "seed": SEED,
        "smoke": args.smoke,
        "num_parameters": module.num_parameters,
        "train_sequences": len(train_loader.dataset),
        "val_sequences": len(val_loader.dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "max_steps": max_steps,
        "precision": precision,
        "hparams": dict(module.hparams),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pytorch_lightning": pl.__version__,
            "numpy": np.__version__,
        },
    }
    print(json.dumps(record, indent=2, default=str))

    trainer.fit(module, train_loader, val_loader)

    best_score = checkpoint.best_model_score
    record["best_val_loss"] = None if best_score is None else float(best_score)
    record["best_checkpoint"] = str(checkpoint.best_model_path) or None
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")

    if record["best_val_loss"] is None:
        print("\nno checkpoint selected: val_loss was never logged")
        print("(validation did not run before training stopped)")
        return 1

    print(f"\nbest val_loss {record['best_val_loss']:.4f}")
    print(f"checkpoint     {record['best_checkpoint']}")
    print(f"run record     {run_dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
