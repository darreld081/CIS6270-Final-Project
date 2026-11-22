"""Generate sequences from a trained checkpoint.

    python scripts/sample.py --checkpoint runs/base/best.ckpt
    python scripts/sample.py --checkpoint runs/smoke/best.ckpt --num-samples 64 --steps 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dfm.alphabet import SEQ_LENGTH, decode_many
from dfm.module import MaskedDFM
from dfm.sampler import sample

# --- settings ---------------------------------------------------------------
NUM_SAMPLES = 1024
NUM_STEPS = 128  # denoiser-call budget K
BATCH_SIZE = 256
SEED = 0
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = MaskedDFM.load_from_checkpoint(checkpoint, map_location=device)
    module.eval()

    out_dir = args.out or checkpoint.parent / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"K{args.steps}_seed{args.seed}_n{args.num_samples}"

    started = time.perf_counter()
    sequences, trace = sample(
        module.model,
        num_samples=args.num_samples,
        seq_length=SEQ_LENGTH,
        num_steps=args.steps,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        return_trace=True,
    )
    elapsed = time.perf_counter() - started

    meta = {
        "checkpoint": str(checkpoint),
        "device": device,
        "num_samples": int(args.num_samples),
        "num_steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "seconds": elapsed,
        "ms_per_sequence": 1000.0 * elapsed / args.num_samples,
        "denoiser_calls_per_sample": int(args.steps),
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    np.save(out_dir / f"{stem}.npy", sequences)
    strings = decode_many(sequences)
    (out_dir / f"{stem}.txt").write_text("\n".join(strings) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.trace.json").write_text(json.dumps(dict(trace), indent=2) + "\n")

    gc = float(((sequences == 1) | (sequences == 2)).mean())
    unique = len(set(strings)) / len(strings)
    print(f"checkpoint   {checkpoint}")
    print(f"device       {device}")
    print(f"samples      {sequences.shape}  dtype={sequences.dtype}")
    print(f"denoiser calls per sample  {args.steps}")
    print(f"GC content   {gc:.4f}")
    print(f"unique       {unique:.4f}")
    print(f"ms/sequence  {meta['ms_per_sequence']:.2f}")
    print(f"wrote        {out_dir}/{stem}.{{npy,txt,trace.json,meta.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
