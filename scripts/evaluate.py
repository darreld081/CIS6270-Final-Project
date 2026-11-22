"""Score generated sequences against held-out K562 and write metrics.json.

    python scripts/evaluate.py --samples runs/base/samples/K128_seed0_n1024.npy

Writes metrics.json and a two-panel diagnostic figure beside the samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dfm.data import load_split  # noqa: E402
from dfm.metrics import evaluate, gc_content  # noqa: E402

# --- settings ---------------------------------------------------------------
KMER_SIZES = (3, 4, 6)
NUM_PAIRS = 100_000
SEED = 0

# Categorical slots 1 and 2 of the reference palette, in fixed order.
MODEL_COLOR = "#2a78d6"  # blue
REFERENCE_COLOR = "#eb6834"  # orange
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e3e2df"
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def style_axes(ax) -> None:
    """Recessive grid and axes; ink stays in text tokens, never series color."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.title.set_color(INK)


def make_figure(generated: np.ndarray, reference: np.ndarray, report: dict, out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    # --- GC content: two distributions, so a legend is required -------------
    bins = np.linspace(0.2, 0.8, 49)
    left.hist(
        gc_content(reference),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=REFERENCE_COLOR,
        label="Held-out K562",
    )
    left.hist(
        gc_content(generated),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=MODEL_COLOR,
        label="Generated",
    )
    left.set_title("GC content", fontsize=11, loc="left", pad=12)
    left.set_xlabel("fraction G or C", fontsize=9, color=INK_MUTED)
    left.set_ylabel("density", fontsize=9, color=INK_MUTED)
    legend = left.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    style_axes(left)

    # --- k-mer JSD against the uniform baseline -----------------------------
    ks = list(report["kmer_jsd"])
    model = [report["kmer_jsd"][k] for k in ks]
    baseline = [report["kmer_jsd_uniform_baseline"][k] for k in ks]

    positions = np.arange(len(ks), dtype=float)
    width = 0.38
    gap = 0.012  # ~2px surface gap between adjacent fills

    right.bar(
        positions - width / 2 - gap,
        baseline,
        width,
        color=REFERENCE_COLOR,
        label="Uniform random",
    )
    right.bar(
        positions + width / 2 + gap, model, width, color=MODEL_COLOR, label="Generated"
    )

    # Six bars only: direct-label all of them, the key numbers of the report.
    headroom = max(baseline + model) or 1.0
    for x, value in zip(positions - width / 2 - gap, baseline):
        right.text(x, value + headroom * 0.03, f"{value:.3f}", ha="center", fontsize=8, color=INK_MUTED)
    for x, value in zip(positions + width / 2 + gap, model):
        right.text(x, value + headroom * 0.03, f"{value:.3f}", ha="center", fontsize=8, color=INK)

    right.set_xticks(positions)
    right.set_xticklabels([f"k = {k}" for k in ks])
    right.set_ylim(0, headroom * 1.18)
    right.set_title(
        "k-mer divergence from held-out  (lower is better)", fontsize=11, loc="left", pad=12
    )
    right.set_ylabel("Jensen-Shannon divergence (bits)", fontsize=9, color=INK_MUTED)
    legend = right.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK)
    style_axes(right)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--split", default="test", help="reference split")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    samples_path = args.samples
    if not samples_path.is_absolute():
        samples_path = PROJECT_ROOT / samples_path
    if not samples_path.is_file():
        raise SystemExit(f"samples not found: {samples_path}")

    generated = np.load(samples_path)
    reference = load_split(PROCESSED_DIR, args.split)
    train = load_split(PROCESSED_DIR, "train")

    report = evaluate(
        generated,
        reference,
        train=train,
        kmer_sizes=KMER_SIZES,
        num_pairs=NUM_PAIRS,
        seed=args.seed,
    )
    report["samples_file"] = str(samples_path)
    report["reference_split"] = args.split

    meta_path = samples_path.with_suffix(".meta.json")
    if meta_path.is_file():
        report["sampling"] = json.loads(meta_path.read_text())

    out_json = samples_path.with_suffix(".metrics.json")
    out_png = samples_path.with_suffix(".metrics.png")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_figure(generated, reference, report, out_png)

    print(f"samples    {samples_path.name}  n={report['num_generated']}")
    print(f"reference  {args.split}  n={report['num_reference']}")
    print(f"valid      {report['valid']}")
    print()
    print(f"GC  mean  generated {report['gc_generated_mean']:.4f}  "
          f"held-out {report['gc_reference_mean']:.4f}  "
          f"|diff| {report['gc_abs_difference']:.4f}")
    # Spread matters as much as the mean: a model can centre GC correctly while
    # producing a far narrower range than real sequences.
    print(f"GC  std   generated {report['gc_generated_std']:.4f}  "
          f"held-out {report['gc_reference_std']:.4f}  "
          f"ratio {report['gc_std_ratio']:.2f}")
    print()
    print(f"{'k':>3}  {'JSD (bits)':>11}  {'uniform':>9}  {'ratio':>7}")
    for k, value in report["kmer_jsd"].items():
        base = report["kmer_jsd_uniform_baseline"][k]
        ratio = value / base if base else float("nan")
        print(f"{k:>3}  {value:>11.4f}  {base:>9.4f}  {ratio:>7.2f}")
    print()
    print(f"unique fraction        {report['unique_fraction']:.4f}")
    print(f"pairwise Hamming       {report['mean_pairwise_hamming']:.4f}  "
          f"(held-out {report['reference_pairwise_hamming']:.4f})")
    print(f"memorization           {report['memorization_fraction']:.4f}")
    if "sampling" in report:
        print(f"ms/sequence            {report['sampling']['ms_per_sequence']:.2f}  "
              f"({report['sampling']['denoiser_calls_per_sample']} denoiser calls)")
    print()
    print(f"wrote {out_json.name} and {out_png.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
