"""Sequence statistics for comparing generated DNA against held-out K562.

Every function takes plain ``(N, L)`` integer arrays of token ids, so Stage 2
reuses them unchanged and only adds motif success and an
outside-the-motif-span variant on top.

None of these establish regulatory function. They measure whether generated
sequences resemble held-out DHS regions in composition and local structure,
which is a necessary but nowhere near sufficient condition.
"""

from __future__ import annotations

import numpy as np

from .alphabet import NUM_BASES, canonical, decode_many

__all__ = [
    "is_valid_array",
    "gc_content",
    "kmer_counts",
    "kmer_frequencies",
    "jensen_shannon_divergence",
    "kmer_jsd",
    "unique_fraction",
    "mean_pairwise_hamming",
    "memorization_fraction",
    "uniform_random_sequences",
    "frechet_distance",
    "evaluate",
]


def _as_sequences(sequences: np.ndarray) -> np.ndarray:
    """Validate shape, preserving dtype.

    Splits are int8. Widening to int64 here would multiply every downstream
    temporary by eight, which matters because the pairwise-distance estimator
    materializes large index selections.
    """
    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"expected an (N, L) array, got shape {sequences.shape}")
    if sequences.size == 0:
        raise ValueError("expected at least one sequence")
    return sequences


def _as_int(sequences: np.ndarray) -> np.ndarray:
    """Validate and widen to int64, for the base-4 k-mer index only."""
    # int8 overflows the rolling index as soon as k >= 4.
    return _as_sequences(sequences).astype(np.int64, copy=False)


def is_valid_array(sequences: np.ndarray, seq_length: int | None = None) -> bool:
    """Every entry a nucleotide, every row the expected length."""
    sequences = _as_sequences(sequences)
    if seq_length is not None and sequences.shape[1] != seq_length:
        return False
    return bool((sequences >= 0).all() and (sequences < NUM_BASES).all())


def gc_content(sequences: np.ndarray) -> np.ndarray:
    """Per-sequence fraction of G or C, shape ``(N,)``."""
    sequences = _as_sequences(sequences)
    return ((sequences == 1) | (sequences == 2)).mean(axis=1)


def kmer_counts(sequences: np.ndarray, k: int) -> np.ndarray:
    """Counts of every length-``k`` word, shape ``(NUM_BASES ** k,)``.

    Words are indexed in base 4, so index 0 is ``AAA...`` and the last is
    ``TTT...``.
    """
    sequences = _as_int(sequences)
    length = sequences.shape[1]
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if k > length:
        raise ValueError(f"k={k} exceeds sequence length {length}")

    windows = length - k + 1
    index = np.zeros((sequences.shape[0], windows), dtype=np.int64)
    for offset in range(k):
        index = index * NUM_BASES + sequences[:, offset : offset + windows]
    return np.bincount(index.ravel(), minlength=NUM_BASES**k)


def kmer_frequencies(sequences: np.ndarray, k: int) -> np.ndarray:
    """Normalized k-mer distribution, shape ``(NUM_BASES ** k,)``."""
    counts = kmer_counts(sequences, k).astype(np.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("no k-mers found")
    return counts / total


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JSD in bits. Symmetric, and bounded in [0, 1].

    Unlike KL this needs no smoothing: a word absent from both distributions
    contributes exactly zero.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")

    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> float:
        # 0 log 0 is 0; the mixture m is nonzero wherever a is.
        nonzero = a > 0
        return float(np.sum(a[nonzero] * np.log2(a[nonzero] / m[nonzero])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def kmer_jsd(generated: np.ndarray, reference: np.ndarray, k: int) -> float:
    """JSD in bits between two sets' k-mer distributions."""
    return jensen_shannon_divergence(
        kmer_frequencies(generated, k), kmer_frequencies(reference, k)
    )


def unique_fraction(sequences: np.ndarray) -> float:
    """Fraction of distinct sequences; 1.0 means no repeats."""
    sequences = _as_sequences(sequences)
    distinct = np.unique(sequences, axis=0).shape[0]
    return distinct / sequences.shape[0]


def mean_pairwise_hamming(
    sequences: np.ndarray,
    num_pairs: int = 100_000,
    rng: np.random.Generator | None = None,
    chunk_size: int = 8192,
) -> float:
    """Mean normalized Hamming distance over randomly chosen distinct pairs.

    Full pairwise is O(N^2); a random subsample estimates the same quantity.
    Two independent uniform sequences differ at 3/4 of positions, so 0.75 is
    the no-structure ceiling — real genomic loci sit there too, so this metric
    detects mode collapse but cannot tell a good model from a random one.

    Pairs are processed in chunks: selecting all ``num_pairs`` rows at once
    materializes two ``(num_pairs, L)`` arrays, which reached 343 MB for a
    0.2 MB input before this was chunked.
    """
    sequences = _as_sequences(sequences)
    n, length = sequences.shape
    if n < 2:
        raise ValueError("need at least two sequences")
    if num_pairs < 1:
        raise ValueError(f"num_pairs must be positive, got {num_pairs}")
    rng = rng or np.random.default_rng(0)

    differing = 0
    for start in range(0, num_pairs, chunk_size):
        size = min(chunk_size, num_pairs - start)
        i = rng.integers(0, n, size=size)
        j = rng.integers(0, n - 1, size=size)
        j = np.where(j >= i, j + 1, j)  # uniform over j != i, no rejection loop
        differing += int((sequences[i] != sequences[j]).sum())
    return differing / (num_pairs * length)


def memorization_fraction(generated: np.ndarray, reference: np.ndarray) -> float:
    """Fraction of generated sequences exactly matching a reference one.

    Matching is strand-agnostic: a generated sequence counts as a copy if it
    equals a reference sequence or its reverse complement.
    """
    reference_keys = {canonical(s) for s in decode_many(_as_sequences(reference))}
    generated_strings = decode_many(_as_sequences(generated))
    hits = sum(canonical(s) in reference_keys for s in generated_strings)
    return hits / len(generated_strings)


def uniform_random_sequences(
    num_sequences: int, seq_length: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """I.i.d. uniform nucleotides: the baseline that makes a JSD interpretable.

    Without it, "4-mer JSD = 0.02" says nothing about whether the model learned
    anything beyond the alphabet.
    """
    rng = rng or np.random.default_rng(0)
    return rng.integers(0, NUM_BASES, size=(num_sequences, seq_length), dtype=np.int8)


def frechet_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Squared Fréchet distance between Gaussians fitted to two embedding sets.

    ``||mu_a - mu_b||^2 + Tr(S_a) + Tr(S_b) - 2 Tr((S_a S_b)^{1/2})``, the
    formula behind FID and FBD. The trace of the matrix square root is taken
    from the eigenvalues of the symmetric ``S_a^{1/2} S_b S_a^{1/2}``, which
    shares its spectrum with ``S_a S_b`` and avoids a scipy dependency.

    The estimate is biased upward at small N, so only compare scores computed
    from equal-sized sets.
    """
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected (N, D) arrays, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"embedding dims differ: {a.shape[1]} vs {b.shape[1]}")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("need at least two embeddings per set")

    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)

    w, v = np.linalg.eigh(cov_a)
    sqrt_a = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    middle = sqrt_a @ cov_b @ sqrt_a
    trace_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((middle + middle.T) / 2), 0, None)).sum()

    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2.0 * trace_sqrt)


def evaluate(
    generated: np.ndarray,
    reference: np.ndarray,
    train: np.ndarray | None = None,
    kmer_sizes: tuple[int, ...] = (3, 4, 6),
    num_pairs: int = 100_000,
    seed: int = 0,
) -> dict:
    """Full metric set for one batch of generated sequences.

    Args:
        generated: model samples.
        reference: held-out sequences (the test split).
        train: training sequences, for the memorization check.
    """
    generated = _as_sequences(generated)
    reference = _as_sequences(reference)
    rng = np.random.default_rng(seed)

    baseline = uniform_random_sequences(
        reference.shape[0], reference.shape[1], np.random.default_rng(seed)
    )

    generated_gc = gc_content(generated)
    reference_gc = gc_content(reference)

    results: dict = {
        "num_generated": int(generated.shape[0]),
        "num_reference": int(reference.shape[0]),
        "seq_length": int(generated.shape[1]),
        "valid": is_valid_array(generated, reference.shape[1]),
        "gc_generated_mean": float(generated_gc.mean()),
        "gc_generated_std": float(generated_gc.std()),
        "gc_reference_mean": float(reference_gc.mean()),
        "gc_reference_std": float(reference_gc.std()),
        "gc_abs_difference": float(abs(generated_gc.mean() - reference_gc.mean())),
        # < 1 means the model is under-dispersed: right centre, too narrow a
        # spread. Easy to miss when only the means are compared.
        "gc_std_ratio": float(generated_gc.std() / reference_gc.std())
        if reference_gc.std() > 0
        else float("nan"),
        "unique_fraction": unique_fraction(generated),
        "mean_pairwise_hamming": mean_pairwise_hamming(generated, num_pairs, rng),
        "reference_pairwise_hamming": mean_pairwise_hamming(
            reference, num_pairs, np.random.default_rng(seed)
        ),
        "kmer_jsd": {},
        "kmer_jsd_uniform_baseline": {},
    }

    for k in kmer_sizes:
        results["kmer_jsd"][str(k)] = kmer_jsd(generated, reference, k)
        results["kmer_jsd_uniform_baseline"][str(k)] = kmer_jsd(baseline, reference, k)

    if train is not None:
        results["memorization_fraction"] = memorization_fraction(generated, train)

    return results
