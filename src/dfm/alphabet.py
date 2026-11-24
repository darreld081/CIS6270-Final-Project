"""Nucleotide alphabet and token ids for masked discrete flow matching.

The mask token sits *last* in the vocabulary so that model output logits are
always ``logits[..., :NUM_BASES]`` with no index remapping anywhere::

    A = 0   C = 1   G = 2   T = 3   M = 4 (mask)

Sequences are stored as ``int8`` arrays (2.4 MB for 12k x 200) and widened to
``int64`` only when they become tensors.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BASES",
    "TOKENS",
    "NUM_BASES",
    "VOCAB_SIZE",
    "MASK_ID",
    "MASK_CHAR",
    "SEQ_LENGTH",
    "encode",
    "decode",
    "encode_many",
    "decode_many",
    "is_valid",
    "revcomp",
    "canonical",
]

BASES = "ACGT"
MASK_CHAR = "M"
TOKENS = BASES + MASK_CHAR

NUM_BASES = len(BASES)  # 4 output logits
VOCAB_SIZE = len(TOKENS)  # 5 input embeddings
MASK_ID = TOKENS.index(MASK_CHAR)  # 4

#: Sequence length of the DNA-Diffusion DHS dataset.
SEQ_LENGTH = 200

# ASCII -> token id, with -1 marking anything outside the alphabet (notably 'N').
# Lowercase is accepted so that a caller who forgets to uppercase gets correct
# ids rather than a silent wall of -1.
_ENCODE_LUT = np.full(256, -1, dtype=np.int8)
for _i, _c in enumerate(TOKENS):
    _ENCODE_LUT[ord(_c)] = _i
    _ENCODE_LUT[ord(_c.lower())] = _i

_DECODE_LUT = np.frombuffer(TOKENS.encode("ascii"), dtype=np.uint8)

_REVCOMP_TABLE = str.maketrans("ACGTMacgtm", "TGCAMTGCAM")


def _ascii(seq: str) -> np.ndarray:
    return np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)


def encode(seq: str) -> np.ndarray:
    """Encode a sequence string to an ``int8`` array of token ids.

    Raises:
        ValueError: if ``seq`` contains a character outside ``ACGTM``.
    """
    ids = _ENCODE_LUT[_ascii(seq)]
    if (ids < 0).any():
        bad = sorted({seq[i] for i in np.flatnonzero(ids < 0)})
        raise ValueError(f"sequence contains non-alphabet characters: {bad}")
    return ids


def decode(ids: np.ndarray) -> str:
    """Decode a 1-D array of token ids back to a sequence string."""
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise ValueError(f"expected a 1-D array of ids, got shape {ids.shape}")
    if ((ids < 0) | (ids >= VOCAB_SIZE)).any():
        raise ValueError("ids out of range for the alphabet")
    return _DECODE_LUT[ids.astype(np.intp)].tobytes().decode("ascii")


def is_valid(seq: str, length: int | None = None, allow_mask: bool = False) -> bool:
    """Whether ``seq`` is usable as training data.

    By default this means: the requested length (if given) and nucleotides
    only. Mask characters are rejected unless ``allow_mask`` is set, since a
    clean training sequence must never contain one.
    """
    if length is not None and len(seq) != length:
        return False
    if not seq:
        return False
    ids = _ENCODE_LUT[_ascii(seq)]
    if (ids < 0).any():
        return False
    if not allow_mask and (ids == MASK_ID).any():
        return False
    return True


def encode_many(seqs: list[str], length: int | None = None) -> np.ndarray:
    """Encode equal-length sequences into an ``(N, L)`` ``int8`` array."""
    if not seqs:
        return np.empty((0, length or 0), dtype=np.int8)
    if length is None:
        length = len(seqs[0])
    lengths = {len(s) for s in seqs}
    if lengths != {length}:
        raise ValueError(f"expected every sequence to have length {length}, saw {sorted(lengths)}")
    out = np.empty((len(seqs), length), dtype=np.int8)
    for i, s in enumerate(seqs):
        out[i] = encode(s)
    return out


def decode_many(ids: np.ndarray) -> list[str]:
    """Decode an ``(N, L)`` array of token ids to a list of strings."""
    ids = np.asarray(ids)
    if ids.ndim != 2:
        raise ValueError(f"expected a 2-D array of ids, got shape {ids.shape}")
    return [decode(row) for row in ids]


def revcomp(seq: str) -> str:
    """Reverse complement of a sequence string."""
    return seq.translate(_REVCOMP_TABLE)[::-1]


def canonical(seq: str) -> str:
    """Canonical strand-agnostic key: the lexicographic min of the two strands.

    Used to group exact duplicates together with reverse-complement pairs so
    that no such group straddles a train/val/test boundary.
    """
    rc = revcomp(seq)
    return seq if seq <= rc else rc
