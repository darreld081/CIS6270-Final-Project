"""K562 DHS data: download, filter, and split.

The source is the subset shipped with the DNA-Diffusion repository: a 19-column
tab-separated table of 200 bp DNase-I-hypersensitive sites across four cell
lines, ~12k rows each. We keep K562 and hold out chromosome 1 for test and
chromosome 2 for validation, following the released data loader.

Exact duplicates and reverse-complement pairs are grouped under a canonical key
so that no group straddles a split boundary; each group contributes one
sequence.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .alphabet import SEQ_LENGTH, canonical, encode_many, is_valid

__all__ = [
    "SOURCE_URL",
    "RAW_FILENAME",
    "K562_TAG",
    "TEST_CHROMOSOMES",
    "VAL_CHROMOSOMES",
    "SPLITS",
    "download",
    "load_table",
    "build_splits",
    "CELL_TAGS",
    "build_labeled_splits",
]

SOURCE_URL = (
    "https://raw.githubusercontent.com/pinellolab/DNA-Diffusion/main/data/"
    "K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
)
RAW_FILENAME = "dhs.txt"

#: Value of the ``TAG`` column identifying K562 rows.
K562_TAG = "K562_ENCLB843GMH"

#: All four cell lines in the table; a row's label is its index here.
CELL_TAGS = (
    "GM12878_ENCLB441ZZZ",
    "hESCT0_ENCLB449ZZZ",
    K562_TAG,
    "HepG2_ENCLB029COU",
)

TEST_CHROMOSOMES = ("chr1",)
VAL_CHROMOSOMES = ("chr2",)

SPLITS = ("train", "val", "test")

#: Only these columns are read. Naming them explicitly means an upstream schema
#: change fails loudly here rather than silently shifting a column.
USED_COLUMNS = ["chr", "sequence", "TAG"]

#: Most-held-out split wins when a group spans several chromosomes.
_SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
_PRIORITY_SPLIT = {v: k for k, v in _SPLIT_PRIORITY.items()}


def download(raw_dir: Path | str, url: str = SOURCE_URL, force: bool = False) -> Path:
    """Fetch the DHS table into ``raw_dir``, skipping if already present."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / RAW_FILENAME
    if destination.exists() and not force:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(destination)
    return destination


def load_table(path: Path | str) -> pd.DataFrame:
    """Read the raw table, keeping only the columns we use."""
    frame = pd.read_csv(path, sep="\t", usecols=USED_COLUMNS, dtype=str)
    missing = [c for c in USED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"raw table is missing expected columns: {missing}")
    return frame


def _assign_split(chromosome: str) -> str:
    if chromosome in TEST_CHROMOSOMES:
        return "test"
    if chromosome in VAL_CHROMOSOMES:
        return "val"
    return "train"


def build_splits(
    frame: pd.DataFrame,
    cell_tag: str = K562_TAG,
    seq_length: int = SEQ_LENGTH,
) -> dict[str, np.ndarray]:
    """Filter, group, split, and encode.

    Returns a dict mapping each split name to an ``int8[N, seq_length]`` array
    of token ids.
    """
    frame = frame[frame["TAG"] == cell_tag].copy()
    if frame.empty:
        raise ValueError(f"no rows with TAG == {cell_tag!r}")

    frame["sequence"] = frame["sequence"].str.upper()
    frame = frame[frame["sequence"].map(lambda s: is_valid(s, length=seq_length))].copy()
    if frame.empty:
        raise ValueError("every row was dropped by the validity filter")

    # Group exact duplicates together with reverse-complement pairs.
    frame["canonical"] = frame["sequence"].map(canonical)
    frame["split"] = frame["chr"].map(_assign_split)

    # Resolve the split per group before deduplicating, so a sequence that also
    # appears on a held-out chromosome never leaks into train.
    priority = frame["split"].map(_SPLIT_PRIORITY)
    frame["group_split"] = (
        priority.groupby(frame["canonical"]).transform("min").map(_PRIORITY_SPLIT)
    )

    # One representative per group: first occurrence in file order, which keeps
    # the observed reference strand rather than flipping to the canonical one.
    frame = frame.drop_duplicates(subset="canonical", keep="first")

    return {
        name: encode_many(
            frame.loc[frame["group_split"] == name, "sequence"].tolist(), length=seq_length
        )
        for name in SPLITS
    }


def build_labeled_splits(
    frame: pd.DataFrame,
    seq_length: int = SEQ_LENGTH,
    cell_tags: tuple[str, ...] = CELL_TAGS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All four cell lines with labels, for the FBD embedding classifier.

    Uses the same chromosome split and canonical grouping as
    :func:`build_splits`, so every K562 test sequence is also held out from
    the classifier. Groups whose rows carry different labels are dropped as
    ambiguous.

    Returns a dict mapping each split name to ``(X int8[N, L], y int64[N])``,
    where ``y`` indexes into ``cell_tags``.
    """
    frame = frame[frame["TAG"].isin(cell_tags)].copy()
    if frame.empty:
        raise ValueError(f"no rows with TAG in {cell_tags!r}")

    frame["sequence"] = frame["sequence"].str.upper()
    frame = frame[frame["sequence"].map(lambda s: is_valid(s, length=seq_length))].copy()
    frame["label"] = frame["TAG"].map({tag: i for i, tag in enumerate(cell_tags)})
    frame["canonical"] = frame["sequence"].map(canonical)
    frame["split"] = frame["chr"].map(_assign_split)

    groups = frame.groupby("canonical")
    frame = frame[groups["label"].transform("nunique") == 1].copy()
    priority = frame["split"].map(_SPLIT_PRIORITY)
    frame["group_split"] = (
        priority.groupby(frame["canonical"]).transform("min").map(_PRIORITY_SPLIT)
    )
    frame = frame.drop_duplicates(subset="canonical", keep="first")

    out = {}
    for name in SPLITS:
        part = frame[frame["group_split"] == name]
        out[name] = (
            encode_many(part["sequence"].tolist(), length=seq_length),
            part["label"].to_numpy(dtype=np.int64),
        )
    return out
