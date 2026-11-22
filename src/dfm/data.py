"""K562 DHS data: download, filter, split, and load.

The source is the subset shipped with the DNA-Diffusion repository: a 19-column
tab-separated table of 200 bp DNase-I-hypersensitive sites across four cell
lines, ~12k rows each. We keep K562 and hold out chromosome 1 for test and
chromosome 2 for validation, following the released data loader.

Exact duplicates and reverse-complement pairs are grouped under a canonical key
so that no group straddles a split boundary; each group contributes one
sequence. Every count along the way lands in ``manifest.json``, which is the
reproducibility record the project description asks for in section A.1.
"""

from __future__ import annotations

import hashlib
import json
import platform
import urllib.request
from datetime import datetime, timezone
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
    "sha256",
    "load_table",
    "build_splits",
    "save_splits",
    "load_split",
    "load_all_splits",
]

SOURCE_URL = (
    "https://raw.githubusercontent.com/pinellolab/DNA-Diffusion/main/data/"
    "K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
)
RAW_FILENAME = "dhs.txt"

#: Value of the ``TAG`` column identifying K562 rows.
K562_TAG = "K562_ENCLB843GMH"

TEST_CHROMOSOMES = ("chr1",)
VAL_CHROMOSOMES = ("chr2",)

SPLITS = ("train", "val", "test")

#: Only these columns are read. Naming them explicitly means an upstream schema
#: change fails loudly here rather than silently shifting a column.
USED_COLUMNS = ["chr", "sequence", "TAG"]

#: Most-held-out split wins when a group spans several chromosomes.
_SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
_PRIORITY_SPLIT = {v: k for k, v in _SPLIT_PRIORITY.items()}


def sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> tuple[dict[str, np.ndarray], dict]:
    """Filter, group, split, and encode.

    Returns ``(splits, manifest)`` where ``splits`` maps a split name to an
    ``int8[N, seq_length]`` array of token ids.
    """
    counts: dict[str, int] = {"rows_in_table": int(len(frame))}

    frame = frame[frame["TAG"] == cell_tag].copy()
    counts["rows_for_cell_type"] = int(len(frame))
    if frame.empty:
        raise ValueError(f"no rows with TAG == {cell_tag!r}")

    frame["sequence"] = frame["sequence"].str.upper()

    valid = frame["sequence"].map(lambda s: is_valid(s, length=seq_length))
    counts["rows_dropped_invalid"] = int((~valid).sum())
    frame = frame[valid].copy()
    counts["rows_valid"] = int(len(frame))
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

    counts["num_groups"] = int(frame["canonical"].nunique())
    counts["rows_collapsed_as_duplicates"] = int(len(frame) - frame["canonical"].nunique())
    counts["groups_reassigned_by_priority"] = int((frame["split"] != frame["group_split"]).sum())

    # One representative per group: first occurrence in file order, which keeps
    # the observed reference strand rather than flipping to the canonical one.
    frame = frame.drop_duplicates(subset="canonical", keep="first")

    splits: dict[str, np.ndarray] = {}
    for name in SPLITS:
        sequences = frame.loc[frame["group_split"] == name, "sequence"].tolist()
        splits[name] = encode_many(sequences, length=seq_length)
        counts[f"{name}_sequences"] = len(sequences)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cell_type_tag": cell_tag,
        "seq_length": seq_length,
        "test_chromosomes": list(TEST_CHROMOSOMES),
        "val_chromosomes": list(VAL_CHROMOSOMES),
        "grouping": "exact duplicates and reverse-complement pairs share a canonical key",
        "group_split_rule": "most held-out split wins (test > val > train)",
        "counts": counts,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return splits, manifest


def save_splits(
    splits: dict[str, np.ndarray],
    manifest: dict,
    processed_dir: Path | str,
) -> Path:
    """Write ``{train,val,test}.npy`` plus ``manifest.json``."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, array in splits.items():
        np.save(processed_dir / f"{name}.npy", array)
    manifest_path = processed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_split(processed_dir: Path | str, name: str) -> np.ndarray:
    """Load one split as an ``int8[N, L]`` array."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    path = Path(processed_dir) / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py first")
    return np.load(path)


def load_all_splits(processed_dir: Path | str) -> dict[str, np.ndarray]:
    """Load every split."""
    return {name: load_split(processed_dir, name) for name in SPLITS}
