"""Build the K562 train/val/test splits.

    python scripts/prepare_data.py            # download if needed, then build
    python scripts/prepare_data.py --force    # re-download first
"""

from __future__ import annotations

import sys
from pathlib import Path

from dfm.data import (
    SOURCE_URL,
    build_splits,
    download,
    load_table,
    save_splits,
    sha256,
)

# --- settings ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# ----------------------------------------------------------------------------


def main(force: bool = False) -> int:
    print(f"source     {SOURCE_URL}")
    raw_path = download(RAW_DIR, force=force)
    digest = sha256(raw_path)
    print(f"raw file   {raw_path}  ({raw_path.stat().st_size:,} bytes)")
    print(f"sha256     {digest}")

    frame = load_table(raw_path)
    splits, manifest = build_splits(frame)

    manifest["source"] = {
        "url": SOURCE_URL,
        "filename": raw_path.name,
        "sha256": digest,
        "bytes": raw_path.stat().st_size,
    }
    manifest_path = save_splits(splits, manifest, PROCESSED_DIR)

    print()
    for key, value in manifest["counts"].items():
        print(f"{key:<32} {value:>8,}")
    print()
    for name, array in splits.items():
        print(f"{name:<8} {array.shape[0]:>6,} sequences  shape={array.shape} dtype={array.dtype}")
    print(f"\nwrote {PROCESSED_DIR} and {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv[1:]))
