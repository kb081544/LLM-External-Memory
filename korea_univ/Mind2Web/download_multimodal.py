"""Download the full osunlp/Multimodal-Mind2Web dataset (all 4 splits, ~13.6GB) to
Mind2Web/data/multimodal/<split>/000N.parquet, unfiltered -- kept as raw shards so any
domain/website can be used later, not just the Shopping/Travel subset already sampled.

Usage (run from Mind2Web/):
    python download_multimodal.py
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

SPLIT_SHARD_COUNTS = {
    "train": 27,
    "test_domain": 11,
    "test_task": 5,
    "test_website": 4,
}
URL_TEMPLATE = (
    "https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/resolve/refs%2Fconvert%2Fparquet/"
    "default/{split}/{index:04d}.parquet"
)


def main() -> int:
    data_dir = Path("data/multimodal")
    total_shards = sum(SPLIT_SHARD_COUNTS.values())
    done = 0
    for split, count in SPLIT_SHARD_COUNTS.items():
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            done += 1
            out_path = split_dir / f"{index:04d}.parquet"
            if out_path.exists():
                print(f"[{done}/{total_shards}] {split}/{index:04d}.parquet already exists, skipping", flush=True)
                continue
            url = URL_TEMPLATE.format(split=split, index=index)
            started = time.monotonic()
            tmp_path = out_path.with_suffix(".parquet.tmp")
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.rename(out_path)
            elapsed = time.monotonic() - started
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[{done}/{total_shards}] {split}/{index:04d}.parquet: {size_mb:.1f}MB in {elapsed:.1f}s", flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
