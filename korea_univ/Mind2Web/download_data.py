"""Download the Mind2Web train split and cache one domain's subset locally.

Mind2Web (osunlp/Mind2Web on Hugging Face) ships as 8 auto-converted parquet shards
under `refs/convert/parquet`. There is no server-side domain filter, so each shard is
downloaded in full and filtered locally to the requested `domain`. Only the columns the
rest of the pipeline needs are kept.

The train split only contains 3 domains (verified directly from the data, not docs):
Travel (467 tasks), Shopping (281), Entertainment (261). The other ~28 domains in
Mind2Web's full 31-domain taxonomy only exist in the password-gated test split.

Usage (run from Mind2Web/):
    python download_data.py                    # domain=Shopping (default)
    python download_data.py --domain Travel
    python download_data.py --domain Travel --refresh   # force re-download even if cached
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NUM_SHARDS = 8
SHARD_URL = (
    "https://huggingface.co/datasets/osunlp/Mind2Web/resolve/refs%2Fconvert%2Fparquet/"
    "default/partial-train/{index:04d}.parquet"
)
COLUMNS = ["website", "domain", "subdomain", "annotation_id", "confirmed_task", "action_reprs", "actions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--domain", default="Shopping", choices=["Travel", "Shopping", "Entertainment"])
    parser.add_argument(
        "--out-name",
        default=None,
        help="Cache filename inside --data-dir. Default: <domain lowercased>_train.parquet",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-download even if the cache file exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_name = args.out_name or f"{args.domain.lower()}_train.parquet"
    out_path = args.data_dir / out_name

    if out_path.exists() and not args.refresh:
        df = pd.read_parquet(out_path)
        print(
            f"Cache already exists: {out_path.resolve()} ({len(df)} {args.domain} tasks). "
            "Use --refresh to redownload."
        )
        return 0

    frames = []
    for index in range(NUM_SHARDS):
        url = SHARD_URL.format(index=index)
        print(f"[{index + 1}/{NUM_SHARDS}] downloading {url}", flush=True)
        shard = pd.read_parquet(url, columns=COLUMNS)
        shard_domain = shard[shard["domain"] == args.domain]
        print(f"  -> {len(shard_domain)}/{len(shard)} rows are domain={args.domain!r}", flush=True)
        frames.append(shard_domain)

    combined = pd.concat(frames, ignore_index=True)
    duplicate_ids = combined["annotation_id"].duplicated().sum()
    if duplicate_ids:
        print(f"[warn] {duplicate_ids} duplicate annotation_id(s) across shards; keeping first occurrence.")
        combined = combined.drop_duplicates(subset="annotation_id", keep="first").reset_index(drop=True)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)

    print(f"\nSaved {len(combined)} {args.domain} tasks to {out_path.resolve()}")
    print("Tasks per website:")
    print(combined["website"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
