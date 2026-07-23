#!/usr/bin/env python3
"""
Pre-build HF datasets cache from webdataset tar files.
Run this once, then train with --data_dir and --data_cache_dir.
"""
import argparse
import glob
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_cache_dir", required=True)
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.data_dir, "*.tar")))
    assert len(shards) > 0, f"No tar files found in {args.data_dir}"

    print(f"Found {len(shards)} tar files, building cache...")
    ds = load_dataset(
        "webdataset",
        data_files=shards,
        split="train",
        num_proc=1,
        cache_dir=args.data_cache_dir,
    )
    print(f"Done: {len(ds)} samples, columns={ds.column_names}")


if __name__ == "__main__":
    main()