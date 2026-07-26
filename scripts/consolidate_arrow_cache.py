#!/usr/bin/env python3
"""Merge the per-chunk webdataset caches into one save_to_disk dataset.

``build_arrow_cache_parallel.py`` leaves 62 separate cache entries (one per
chunk of shards). Loading them costs ~14 min of metadata parsing per process --
paid by *every* rank, on every start. Consolidating once into a single
``save_to_disk`` directory turns that into a seconds-long memory-mapped load.

This is orthogonal to the sampling policy: it stores the decoded corpus only.
``membership.parquet`` is applied afterwards, so changing alpha/cap/reduction
does not invalidate this output.

The ``pkl`` column (precomputed embeddings, ~36MB/shard) is dropped -- training
never reads it, and ``rebalanced_dataset.py`` discards it anyway.
"""
import argparse
import os
import sys
import time

from datasets import concatenate_datasets, load_dataset

KEEP = {"__key__", "__url__", "jpg", "json", "txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shardlist", required=True)
    ap.add_argument("--cache_dir", required=True,
                    help="per-chunk cache built by build_arrow_cache_parallel.py")
    ap.add_argument("--out_dir", required=True,
                    help="destination for save_to_disk")
    ap.add_argument("--chunk_size", type=int, default=64,
                    help="MUST match the value used at build time")
    ap.add_argument("--num_proc", type=int, default=16,
                    help="workers for the final save_to_disk write")
    ap.add_argument("--num_shards", type=int, default=2000,
                    help="output shard count. Passing this explicitly skips "
                         "save_to_disk's _estimate_nbytes(), a single-threaded "
                         "scan of all rows that stalls for tens of minutes on a "
                         "corpus this size with no progress output.")
    args = ap.parse_args()

    if args.num_shards % args.num_proc:
        # save_to_disk requires num_shards to be divisible by num_proc.
        args.num_shards += args.num_proc - (args.num_shards % args.num_proc)
        print(f"rounded num_shards up to {args.num_shards} "
              f"(must be divisible by num_proc={args.num_proc})", flush=True)

    with open(args.shardlist) as f:
        shards = [l.strip() for l in f if l.strip()]

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    started = time.time()
    parts = []
    n_chunks = (len(shards) + args.chunk_size - 1) // args.chunk_size
    for n, i in enumerate(range(0, len(shards), args.chunk_size)):
        ds = load_dataset("webdataset", data_files=shards[i:i + args.chunk_size],
                          split="train", num_proc=1, cache_dir=args.cache_dir)
        drop = [c for c in ds.column_names if c not in KEEP]
        if drop:
            ds = ds.remove_columns(drop)
        parts.append(ds)
        print(f"[{n+1}/{n_chunks}] rows={len(ds)} "
              f"({time.time()-started:.0f}s)", flush=True)

    merged = concatenate_datasets(parts)
    print(f"merged rows={len(merged)} cols={merged.column_names} "
          f"({time.time()-started:.0f}s)", flush=True)

    print(f"writing {args.num_shards} shards with {args.num_proc} procs...",
          flush=True)
    merged.save_to_disk(args.out_dir, num_proc=args.num_proc,
                        num_shards=args.num_shards)
    print(f"saved to {args.out_dir} in {(time.time()-started)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
