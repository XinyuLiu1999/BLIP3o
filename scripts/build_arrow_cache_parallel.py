#!/usr/bin/env python3
"""Build the webdataset arrow cache in parallel, working around a datasets bug.

``datasets==2.16.1``'s webdataset builder puts *live* tar iterators (open
``ExFileObject`` handles) into ``gen_kwargs``. With ``num_proc>1`` the builder
pickles ``gen_kwargs`` to hand shards to pool workers, which dies with
``TypeError: cannot pickle 'ExFileObject' object`` -- after streaming the whole
corpus and writing nothing.

So parallelism cannot live *inside* one ``load_dataset`` call. Instead each
chunk of shards is built by a separate process with ``num_proc=1`` (the path
that works), and the chunks run concurrently. Training later reads the same
per-chunk caches by passing the same chunk's ``data_files``.

Each chunk is an independent cache entry keyed by its own ``data_files``, so
re-running skips finished chunks and only redoes the missing ones.
"""
import argparse
import os
import subprocess
import sys
import time

CHILD = r"""
import sys
from datasets import load_dataset
shards = sys.argv[2:]
ds = load_dataset("webdataset", data_files=shards, split="train",
                  num_proc=1, cache_dir=sys.argv[1])
print(len(ds), flush=True)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shardlist", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--chunk_size", type=int, default=64,
                    help="shards per chunk (one cache entry per chunk)")
    ap.add_argument("--jobs", type=int, default=16,
                    help="chunks built concurrently")
    args = ap.parse_args()

    with open(args.shardlist) as f:
        shards = [l.strip() for l in f if l.strip()]
    missing = [s for s in shards if not os.path.exists(s)]
    if missing:
        sys.exit(f"{len(missing)} shard(s) missing, first: {missing[0]}")

    chunks = [shards[i:i + args.chunk_size]
              for i in range(0, len(shards), args.chunk_size)]
    print(f"{len(shards)} shards -> {len(chunks)} chunks of "
          f"{args.chunk_size}, {args.jobs} concurrent", flush=True)

    env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1")
    running, done, failed, started = [], 0, [], time.time()

    def reap(block):
        nonlocal done
        while running:
            for i, (idx, p) in enumerate(running):
                rc = p.poll()
                if rc is None:
                    continue
                running.pop(i)
                if rc == 0:
                    done += 1
                    el = time.time() - started
                    rate = done / el if el else 0
                    eta = (len(chunks) - done) / rate / 60 if rate else 0
                    print(f"[{done}/{len(chunks)}] chunk {idx} ok "
                          f"({el/60:.1f}m elapsed, ~{eta:.0f}m left)", flush=True)
                else:
                    failed.append(idx)
                    print(f"[FAIL] chunk {idx} rc={rc}", flush=True)
                break
            else:
                if not block:
                    return
                time.sleep(2)
                continue
            if not block:
                return

    for idx, chunk in enumerate(chunks):
        while len(running) >= args.jobs:
            reap(block=True)
        p = subprocess.Popen(
            [sys.executable, "-c", CHILD, args.cache_dir, *chunk],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running.append((idx, p))

    while running:
        reap(block=True)

    print(f"done={done}/{len(chunks)} failed={len(failed)} "
          f"in {(time.time()-started)/60:.1f}m", flush=True)
    if failed:
        sys.exit(f"failed chunks: {failed}")


if __name__ == "__main__":
    main()
