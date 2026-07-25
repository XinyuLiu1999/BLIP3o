"""Stage 1 — Tag/link the corpus -> per-sample concepts (vocab-cached).

Reads the `.json` sidecar of every webdataset sample (its ``tagging_caption``),
cleans tags with the browser's ``parse_tags``, then resolves the **unique tag
vocabulary** to Bamboo nodes *once* via the HybridMatcher and joins the map back
to samples by string (the mandatory scale trick, plan §Stage 1).

Two intermediates + two outputs are written so the (GPU) resolution step and the
(CPU) tar scan are independently restartable:

    sample_tags.parquet   (sample_key, tags: list<string>)     [scan output]
    vocab.txt             one unique content tag per line       [scan output]
    tag_to_nodes.parquet  (tag, node_id, node_name, score)      [resolve output]
    links.parquet         (sample_key, node_id)                 [join output, Stage-1 deliverable]

Typical run (from BLIP3o repo root, in the `wiki` conda env)::

    python concept_rebalancing/link_corpus.py \
        --tar_dir /cephfs/liuxinyu/BLIP3o-Pretrain-Long-Caption-filtered-recaptioned \
        --output_dir concept_rebalancing/runs/blip3o_pretrain \
        --num_workers 32

Re-run with ``--resume`` to skip stages whose outputs already exist. The GPU
resolve step needs the browser's `wiki` env and its prebuilt caches (see
``rebalance/linker.py``); the tar scan needs neither and can run anywhere.
"""

import argparse
import os
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rebalance.linker import (  # noqa: E402
    DEFAULT_BROWSER_ROOT,
    VocabLinker,
    ensure_browser_on_path,
    is_abstract_node,
    parse_tags,
)

# Process-global set by the worker initializer so each fork imports the browser
# tag cleaner exactly once.
_BROWSER_ROOT = DEFAULT_BROWSER_ROOT


def _worker_init(browser_root: str):
    global _BROWSER_ROOT
    _BROWSER_ROOT = browser_root
    ensure_browser_on_path(browser_root)


def _scan_one_tar(tar_path: str):
    """Return (records, local_vocab) for one tar.

    ``records`` = list of (sample_key, tuple(tags)); ``local_vocab`` = set of the
    unique content tags seen in this tar. Reads only the `.json` members.
    """
    import json

    records = []
    local_vocab = set()
    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                sample_key = os.path.basename(member.name)[: -len(".json")]
                try:
                    meta = json.load(tf.extractfile(member))
                except Exception:
                    continue
                caption = meta.get("tagging_caption") or ""
                tags = parse_tags(caption, _BROWSER_ROOT)
                records.append((sample_key, tuple(tags)))
                local_vocab.update(tags)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] error scanning {tar_path}: {e}")
    return records, local_vocab


def scan_corpus(tar_paths, num_workers, browser_root, sample_tags_path, vocab_path):
    """Scan every tar -> sample_tags.parquet + vocab.txt."""
    print(f"[scan] {len(tar_paths)} tars, {num_workers} workers")
    keys, tag_lists = [], []
    vocab = set()
    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=_worker_init, initargs=(browser_root,)
    ) as ex:
        for i, (records, local_vocab) in enumerate(ex.map(_scan_one_tar, tar_paths)):
            for sample_key, tags in records:
                keys.append(sample_key)
                tag_lists.append(list(tags))
            vocab.update(local_vocab)
            if (i + 1) % 50 == 0:
                print(f"[scan]   {i + 1}/{len(tar_paths)} tars, "
                      f"{len(keys)} samples, {len(vocab)} unique tags")

    table = pa.table({
        "sample_key": pa.array(keys, type=pa.string()),
        "tags": pa.array(tag_lists, type=pa.list_(pa.string())),
    })
    pq.write_table(table, sample_tags_path)
    with open(vocab_path, "w") as f:
        for tag in sorted(vocab):
            f.write(tag + "\n")
    print(f"[scan] wrote {sample_tags_path} ({len(keys)} samples) "
          f"and {vocab_path} ({len(vocab)} unique tags)")
    return sorted(vocab)


def _t2n_to_table(tag_to_nodes):
    """Flatten {tag: [(nid, name, score), ...]} -> a pyarrow tag_to_nodes table."""
    tags, node_ids, node_names, scores = [], [], [], []
    for tag, matches in tag_to_nodes.items():
        for nid, name, score in matches:
            if is_abstract_node(name):  # node-name backstop (see rebalance.linker)
                continue
            tags.append(tag)
            node_ids.append(nid)
            node_names.append(name)
            scores.append(float(score))
    return pa.table({
        "tag": pa.array(tags, type=pa.string()),
        "node_id": pa.array(node_ids, type=pa.string()),
        "node_name": pa.array(node_names, type=pa.string()),
        "score": pa.array(scores, type=pa.float32()),
    })


def _resolve_shard_worker(payload):
    """Resolve one vocab shard on one GPU; write its own parquet, return stats.

    Runs in a **fresh spawned process**: it pins ``CUDA_VISIBLE_DEVICES`` *before*
    torch/backend are imported (VocabLinker imports them lazily inside .load()),
    so each worker owns exactly one GPU as cuda:0.
    """
    idx, gpu_id, shard, browser_root, top_k, min_similarity, keep_margin, out_path = payload
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ensure_browser_on_path(browser_root)
    print(f"[resolve]   shard {idx} on GPU {gpu_id}: {len(shard)} tags", flush=True)
    linker = VocabLinker(browser_root).load()
    try:
        t2n = linker.resolve_vocabulary(
            shard, top_k=top_k, min_similarity=min_similarity, keep_margin=keep_margin
        )
    finally:
        linker.close()
    table = _t2n_to_table(t2n)
    pq.write_table(table, out_path)
    n_matched = sum(1 for m in t2n.values() if m)
    return out_path, len(shard), table.num_rows, n_matched


def resolve_vocab(vocab, browser_root, tag_to_nodes_path, top_k, min_similarity,
                  keep_margin, gpus=(0,)):
    """Resolve the unique vocabulary -> tag_to_nodes.parquet.

    With a single GPU this is one in-process HybridMatcher pass. With multiple
    GPUs the vocabulary is round-robin sharded across GPU-pinned worker processes
    (each loads its own node-vector index) and the shard parquets are merged —
    the resolve step is embarrassingly parallel across tags.
    """
    gpus = list(gpus)
    if len(gpus) <= 1:
        print(f"[resolve] resolving {len(vocab)} unique tags via HybridMatcher "
              f"(1 GPU: {gpus[0] if gpus else 'auto'})")
        linker = VocabLinker(browser_root).load()
        try:
            t2n = linker.resolve_vocabulary(
                vocab, top_k=top_k, min_similarity=min_similarity, keep_margin=keep_margin
            )
        finally:
            linker.close()
        table = _t2n_to_table(t2n)
        pq.write_table(table, tag_to_nodes_path)
        n_matched = sum(1 for m in t2n.values() if m)
        print(f"[resolve] wrote {tag_to_nodes_path}: {table.num_rows} tag->node pairs, "
              f"{n_matched}/{len(vocab)} tags matched")
        return tag_to_nodes_path

    import multiprocessing as mp
    n = len(gpus)
    print(f"[resolve] resolving {len(vocab)} unique tags across {n} GPUs {gpus} "
          f"(round-robin sharded)")
    shards = [vocab[i::n] for i in range(n)]  # round-robin -> balanced shard sizes
    payloads = [
        (i, gpus[i], shards[i], browser_root, top_k, min_similarity, keep_margin,
         tag_to_nodes_path + f".shard{i}")
        for i in range(n)
    ]
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
        for r in ex.map(_resolve_shard_worker, payloads):
            results.append(r)

    merged = pa.concat_tables([pq.read_table(r[0]) for r in results])
    pq.write_table(merged, tag_to_nodes_path)
    for r in results:
        os.remove(r[0])
    n_pairs = sum(r[2] for r in results)
    n_matched = sum(r[3] for r in results)
    print(f"[resolve] wrote {tag_to_nodes_path}: {n_pairs} tag->node pairs, "
          f"{n_matched}/{len(vocab)} tags matched (merged from {n} shards)")
    return tag_to_nodes_path


def join_links(sample_tags_path, tag_to_nodes_path, links_path):
    """Join sample tags x tag_to_nodes -> links.parquet (sample_key, node_id)."""
    print("[join] joining sample tags to resolved nodes")
    t2n_table = pq.read_table(tag_to_nodes_path)
    tag_col = t2n_table.column("tag").to_pylist()
    node_col = t2n_table.column("node_id").to_pylist()
    tag_to_nodes = {}
    for tag, nid in zip(tag_col, node_col):
        tag_to_nodes.setdefault(tag, []).append(nid)

    st = pq.read_table(sample_tags_path)
    sample_keys = st.column("sample_key").to_pylist()
    tag_lists = st.column("tags").to_pylist()

    out_keys, out_nodes = [], []
    n_with_concept = 0
    for key, tags in zip(sample_keys, tag_lists):
        nodes = set()
        for tag in tags:
            nodes.update(tag_to_nodes.get(tag, ()))
        if nodes:
            n_with_concept += 1
        for nid in nodes:
            out_keys.append(key)
            out_nodes.append(nid)

    table = pa.table({
        "sample_key": pa.array(out_keys, type=pa.string()),
        "node_id": pa.array(out_nodes, type=pa.string()),
    })
    pq.write_table(table, links_path)
    avg = len(out_keys) / max(1, n_with_concept)
    print(f"[join] wrote {links_path}: {len(out_keys)} links across "
          f"{n_with_concept}/{len(sample_keys)} samples with >=1 concept "
          f"(avg {avg:.2f} nodes/linked-sample)")


def main():
    ap = argparse.ArgumentParser(description="Stage 1: tag/link the corpus (vocab-cached).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tar_dir", help="Directory to recursively search for .tar files.")
    src.add_argument("--tar_list", help="Text file with one tar path per line.")
    ap.add_argument("--output_dir", required=True, help="Where to write intermediates + links.parquet.")
    ap.add_argument("--browser_root", default=DEFAULT_BROWSER_ROOT,
                    help="semantic_image_browser checkout (for backend.* + caches).")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--gpus", default="0",
                    help="Comma-separated GPU ids for the resolve step, e.g. '0,1'. "
                         "Multiple GPUs shard the tag vocabulary across pinned worker "
                         "processes. Do NOT set CUDA_VISIBLE_DEVICES when using this.")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--min_similarity", type=float, default=0.65)
    ap.add_argument("--keep_margin", type=float, default=0.03)
    ap.add_argument("--max_shards", type=int, default=None,
                    help="Cap shards scanned (smoke tests).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip stages whose output already exists.")
    ap.add_argument("--skip_resolve", action="store_true",
                    help="Only scan tars (no GPU); resolve/join later.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sample_tags_path = os.path.join(args.output_dir, "sample_tags.parquet")
    vocab_path = os.path.join(args.output_dir, "vocab.txt")
    tag_to_nodes_path = os.path.join(args.output_dir, "tag_to_nodes.parquet")
    links_path = os.path.join(args.output_dir, "links.parquet")

    if args.tar_list:
        with open(args.tar_list) as f:
            tar_paths = [l.strip() for l in f if l.strip()]
    else:
        tar_paths = sorted(str(p) for p in Path(args.tar_dir).rglob("*.tar"))
    if args.max_shards is not None:
        tar_paths = tar_paths[: args.max_shards]

    # Stage 1a: scan
    if args.resume and os.path.exists(sample_tags_path) and os.path.exists(vocab_path):
        print(f"[scan] resume: reusing {sample_tags_path} / {vocab_path}")
        with open(vocab_path) as f:
            vocab = [l.strip() for l in f if l.strip()]
    else:
        vocab = scan_corpus(tar_paths, args.num_workers, args.browser_root,
                            sample_tags_path, vocab_path)

    if args.skip_resolve:
        print("[done] scan complete (--skip_resolve); run again without it to resolve+join.")
        return

    # Stage 1b: resolve vocabulary (GPU)
    if args.resume and os.path.exists(tag_to_nodes_path):
        print(f"[resolve] resume: reusing {tag_to_nodes_path}")
    else:
        gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
        resolve_vocab(vocab, args.browser_root, tag_to_nodes_path,
                      args.top_k, args.min_similarity, args.keep_margin, gpus=gpus)

    # Stage 1c: join
    join_links(sample_tags_path, tag_to_nodes_path, links_path)
    print(f"[done] Stage 1 complete -> {links_path}")


if __name__ == "__main__":
    main()
