"""Stage 6 — Training-time multiplicity expansion (experiment mode extension).

``blip3o/data/dataset.py`` experiment mode filters HF rows by a membership *set*
(one row = one sample). This module adds the multiplicity generalization the
plan calls for (§Stage 6) as a **subclass**, so nothing in the core trainer is
modified:

1. load ``shardlist.txt`` shards, filter to the membership keys (unchanged),
2. build ``count_map = {sample_key: count}`` from ``membership.parquet``,
3. build ``index_map = [row_idx repeated count times]``,
4. ``__len__ = len(index_map)``; ``__getitem__(i)`` -> ``index_map[i]`` -> HF row,
5. shuffle ``index_map`` (seed 42), not the HF dataset, so repeats disperse.

Repeats are cheap index entries, not duplicated image bytes (HF rows stay
memory-mapped). Works with the stock ``group_by_modality_length`` sampler because
all modality lengths are constant.

Enable it by making the trainer resolve ``--dataset_cls rebalanced`` to
:class:`LazySupervisedRebalancedDataset`. Either call :func:`register` at import
(monkeypatches ``blip3o.data.dataset.get_dataset_cls`` without editing it), or add
one line to that function. See ``README.md``.
"""

import os
import random

import pyarrow.parquet as pq
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk

from blip3o.data.dataset import LazySupervisedMixDataset
from blip3o.utils import rank0_print


# Columns the trainer consumes downstream; everything else is dropped to keep the
# arrow table lean. Mirrors the base experiment-mode keep-set.
_KEEP_COLS = {"image", "txt", "json", "type", "id", "__key__", "__url__"}


def _load_count_map(membership_path):
    """membership.parquet -> {sample_key: count}."""
    table = pq.read_table(membership_path, columns=["sample_key", "count"])
    keys = table.column("sample_key").to_pylist()
    counts = table.column("count").to_pylist()
    return {k: int(c) for k, c in zip(keys, counts)}


class LazySupervisedRebalancedDataset(LazySupervisedMixDataset):
    """Experiment-mode dataset with per-sample multiplicity (integer copies)."""

    def __init__(self, tokenizer, data_path, data_args):
        # Deliberately does NOT call super().__init__: the base __init__ builds a
        # membership *set* and shuffles the HF dataset. We reproduce its loading
        # then add multiplicity expansion and shuffle the index map instead.
        self.data_args = data_args
        self.caption_key = getattr(data_args, "caption_key", "txt")
        rank0_print(f"caption_key: {self.caption_key}")

        experiment_dir = getattr(data_args, "experiment_dir", None)
        if experiment_dir is None:
            raise ValueError("rebalanced dataset requires data_args.experiment_dir")

        shardlist_path = os.path.join(experiment_dir, "shardlist.txt")
        membership_path = os.path.join(experiment_dir, "membership.parquet")
        if not os.path.exists(membership_path):
            raise FileNotFoundError(
                f"{membership_path} not found. The rebalanced dataset needs a "
                f"multiplicity membership.parquet (materialize_rebalanced.py), "
                f"not the flat membership.txt."
            )

        with open(shardlist_path) as f:
            shards = [l.strip() for l in f if l.strip()]
        count_map = _load_count_map(membership_path)

        rank0_print(f"Loading rebalanced experiment from {experiment_dir}")
        rank0_print(f"  shards: {len(shards)}, membership: {len(count_map)} distinct samples, "
                    f"expanded: {sum(count_map.values())}")

        cache_dir = getattr(data_args, "data_cache_dir", None)
        num_proc = getattr(data_args, "num_loading_workers", 32)
        load_num_proc = 1 if cache_dir is not None else num_proc

        # datasets==2.16.1's webdataset builder cannot run with num_proc>1: it
        # stores live tar handles in gen_kwargs, which fail to pickle
        # ("cannot pickle 'ExFileObject'"). scripts/build_arrow_cache_parallel.py
        # therefore prebuilds the cache one chunk of shards at a time. Each chunk
        # is keyed by its own data_files, so we must load with the *same* chunking
        # to hit that cache -- one 3909-shard call would miss it and regenerate
        # the whole corpus single-process (~6h).
        #
        # BLIP3O_ARROW_CONSOLIDATED opts into a save_to_disk corpus built by
        # scripts/consolidate_arrow_cache.py. Measured at 2000 shards it loaded
        # in 12.9 min versus ~14 min for the per-chunk path -- not worth the
        # 911GB, so it stays off by default. A much smaller num_shards may pay
        # off; it has not been measured.
        consolidated = os.environ.get("BLIP3O_ARROW_CONSOLIDATED")
        chunk_size = int(os.environ.get("BLIP3O_ARROW_CHUNK_SIZE", "64"))
        if consolidated:
            if not os.path.isdir(consolidated):
                raise FileNotFoundError(
                    f"BLIP3O_ARROW_CONSOLIDATED={consolidated} is not a directory. "
                    f"Build it with scripts/consolidate_arrow_cache.py, or unset "
                    f"the variable to fall back to the per-chunk cache."
                )
            train_dataset = load_from_disk(consolidated)
        elif cache_dir is not None and chunk_size > 0 and len(shards) > chunk_size:
            parts = []
            for i in range(0, len(shards), chunk_size):
                parts.append(load_dataset(
                    "webdataset",
                    data_files=shards[i:i + chunk_size],
                    split="train",
                    num_proc=1,
                    cache_dir=cache_dir,
                ))
            train_dataset = concatenate_datasets(parts)
        else:
            train_dataset = load_dataset(
                "webdataset",
                data_files=shards,
                split="train",
                num_proc=load_num_proc,
                cache_dir=cache_dir,
            )
        rank0_print(f"Loaded raw experiment dataset: {len(train_dataset)} samples")

        before_count = len(train_dataset)

        # NOTE: do *not* use .filter() here. It rewrites every surviving row --
        # including the ~380GB of jpg bytes -- to a new arrow file just to drop
        # 29% of them, measured at ~450 rows/s/worker (~3h), re-paid by every
        # rank on every restart. The multiplicity expansion below already skips
        # rows whose count is 0, so dropping them from the table buys nothing.
        # Reading the single __key__ column is all the membership test needs.
        keys = train_dataset["__key__"]
        kept = sum(1 for k in keys if count_map.get(k, 0) > 0)
        rank0_print(f"  membership: {before_count} rows -> {kept} kept "
                    f"(no row rewrite; zero-count rows are skipped below)")

        if "jpg" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("jpg", "image")
        elif "png" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("png", "image")
        train_dataset = train_dataset.remove_columns(
            [c for c in train_dataset.column_names if c not in _KEEP_COLS]
        )

        # ---- multiplicity expansion: build the index map ----
        # Each kept row's index is repeated `count` times. Rows absent from the
        # membership map contribute 0 copies -- that is what drops the 29% the
        # old .filter() call used to materialize.
        index_map = []
        for row_idx, key in enumerate(keys):
            c = count_map.get(key, 0)
            if c > 0:
                index_map.extend([row_idx] * c)

        # Shuffle the INDEX MAP (not the HF dataset) so a sample's copies land in
        # different batches. Deterministic seed matches the base class (42).
        rng = random.Random(42)
        rng.shuffle(index_map)

        self.tokenizer = tokenizer
        self.list_data_dict = train_dataset          # unexpanded, memory-mapped
        self.index_map = index_map                    # expanded positions
        self.modality = torch.tensor(0)               # 0 = und, 1 = gen

        rank0_print(
            f"finish loading rebalanced experiment: {len(train_dataset)} rows, "
            f"{len(index_map)} training instances (expanded)"
        )

    def __len__(self):
        return len(self.index_map)

    @property
    def lengths(self):
        return [128] * len(self.index_map)

    @property
    def modality_lengths(self):
        return [128] * len(self.index_map)

    def __getitem__(self, i):
        # Map the expanded position to the underlying HF row, then reuse the base
        # item builder verbatim (image decode + T2I conversation + tokenization).
        return super().__getitem__(self.index_map[i])


def register():
    """Monkeypatch ``blip3o.data.dataset.get_dataset_cls`` to know 'rebalanced'.

    Call once at trainer startup (e.g. from train.py) so ``--dataset_cls
    rebalanced`` resolves here without editing the core function. Idempotent.
    """
    import blip3o.data.dataset as base

    if getattr(base.get_dataset_cls, "_rebalanced_patched", False):
        return
    _orig = base.get_dataset_cls

    def get_dataset_cls(name):
        if name == "rebalanced":
            return LazySupervisedRebalancedDataset
        return _orig(name)

    get_dataset_cls._rebalanced_patched = True
    base.get_dataset_cls = get_dataset_cls
    # make_supervised_data_module captured the original by reference; repoint it.
    base.make_supervised_data_module.__globals__["get_dataset_cls"] = get_dataset_cls
