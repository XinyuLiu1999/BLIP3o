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
from datasets import load_dataset

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

        train_dataset = load_dataset(
            "webdataset",
            data_files=shards,
            split="train",
            num_proc=load_num_proc,
            cache_dir=cache_dir,
        )
        rank0_print(f"Loaded raw experiment dataset: {len(train_dataset)} samples")

        before_count = len(train_dataset)
        keyset = set(count_map.keys())
        train_dataset = train_dataset.filter(
            lambda sample: sample["__key__"] in keyset,
            num_proc=num_proc,
        )
        rank0_print(f"  filtered: {before_count} -> {len(train_dataset)}")

        if "jpg" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("jpg", "image")
        elif "png" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("png", "image")
        train_dataset = train_dataset.remove_columns(
            [c for c in train_dataset.column_names if c not in _KEEP_COLS]
        )

        # ---- multiplicity expansion: build the index map ----
        # Each kept row's index is repeated `count` times. Missing keys (should
        # not happen after the filter) contribute 0 copies.
        keys = train_dataset["__key__"]
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
