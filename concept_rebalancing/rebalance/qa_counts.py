"""An in-memory ``links.parquet``-backed store that satisfies the subset of the
browser ``DatabaseStore`` interface that ``CompositionAnalyzer`` calls.

``CompositionAnalyzer`` (semantic_image_browser/backend/composition.py) is the
canonical implementation of the 11-GROUP DAG attribution and the frequency-skew
stats we want for the Stage-1b QA gate and the Stage-6a distribution audit. It
touches the DB through exactly three methods::

    get_all_node_counts()  -> {node_id: direct_count}
    get_total_images()     -> int
    count_images_for_buckets({bucket_id: [node_id, ...]}) -> {bucket_id: distinct images}

By providing those three from ``links.parquet`` we get the *entire* composition +
skew machinery for free, with no DuckDB and no modification to the browser code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import pyarrow.parquet as pq


class LinksCountStore:
    """Duck-typed stand-in for ``DatabaseStore``, built from ``links.parquet``.

    ``links.parquet`` has columns ``(sample_key, node_id)`` — one row per
    (sample, concept) link. Distinct-sample counts and distinct-image bucket
    counts are computed in memory.
    """

    def __init__(
        self,
        node_to_samples: Dict[str, set],
        total_images: int,
    ):
        self._node_to_samples = node_to_samples
        self._direct = {nid: len(s) for nid, s in node_to_samples.items()}
        self._total_images = total_images

    # ---- constructors --------------------------------------------------- #
    @classmethod
    def from_links_parquet(
        cls, links_path: str, total_images: Optional[int] = None
    ) -> "LinksCountStore":
        table = pq.read_table(links_path, columns=["sample_key", "node_id"])
        sample_keys = table.column("sample_key").to_pylist()
        node_ids = table.column("node_id").to_pylist()

        node_to_samples: Dict[str, set] = defaultdict(set)
        linked_samples = set()
        for key, nid in zip(sample_keys, node_ids):
            node_to_samples[nid].add(key)
            linked_samples.add(key)

        if total_images is None:
            # Distinct samples that carry >=1 concept. If the caller knows the
            # true corpus size (incl. no-concept samples) pass it explicitly.
            total_images = len(linked_samples)
        return cls(dict(node_to_samples), total_images)

    # ---- DatabaseStore surface used by CompositionAnalyzer -------------- #
    def get_all_node_counts(self) -> Dict[str, int]:
        return dict(self._direct)

    def get_total_images(self) -> int:
        return self._total_images

    def count_images_for_buckets(
        self, bucket_nodes: Dict[str, List[str]]
    ) -> Dict[str, int]:
        out: Dict[str, int] = {bid: 0 for bid in bucket_nodes}
        for bid, nodes in bucket_nodes.items():
            seen: set = set()
            for nid in set(nodes):
                s = self._node_to_samples.get(nid)
                if s:
                    seen |= s
            out[bid] = len(seen)
        return out

    # ---- convenience ---------------------------------------------------- #
    def node_counts(self) -> Dict[str, int]:
        """Alias for the Stage-2 per-concept frequency table (N_c)."""
        return dict(self._direct)


def build_analyzer(links_path: str, browser_root: str, total_images: Optional[int] = None):
    """Construct a ``CompositionAnalyzer`` over ``links.parquet``.

    Loads the Bamboo taxonomy + embedding index (for keyword-seed presence) from
    the browser caches and wires them to a :class:`LinksCountStore`.
    """
    from rebalance.linker import VocabLinker  # local import: optional GPU deps

    linker = VocabLinker(browser_root).load()
    from backend.composition import CompositionAnalyzer  # noqa: E402

    store = LinksCountStore.from_links_parquet(links_path, total_images=total_images)
    analyzer = CompositionAnalyzer(
        taxonomy=linker.taxonomy,
        db=store,
        bamboo_json_path=linker.paths["bamboo_json"],
        dag_cache_path=linker.paths["dag_cache"],
        embedding_index=linker.embedding_index,
    )
    return analyzer, linker
