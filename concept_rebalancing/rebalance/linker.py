"""Stage 1 helper: load the Bamboo taxonomy + node embedding index and resolve a
tag vocabulary to concept nodes via the `semantic_image_browser` HybridMatcher.

This reuses the *exact* linking path deployed in the browser
(`parse_tags` -> `HybridMatcher`) but drives it from a **tag vocabulary** rather
than a DuckDB image table, which is the mandatory scale trick (plan §Stage 1):
tags repeat massively, so we resolve each unique tag once and join the map back
to samples by string.

Requires the browser's runtime environment (the `wiki` conda env: torch+cuda,
sentence-transformers, lancedb) and its prebuilt caches:

    <browser>/data/bamboo_V4.json
    <browser>/data/taxonomy_cache_bamboo.pkl
    <browser>/data/embedding_index_bamboo.lance

Nothing here writes to the browser DB — it only reads the taxonomy + vectors.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

# Default location of the semantic_image_browser checkout and its caches. Override
# with --browser_root on the CLI drivers if it moves.
DEFAULT_BROWSER_ROOT = "/cephfs/liuxinyu/semantic_image_browser"


def _browser_paths(browser_root: str) -> Dict[str, str]:
    data = os.path.join(browser_root, "data")
    return {
        "root": browser_root,
        "bamboo_json": os.path.join(data, "bamboo_V4.json"),
        "taxonomy_cache": os.path.join(data, "taxonomy_cache_bamboo.pkl"),
        "embedding_cache": os.path.join(data, "embedding_index_bamboo.lance"),
        "dag_cache": os.path.join(data, "composition_dag.pkl"),
    }


class VocabLinker:
    """Loads taxonomy + embedding index once; resolves tag vocabularies to nodes."""

    def __init__(self, browser_root: str = DEFAULT_BROWSER_ROOT):
        self.paths = _browser_paths(browser_root)
        # Make `import backend.*` resolve against the browser checkout.
        if browser_root not in sys.path:
            sys.path.insert(0, browser_root)

        self.taxonomy = None
        self.embedding_index = None
        self._matcher = None

    # ---- loading -------------------------------------------------------- #
    def load(self) -> "VocabLinker":
        """Load the Bamboo taxonomy and attach the prebuilt node vector store."""
        from backend.taxonomy import TaxonomyTree
        from backend.embeddings import EmbeddingIndex, EmbeddingConfig

        p = self.paths
        if not os.path.exists(p["bamboo_json"]) and not os.path.exists(p["taxonomy_cache"]):
            raise FileNotFoundError(
                f"Neither bamboo JSON ({p['bamboo_json']}) nor taxonomy cache "
                f"({p['taxonomy_cache']}) found under {p['root']}."
            )

        print(f"[linker] loading Bamboo taxonomy (cache: {p['taxonomy_cache']})")
        self.taxonomy = TaxonomyTree.load_from_bamboo(
            json_path=p["bamboo_json"],
            cache_path=p["taxonomy_cache"],
        )

        print(f"[linker] attaching embedding index: {p['embedding_cache']}")
        self.embedding_index = EmbeddingIndex(config=EmbeddingConfig())
        self.embedding_index.load(p["embedding_cache"])
        if not self.embedding_index.node_ids:
            raise RuntimeError(
                f"Embedding cache at {p['embedding_cache']} is empty; "
                f"build it with the browser first."
            )
        # Sanity: the cache must cover the active taxonomy id-space.
        if self.taxonomy.root_id not in self.embedding_index.node_names:
            raise RuntimeError(
                "Embedding cache does not match the active taxonomy "
                f"(root {self.taxonomy.root_id} missing). Rebuild the browser cache."
            )
        return self

    def _get_matcher(self):
        if self._matcher is None:
            from backend.lexical_match import HybridMatcher
            self._matcher = HybridMatcher(self.embedding_index, self.taxonomy)
        return self._matcher

    # ---- resolution ----------------------------------------------------- #
    def resolve_vocabulary(
        self,
        tags: List[str],
        top_k: int = 3,
        min_similarity: float = 0.30,
        keep_margin: float = 0.03,
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Resolve unique *content* tags -> node matches (HybridMatcher.match).

        Returns ``{tag: [(node_id, node_name, score), ...]}`` — the same shape the
        browser stores. Tags are expected to be already cleaned by
        :func:`backend.tags.parse_tags`; pass raw content tags only.
        """
        if not tags:
            return {}
        matcher = self._get_matcher()
        return matcher.match(
            tags,
            top_k=top_k,
            min_similarity=min_similarity,
            keep_margin=keep_margin,
        )

    def close(self) -> None:
        if self.embedding_index is not None:
            self.embedding_index.close()


def ensure_browser_on_path(browser_root: str = DEFAULT_BROWSER_ROOT) -> None:
    """Put the browser checkout on sys.path so ``import backend.*`` resolves.

    Idempotent and safe to call from worker processes (which do not inherit the
    parent's sys.path mutation)."""
    if browser_root not in sys.path:
        sys.path.insert(0, browser_root)


# --- concept-rebalancing-local descriptor stoplist extension ---------------- #
# The browser's ``backend.tags.STOPTAGS`` is exact-match, so *compound* tags whose
# head noun is an abstract descriptor slip through and the HybridMatcher's
# head-noun rule then resolves them to a non-depictable node:
#     "peaceful atmosphere" -> atmosphere   "architectural view" -> view
#     "urban scene"         -> scene        "ornate details"     -> detail
#     "beach setting"       -> setting      "ancient style"      -> manner
# These carry no subject and only pollute the per-concept counts the schedule is
# built from. We drop them here — the same "drop before matching" philosophy as
# STOPTAGS — WITHOUT modifying the browser (README "separate package" invariant).
# Nothing depictable is lost: the browser's head-noun resolver already sends e.g.
# "mountain view" -> view (not mountain), so these compounds never yielded a
# concrete node to begin with.
_ABSTRACT_STOP_EXACT = {
    "atmosphere", "scene", "view", "setting", "detail", "details", "moment",
    "expression", "expressions", "perspective", "angle", "texture", "textures",
    "mood", "vibe", "vibes", "backdrop", "surroundings", "environment",
    "aesthetic", "aesthetics", "manner", "style", "ambiance", "ambience",
    "shot", "fashion",
}
# Applied to any tag *ending* with one of these (space-prefixed so single words
# like "hairstyle"/"freestyle" are untouched). Deliberately excludes bare " light"
# and content heads; every entry here names a non-depictable descriptor head
# (photographic shot type, time/weather framing, abstract mood/aesthetic). The
# head-noun resolver already sends these compounds to the abstract node, not the
# modifier — e.g. "sunny day"->day, "close-up shot"->shot — so no subject is lost.
_ABSTRACT_STOP_SUFFIX = (
    " atmosphere", " atmospheres", " scene", " scenes", " view", " views",
    " setting", " settings", " detail", " details", " moment", " moments",
    " expression", " expressions", " perspective", " perspectives", " angle",
    " angles", " texture", " textures", " mood", " vibe", " backdrop",
    " background", " surroundings", " environment", " style", " aesthetic",
    " aesthetics", " shot", " shots", " day", " days", " fashion",
)


def _is_abstract_descriptor(tag: str) -> bool:
    t = tag.strip().lower()
    return t in _ABSTRACT_STOP_EXACT or t.endswith(_ABSTRACT_STOP_SUFFIX)


# Node-name backstop. The tag-level stoplist above runs pre-resolution (cheap,
# saves GPU) and catches the bulk, but a few phrasings still resolve to an
# abstract node no surface rule can predict: head-noun-at-front "X of Y"
# ("texture of water", "Day of the Dead"), synonym drift ("scenic panorama"->view,
# "mixed styles"->manner), and OCR text ("FORT VIEW"). We drop links to these node
# *names* regardless of the tag that produced them. Kept tight to confirmed
# non-depictable descriptors — content nouns like "structure"/"line"/"event" are
# deliberately NOT here.
ABSTRACT_NODE_NAMES = {
    "atmosphere", "scene", "view", "setting", "settings", "detail", "details",
    "moment", "expression", "perspective", "angle", "texture", "mood", "vibe",
    "backdrop", "background", "surroundings", "environment", "aesthetic",
    "manner", "style", "ambiance", "ambience", "shot", "fashion", "day",
    "hour", "preparation",
}


def is_abstract_node(node_name: str) -> bool:
    return node_name.strip().lower() in ABSTRACT_NODE_NAMES


def parse_tags(tagging_caption: str, browser_root: str = DEFAULT_BROWSER_ROOT) -> List[str]:
    """Re-export of the browser's tag cleaner so drivers need one import.

    Drops OCR quotes, bare numerics and STOPTAGS descriptors — must match the
    matcher's expectations exactly, so we defer to the browser implementation —
    then applies the concept-rebalancing-local abstract-head-noun stoplist above.
    """
    ensure_browser_on_path(browser_root)
    from backend.tags import parse_tags as _parse_tags
    return [t for t in _parse_tags(tagging_caption) if not _is_abstract_descriptor(t)]
