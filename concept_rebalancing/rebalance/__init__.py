"""Concept rebalancing for BLIP3o pretraining.

Offline resampling of the pretraining corpus at the *concept* (Bamboo node)
level: downsample over-represented head concepts, retain the mid band, and
oversample tail concepts, then materialize a rebalanced experiment the existing
BLIP3o trainer can consume.

See ``docs/concept_rebalancing_plan.md`` for the design. The package is a
*separate* implementation that reuses — but does not modify — the existing
``scripts/data_pipeline`` and ``semantic_image_browser`` code.

Modules
-------
- ``multiplicity`` : the per-sample multiplicity abstraction (§4/§5 of the plan)
                     — the shared hash + schedule + realize primitives.
- ``schedule``     : per-concept multiplicity schedule + rarest-wins reduction
                     (Stages 3-4).
- ``linker``       : loads the Bamboo taxonomy + embedding index and resolves a
                     tag vocabulary to concept nodes via HybridMatcher (Stage 1).
- ``qa_counts``    : an in-memory ``links.parquet``-backed store that satisfies
                     the ``CompositionAnalyzer`` DB interface for the QA gate and
                     the distribution audit (Stages 1b / 6a).
"""
