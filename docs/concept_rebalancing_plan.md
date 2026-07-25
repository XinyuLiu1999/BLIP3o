# Concept Rebalancing for BLIP3o Pretraining — Implementation Plan

Status: **design locked, not yet implemented** · Date: 2026-07-23
Owner: xinyu · Target data: `BLIP3o-Pretrain-Long-Caption-filtered-recaptioned`

---

## 1. Goal

The pretraining corpus is extremely long-tailed at the *concept* level (cf. the
reference: ~2% of categories carry >30% of frequency, ~50% appear <100K times).
We want to **resample the corpus offline** — downsample over-represented head
concepts, fully retain the mid band, and **oversample tail concepts** — then
train BLIP3o on the rebalanced set and measure the effect, especially on
tail-concept generation.

This document is the end-to-end plan connecting three existing pieces:

- **The data** — `BLIP3o-Pretrain-Long-Caption-filtered-recaptioned`, webdataset
  `.tar` shards. Each sample is `{uuid}.{jpg,txt,json,pkl}`. The `.json` sidecar
  already contains a **`tagging_caption`** (VLM comma-separated tags) plus quality
  scores; the `.pkl` is a precomputed feature vector.
- **`semantic_image_browser`** — maps `tagging_caption` → Bamboo taxonomy nodes
  (concepts) via `parse_tags` → `HybridMatcher`, and already computes the
  long-tail skew (concentration curve, Gini) and an 11-group composition view.
- **`BLIP3o/scripts/data_pipeline`** — the offline experiment pipeline
  (`index_tars` → `enrich_index` → `materialize`) that produces a
  `membership` + `shardlist` an experiment trains on via
  `data_args.experiment_dir` (see `blip3o/data/dataset.py`, "experiment mode").

## 2. Why this is feasible (the key alignment)

**Identity is shared with no fuzzy matching.** Everything keys on the UUID stem:

```
webdataset __key__  ==  {uuid}.json filename stem  ==  BLIP3o sample_key  ==  the tagged unit
```

`index_tars.compute_priority` and the tagging both live on `sample_key`, so
concept labels join back to training samples exactly. The `tagging_caption`
format in this corpus is precisely what `semantic_image_browser/backend/tags.py`
was tuned on (count prefixes like `1 male kayaker`, OCR quotes, and the
`high resolution`/`sharp focus`/`photograph` descriptor boilerplate that
`STOPTAGS` drops).

## 3. Locked design decisions

1. **Category-level review is a QA gate, not the sampling granularity.** After
   linking, inspect the 11 composition GROUPS + subcategories to catch gross
   linker errors (homonym/descriptor inflation — the documented `blue`→butterfly,
   `overcast`→fishing-cast class of bug) *before* trusting the counts. The
   **schedule operates on per-concept (Bamboo node) counts**; categories are only
   the sanity checkpoint.
2. **Multi-label reduction = rarest-wins.** A sample carries several concepts.
   Its resampling multiplicity is governed by its **rarest** concept (the one
   demanding the most representation).
3. **Tail oversampling is required.** A pure subset (the current membership-set)
   is insufficient. We generalize membership to an integer **multiplicity per
   sample** (0 = drop, 1 = keep, ≥2 = oversample), which unifies drop /
   downsample / retain / oversample under one abstraction.

## 4. The unifying abstraction: per-sample multiplicity `m`

Every sample gets a real-valued target multiplicity `m ≥ 0`, realized to an
integer count deterministically:

```
h      = hash01(sample_key)          # uniform [0,1), = compute_priority / 2**63
base   = floor(m)
frac   = m - base
count  = base + (1 if h < frac else 0)
```

- `m = 0`      → dropped
- `0 < m < 1`  → probabilistic keep (downsample):  count ∈ {0,1}, E[count]=m
- `m = 1`      → retained
- `m > 1`      → oversampled: `base` guaranteed copies + 1 probabilistic

Determinism comes entirely from `sample_key` (shared sha256 primitive), so the
resampled set is reproducible and re-derivable from the schedule alone.

## 5. Pipeline stages

### Stage 0 — Index the shards (existing code)
`index_tars.py --tar_dir <corpus> --output index.parquet`
→ `(sample_key, tar_path, priority)`. Unchanged.

### Stage 1 — Tag/link the corpus → per-sample concepts (existing linker, new input + a vocab cache)
Run `parse_tags` → `HybridMatcher` (the `semantic_image_browser` linking path)
over every sample's `tagging_caption`. Output: `links.parquet` =
`(sample_key, node_id)` (many-to-many, avg ~1.2 nodes/tag after margin-keep).

**Scale trick (mandatory).** ~640K samples are already on disk (39 GB / 130
shards, ~16K samples/GB) and the full set is far larger; naive per-image
embedding won't scale. Tags repeat massively, so:
1. collect the **unique tag vocabulary** across the corpus,
2. resolve each unique tag → node(s) **once** (lexical-first; GPU embedding only
   for the lexical-miss fallback),
3. join the resolved map back to samples by string.

This is the same HybridMatcher, just amortized over the vocabulary (~1e6 unique
tags) instead of 1e7+ images.

### Stage 1b — Category-level QA gate (decision #1)
Aggregate `links.parquet` into the 11 composition GROUPS (reuse
`composition.py`'s DAG attribution) and eyeball the group + subcategory shares.
**Go/no-go:** if a GROUP is implausibly inflated, fix `STOPTAGS`/matcher and
re-link before proceeding. Do **not** compute rates on a polluted link set.

### Stage 2 — Per-concept frequency table
`counts.parquet` = `(node_id, N_c)` where `N_c` = distinct samples linked to
concept `c`. This is exactly `node_image_counts` / composition.py's `direct_count`.

### Stage 3 — Per-concept multiplicity schedule
Map each concept's count `N_c` to a target multiplicity `m_c`. Piecewise,
continuous at the boundaries, all constants **tunable and calibrated** against
the Stage-6a audit:

```
head  (N_c ≥ N_high) :  m_c = max(r_min, a_head * 2 / log10(N_c))     # < 1  downsample
mid   (N_low ≤ N_c < N_high) :  m_c = 1.0                             # retain
tail  (N_c < N_low)  :  m_c = min(M_max, (N_low / N_c) ** gamma)      # > 1  oversample
```

Starting defaults (calibrate later):
`N_high = 100_000` (from the reference), `N_low = 20_000`,
`a_head = log10(N_high)/2 = 2.5` (⇒ `m_c = 5/log10(N_c)`, so `m_c(1e5)=1`,
`m_c(1e7)≈0.71`), `r_min = 0.1` (floor so heads never vanish),
`gamma = 0.5` (sub-linear so singletons aren't boosted 100×), `M_max = 4` (cap).

The `head` branch is the reference's `N_sample ∝ 2/log(Count)`; the `tail` branch
is its symmetric oversampling extension, which the reference omits because it
never oversamples.

### Stage 4 — Per-sample multiplicity (rarest-wins, decision #2)
Join `links` × `m_c`. For each `sample_key`:

```
m(sample) = max over its concepts c of  m_c          # rarest concept wins
m(sample) = 1.0   if the sample has no content concept (all tags STOPTAGS-dropped)
```

`max` because rarer concepts have larger `m_c`; the rarest concept sets the bar.
Output `sample_multiplicity.parquet` = `(sample_key, m)`.

### Stage 5 — Materialize the resampled membership (extend `materialize.py`)
Realize integer `count` per sample via the Stage-4 hash rule; keep rows with
`count ≥ 1`. Write:

- `membership.parquet` — `(sample_key, count)`  (**replaces** the flat
  `membership.txt`; count carries the multiplicity)
- `shardlist.txt` — unique `tar_path`s over kept samples (unchanged)
- `config.yaml` — as today **plus** `actual_samples = Σ count` (the *expanded*
  total, so `run_experiment.sh`'s `MAX_STEPS = actual_samples / global_batch`
  budgets the extra tail copies correctly), and a `membership_hash`.

### Stage 6 — Train (extend `dataset.py` experiment mode for multiplicity)
`blip3o/data/dataset.py` currently filters HF rows by a membership **set** (one
row = one sample). Add multiplicity expansion:

1. load `shardlist` shards, filter to `count_map.keys()` (unchanged filter),
2. build `count_map = {sample_key: count}` from `membership.parquet`,
3. build `self.index_map = [row_idx repeated count times for each kept row]`,
4. `__len__ = len(index_map)`; `__getitem__(i)` → `index_map[i]` → HF row,
5. shuffle **`index_map`** (seed 42), not the HF dataset, so repeats disperse.

Repeats are cheap **index entries**, not duplicated image bytes (HF rows stay
memory-mapped). No trainer/sampler surgery; works with the existing
`group_by_modality_length` sampler because all modality lengths are constant.

Then train each arm exactly as today: `sbatch run_experiment.sh <experiment_dir>`.

## 6. Evaluation plan

### 6a. Distribution-level (cheap, pre-training — the browser is the instrument)
- **Before/after concentration curve + Gini** (composition.py skew): expect a
  measurable Gini drop and a flatter head.
- **Realized vs intended keep-rate per concept**: for each node, realized
  `Σcount over its samples / N_c` vs the schedule. **This is the primary guard
  against rarest-wins coupling** — a head concept that always co-occurs with tails
  can get re-inflated by oversampling; this audit catches it, and the caps
  (`M_max`, `r_min`) / constants are re-calibrated until realized ≈ intended.
- **Group-level mass shift** across the 11 GROUPS: confirm subjects moved, not
  boilerplate.
- **Budget accounting**: `Σcount`, effective epochs at fixed compute.

### 6b. Model-level (the real question) — controlled ablation
The pipeline makes each arm a different `experiment_dir`, hash-pinned:

- **Arm A — baseline** (full / uniform).
- **Arm B — rebalanced** (this plan).
- **Arm C — baseline subsampled to Arm B's sample count** (critical control:
  isolates *rebalancing* from *dataset size / epoch count*).

Fix total training steps/tokens across arms.

**Metrics (BLIP3o is text→image generation):**
- **Stratified tail eval (sharpest test).** Build an eval prompt suite of Bamboo
  concepts **stratified by original frequency** (head / mid / tail), score
  generation per stratum. Hypothesis: **tail-concept generation improves at ~no
  cost to head.** Report the **floor** (min / 10th-pct per-concept), not just the
  mean. The tagging already gives the concept→frequency map to define strata.
- **Overall quality / alignment:** FID / CLIP-FID vs a held-out reference;
  CLIPScore and/or VLM-judge alignment on a fixed prompt suite.
- **Diversity:** within-prompt LPIPS/feature variance for head concepts — verify
  downsampling didn't collapse variety.
- **External** (secondary): GenEval / DPG-Bench / T2I-CompBench.
- **Targeted VLM-judge preference** on the tail prompt suite (cheap).

## 7. Code inventory — what's new vs reused

| Component | Status |
|---|---|
| `index_tars.py` | reuse as-is |
| Tagging/linking (`parse_tags`+`HybridMatcher`) | reuse; **new**: corpus input + vocab-cache driver script |
| Category QA aggregation | reuse `composition.py` |
| Per-concept counts | reuse (`node_image_counts` logic) |
| Schedule + rarest-wins scorer (Stages 3–4) | **new** small script → `sample_multiplicity.parquet` |
| `materialize.py` | **extend**: emit `(sample_key, count)` + expanded `actual_samples` |
| `enrich_index.py` | optional (only if joining quality-score filters too) |
| `blip3o/data/dataset.py` experiment mode | **extend**: multiplicity `index_map` expansion |
| `run_experiment.sh` | reuse (reads `actual_samples` already) |

Net new code: one scorer script, a `materialize` extension, and a ~15-line
`dataset.py` change. Everything else is existing code on new inputs.

## 8. Open items before coding
- Calibrate schedule constants against a first Stage-6a audit (iterate `M_max`,
  `gamma`, `a_head`, `r_min`).
- Confirm corpus fully downloaded before Stage 0 (`.obs.temp` files were still in
  flight on 2026-07-23).
- Decide whether quality-score gating (`aesthetic`, `nsfw`, `watermark` from the
  `.json`) is applied *before* resampling (recommended: filter first via
  `enrich_index` + a `filter_expression`, then rebalance the survivors).

---

## 9. Detailed end-to-end pipeline walkthrough

This traces the whole system, following one real sample from raw shard to a
training batch, and explains what each stage guarantees.

### The sample
In `shard-000001.tar`:
`cd3cd09d12de4c48a72a24a8de952a5e.{jpg,txt,json,pkl}`. Its `.json` contains,
among quality scores:

```
tagging_caption: "photograph, high resolution, sharp focus, 1 male kayaker,
  adult male, orange helmet, yellow and black drysuit, green life vest,
  intense action, whitewater rapids, turbulent river, red and yellow kayak,
  double-bladed paddle, dynamic splash, water droplets, overhead angle,
  close-up, high contrast, energetic, 'S' logo, 'AQUATEC' text"
```

Its identity everywhere downstream is `sample_key =
cd3cd09d12de4c48a72a24a8de952a5e`.

### Step A — Index (`index_tars.py`)
Every shard is scanned; each unique stem becomes a catalog row
`(sample_key, tar_path, priority)`, where
`priority = sha256(sample_key) mod 2^63`. Our kayaker gets one row pointing at
`shard-000001.tar`. `priority/2^63` is a fixed uniform-random number `h` for this
sample — the single source of randomness reused by *both* the pipeline and the
resampling draw, which is what makes the whole thing reproducible.

### Step B — Tag → concept linking (browser's HybridMatcher, vocab-cached)
`parse_tags` first cleans the caption: it drops OCR quotes (`'S' logo`,
`'AQUATEC' text`), bare numerics, and `STOPTAGS` descriptors — here `photograph`,
`high resolution`, `sharp focus`, `overhead angle`, `close-up`, `high contrast`
all fall out (they are the highest-frequency, least-informative tags and match
misleading homonym nodes). What survives are **content** tags: `male kayaker`
(count-prefix `1 ` stripped), `orange helmet`, `drysuit`, `life vest`,
`whitewater rapids`, `turbulent river`, `kayak`, `double-bladed paddle`,
`water droplets`, …

Each surviving tag is resolved to Bamboo node(s) by the HybridMatcher: a tag that
*names* a node resolves lexically (exact/singularized/count-stripped), and the
embedding only **reranks** the lexical candidates (so homonyms resolve to the
right sense); head-noun compounds keep their general node *and* may gain a
confident distinct concept (`golden retriever puppy` → both `puppy` and
`golden_retriever`). Because we resolve the **unique vocabulary once** and join by
string, this is affordable at corpus scale.

Result for this sample: a set of concept nodes, e.g. `{kayak, rapids/river,
paddle, helmet, life_vest, water_droplet, …}`. Written to `links.parquet` as
`(sample_key, node_id)` pairs.

### Step C — Category QA gate (composition.py)
All links roll up into the 11 visual GROUPS via the Bamboo **DAG** ancestor
closure (not the pruned tree). We *read* the group/subcategory shares to confirm
the linker isn't inflating a category through a homonym or descriptor leak. Our
kayaker contributes to *Activities* (kayaking) and *Objects* (kayak, paddle,
helmet). This is a human checkpoint; if it looks wrong we fix matching and
re-link. It does **not** feed the schedule — the schedule uses per-node counts.

### Step D — Concept counts (`counts.parquet`)
Across the whole corpus we count, per node, how many distinct samples link to it:
`N_c`. Suppose `kayak` is a tail concept (`N_kayak = 8_000`), while `water` /
`river` is a head concept (`N_river = 3_000_000`), and `helmet` is mid
(`N_helmet = 50_000`).

### Step E — Per-concept multiplicity (schedule, Stage 3)
Apply the piecewise schedule:
- `river` (head, 3e6): `m = 5/log10(3e6) = 5/6.48 ≈ 0.77` → downsample.
- `helmet` (mid, 5e4): between `N_low` and `N_high` → `m = 1.0`.
- `kayak` (tail, 8e3 < 20e3): `m = min(4, (20000/8000)^0.5) = min(4, 1.58) = 1.58`
  → oversample.

### Step F — Per-sample multiplicity (rarest-wins, Stage 4)
The sample's concepts have `m ∈ {0.77 (river), 1.0 (helmet), 1.58 (kayak), …}`.
**Rarest-wins = max = 1.58** (the rare `kayak` concept governs). So this sample's
target multiplicity is `m(sample) = 1.58`. Written to
`sample_multiplicity.parquet`.

*(Why max: a sample depicting a rare subject should be preserved/boosted even if
it also contains common ones — otherwise downsampling the common `river` would
throw away scarce `kayak` pixels. The realized-count audit in 6a is the guard
against the opposite failure — `river` riding tail co-occurrences back up.)*

### Step G — Realize to an integer count (`materialize.py`, Stage 5)
Deterministic draw with this sample's fixed `h` (from Step A):
`base = floor(1.58) = 1`, `frac = 0.58`, `count = 1 + (1 if h < 0.58 else 0)`.
So this sample appears **1 or 2 times** depending on its hash — say `count = 2`.
We write `(cd3cd09d…, 2)` to `membership.parquet`, add `shard-000001.tar` to
`shardlist.txt`, and accumulate `2` into `config.yaml.actual_samples`.

A pure head sample (e.g. `m = 0.77`, `base = 0`, `frac = 0.77`) would get
`count ∈ {0,1}` — i.e. dropped with prob 0.23. Same mechanism, whole range.

### Step H — Training-time expansion (`dataset.py` experiment mode, Stage 6)
The trainer loads the shards in `shardlist.txt`, filters HF rows to the
membership keys, then builds `index_map` where each kept row index is repeated
`count` times — our kayaker's row index appears **twice**. `__len__` is the
expanded total (matching `actual_samples`, so `MAX_STEPS` is right). `index_map`
is shuffled (seed 42) so the two copies land in different batches. `__getitem__`
maps a position → row → decodes the jpg, builds the T2I conversation from the
caption, returns tensors. The two copies are identical images seen twice per
epoch — genuine oversampling, at the cost of two ints, not two images.

### Step I — Train the ablation arms
Arm A (baseline), Arm B (this membership), Arm C (baseline subsampled to B's
size) are three `experiment_dir`s handed to the *same* `run_experiment.sh`. Each
is fully described by its `membership.parquet` + `membership_hash`, so runs are
reproducible and comparable.

### Step J — Evaluate
Run 6a (distribution audit — verify Gini dropped and realized≈intended, re-tune
constants if not) and 6b (stratified tail eval + FID/CLIP/diversity across arms).
The rebalancing hypothesis is confirmed if tail-concept generation improves in
Arm B over Arms A and C without a head-concept regression.

### Invariants that make the whole pipeline trust-worthy
- **One identity** (`sample_key`) from shard to batch; no fuzzy joins.
- **One randomness source** (`hash01(sample_key)`) for both catalog priority and
  the resampling draw ⇒ any membership is re-derivable from the schedule.
- **Counts, not weights**, carry multiplicity ⇒ works with the stock trainer and
  step-budget math; drop/downsample/retain/oversample are one formula.
- **Repeats are index entries**, not copied bytes ⇒ oversampling is nearly free.

---

## Appendix — Stage-1b QA audit of the first real run (2026-07-24)

First end-to-end run on `BLIP3o-Pretrain-Long-Caption-filtered-recaptioned`
(`runs/blip3o_pretrain/`: 19,755,541 images, 218,641,343 links, 58,834 active
concepts). We audited `qa_report.json` against the raw parquet, investigated a
systematic linker-quality fix, and — importantly — **measured whether any of it
matters to the schedule.** Verdict up front: **the QA report is arithmetically
correct; the link set has real false-friend errors; those errors are effectively
inert to the rebalancing, so we do NOT relink.**

### A. The report is arithmetically sound
Independently reproduced every group mass, membership mass, the visual/non-visual
split (75.4% / 24.6%), and the skew stats (Gini 0.975, top-2% = 84% of tag mass)
from `links.parquet` + the DAG — all matched to the digit. `total_mass` = link
rows, `total_images` = sample rows. The aggregation code is trustworthy.

### B. But the link set has systemic false-friend mislinks
Bare lexical matches at the 0.98 floor collapse a high-frequency tag onto the
wrong *sense* (the embedding can only rerank, never override). Confirmed by
tracing feeder tags + co-occurrence context. Three failure classes:

1. **Homonym mislinks** (tag names the wrong sense) — the wrong node holds the
   mass while the correct sense sits at `Nc≈0`:

   | tag | links to | should be | mass |
   |---|---|---|---|
   | `table` | "data arranged in rows/columns" | furniture | 759k |
   | `plant` | "buildings for industrial labor" (factory) | botanical | 322k |
   | `glass` | the material | drinking glass | 256k |
   | `female` | "an animal that produces gametes" | person | 1.16M |
   | `car` | "where passengers ride up and down" (elevator) | automobile | 860k |
   | `pool` | pocket billiards | swimming pool | 78k |
   | `nail` | construction nail | fingernail | 75k |
   | `jersey` | Jersey dairy cattle | shirt | 137k |
   | `board` | committee | plank | 12k |

   Asymmetry tell: `male`→person (correct) but `female`→animal (wrong). 1,505
   polysemous linked head tags (`Nc≥3000`) exist — the problem is broad, and it
   touches two top-40 concepts (`table`, `plant`).
2. **Descriptor leaks** — mostly handled by `STOPTAGS` (bare `blue`/`photograph`
   confirmed dropped); residual leaks arrive via compounds (`X in blue`→butterfly).
3. **Taxonomy naming bug** — the automobile node `n02958343` has primary name
   `"car side"`, so the tag `"car"` scores it 0.843 and loses to the elevator node
   named exactly `"car"` (1.000). Bad node metadata; one-field fix.

These distort the composition **report's grouping** (Animals 45% inflated by
`female`; Plants 36% by `hair`/`cloud`/`skin`; Design 23% by `smile`) but that
grouping never feeds the schedule.

### C. A category-free detector: what failed, what worked
Goal: catch mislinks without relying on the 11 hand-curated anchors (which only
expose errors that happen to land in a category). Tested on a 6% corpus sample
(1.23M images) against labeled mislinks + correct controls:

- **Co-occurrence + DAG taxonomic distance — FALSIFIED.** Co-occurrence is a
  *scene* relation (`stripe`+`tiger`), not is-a; it punishes correct links.
- **Co-occurrence + node-embedding coherence — FALSIFIED.** Node vectors are
  built from name+synonyms only, so a surface-string homonym (`car`, `female`)
  embeds next to its misused context and looks coherent.
- **Context-vs-gloss WSD — VALIDATED.** Match a tag's co-occurrence context to
  each candidate *sense's definition* (in `bamboo_V4.json` `id2desc`, currently
  discarded). On a 41-tag labeled set: raw argmax fixed 12/13 wrong, kept 17/28
  correct (11 regressions). Adding a **concreteness guard** (never overturn a
  concrete incumbent for an abstract challenger, via physical-entity ancestry
  `n00001930`) held 12/13 fixes and cut regressions to 8 — of which node-level
  inspection showed ~6 benign co-named sub-senses and **2 genuine** (`male`→animal,
  `hair`→plant). Residual hard core = concrete-vs-concrete same-surface homonyms,
  needing a sense-frequency / corpus-support tie-break on small gloss margins.

Conclusion: a two-guard (concreteness + support) gloss-WSD, run as a
**propose-and-review** Stage-1c (not silent auto-apply), is the right design *if*
link precision ever becomes the goal. It did not, because of D.

### D. Decisive measurement — the mislinks are inert to the schedule
The schedule keys on per-node `N_c` via `m(sample) = max_c m_c` (rarest-wins).
A mislink only matters if the mislinked node *decides* a sample's multiplicity.
Measured on the 6% sample — fraction of samples containing each mislink where it
is the unique rarest concept:

| mislink | band | m_c | decides m |
|---|---|---|---|
| female→animal (1.16M) | HEAD | 0.82 | **0.0%** |
| car→elevator (860k) | HEAD | 0.84 | **0.0%** |
| table→data (759k) | HEAD | 0.85 | **0.0%** |
| plant→factory (322k) | HEAD | 0.91 | 0.2% |
| jersey→cattle (137k) | HEAD | 0.97 | 0.5% |
| pool / nail (75–78k) | MID | 1.00 | 3–4% |
| board→committee (12k) | TAIL | 1.27 | 39% |

Two compounding reasons they don't matter: (1) rarest-wins over ~11 concepts/image
means a correctly-linked rarer tag almost always sets the copy count — the head
mislink is a passenger 0.0–0.5% of the time; (2) head→head is **band-preserving**
(the correct sense is also head, so `m_c` moves <0.2 within the flat head band and
the integer count usually doesn't change). The only per-sample effect is the lone
tail mislink `board→committee` — ~12k images at a mild 1.27×, ~3k extra copies in
a 19.7M corpus. Noise. Rebalancing is robust to link noise **by construction**: it
only cares about frequency band, and the one dangerous mode — a high-frequency tag
mislinking to a genuine *tail* node (spurious junk oversampling) — does not occur
at scale here.

### E. Decisions moving forward
1. **Do not relink; do not build the WSD relinker.** Proven not worth it for the
   rebalancing. Proceed to `build_schedule` → `materialize` → `audit` on the
   current `counts.parquet`.
2. **Keep one cheap guard (Stage 2/3):** assert no high-frequency polysemous tag
   feeds a *tail* (oversampled) node — the only path from link noise to a
   corrupted training mix. **Re-run whenever schedule constants change** (lowering
   `N_low` / raising `M_max` pulls mid mislinks like `pool`/`nail` toward the tail).
   Currently clean.
3. **Evaluation (§6b):** hand-curate the tail-concept eval list rather than
   trusting auto-links for eval buckets — sidesteps link noise for the deliverable.
4. **Zero-cost housekeeping:** footnote the known head mislinks (`table`=data-table,
   `plant`=factory) in the QA report; optionally fix the `n02958343` "car side"
   primary-name bug.

Method note (reproducible): 6% co-occurrence sample = `links.parquet` rows whose
`sample_key` starts with hex `0`; `m_c` from the Stage-3 defaults
(`N_high=100000, N_low=20000, a_head=2.5, r_min=0.1, gamma=0.5, M_max=4`); glosses
+ node vectors from the `wiki` env caches under `semantic_image_browser/data/`.

## Appendix — Schedule calibration against the real N_c distribution (2026-07-24)

Before running Stages 2–5 we measured the actual per-concept count distribution on
`runs/blip3o_pretrain/` and found the plan's *starting* constants (`N_low=20000,
M_max=4`) badly mis-tuned for this corpus. **The example config is not
rebalancing — it is a near-uniform 2× duplication of the whole dataset.** Revised
starting point below. (Stages 2–5 + 6a are pure pyarrow arithmetic — no GPU,
`pyarrow`+`pyyaml` only — so this was calibrated by re-running the reduction, not
guessed.)

### A. The concept distribution is far more bottom-heavy than assumed
58,834 active concepts over 19,755,541 images. Per-concept distinct-sample count
`N_c`:

| p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| **8** | 92 | 1,310 | 5,755 | 67,005 | 3,509,499 |

The median concept has **8 samples**. So `N_low=20000` lands at ≈ p97 — it labels
**57,443 / 58,834 concepts (97.6%) as "tail"** to be oversampled. The head is
genuinely small and heavy: the 405 concepts with `N_c ≥ 100000` carry **66% of all
link mass**, confirming `N_high=100000` is well placed.

### B. Rarest-wins turns that mislabeling into blanket duplication
Reduction is *rarest-wins = max* over ~11 concepts/image, so a sample only needs
**one** sub-`N_low` concept to be oversampled. With 97.6% of concepts below the
threshold, almost every image qualifies. Full simulation over all 218,641,343
links (per-sample max `m`, then `Σm`):

| config (`N_low`, `M_max`, `γ`) | samples oversampled | expanded total | pinned at `M_max` |
|---|---|---|---|
| **example** 20000, 4, 0.5 | **69.0%** | **×1.99** | 16.0% |
| 2000, 3, 0.5 | 21.8% | ×1.19 | 4.6% |
| **1000, 3, 0.5 (chosen)** | 13.7% | ×1.10 | 1.8% |

The example ×1.99 is indiscriminate: it copies nearly the whole corpus rather than
concentrating budget on genuinely rare concepts.

### C. Chosen starting constants
```
--n_high 100000 --n_low 1000 --a_head 2.5 --r_min 0.1 --gamma 0.5 --m_max 3.0
```
- **`N_high=100000`, `a_head=2.5`, `r_min=0.1`** — unchanged. Top 405 concepts =
  66% of mass are the true head; `a_head=2.5` keeps `m_c(N_high)=1` (mid boundary
  continuous). Head downsampling stays gentle by the reference `2/log10(N_c)` law
  (biggest concept 3.5M → `m_c=0.76`).
- **`N_low=1000`** (was 20000) — ≈ p88 of concepts. Reserves oversampling for the
  genuinely under-represented (<1000 of 19.7M images); concepts with a few
  thousand samples have enough signal and stay at `m=1`. Cuts oversampled samples
  from 69% → 13.7%.
- **`M_max=3.0`** (was 4) — large singleton population (`p25=2`; every concept with
  `N_c ≤ ~110` hits the cap at `N_low=1000, γ=0.5`), so the cap, not `γ`, governs
  the tail. Lowering to 3 holds pinned samples at 1.8% and expansion at ×1.10.
- **`γ=0.5`** — unchanged (sub-linear; cap governs singletons anyway).

Per-concept mass shift under the chosen config (head/mid/tail link mass, before →
after, ignoring rarest-wins coupling): head 66.3% → 62.3%, mid 32.2% → 34.7%,
tail 1.5% → 3.0% — a real flattening, not blanket duplication.

### D. Still a *starting* point — calibrate with Stage 6a
These reduce indiscriminate oversampling but are not final. Run
`build_schedule → materialize → audit` and check `head_reinflation_offenders`
(a head concept riding tail co-occurrences back up → lower `M_max` / raise `r_min`)
and the Gini drop before training. Head downsampling is deliberately mild; if the
audit shows heads still dominant that is the knob to revisit next. Cross-reference
the previous appendix's guard (§E.2): `N_low=1000` pulls mid mislinks like
`pool`/`nail` (75–78k, still MID) no closer to the tail, but `board→committee`
(12k) remains the lone tail mislink — re-run the guard since a constant changed.

Method note (reproducible): `N_c` from `runs/blip3o_pretrain/counts.parquet`;
per-sample simulation joins `node_multiplicity` onto all `links.parquet` rows
(arrow dictionary-encoded to int32 to stay under the 50 GB cgroup) and takes the
group-max per `sample_key`, adding `m=1` for the 9 no-concept images
(19,755,541 − 19,755,532).
