# Hierarchical CosFace: Multi-Level Coin Identification

**Status:** Draft v2
**Date:** 2026-04-10
**Context:** HDBSCAN vs CosFace comparison showed CosFace wins authority retrieval (+6.7pp, no noise gap), HDBSCAN retains neighborhood context on ambiguous bleed clusters. Production should be CosFace-first, cluster-aware, retrieval-oriented.

## Problem

The current identify endpoint uses HDBSCAN cluster-vote (KNN-10 on raw 768-d DINOv2). This works but:
- 14-30% of coins are noise (no prediction possible)
- Period purity is moderate (~75% median across clusters)
- Authority discrimination is weak (47.7% on 36-class Roman Imperial)

HDBSCAN's drill-down goes three levels deep (initial cluster -> purity-gated sub-cluster -> KNN noise rescue). CosFace should match that depth, with each level producing a ranked candidate list within its parent scope.

## Core Principle: Ranking-First

Each level is a **scoped retrieval space**, not a classifier. The CosFace margin loss learns a 128-d projected embedding space where cosine similarity is meaningful within that scope. Every level returns **top-k ranked candidates with scores**, preserving ambiguity through the full pipeline. The endpoint never collapses to a single answer internally — it shortlists, ranks, and lets downstream consumers (UI, API callers) decide resolution.

The 81-class Roman Imperial authority head is a **common-authority ranking engine**: it covers the ~81 authorities with sufficient training data. Long-tail authorities (the remaining ~220 with <50 coins) are not in the head's class set — they are handled by falling through to KNN retrieval against cluster neighbors in the projected space. The head ranks what it can; retrieval covers what it can't.

## Architecture

```
Query coin (768-d DINOv2 embedding)
|
+-- Level 1: PERIOD RANKING (11 classes, 126k coins)
|   768-d -> 128-d projection -> cosine score against 11 period centroids
|   Output: top-3 period candidates with scores
|   Default: always route top-3 (heads are cheap, cascading errors are not)
|
+-- Level 2a: AUTHORITY/ISSUER RANKING (per-period, routed from top-3 periods)
|   768-d -> 128-d projection -> cosine score against authority centroids
|   Run for each of top-3 periods that has a trained head
|   Output: top-5 authority candidates per routed period, re-ranked by combined score
|   Roman Imperial: "common-authority ranker" (81 classes, long-tail via retrieval)
|   Greek: "primary-label ranker" (cities/mints/issuers, not rulers)
|
+-- Level 2b: DENOMINATION RANKING (per-period, parallel with authority)
|   768-d -> 128-d projection -> cosine score against denomination centroids
|   Output: top-3 denomination candidates per routed period
|
+-- Cluster Context (HDBSCAN)
    KNN-10 vote on raw 768-d -> cluster assignment
    Cluster purity, bleed status, and known hard pairs inform trust scoring
    Output: cluster_id, cluster neighbors, trust signal
```

Each level is an independent `CosFaceHead(768 -> 128-d)` trained only on coins within its parent scope. The 128-d projections at each level are optimized for different discrimination tasks. At inference, the top-3 period routing means 2-3 authority heads fire in parallel — still sub-millisecond per head on CPU.

## Data Availability Per Level

### Level 1: Period (ready)

| Period | Coins | Status |
|--------|------:|--------|
| roman_imperial | 58,629 | Tested: 86.7% top-1, 96.8% top-3 |
| greek | 40,891 | Tested |
| roman_provincial | 8,511 | Tested |
| roman_republican | 4,347 | Tested |
| byzantine | 3,567 | Tested |
| medieval | 3,526 | Tested |
| central_asian | 1,881 | Tested |
| oriental_greek | 1,569 | Tested |
| islamic | 1,481 | Tested |
| celtic | 1,446 | Tested |
| persian | 423 | Tested |

### Level 2a: Authority/Issuer Ranking (per-period)

| Period | Head Name | Labeled | Classes >= 50 | Semantics | Notes |
|--------|-----------|--------:|------:|-----------|-------|
| roman_imperial | common-authority ranker | 20,223 | 81 | Rulers (emperors) | Best coverage. 36-class tested at 54.4% top-1, 80.3% top-5 |
| greek | primary-label ranker | 14,919 | 44 | Cities, mints, issuers | NOT rulers. "Athens", "Syracuse", "Alexander". "Circa" excluded. |
| roman_provincial | authority ranker | 4,040 | 22 | Rulers (same as RI) | Overlapping names with RI — same rulers, different coin styles |
| byzantine | authority ranker | 2,935 | 15 | Rulers | Verbose labels need normalization |
| roman_republican | — | 1,313 | 3 | — | Not viable, retrieval-only |
| medieval | — | 1,669 | 5 | — | Not viable, retrieval-only |

**Long-tail contract:** For periods without a trained head, and for authorities below the 50-coin threshold within periods that do have heads, identification falls through to KNN retrieval in the period-level projected space against cluster neighbors. The head ranks what it can; retrieval covers what it can't.

### Level 2b: Denomination Ranking (per-period)

| Period | Labeled | Classes >= 50 |
|--------|--------:|------:|
| roman_imperial | 17,083 | 18 |
| greek | 15,801 | 14 |
| roman_provincial | 4,162 | 12 |
| byzantine | 2,902 | 8 |
| roman_republican | 3,102 | 4 |

Denomination heads run in parallel with authority — no dependency. A coin's denomination (Denarius, Tetradrachm, Follis) is visually distinct independent of who issued it.

## Inference Flow

```python
def identify(embedding_768d):
    # Level 1: Top-3 period candidates (always route all 3)
    period_logits = period_head.score(embedding_768d)        # (11,) cosine scores
    period_probs = softmax(period_logits)
    top3_periods = argsort(period_probs)[-3:][::-1]          # descending

    period_candidates = [
        {"period": class_map[p], "score": float(period_probs[p])}
        for p in top3_periods
    ]

    # Level 2a: Authority ranking — run for each routed period with a head
    authority_candidates = []
    for p_idx in top3_periods:
        period_name = class_map[p_idx]
        period_score = float(period_probs[p_idx])

        if period_name in authority_heads:
            head = authority_heads[period_name]
            auth_logits = head.score(embedding_768d)          # (N_auth,) cosine scores
            auth_probs = softmax(auth_logits)
            top5_auth = argsort(auth_probs)[-5:][::-1]

            for a_idx in top5_auth:
                combined = period_score * float(auth_probs[a_idx])
                authority_candidates.append({
                    "period": period_name,
                    "authority": head.class_map[a_idx],
                    "authority_score": float(auth_probs[a_idx]),
                    "combined_score": combined,
                })

    # Re-rank authority candidates by combined score, keep top-5
    authority_candidates.sort(key=lambda x: -x["combined_score"])
    authority_candidates = authority_candidates[:5]

    # Level 2b: Denomination ranking — same routing
    denomination_candidates = []
    for p_idx in top3_periods:
        period_name = class_map[p_idx]
        period_score = float(period_probs[p_idx])

        if period_name in denomination_heads:
            head = denomination_heads[period_name]
            denom_logits = head.score(embedding_768d)
            denom_probs = softmax(denom_logits)
            top3_denom = argsort(denom_probs)[-3:][::-1]

            for d_idx in top3_denom:
                combined = period_score * float(denom_probs[d_idx])
                denomination_candidates.append({
                    "period": period_name,
                    "denomination": head.class_map[d_idx],
                    "denomination_score": float(denom_probs[d_idx]),
                    "combined_score": combined,
                })

    denomination_candidates.sort(key=lambda x: -x["combined_score"])
    denomination_candidates = denomination_candidates[:3]

    # Cluster context
    cluster = get_cluster_context(embedding_768d)

    # Trust scoring
    trust = compute_trust(period_candidates, authority_candidates, cluster)

    return {
        "periods": period_candidates,              # top-3
        "authorities": authority_candidates,        # top-5, cross-period re-ranked
        "denominations": denomination_candidates,   # top-3, cross-period re-ranked
        "cluster": cluster,
        "trust": trust,
    }
```

## Production Response Schema

```json
{
  "periods": [
    {"period": "roman_imperial", "score": 0.91},
    {"period": "roman_provincial", "score": 0.06},
    {"period": "greek", "score": 0.02}
  ],
  "authorities": [
    {"period": "roman_imperial", "authority": "Hadrian", "authority_score": 0.42, "combined_score": 0.38},
    {"period": "roman_imperial", "authority": "Trajan", "authority_score": 0.18, "combined_score": 0.16},
    {"period": "roman_imperial", "authority": "Antoninus Pius", "authority_score": 0.11, "combined_score": 0.10},
    {"period": "roman_provincial", "authority": "Hadrian", "authority_score": 0.35, "combined_score": 0.02},
    {"period": "roman_imperial", "authority": "Marcus Aurelius", "authority_score": 0.08, "combined_score": 0.07}
  ],
  "denominations": [
    {"period": "roman_imperial", "denomination": "Denarius", "denomination_score": 0.67, "combined_score": 0.61},
    {"period": "roman_imperial", "denomination": "Sestertius", "denomination_score": 0.15, "combined_score": 0.14},
    {"period": "roman_provincial", "denomination": "Bronze", "denomination_score": 0.41, "combined_score": 0.02}
  ],
  "cluster": {
    "cluster_id": "671-23",
    "cluster_period_majority": "roman_imperial",
    "cluster_period_purity": 0.89,
    "cluster_authority_majority": "Hadrian",
    "cluster_bleed_status": "clean",
    "top_neighbors": ["coin_134521", "coin_89432", "coin_201004"]
  },
  "trust": {
    "level": "high",
    "period_agreement": true,
    "authority_agreement": true,
    "cluster_purity_weight": 0.89,
    "flags": []
  },
  "routed_heads": ["period", "roman_imperial_authority", "roman_imperial_denomination"]
}
```

## Trust Scoring

Trust is not just "CosFace agrees/disagrees with cluster." Three factors:

### 1. CosFace-Cluster Agreement
Does the top CosFace period/authority match the cluster's majority label?

### 2. Cluster Quality
A disagreement with a **high-purity cluster** (>90%) is a strong signal — either CosFace is wrong or this coin is a genuine outlier. A disagreement with a **bleed cluster** (<75% purity) is weak signal — the cluster itself is unreliable.

### 3. Known Hard Pairs
Certain authority pairs are known-ambiguous from prior analysis (e.g., Diocletian/Maximian, separation=0.060). If the top-2 CosFace authority candidates are a known hard pair, flag it — ambiguity is expected, not a failure.

```python
def compute_trust(period_candidates, authority_candidates, cluster):
    top_period = period_candidates[0]["period"]
    top_period_score = period_candidates[0]["score"]
    cluster_period = cluster["cluster_period_majority"]
    cluster_purity = cluster["cluster_period_purity"]

    period_agreement = (top_period == cluster_period)

    # Weight disagreement by cluster purity
    # Disagreeing with a 95% pure cluster is much worse than with a 60% one
    if not period_agreement:
        disagreement_weight = cluster_purity  # 0.0-1.0
    else:
        disagreement_weight = 0.0

    # Check for known hard pairs in top-2 authority candidates
    flags = []
    if len(authority_candidates) >= 2:
        pair = frozenset([authority_candidates[0]["authority"],
                          authority_candidates[1]["authority"]])
        if pair in KNOWN_HARD_PAIRS:
            flags.append(f"known_hard_pair:{sorted(pair)}")

    # Composite trust
    if top_period_score > 0.8 and period_agreement:
        level = "high"
    elif top_period_score > 0.8 and disagreement_weight < 0.75:
        level = "high"      # confident CosFace, low-quality cluster disagrees — trust CosFace
    elif period_agreement:
        level = "medium"
    elif disagreement_weight > 0.85:
        level = "low"       # confident cluster disagrees — surface both
    else:
        level = "medium"    # neither side is confident

    return {
        "level": level,
        "period_agreement": period_agreement,
        "authority_agreement": check_authority_agreement(authority_candidates, cluster),
        "cluster_purity_weight": cluster_purity,
        "flags": flags,
    }
```

## Retrieval Assets

Each trained head exports three files for production serving:

| File | Size | Purpose |
|------|------|---------|
| `{scope}_head.pt` | ~386 KB | `nn.Linear(768, 128)` projection weights |
| `{scope}_centroids.npy` | ~N x 128 x 4 B | L2-normalized margin prototypes (class centroids in projected space) |
| `{scope}_meta.json` | ~2 KB | Class map, training config, scope definition |

**Scoped naming convention:**
- `period_head.pt`, `period_centroids.npy`, `period_meta.json`
- `roman_imperial_authority_head.pt`, `roman_imperial_authority_centroids.npy`, ...
- `greek_primary_label_head.pt`, `greek_primary_label_centroids.npy`, ...
- `roman_imperial_denomination_head.pt`, ...

Total for full hierarchy (1 period + 4 authority/issuer + 4 denomination = 9 heads): ~4 MB.

### Query-Time Retrieval

At query time, each head serves two retrieval modes:

**Mode 1: Centroid ranking** (fast, always available)
Project query to 128-d, cosine against centroids. Returns ranked candidates with scores. This is the `.score()` call in the inference flow — one matmul per head.

**Mode 2: KNN retrieval** (richer, for long-tail and context)
Project query to 128-d, cosine against pre-projected index of all coins in that scope. Returns top-k neighbor coins with scores. Used when:
- The authority/issuer is below the head's class threshold (long-tail)
- Trust is low and cluster neighbors are needed for context
- The caller wants visual matches, not just labels

Pre-projected indices are scoped: only the period-level index covers all 128k coins. Authority-level indices cover only the coins within that period. These are precomputed at export time.

| Index | Coins | Size | Precomputed At |
|-------|------:|-----:|----------------|
| Period (all) | 128,306 | 63 MB | Export time |
| RI Authority | 58,629 | 29 MB | Export time |
| Greek Primary Label | 40,891 | 20 MB | Export time |
| RI Denomination | 58,629 | 29 MB | Shares projection with RI Authority if same head; otherwise separate |

## Label Semantics By Period

| Period | Level 2a Name | Label Type | Examples | Notes |
|--------|---------------|------------|----------|-------|
| roman_imperial | Authority Ranker | Rulers (emperors) | Hadrian, Trajan, Caracalla | 81 common authorities; ~220 long-tail via retrieval |
| greek | Primary Label Ranker | Cities, mints, issuers | Athens, Syracuse, Alexander, Corinth | NOT rulers. "Circa" excluded. "Alexander" = Alexander III unless disambiguated |
| roman_provincial | Authority Ranker | Rulers (same as RI) | Caracalla, Hadrian, Septimius Severus | Same people as RI, different coin traditions. Cross-link in response. |
| byzantine | Authority Ranker | Rulers | Justinian I, Phocas, Heraclius | Verbose labels normalized (strip co-regent suffixes) |

The response schema uses `"authority"` as the field name for all periods for API consistency, but `routed_heads` reveals which head produced it, and the meta.json for each head documents its label semantics.

## Known Limitations

1. **Label sparsity:** Only 34-47% of coins have authority labels per period. Heads train on the labeled subset and rank all coins. Coins from unlabeled authorities will get assigned to the nearest known authority — the score indicates confidence.

2. **Long-tail authorities:** The RI head covers 81 of ~300 known authorities. The remaining ~220 have too few labeled coins to train on. For these, the head will still produce a ranking (nearest common authority), but the correct answer isn't in the candidate set. The KNN retrieval fallback in projected space handles this — if the top centroid score is low, surface neighbors instead of labels.

3. **Cross-period authority overlap:** Caracalla appears in roman_imperial (767) and roman_provincial (325). Because routing is top-3, both heads fire and both Caracalla entries appear in the response. The re-ranking by combined score naturally handles this — the higher-confidence period wins, but both are visible.

4. **Denomination ambiguity:** "Bronze", "AE", "Copper" are vague catch-all denominations. Pre-training label normalization should merge these. Denomination ranking is inherently lower-resolution than authority ranking.

5. **Greek label quality:** "Circa" is a date prefix, not an issuer — excluded. "Alexander" is ambiguous. "Macedon" is a region. The Greek head's value is in retrieval geometry (putting Athenian owls near each other), not in label accuracy. Frame it as "visually similar coins issued under this label" rather than "this coin was definitely issued by Athens."

## Training Plan

All training on CPU. Each head is `CosFaceHead(768, 128)` — 98k params, trains in 1-3 minutes on frozen embeddings.

### Phase 1: Period head (done)
- 11 classes, 126k coins
- Trained in hdbscan_vs_cosface.py
- Save as proper checkpoint with export

### Phase 2: Roman Imperial heads (done)

**RI common-authority ranker**
- 83 classes (authorities >= 50 coins), 29k coins, 80/10/10 split
- Top-1: 49.8%, Top-5: 77.6%, KNN-10: 50.1% (train-reference protocol)
- Retrieval evaluation was re-run under a consistent train-reference / test-query protocol. Under matched methodology, the 83-class CosFace projected space performs essentially identically to the 36-class version (KNN-10: 48.8% vs 48.6%) and significantly outperforms raw DINOv2 (39.7%), confirming that an earlier lower KNN result was a methodology artifact.
- Exported: `roman_imperial_authority_{head,centroids,meta}.*`

**RI denomination ranker**
- 17 classes, 26k coins
- Top-1: 77.4%, Top-5: 95.4%, KNN-10: 78.8%
- Exported: `roman_imperial_denomination_{head,centroids,meta}.*`

### Phase 3: Wire MVP into visual_search
- Period + RI authority + RI denomination
- Top-3 period routing
- Response schema as specified above
- Evaluate end-to-end against current HDBSCAN pipeline

### Phase 4: Extend to other periods
- Greek primary-label ranker (label cleanup first: exclude "Circa", normalize "Alexander")
- Roman Provincial authority ranker
- Byzantine authority ranker (label normalization: strip co-regent suffixes)
- Denomination rankers for each

### Phase 5: Trust scoring + cluster integration
- Purity-weighted cluster agreement
- Known hard-pair flagging
- Long-tail retrieval fallback wiring

## Success Criteria

| Metric | Current (HDBSCAN) | Target (Hierarchical CosFace) |
|--------|-------------------|-------------------------------|
| Period top-1 | 89.5% (clustered only) | >85% (all coins) |
| Period top-3 | N/A | >96% (all coins) |
| RI Authority top-1 | 47.7% (clustered only) | >45% top-1 (all, 81-class) |
| RI Authority top-5 | N/A | >75% (all coins) |
| Coverage | 70-77% (noise excluded) | 100% |
| Inference latency | ~5ms (KNN vote) | ~2ms (3-9 matmuls) |
| Noise coins handled | No | Yes |
| Long-tail authorities | No prediction | KNN retrieval fallback |
| Response ambiguity | Single label | Top-k ranked candidates |
