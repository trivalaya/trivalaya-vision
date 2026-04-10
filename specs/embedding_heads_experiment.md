# Obverse Embedding Head Comparison — Experiment Spec

**Location:** `trivalaya-vision/embedding_heads/`
**Framework:** PyTorch
**Goal:** Determine whether a metric/margin head on frozen DINOv2 obverse embeddings meaningfully improves authority discrimination over a linear probe.

---

## Context

A linear probe on frozen 768-d DINOv2 obverse embeddings achieves 84.3% top-1 on 10-way authority classification. Centroid-based cosine similarity achieves only 56.5%. The gap proves the signal exists but is inaccessible to cosine similarity. This experiment tests whether a learned projection can close that gap for production use.

---

## Dataset

Reuse the exact same confusion subset and splits from the linear probe experiment:
- Same 10 authorities, same coin_ids, same 80/10/10 stratified split
- Input: precomputed 768-d obverse DINOv2 embeddings (frozen, no image loading needed)
- If embeddings are already saved as .npy or in a DB table, load directly — no GPU needed for data prep

---

## Three Heads to Compare

### Head A: Linear Probe (baseline)

```
Linear(768 → num_classes)
```

- CrossEntropyLoss
- This is the existing result (84.3%) — rerun for identical-split consistency

### Head B: Small MLP

```
Linear(768 → 256) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(256 → num_classes)
```

- CrossEntropyLoss
- Tests whether a nonlinear projection extracts more signal

### Head C: ArcFace / CosFace

```
Linear(768 → 128)  →  L2-normalize  →  ArcFace margin layer (num_classes, s=30, m=0.50)
```

- ArcFaceLoss (additive angular margin on the correct class logit before softmax)
- This reshapes the embedding space so cosine similarity becomes discriminative
- After training, the 128-d normalized output IS the new embedding — cosine similarity on these should approach probe accuracy
- Also test CosFace (multiplicative margin, m=0.35) as a variant if time permits

**ArcFace implementation note:** Use `torch.nn.functional` — the margin is applied only to the ground-truth class logit:
```python
# pseudo:
cos_theta = F.linear(F.normalize(features), F.normalize(weight))
cos_theta_target = cos_theta[target_class] 
cos_theta_m = cos(acos(cos_theta_target) + m)  # additive angular margin
logits = s * cos_theta_with_margin_applied
loss = CrossEntropyLoss(logits, target)
```

---

## Training (all heads)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 (linear, MLP), 1e-2 (ArcFace — needs higher lr for the weight matrix) |
| LR scheduler | ReduceLROnPlateau(patience=3, factor=0.5) |
| Batch size | 1024+ (these are tiny 768-d vectors, no GPU memory pressure) |
| Epochs | 50 max |
| Early stopping | patience=7 on val metric (val_loss for A/B, val_top1 for C) |
| Weight decay | 1e-4 |

- Train on CPU or GPU — embeddings-only training is fast either way
- Save best checkpoint per head

---

## Evaluation — Report Per Head

Run on held-out 10% test set:

### Core Metrics
- **Top-1 accuracy**
- **Top-3 accuracy**
- **Top-5 accuracy**

### Per-Class
- **Per-class recall** (table, sorted worst to best)
- **Per-class precision**

### Confusion Analysis
- **Confusion matrix** (saved as image)
- **Hardest pairs table:** For each head, list the top-5 most confused (A→B) pairs with error counts
- **Delta vs linear baseline:** For each pair, show whether MLP/ArcFace improved, worsened, or held steady

### ArcFace-Specific (Head C only)
- After training, extract the 128-d L2-normalized embeddings for all test coins
- Compute **cosine similarity top-1 accuracy** on these new embeddings (KNN with K=1)
- Compare to the original 768-d cosine similarity (56.5%) — this measures whether the learned space makes cosine useful
- Also compute **KNN K=10 accuracy** on the new 128-d embeddings for direct comparison to the 67.9% baseline

---

## Output Summary

Generate a single summary table:

```
┌──────────┬───────┬───────┬───────┬────────────────┬────────────────┐
│ Head     │ Top-1 │ Top-3 │ Top-5 │ Worst Pair     │ Worst Pair Err │
├──────────┼───────┼───────┼───────┼────────────────┼────────────────┤
│ Linear   │       │       │       │                │                │
│ MLP      │       │       │       │                │                │
│ ArcFace  │       │       │       │                │                │
│ CosFace  │       │       │       │                │                │
└──────────┴───────┴───────┴───────┴────────────────┴────────────────┘
```

And a separate table for ArcFace embedding-space metrics:

```
┌────────────────────────────┬──────────┬───────────┐
│ Metric                     │ Original │ ArcFace   │
├────────────────────────────┼──────────┼───────────┤
│ Cosine KNN-1 accuracy      │ (≈56.5%) │           │
│ Cosine KNN-10 accuracy     │ (≈67.9%) │           │
│ Probe/head top-1 accuracy  │ 84.3%    │           │
└────────────────────────────┴──────────┴───────────┘
```

---

## File Structure

```
trivalaya-vision/
└── embedding_heads/
    ├── config.py              # paths, hyperparameters, authority list
    ├── dataset.py             # loads precomputed 768-d embeddings + labels
    ├── heads.py               # LinearHead, MLPHead, ArcFaceHead classes
    ├── arcface.py             # ArcFace/CosFace margin layer implementation
    ├── train.py               # unified training loop, selects head by CLI arg
    ├── evaluate.py            # full evaluation suite, generates tables + plots
    ├── compare.py             # loads all checkpoints, produces summary tables
    ├── checkpoints/
    │   ├── linear_best.pt
    │   ├── mlp_best.pt
    │   └── arcface_best.pt
    └── outputs/
        ├── confusion_linear.png
        ├── confusion_mlp.png
        ├── confusion_arcface.png
        ├── summary_table.txt
        └── hardest_pairs.txt
```

---

## Execution

```bash
# All three can run on the droplet CPU — no GPU needed
python -m embedding_heads.train --head linear
python -m embedding_heads.train --head mlp
python -m embedding_heads.train --head arcface

# Optional
python -m embedding_heads.train --head cosface

# Compare
python -m embedding_heads.compare
```

---

## Decision Framework

After results:

- If **MLP ≈ Linear**: nonlinearity doesn't help, signal is linearly separable → use linear head in production (simplest)
- If **MLP > Linear by 3+pp**: worth the small complexity → use MLP
- If **ArcFace cosine KNN-10 approaches probe accuracy (>80%)**: the embedding space has been reshaped successfully → integrate ArcFace embeddings into the identify endpoint, replacing raw DINOv2 cosine for authority matching
- If **ArcFace cosine KNN-10 stays low (<75%)**: margin learning helps classification but doesn't transfer to retrieval → use probe/MLP as a post-hoc classifier on raw embeddings instead

---

## Dependencies

torch, scikit-learn, matplotlib, numpy. No new dependencies beyond what's already in the pipeline.
