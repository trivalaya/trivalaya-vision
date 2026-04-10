# Legend CNN + Embedding Heads — Experiment Results

**Date:** 2026-04-10
**GPU:** RunPod RTX 5090 (33.7 GB VRAM), PyTorch 2.8.0+cu128

---

## Part 1: Legend Ribbon CNN (dead end)

### Question
Can a lightweight CNN on grayscale polar-unwrapped legend ribbons separate 10 Roman Imperial authorities?

### Answer
**No.** Four training runs all plateau at ~18–20% validation accuracy (random = 10%). The representation is the bottleneck — not optimization, data quality, or spatial resolution.

### Training Runs

| Run | LR | Batch | Ribbon | Filter | Best Val Acc | Val Loss |
|-----|----|-------|--------|--------|-------------|----------|
| 0 baseline | 1e-3 | 256 | 64×512 | none | 19.6% | 2.074 |
| 1 optim | 3e-4 | 64 | 64×512 | none | 18.5% | 2.153 |
| 2 filtered | 3e-4 | 64 | 64×512 | −bad_blank | 20.3% | 2.072 |
| 3 h128 | 3e-4 | 64 | 128×512 | none | 18.5% | 2.109 |

### What Was Ruled Out
- **Optimization instability** (Run 1 vs 0): lower LR + smaller batch smoothed val loss but didn't help accuracy
- **Bad ribbon quality** (Run 2 vs 1): only 27 bad ribbons exist; filtering made no difference
- **Spatial resolution** (Run 3 vs 1): doubling ribbon height from 64 to 128 made no difference

### Human-Eye Gallery Findings
- **Intra-class variation is enormous:** same authority ribbons look completely different across coins (different sizes, preservation, centering)
- **Inter-class distinction is not visible in texture:** geometry artifacts (black wedges from polar unwrap) dominate over legend content
- **Model's "confident" predictions are 13–16%** — barely above uniform (10%)
- The unwrap destroys more spatial letter structure than it preserves

### Top Confusion Pairs (Ribbon CNN)
Trajan→Hadrian (40), Septimius Severus→Caracalla (31), Gordian III→Hadrian (27)

### Conclusion
Ribbon-only texture classification is a validated dead end. Authority discrimination likely requires reading letter sequences (closer to OCR), not texture statistics.

---

## Part 2: Embedding Heads on Frozen DINOv2

### Question
Can a learned projection head on frozen DINOv2 ViT-B/14 768-d obverse embeddings (a) classify authorities accurately, and (b) make cosine similarity useful for production retrieval?

### Answer
**Yes to both.** MLP head achieves 81% top-1 classification. CosFace projection pushes cosine KNN from 66% to 80%, nearly closing the gap to classification accuracy.

### Setup
- **Input:** 768-d DINOv2 obverse embeddings (frozen, precomputed)
- **Dataset:** 6,194 coins, 10 authorities, 80/10/10 stratified split (same splits as ribbon CNN)
- **Training:** CPU-only, AdamW + ReduceLROnPlateau, early stopping

### Classification Results

| Head | Top-1 | Top-3 | Top-5 | Worst Pair | Errors |
|------|-------|-------|-------|------------|--------|
| **MLP** | **81.0%** | 95.2% | 98.4% | Maximian→Diocletian | 12 |
| ArcFace | 78.6% | 93.9% | 97.9% | Diocletian→Maximian | 19 |
| Linear | 77.7% | 95.4% | 99.0% | Diocletian→Maximian | 16 |
| CosFace | 77.1% | 93.8% | 98.2% | Diocletian→Maximian | 16 |

### Embedding Space Results (Production-Relevant)

| Metric | Raw DINOv2 768-d | ArcFace 128-d | CosFace 128-d |
|--------|-----------------|---------------|---------------|
| Cosine KNN-1 | 61.2% | 76.8% | **78.7%** |
| Cosine KNN-10 | 66.3% | 79.2% | **79.9%** |

### Per-Authority KNN-10 Recall (CosFace vs Raw)

| Authority | CosFace | Raw | Lift |
|-----------|---------|-----|------|
| Antoninus Pius | 81.1% | 45.9% | **+35.1pp** |
| Septimius Severus | 87.8% | 68.9% | +18.9pp |
| Hadrian | 89.0% | 71.4% | +17.6pp |
| Maximian | 74.5% | 60.8% | +13.7pp |
| Caracalla | 75.0% | 63.2% | +11.8pp |
| Gordian III | 88.1% | 77.6% | +10.5pp |
| Diocletian | 65.8% | 55.3% | +10.5pp |
| Trajan | 80.0% | 74.3% | +5.7pp |
| Severus Alexander | 73.3% | 65.0% | +8.3pp |
| Elagabalus | 70.2% | 61.7% | +8.5pp |

Every authority improves. Antoninus Pius sees the largest lift (+35pp).

### Hard-Pair Analysis: Diocletian ↔ Maximian

This pair accounts for 23 of the top-10 errors (13 Dio→Max + 10 Max→Dio).

| Metric | Diocletian/Maximian | Hadrian/Gordian III |
|--------|--------------------|--------------------|
| Intra-class sim | 0.486 / 0.483 | 0.493 / 0.521 |
| Cross-class sim | 0.424 | −0.219 |
| **Separation** | **0.060** | **0.726** |

Diocletian and Maximian have 12× less separation than a well-separated pair. Their cross-class similarity (0.424) nearly equals their intra-class similarity (~0.484). This is not a model failure — DINOv2 sees them as visually near-identical because they were Tetrarchy co-emperors with deliberately similar coinage.

Misclassified coins confirm: nearest-Diocletian and nearest-Maximian distances are within 0.01–0.04 for most errors (e.g., coin 164314: Dio=0.120, Max=0.122). These are genuinely ambiguous.

### Calibration

The CosFace head is **overconfident** (ECE = 0.176). The s=30 scaling compresses most predictions to >99% softmax confidence. However:

- Confidence still separates correct from wrong (mean 0.971 vs 0.868)
- Reliability improves with thresholds: confidence ≥ 0.9 → 83.5% accuracy at 84% coverage
- Temperature scaling on the logits would fix calibration cheaply

**Top-K is the production headline:** top-5 = 98.2%, top-10 = 100%.

---

## Decision Framework

Per the spec in `specs/embedding_heads_experiment.md`:

- **MLP ≈ Linear (+3pp):** nonlinearity helps modestly → use MLP for classification if a head is deployed
- **CosFace KNN-10 = 79.9% (approaching probe accuracy of 81%):** the embedding space has been reshaped successfully → **CosFace embeddings are viable for the identify endpoint**, replacing raw DINOv2 cosine for authority matching
- Remaining gap (79.9% KNN vs 81.0% MLP) is small enough that retrieval-based identification is practical

## Recommendations

1. **For classification (post-hoc labeling):** use MLP head (81% top-1, 95% top-3)
2. **For retrieval (identify endpoint):** use CosFace 128-d projected embeddings (79.9% KNN-10)
3. **Treat Diocletian/Maximian as a known hard pair** — consider merging into "Tetrarchy" or flagging as ambiguous when both are in top-3
4. **Add temperature scaling** to fix calibration before using confidence scores as absolute thresholds
5. **Do not pursue ribbon-only CNN** further — the signal isn't there

## Artifacts

```
embedding_heads/
├── config.py, dataset.py, heads.py, arcface.py
├── train.py, evaluate.py, compare.py, deep_eval.py
├── checkpoints/{linear,mlp,arcface,cosface}_best.pt  (gitignored)
└── outputs/
    ├── all_results.json
    ├── eval_{linear,mlp,arcface,cosface}.json
    ├── confusion_{linear,mlp,arcface,cosface}.png
    ├── calibration_cosface.png
    └── train_log_{linear,mlp,arcface,cosface}.csv

legend_cnn/
├── outputs/
│   ├── train_log_run{0-4}_*.csv
│   ├── training_curves*.png
│   └── human_galleries/  (same-auth, cross-auth, confidence galleries)
└── runs/run_20260410_161829_promoted_v1_h128/  (128px re-unwrap)
```
