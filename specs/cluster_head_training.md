# Cluster-Driven Head Model Training

**Status**: Active
**Date**: 2026-04-11
**Context**: Pipeline spec `trivalaya-pipeline/specs/head_models_v1.md` (v4) defines the
cluster analysis and candidate discovery. This spec covers what happens in the vision
repo: receiving scope data, training CosFace heads, evaluating, calibrating, and
exporting assets for the pipeline to consume.

---

## 1. Input Contract

The pipeline repo's `prepare_head_training_data.py` writes three files per scope to
`embedding_heads/data/`:

```
data/{scope}_subset.csv      — coin_id, label_int, authority
data/{scope}_splits.json     — {scope, seed, train: [...], val: [...], test: [...]}
data/{scope}_class_map.json  — {"0": "Caracalla", "1": "Diadumenian", ...}
```

**Splits are leakage-safe** (grouped by auction lot + near-duplicate coins). The vision
repo trusts these — it does not re-split. Cross-cluster pooled coins are oversampled
2-3x in the subset CSV by the pipeline; the vision repo sees them as normal rows.

**Embeddings** are read from the pipeline's prod directory (path in `config.py`):
```
/root/trivalaya-pipeline/cluster_output_prod/obv_features_768.npy  (128,306 × 768)
```

The subset CSV's `coin_id` column maps to rows in the enriched CSV, which maps 1:1 to
rows in the embedding array. `dataset.py`'s `build_dataloaders_from_scope()` handles
this mapping — it already works for the existing `ri_authority`, `ri_denomination`, and
`period` scopes.

### Current Scopes (from Run 4 discovery)

| Scope | N-way | Train | Val | Test | KNN Baseline |
|-------|-------|-------|-----|------|-------------|
| `auth_ri_severan_13way` | 13 | 5,504 | 635 | 615 | 59.3% |
| `auth_ri_nerva_antonine_12way` | 12 | 4,457 | 533 | 546 | 53.4% |
| `auth_ri_cluster_66_8way` | 8 | 4,986 | 598 | 597 | 68.7% |
| `auth_ri_late_roman_8way` | 8 | 3,032 | 318 | 302 | 45.0% |
| `auth_ri_3rd_century_crisis_8way` | 8 | 1,388 | 171 | 172 | 54.5% |
| `auth_ri_julio_claudian_3way` | 3 | 1,332 | 149 | 159 | 69.7% |
| `auth_ri_cluster_91_2way` | 2 | 787 | 88 | 96 | 59.5% |
| `auth_ri_illyrian_emperors_4way` | 4 | 679 | 92 | 81 | 72.1% |
| `auth_ri_civil_war_68_3way` | 3 | 489 | 59 | 60 | 70.2% |
| `auth_byz_cluster_99_6way` | 6 | 374 | 49 | 47 | 79.6% |

## 2. Code Changes Required

### 2a. train.py — add `--scope` argument

Current interface uses `--version {v1, v2}`. Add `--scope` as the primary interface
for cluster-driven heads:

```
python -m embedding_heads.train --scope auth_ri_severan_13way --head cosface
```

When `--scope` is provided:
- Call `build_dataloaders_from_scope(scope)` instead of `build_dataloaders(version)`
- Checkpoint saves to `checkpoints/{scope}_cosface_best.pt`
- Train log saves to `outputs/train_log_{scope}.csv`
- All other training logic (optimizer, scheduler, early stopping) unchanged

`--version` remains for backward compatibility with v1/v2 scopes.

Implementation: ~10 lines in `main()` — an if/else on `args.scope` before the
dataloader call, and a tweak to the checkpoint/log naming.

### 2b. export.py — add `--scope` argument for cluster-driven heads

Current scoped export handles `period`, `ri_authority`, `ri_denomination` with
hardcoded prefix mappings. Generalize:

```
python -m embedding_heads.export --scope auth_ri_severan_13way \
    --output-dir /root/trivalaya-pipeline/cluster_output_prod/head_models/
```

When `--scope` is provided:
- Load checkpoint from `checkpoints/{scope}_cosface_best.pt`
- Load class map from `data/{scope}_class_map.json`
- Load splits from `data/{scope}_splits.json` for temperature calibration
- Output files use scope as prefix:
  - `{scope}_head.pt` — Linear(768 → 128) weights
  - `{scope}_centroids.npy` — (num_classes, 128) L2-normed centroids
  - `{scope}_meta.json` — metadata (see §5 for schema)

No precomputed full-dataset projections (unlike legacy export). Per-scope projections
are unnecessary — the pipeline computes projections on demand for the coins that
route to each head.

### 2c. config.py — EMBED_DIM parameterization

Currently `EMBED_DIM = 768` is a module-level constant used by `heads.py`. For the
§6c combined-embedding experiment (1536-d), make this configurable:

```python
EMBED_DIM = 768  # default, override via --embed-dim CLI arg or scope config
```

This is a future change for the obv+rev experiment. Not needed for the initial 10
scopes (all obverse-only, 768-d).

## 3. Training Protocol

### 3a. Per-Scope Training

For each scope, training is a single command:

```bash
python -m embedding_heads.train --scope {scope} --head cosface
```

**Hyperparameters** (defaults, carried over from existing infrastructure):

| Param | Value | Notes |
|-------|-------|-------|
| Architecture | CosFace | `heads.py:CosFaceHead(768 → 128, s=30.0, m=0.35)` |
| Optimizer | AdamW | lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=3 |
| Epochs | 200 max | Early stopping at patience=7 |
| Batch size | 1024 | Fits in GPU memory easily (embeddings only) |
| Early stop metric | val accuracy | For CosFace heads |

These defaults have been validated on the existing 83-class `ri_authority` scope. The
dynasty-scoped heads (3-13 way) are easier problems — same defaults should work without
sweeping.

### 3b. Sweep (optional, for P0 scopes)

If a P0 scope underperforms expectations, run the existing sweep:

```bash
python -m embedding_heads.sweep_cosface --scope {scope}
```

The sweep grid (from `sweep_cosface.py`): lr ∈ {1e-2, 5e-3}, m ∈ {0.35, 0.50},
s ∈ {30, 64}, dim ∈ {128, 256}. 16 configurations, ~2 min each on GPU.

### 3c. Training Time Estimates

All training is on frozen 768-d embeddings (no image loading, no backbone forward pass).
On a RunPod A40/A100:

| Scope | Train size | Est. time | Notes |
|-------|-----------|-----------|-------|
| auth_ri_severan_13way | 5,504 | ~2 min | Largest scope |
| auth_ri_nerva_antonine_12way | 4,457 | ~2 min | |
| auth_ri_cluster_66_8way | 4,986 | ~2 min | |
| auth_ri_late_roman_8way | 3,032 | ~1.5 min | |
| All 10 scopes | — | ~15 min total | Sequential |

CPU training is also feasible (~5-10 min per scope) since it's linear-layer-only.

## 4. Evaluation Protocol

After training, evaluate each scope with both standard and deep evaluation:

### 4a. Standard Evaluation

```bash
python -m embedding_heads.evaluate --scope {scope} --head cosface
```

Produces `outputs/eval_{scope}.json` with:
- Top-1, top-3, top-5 accuracy
- Per-class recall and precision
- Confusion matrix (PNG)
- KNN-1/10 on projected 128-d vs raw 768-d (lift measurement)

### 4b. Deep Evaluation (calibration + hard pairs)

```bash
python -m embedding_heads.deep_eval --scope {scope}
```

Produces calibration analysis:
- **ECE** (expected calibration error) — target < 0.10
- Top-K accuracy (K=1,2,3,5,10)
- Confidence distribution (correct vs wrong predictions)
- Reliability at confidence thresholds (0.3-0.9)
- Calibration curve plot

And hard-pair analysis:
- Top-10 most confused pairs
- Per-class KNN recall on projected space
- Misclassified coin IDs (for spot-checking)

### 4c. Deployment Gates

A scope passes to export only if **all** gates pass:

| Gate | Threshold | Source |
|------|-----------|--------|
| Test top-1 accuracy | >= 70% | evaluate |
| Accuracy over KNN baseline | >= 5pp | evaluate vs discovery KNN |
| ECE | < 0.10 | deep_eval |

If ECE > 0.10, try temperature scaling (default) → isotonic regression → margin
abstention (see §5 calibration). Record which method achieves ECE < 0.10.

If top-1 < 70% but top-2 >= 85%, the head is still useful for narrowing — flag it as
`deployment_status: "candidate"` rather than rejecting outright.

If top-1 < KNN baseline + 5pp, the head doesn't justify its complexity. Record as
`status: "knn_sufficient"` — the pipeline will use KNN fallback for these clusters.

## 5. Export Deliverables

For each scope that passes deployment gates:

```bash
python -m embedding_heads.export --scope {scope} \
    --output-dir /root/trivalaya-pipeline/cluster_output_prod/head_models/
```

### 5a. Exported Files

```
head_models/{scope}_head.pt           — Linear(768, 128) state_dict (weight + bias)
head_models/{scope}_centroids.npy     — (num_classes, 128) L2-normed class centroids
head_models/{scope}_meta.json         — metadata (schema below)
```

**Total per scope**: ~400 KB. All 10 scopes: ~4 MB.

### 5b. Meta JSON Schema

```json
{
  "scope": "auth_ri_severan_13way",
  "axis": "authority",
  "num_classes": 13,
  "embedding_dim": 128,
  "embedding_input": "obverse",
  "class_names": ["Caracalla", "Diadumenian", ...],

  "performance": {
    "test_accuracy": 0.743,
    "top2_accuracy": 0.891,
    "top5_accuracy": 0.964,
    "knn_baseline": 0.593,
    "accuracy_margin": 0.150
  },

  "calibration": {
    "method": "temperature_scaling",
    "temperature": 0.082,
    "ece": 0.047
  },

  "provenance": {
    "checkpoint": "checkpoints/auth_ri_severan_13way_cosface_best.pt",
    "split_seed": 42,
    "train_size": 5504,
    "val_size": 635,
    "test_size": 615,
    "training_config": {
      "head": "cosface",
      "lr": 0.001,
      "s": 30.0,
      "m": 0.35,
      "proj_dim": 128,
      "epochs_trained": 47,
      "best_val_acc": 0.756
    },
    "cluster_run_id": "run4_20260407",
    "feature_file": "obv_features_768.npy"
  }
}
```

### 5c. Calibration Method

The export script fits calibration during export:

1. Project validation embeddings through the head → cosine scores against centroids
2. Fit **temperature T** by minimizing NLL on validation labels
3. Compute ECE on test set with temperature-scaled softmax
4. If ECE < 0.10 → use `temperature_scaling`, record T
5. If ECE >= 0.10 → fit **isotonic regression** on validation scores, re-check ECE
6. If still >= 0.10 → use **margin abstention**: learn a threshold on (top1 - top2)
   score below which the head abstains. Record threshold.
7. Record chosen method + params in meta JSON

The pipeline reads `calibration.method` and `calibration.temperature` (or
`calibration.isotonic_bins` or `calibration.abstention_threshold`) at inference time.

## 6. Handoff to Pipeline

After export, assets are copied back to the pipeline's droplet:

```bash
# On RunPod, after training + export:
rsync -avz /root/trivalaya-pipeline/cluster_output_prod/head_models/ \
    droplet:/root/trivalaya-pipeline/cluster_output_prod/head_models/
```

The pipeline's `register_head_models.py` then:
1. Reads each `{scope}_meta.json`
2. Checks deployment gates (accuracy, KNN margin, ECE)
3. Writes `head_model` + `cluster_model_xref` DB rows
4. Sets `deployment_status = 'active'`

The pipeline's `visual_search/app.py` loads assets at startup from `head_models/`.

## 7. Batch Training Script

For running all 10 scopes in sequence (the common case on a RunPod burst):

```bash
#!/bin/bash
# train_all_scopes.sh — run from trivalaya-vision root

SCOPES=(
  auth_ri_severan_13way
  auth_ri_nerva_antonine_12way
  auth_ri_cluster_66_8way
  auth_ri_late_roman_8way
  auth_ri_3rd_century_crisis_8way
  auth_ri_julio_claudian_3way
  auth_ri_cluster_91_2way
  auth_ri_illyrian_emperors_4way
  auth_ri_civil_war_68_3way
  auth_byz_cluster_99_6way
)

OUTPUT_DIR="/root/trivalaya-pipeline/cluster_output_prod/head_models"
mkdir -p "$OUTPUT_DIR"

for scope in "${SCOPES[@]}"; do
  echo "=== Training: $scope ==="
  python -m embedding_heads.train --scope "$scope" --head cosface

  echo "=== Evaluating: $scope ==="
  python -m embedding_heads.evaluate --scope "$scope" --head cosface
  python -m embedding_heads.deep_eval --scope "$scope"

  echo "=== Exporting: $scope ==="
  python -m embedding_heads.export --scope "$scope" --output-dir "$OUTPUT_DIR"

  echo ""
done

echo "=== All scopes complete. Assets in $OUTPUT_DIR ==="
ls -la "$OUTPUT_DIR"
```

Estimated total time: ~15 min on GPU, ~60 min on CPU.

## 8. What's Not Changing

- **heads.py** — CosFaceHead architecture unchanged. Same `Linear(768 → 128)` +
  `CosFaceMargin(128, num_classes, s=30, m=0.35)`.
- **dataset.py** — `build_dataloaders_from_scope()` already works. No changes needed.
- **Training loop** — same optimizer, scheduler, early stopping, logging. Only the
  scope-routing CLI arg is new.
- **Checkpoint format** — same `{epoch, head_name, model_state_dict, val_loss, val_acc,
  num_classes, class_map}`. Scope name embedded in filename.

## 9. Success Criteria

Training is successful when:

1. All 10 scopes train without error
2. At least 7 of 10 pass the deployment gate (top-1 >= 70%, +5pp over KNN, ECE < 0.10)
3. The 4 largest dynasty scopes (Severan, Nerva-Antonine, Late Roman, 3rd Century Crisis)
   all pass — these cover the most coins and have the most headroom
4. Exported meta JSON is parseable by the pipeline's `register_head_models.py`
5. Per-scope eval JSON and calibration plots are saved for review

**Expected accuracy ranges** (based on KNN baselines and the fact that tight CosFace
heads typically add 10-20pp over KNN on authority discrimination):

| Scope | KNN | Expected Head | Headroom |
|-------|-----|--------------|----------|
| auth_ri_late_roman_8way | 45.0% | 60-70% | High |
| auth_ri_nerva_antonine_12way | 53.4% | 65-75% | High |
| auth_ri_3rd_century_crisis_8way | 54.5% | 65-75% | High |
| auth_ri_severan_13way | 59.3% | 70-80% | Good |
| auth_ri_cluster_91_2way | 59.5% | 70-80% | Good |
| auth_ri_cluster_66_8way | 68.7% | 75-85% | Moderate |
| auth_ri_julio_claudian_3way | 69.7% | 78-85% | Moderate |
| auth_ri_civil_war_68_3way | 70.2% | 78-85% | Moderate |
| auth_ri_illyrian_emperors_4way | 72.1% | 78-85% | Moderate |
| auth_byz_cluster_99_6way | 79.6% | 82-88% | Tight |
