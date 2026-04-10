# Legend Stylometry PoC — Implementation Spec

**Location:** `trivalaya-vision/legend_cnn/`
**Framework:** PyTorch
**Goal:** Validate that a lightweight CNN can distinguish visually similar Roman authorities by classifying the visual texture of their unwrapped coin legends.

---

## Step 1: Confusion Subset Selection

**File:** `legend_cnn/select_subset.py`

### Data Model

```
ml_coin_dataset.coin_id  →  coin_cluster_xref.coin_id  (filter by run_id)
ml_coin_dataset.coin_id  →  coins.id  →  coin_detections.coin_id  →  highres_path (512x512)
```

- `authority` lives on `ml_coin_dataset`
- `coin_cluster_xref` has `run_id` — use the latest production run
- 512x512 images are on DO Spaces at `coin_detections.highres_path`
- Public URL prefix: `https://trivalaya-data.sfo3.digitaloceanspaces.com/`
- Transparent masks at `coin_detections.transparent_path` (RGBA PNGs, hard alpha)

### Logic

1. Query deepest sub-split level from `coin_cluster_xref` for the latest `run_id`
2. For each leaf cluster, compute authority distribution
3. Rank clusters by **authority entropy** (Shannon) — high entropy = still mixed
4. From worst clusters, extract top co-occurring authority pairs
5. Select 5–10 authorities with ~1,000+ images each, targeting ~10,000 total
6. Filter to rows that have a valid `highres_path` in `coin_detections`
7. Output: `legend_cnn/data/confusion_subset.csv` with columns:
   - `coin_id, authority, authority_label_int, highres_url, transparent_url, cluster_id`

### Notes

- Period filter: Roman Imperial only (these are the 3rd-century confusion cases)
- If an authority has fewer than 200 images, skip it — not enough for training
- Print summary: authority counts, total images, cluster overlap stats

---

## Step 2: Polar Unwrap Pipeline

**File:** `legend_cnn/unwrap.py`

### Inputs

- 512×512 coin images from DO Spaces (via `highres_url`)
- Transparent masks from DO Spaces (via `transparent_url`) for center/radius detection

### Pipeline Per Image

```python
# 1. Download 512x512 image + transparent mask
# 2. Find center and radius from the transparent mask:
#    - Convert alpha channel to binary mask
#    - Find largest contour
#    - cv2.minEnclosingCircle → (cx, cy), r
# 3. Define legend annulus:
#    - inner_r = 0.70 * r
#    - outer_r = 0.98 * r
# 4. Polar unwrap:
#    - raw = cv2.warpPolar(gray, dsize=(512, 64), center, outer_r, WARP_POLAR_LINEAR)
#    - CRITICAL: warpPolar maps angle→Y, radius→X by default.
#      Transpose immediately: ribbon = raw.T  (now angle runs along width, radius along height)
#    - After transpose, shape is (64, 512) — radius is rows, angle is columns
#    - Crop to the radial band: rows corresponding to inner_r..outer_r
#      (i.e., row_start = int(64 * inner_r / outer_r), keep row_start:64)
# 5. Resize result to exactly 64 × 512 (H × W)
# 6. Normalize to [0, 1] float32
# 7. Save as .npy to legend_cnn/data/ribbons/{coin_id}.npy
```

### Implementation Details

- Use `boto3` or `requests` to download from Spaces (public URL, no auth needed if public)
- Batch with multiprocessing or `concurrent.futures` (I/O bound on downloads)
- Skip and log failures (missing images, zero-radius detections, etc.)
- Update CSV with `ribbon_path` column

### Validation Checkpoint

Before proceeding to Step 3, generate a visual gallery:
- `legend_cnn/validate_ribbons.py`
- Randomly sample 20 ribbons per authority
- Output a grid image (`legend_cnn/data/ribbon_gallery.png`) showing ribbons grouped by authority
- Visual check: legends should appear as horizontal texture bands, consistent positioning, no major artifacts
- Log statistics: mean radius, center offset from image center, failure rate

**STOP HERE and show the gallery to Jorg before proceeding to Step 3.**

---

## Step 3: CNN Model + Training

### Dataset — `legend_cnn/dataset.py`

```python
class LegendRibbonDataset(torch.utils.data.Dataset):
    """
    Reads pre-computed .npy ribbons.
    Returns (tensor[1, 64, 512], authority_label_int)
    """
```

- Load CSV from Step 1 (filtered to rows with valid `ribbon_path`)
- Stratified 80/10/10 split, seeded (`random_state=42`)
- Training augmentations:
  - **Horizontal circular shift** via `torch.roll(tensor, shifts=random_int, dims=-1)` — simulates rotation on coin, seamlessly wraps from right edge back to left edge
  - **Brightness/contrast jitter** (small: ±10%)
  - NO vertical flip (radial direction matters)
  - NO heavy augmentation
- Validation/test: no augmentation, just normalize

### Architecture — `legend_cnn/model.py`

```python
class LegendCNN(nn.Module):
    """
    Input: (batch, 1, 64, 512) grayscale ribbons
    Output: (batch, num_classes) logits
    """
```

```
Block 1 (Texture):  Conv2d(1→32,  5×5, pad=2) → BN → ReLU → MaxPool(2×2)   →  32 × 32 × 256
Block 2 (Pattern):  Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)   →  64 × 16 × 128
Block 3 (Pattern):  Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 128 ×  8 ×  64
Block 4 (Features): Conv2d(128→256, 3×3, pad=1) → BN → ReLU → AdaptiveAvgPool(1×4) → 256 × 1 × 4

Flatten → 1024
Linear(1024 → 128) → ReLU → Dropout(0.5)
Linear(128 → num_classes)
```

### Training — `legend_cnn/train.py`

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| LR scheduler | ReduceLROnPlateau(patience=3, factor=0.5, monitor=val_loss) |
| Batch size | 256 (adjust up on RTX 5090) |
| Loss | CrossEntropyLoss |
| Epochs | 50 max |
| Early stopping | patience=5 on val_loss |

- Log per epoch: train_loss, val_loss, val_accuracy, learning_rate
- Save best checkpoint by val_loss to `legend_cnn/checkpoints/best.pt`
- Save training curves to `legend_cnn/outputs/training_curves.png`

### Evaluation — `legend_cnn/evaluate.py`

Run on the 10% held-out test set:
1. **Accuracy** — target >85%
2. **Confusion matrix** — saved as image, flag any pair with >20% cross-confusion
3. **Per-class precision/recall/F1**
4. **Inference timing** — measure single forward pass on 64×512 input, target <10ms

---

## File Structure

```
trivalaya-vision/
└── legend_cnn/
    ├── select_subset.py      # Step 1: mine confusion authorities from DB
    ├── unwrap.py              # Step 2: polar unwrap pipeline
    ├── validate_ribbons.py    # Step 2: visual validation gallery
    ├── dataset.py             # Step 3: PyTorch dataset + splits
    ├── model.py               # Step 3: CNN architecture
    ├── train.py               # Step 3: training loop
    ├── evaluate.py            # Step 3: test evaluation + confusion matrix
    ├── config.py              # shared constants (paths, Spaces URL, etc.)
    ├── data/
    │   ├── confusion_subset.csv
    │   ├── ribbons/           # .npy files per coin
    │   └── ribbon_gallery.png
    ├── checkpoints/
    │   └── best.pt
    └── outputs/
        ├── training_curves.png
        └── confusion_matrix.png
```

---

## Execution Order

1. Run `select_subset.py` on the droplet (needs DB access)
2. Run `unwrap.py` on the droplet (downloads from Spaces, writes ribbons locally)
3. Run `validate_ribbons.py` — **show gallery to Jorg, get approval**
4. Copy `data/` directory to RunPod burst instance
5. Run `train.py` on RunPod (RTX 5090)
6. Run `evaluate.py` on RunPod
7. Review results

## DB Connection

Use existing Trivalaya DB connection pattern (MySQL on the droplet). `select_subset.py` should use the same connection config as the rest of the pipeline.

## Dependencies

Add to requirements: `torch`, `torchvision`, `scikit-learn` (for stratified split + metrics), `matplotlib` (for plots), `boto3` or `requests` (for Spaces downloads). OpenCV is already present.
