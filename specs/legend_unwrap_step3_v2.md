# Step 3: CNN Model + Training — Updated Spec

**Location:** `trivalaya-vision/legend_cnn/`
**Framework:** PyTorch
**Goal:** Train a lightweight CNN on the promoted legend-ribbon dataset to test whether unwrapped legend texture can distinguish visually similar Roman authorities.

## Preconditions

Step 3 begins only after the following are complete:

* `confusion_subset.csv` has been created and validated
* the unwrap evaluation harness has been run
* a promoted unwrap config has been selected
* ribbons have been generated using the promoted config
* baseline review and promotion notes have been saved
* ribbon quality and geometry/process metadata are available in run artifacts

### Promoted unwrap config for Step 3

Use the promoted configuration as the canonical ribbon generator for training input:

* `inner_r_ratio = 0.68`
* `outer_r_ratio = 0.92`
* `center_method = moments`
* `max_center_offset = 200`

Training must use ribbons generated from this promoted configuration, not from earlier baseline or experimental configs.

---

## Step 3A: Training Dataset Freeze

**Files:**

* `legend_cnn/dataset.py`
* `legend_cnn/data/confusion_subset.csv`
* promoted run artifact:

  * `runs/<promoted_run>/ribbon_stats.csv`

### Objective

Construct a training-ready dataset from the promoted run outputs while filtering out failed unwraps and preserving useful metadata for later analysis.

### Source of truth

The training dataset must be built from the promoted run’s `ribbon_stats.csv`, joined back to the original subset metadata as needed.

### Inclusion rules

Include only rows where:

* `status = success`
* `ribbon_path` exists
* ribbon file loads correctly
* `authority_label_int` is present

### Exclusion rules

Exclude rows where:

* `status = failed`
* ribbon file is missing or unreadable
* label is missing
* authority class falls below a minimum retained sample threshold after filtering

### Quality handling

For v1 training, include all successful ribbons regardless of `quality_bucket`, but preserve:

* `quality_bucket`
* `quality_score`
* `used_fallback`
* `geometry_flags` if present
* `center_method`

These fields are not part of the model input, but must remain available for analysis.

### Optional filtered variant

Also support an optional stricter training view for experiments:

* exclude `bad_blank`
* optionally exclude `bad_low_signal`

This is not the default training set, but should be easy to enable for follow-up experiments.

### Output splits

Create a stratified split by authority:

* train: 80%
* validation: 10%
* test: 10%

Requirements:

* seeded split (`random_state = 42`)
* no overlap between splits
* class stratification required
* split manifest should be saved for reproducibility

### Split artifact

Save:

`legend_cnn/data/splits_v1.json`

Containing:

* split version
* promoted run id
* random seed
* list of coin_ids per split

---

## Step 3B: Dataset Class

**File:** `legend_cnn/dataset.py`

### Class

```python
class LegendRibbonDataset(torch.utils.data.Dataset):
    """
    Reads promoted .npy ribbons and returns:
      - tensor of shape [1, 64, 512]
      - authority label int
      - metadata dict for analysis/debug
    """
```

### Required behavior

For each item, return:

* `image_tensor`: `float32`, shape `[1, 64, 512]`
* `label`: integer authority class
* `meta`: dictionary containing at least:

  * `coin_id`
  * `authority`
  * `quality_bucket`
  * `quality_score`
  * `used_fallback`
  * `center_method`

### Input normalization

* ribbons are already stored as `[0,1]` float32
* no ImageNet normalization
* no per-image z-score normalization in v1
* use direct tensor loading unless a later experiment proves otherwise

### Training augmentations

Apply only on the training split:

#### Required

* **horizontal circular shift**

  * implemented with `torch.roll(..., dims=-1)`
  * simulates coin rotation
  * must wrap seamlessly

#### Allowed light augmentations

* brightness jitter: about ±10%
* contrast jitter: about ±10%
* light additive noise only if needed later

#### Forbidden in v1

* vertical flip
* arbitrary crop
* geometric warping
* heavy blur
* strong elastic transforms

### Validation/test behavior

* no augmentation
* deterministic loading only

---

## Step 3C: CNN Architecture

**File:** `legend_cnn/model.py`

### Objective

Use a lightweight grayscale CNN designed for horizontal texture and repeated local legend patterns, not a large generic backbone.

### Model

```python
class LegendCNN(nn.Module):
    """
    Input: [B, 1, 64, 512]
    Output: [B, num_classes]
    """
```

### Baseline architecture

```text
Block 1: Conv2d(1 -> 32, kernel=5x5, padding=2)
         BatchNorm2d
         ReLU
         MaxPool2d(2x2)
         Output: 32 x 32 x 256

Block 2: Conv2d(32 -> 64, kernel=3x3, padding=1)
         BatchNorm2d
         ReLU
         MaxPool2d(2x2)
         Output: 64 x 16 x 128

Block 3: Conv2d(64 -> 128, kernel=3x3, padding=1)
         BatchNorm2d
         ReLU
         MaxPool2d(2x2)
         Output: 128 x 8 x 64

Block 4: Conv2d(128 -> 256, kernel=3x3, padding=1)
         BatchNorm2d
         ReLU
         AdaptiveAvgPool2d(1x4)
         Output: 256 x 1 x 4

Flatten -> 1024
Linear(1024 -> 128)
ReLU
Dropout(0.5)
Linear(128 -> num_classes)
```

### Rationale

This architecture is retained for v1 because it is:

* small enough to train quickly
* appropriate for grayscale ribbon inputs
* structured to learn local texture and longer horizontal pattern structure
* simple enough that training behavior will be interpretable

### Optional future variants

Out of scope for initial Step 3, but allowed later:

* shallower CNN
* depthwise-separable CNN
* 1D-over-width hybrid model
* small ResNet-style model

---

## Step 3D: Training Loop

**File:** `legend_cnn/train.py`

### Objective

Train the CNN on promoted ribbons and produce a reproducible checkpoint plus training diagnostics.

### Training parameters

| Parameter          | Value                         |
| ------------------ | ----------------------------- |
| Optimizer          | AdamW                         |
| Learning rate      | 1e-3                          |
| Scheduler          | ReduceLROnPlateau             |
| Scheduler patience | 3                             |
| Scheduler factor   | 0.5                           |
| Monitor            | validation loss               |
| Batch size         | 256                           |
| Epochs             | 50 max                        |
| Early stopping     | patience 5 on validation loss |
| Loss               | CrossEntropyLoss              |

### Class imbalance

If class imbalance remains meaningful after filtering, support:

* either class weights in `CrossEntropyLoss`
* or balanced sampling

Default v1 behavior:

* start without weighting if class counts are reasonably close
* add weights only if class skew is clearly hurting minority recall

### Device behavior

* train on GPU when available
* log device info at startup
* allow automatic batch-size reduction if VRAM is insufficient

### Reproducibility

Set and log:

* Python random seed
* NumPy seed
* PyTorch seed

At minimum use:

* `seed = 42`

### Per-epoch logging

Log at least:

* epoch
* train loss
* validation loss
* validation accuracy
* learning rate
* epoch time

Save logs to:

`legend_cnn/outputs/train_log.csv`

### Checkpointing

Save:

* best model by validation loss
* optional latest checkpoint

Required path:

`legend_cnn/checkpoints/best.pt`

Checkpoint metadata must include:

* epoch
* model state dict
* optimizer state dict
* scheduler state dict
* validation loss
* validation accuracy
* class mapping
* promoted run id
* split version

### Training curves

Save:

`legend_cnn/outputs/training_curves.png`

Must include at least:

* train loss
* validation loss
* validation accuracy

---

## Step 3E: Evaluation

**File:** `legend_cnn/evaluate.py`

### Objective

Evaluate the trained model on the held-out test split and determine whether legend ribbons contain usable authority-discriminating signal.

### Primary metrics

Compute on the 10% held-out test set:

* overall accuracy
* confusion matrix
* per-class precision
* per-class recall
* per-class F1

### Success target

Initial PoC target:

* **accuracy > 85%**

This is a directional success target, not a hard go/no-go boundary. A lower result may still be informative if confusion structure is meaningful.

### Required outputs

Save:

* `legend_cnn/outputs/confusion_matrix.png`
* `legend_cnn/outputs/classification_report.json`
* `legend_cnn/outputs/test_predictions.csv`

### `test_predictions.csv` columns

Include at least:

* `coin_id`
* `true_label`
* `true_authority`
* `pred_label`
* `pred_authority`
* `correct`
* `quality_bucket`
* `quality_score`
* `used_fallback`
* `center_method`

This is important because it allows post-hoc analysis of whether errors correlate with ribbon quality or unwrap process flags.

### Error analysis requirements

Flag:

* any authority pair with >20% cross-confusion
* classes with weak recall
* whether misclassifications cluster in:

  * `bad_blank`
  * `bad_low_signal`
  * fallback-derived ribbons

### Secondary analyses

Required post-eval slices:

1. accuracy by `quality_bucket`
2. accuracy for `used_fallback = true` vs `false`
3. confusion concentrated in specific authority pairs
4. top confident wrong predictions

These analyses matter because the unwrap harness already showed that process reliability and visual quality are distinct axes.

### Inference timing

Measure approximate single-sample inference latency on a `64x512` ribbon.

Target:

* under 10 ms on GPU is nice
* not mission-critical for PoC

---

## Step 3F: Step 3 Decision Criteria

### Positive outcome

Step 3 is considered successful if most of the following are true:

* test accuracy is strong enough to show real discriminative signal
* confusion matrix is structured rather than chaotic
* some authority pairs are clearly separable from legend texture alone
* poor ribbons underperform good ribbons in a way that matches expectations
* fallback-derived ribbons are not universally useless

### Weak but still useful outcome

Step 3 is still informative if:

* overall accuracy is moderate rather than excellent
* confusion concentrates in a few historically/visually close authority pairs
* model performance improves materially when excluding worst-quality ribbons

That would suggest the unwrap pipeline is partly working but label signal or ribbon quality is still limiting.

### Negative outcome

Step 3 is likely a failure signal if:

* test accuracy is near chance
* confusion is broadly uniform
* quality slices show no difference between good and bad ribbons
* predictions appear insensitive to ribbon structure

That would suggest the legend-ribbon representation is not capturing enough authority-specific signal.

---

## Step 3G: File Structure Additions

```text
trivalaya-vision/
└── legend_cnn/
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── data/
    │   ├── confusion_subset.csv
    │   ├── splits_v1.json
    │   └── ribbons/
    ├── checkpoints/
    │   └── best.pt
    └── outputs/
        ├── train_log.csv
        ├── training_curves.png
        ├── confusion_matrix.png
        ├── classification_report.json
        └── test_predictions.csv
```

---

## Step 3H: Execution Order

1. Generate final ribbons using the promoted unwrap config
2. Freeze train/val/test splits
3. Implement and verify `LegendRibbonDataset`
4. Implement `LegendCNN`
5. Run training
6. Save best checkpoint
7. Run held-out evaluation
8. Review confusion matrix and per-slice error analysis
9. Decide whether to:

   * proceed to refinement,
   * test filtered training subsets,
   * or conclude the PoC signal is insufficient

---

## Step 3I: Immediate Implementation Notes

### Recommended first training run

Use the full successful promoted-ribbon set with no extra filtering except `status = success`.

Why:

* establishes the real baseline
* preserves the harness philosophy of measuring rather than prematurely cleaning
* allows later comparison against filtered variants

### Recommended first follow-up experiment

If results are only middling, run:

* same architecture
* same split
* exclude `bad_blank`

This is the cleanest ablation to test whether poor unwrap quality is still suppressing model performance.

### Important rule

Do not change both:

* ribbon filtering policy
* and model architecture

at the same time in the first follow-up round.

Keep one axis stable so results remain interpretable.

---