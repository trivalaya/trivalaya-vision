# Cluster-Driven Head Training Results

**Date**: 2026-04-11
**Spec**: `specs/cluster_head_training.md`
**Hyperparams**: defaults (lr=1e-3, s=30, m=0.35, dim=128, CosFace)

---

## Gate Results

Gates: top-1 >= 70%, +5pp over KNN baseline, ECE < 0.10

| Scope | N-way | Top-1 | Top-2 | Top-5 | KNN BL | Margin | Status |
|-------|-------|-------|-------|-------|--------|--------|--------|
| auth_ri_cluster_66_8way | 8 | 81.7% | 90.5% | 98.2% | 68.7% | +13.0pp | **PASS** |
| auth_ri_julio_claudian_3way | 3 | 78.6% | 89.9% | 100% | 69.7% | +8.9pp | **PASS** |
| auth_ri_severan_13way | 13 | 75.8% | 88.0% | 95.8% | 59.3% | +16.5pp | **PASS** |
| auth_ri_nerva_antonine_12way | 12 | 75.8% | 87.5% | 94.1% | 53.4% | +22.4pp | **PASS** |
| auth_ri_3rd_century_crisis_8way | 8 | 75.0% | 84.3% | 96.5% | 54.5% | +20.5pp | **PASS** |
| auth_ri_illyrian_emperors_4way | 4 | 74.1% | 79.0% | 100% | 72.1% | +2.0pp | knn_sufficient |
| auth_byz_cluster_99_6way | 6 | 72.3% | 93.6% | 100% | 79.6% | -7.3pp | knn_sufficient |
| auth_ri_late_roman_8way | 8 | 57.6% | 74.5% | 94.7% | 45.0% | +12.6pp | FAIL (top-1) |
| auth_ri_cluster_91_2way | 2 | 53.1% | 100% | 100% | 59.5% | -6.4pp | candidate |
| auth_ri_civil_war_68_3way | 3 | 40.0% | 70.0% | 100% | 70.2% | -30.2pp | knn_sufficient |

**5/10 pass** (spec target: 7/10). 3 of 4 P0 dynasty scopes pass (Late Roman misses).

Calibration verified on Severan export: ECE = 0.020, temperature = 3.70.

## Per-Scope Notes

### Passing scopes

- **cluster_66** (81.7%): Best performer. Hardest pair: Caracalla -> Septimius Severus (9).
- **julio_claudian** (78.6%): Augustus 88.4%, Nero 82.0%, Tiberius 57.5%. Tiberius confused with both.
- **severan** (75.8%): Hardest pair: Caracalla -> Elagabalus (14). Julia Soaemias 0% recall (3 test samples).
- **nerva_antonine** (75.8%): Hardest pair: Antoninus Pius -> Hadrian (13). Faustina Senior 0% (7 test).
- **3rd_century_crisis** (75.0%): Hardest pair: Valerian I -> Gallienus (12). Father/son confusion.

### Failing scopes

- **illyrian_emperors** (74.1%): Model collapsed to all-Probus predictions. Probus = 74% of test set. Macro F1 = 0.21. Needs class-balanced sampling or sweep.
- **byz_cluster_99** (72.3%): KNN baseline already 79.6% — well-separated cluster, head adds nothing.
- **civil_war_68** (40.0%): 489 training samples, 3 near-identical emperors (Galba/Otho/Vitellius). Far below KNN baseline. Use KNN fallback.
- **cluster_91** (53.1%): Diocletian/Maximian 2-way. Unseparable at the embedding level.

## Late Roman / Tetrarchy Analysis

### Per-class recall (default params)

| Class | Test N | Recall | Primary confusion |
|-------|--------|--------|-------------------|
| Constantine I | 73 | 90.4% | Clean |
| Licinius I | 32 | 65.6% | -> Maximian |
| Maximian | 60 | 66.7% | -> Diocletian |
| Constantius I | 26 | 50.0% | -> Maximian, Diocletian |
| Diocletian | 59 | 37.3% | -> Maximian (22 misses) |
| Galerius | 24 | 29.2% | -> Maximian (9 misses) |
| Maximinus II | 20 | 25.0% | -> Constantine I |
| Severus II | 8 | 0.0% | All misclassified (tiny class) |

**Constantine I separates cleanly** (90.4% recall) because his iconography shifted
post-Tetrarchy. This is the actionable signal in this scope.

The 4 Tetrarchs (Diocletian, Maximian, Galerius, Constantius I) are
representation-limited: they deliberately used near-identical portrait types to project
collective rule ideology. DINOv2 encodes what's visually there, and what's visually
there is the same face repeated four times.

### Sweep test (s=64, m=0.50, dim=256, lr=5e-3)

Top-1: 55.6% (vs 57.6% baseline). Dio->Max reduced 22->18, Diocletian recall improved
37.3%->49.2%, but Galerius dropped 29.2%->16.7%. Higher margin helps the hardest pair
at the expense of other classes. Not a hyperparameter problem.

### Options for Late Roman

1. **Split scope**: 2-tier head — first Constantine+Licinius vs Tetrarchs, then finer
   Tetrarch separation where the head still adds +12.6pp over KNN.
2. **Reverse type features**: Tetrarch reverse types vary more than portraits. obv+rev
   1536-d concatenation (spec §6c) could provide the missing signal.
3. **Accept candidate status**: 57.6% + top-2=74.5% still narrows candidates for the
   pipeline even without clearing the 70% gate.

## File Locations

| Asset | Path |
|-------|------|
| Scope data | `embedding_heads/data/{scope}_subset.csv`, `_splits.json`, `_class_map.json` |
| Checkpoints | `embedding_heads/checkpoints/{scope}_cosface_best.pt` |
| Eval summary | `embedding_heads/outputs/eval_all_scopes.json` |
| Per-scope eval | `embedding_heads/outputs/eval_{scope}.json` |
| Train logs | `embedding_heads/outputs/train_log_{scope}.csv` |
| Confusion plots | `embedding_heads/outputs/confusion_{scope}.png` |
| Export (via CLI) | `{scope}_head.pt`, `{scope}_centroids.npy`, `{scope}_meta.json` |
