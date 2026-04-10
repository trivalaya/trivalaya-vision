#!/usr/bin/env python3
"""Deep evaluation: per-authority KNN recall, hard-pair analysis, calibration.

    python -m embedding_heads.deep_eval
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from embedding_heads.config import (
    CHECKPOINT_DIR, CONFUSION_SUBSET_PATH, OUTPUT_DIR, SPLITS_PATH,
)
from embedding_heads.dataset import extract_subset_embeddings
from embedding_heads.heads import build_head


def load_cosface():
    ck = torch.load(CHECKPOINT_DIR / "cosface_best.pt", map_location="cpu", weights_only=False)
    model = build_head("cosface", ck["num_classes"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    class_map = {int(k): v for k, v in ck["class_map"].items()}
    return model, class_map


def collect_splits(model):
    """Collect projected embeddings and labels for train/test splits."""
    embed_dict, _ = extract_subset_embeddings()

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    subset = pd.read_csv(CONFUSION_SUBSET_PATH)
    coin_to_label = dict(zip(subset["coin_id"], subset["authority_label_int"]))
    coin_to_auth = dict(zip(subset["coin_id"], subset["authority"]))

    def collect(ids):
        embs, proj, labels, auths, cids = [], [], [], [], []
        with torch.no_grad():
            for cid in ids:
                if cid in embed_dict and cid in coin_to_label:
                    raw = torch.from_numpy(embed_dict[cid]).float().unsqueeze(0)
                    projected = model.embed(raw).squeeze(0).numpy()
                    embs.append(embed_dict[cid])
                    proj.append(projected)
                    labels.append(coin_to_label[cid])
                    auths.append(coin_to_auth[cid])
                    cids.append(cid)
        return (np.array(embs), np.array(proj), np.array(labels),
                auths, cids)

    train_data = collect(splits["train"])
    test_data = collect(splits["test"])
    return train_data, test_data


# ── 1. Per-Authority KNN Recall ─────────────────────────────────────


def per_authority_knn_recall(train_data, test_data, class_map):
    """KNN-1 and KNN-10 recall per authority for CosFace projected embeddings."""
    _, train_proj, train_labels, _, _ = train_data
    _, test_proj, test_labels, test_auths, _ = test_data

    print("=" * 70)
    print("1. PER-AUTHORITY COSINE KNN RECALL (CosFace 128-d)")
    print("=" * 70)

    header = f"{'Authority':<22} {'N':>4} {'KNN-1':>7} {'KNN-10':>7} {'Raw-1':>7} {'Raw-10':>7}"
    print(header)
    print("-" * len(header))

    train_raw, _, _, _, _ = train_data
    test_raw, _, _, _, _ = test_data

    for k in [1, 10]:
        knn_proj = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_proj.fit(train_proj, train_labels)

        knn_raw = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_raw.fit(train_raw, train_labels)

        if k == 1:
            pred_proj_1 = knn_proj.predict(test_proj)
            pred_raw_1 = knn_raw.predict(test_raw)
        else:
            pred_proj_10 = knn_proj.predict(test_proj)
            pred_raw_10 = knn_raw.predict(test_raw)

    results = []
    for label_int in sorted(class_map.keys()):
        auth = class_map[label_int]
        mask = test_labels == label_int
        n = mask.sum()

        knn1_acc = (pred_proj_1[mask] == test_labels[mask]).mean()
        knn10_acc = (pred_proj_10[mask] == test_labels[mask]).mean()
        raw1_acc = (pred_raw_1[mask] == test_labels[mask]).mean()
        raw10_acc = (pred_raw_10[mask] == test_labels[mask]).mean()

        results.append({
            "authority": auth, "n": int(n),
            "knn1": knn1_acc, "knn10": knn10_acc,
            "raw1": raw1_acc, "raw10": raw10_acc,
            "lift1": knn1_acc - raw1_acc,
            "lift10": knn10_acc - raw10_acc,
        })

    results.sort(key=lambda r: r["knn10"])

    for r in results:
        print(f"{r['authority']:<22} {r['n']:>4} {r['knn1']:>7.1%} {r['knn10']:>7.1%} "
              f"{r['raw1']:>7.1%} {r['raw10']:>7.1%}")

    print(f"\n{'TOTAL':<22} {len(test_labels):>4} "
          f"{(pred_proj_1 == test_labels).mean():>7.1%} "
          f"{(pred_proj_10 == test_labels).mean():>7.1%} "
          f"{(pred_raw_1 == test_labels).mean():>7.1%} "
          f"{(pred_raw_10 == test_labels).mean():>7.1%}")

    print("\nBiggest CosFace KNN-10 lifts over raw:")
    for r in sorted(results, key=lambda x: -x["lift10"])[:5]:
        print(f"  {r['authority']}: +{r['lift10']:.1%} ({r['raw10']:.1%} -> {r['knn10']:.1%})")

    return results


# ── 2. Hard-Pair Analysis ───────────────────────────────────────────


def hard_pair_analysis(train_data, test_data, class_map):
    """Detailed analysis of the Diocletian/Maximian confusion and other hard pairs."""
    _, train_proj, train_labels, _, _ = train_data
    _, test_proj, test_labels, test_auths, test_cids = test_data

    print("\n" + "=" * 70)
    print("2. HARD-PAIR ANALYSIS")
    print("=" * 70)

    # KNN-10 confusion matrix on projected embeddings
    knn = KNeighborsClassifier(n_neighbors=10, metric="cosine")
    knn.fit(train_proj, train_labels)
    preds = knn.predict(test_proj)

    class_names = [class_map[i] for i in sorted(class_map.keys())]
    cm = confusion_matrix(test_labels, preds)

    # Find all off-diagonal errors
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], cm[i, j]))
    pairs.sort(key=lambda x: -x[2])

    print("\nKNN-10 confusion pairs (top 10):")
    for true_cls, pred_cls, count in pairs[:10]:
        print(f"  {true_cls:>22} -> {pred_cls:<22} {count:>3}")

    # Deep dive: Diocletian vs Maximian
    print("\n--- Diocletian <-> Maximian Deep Dive ---")

    dio_idx = [i for i, a in enumerate(class_names) if a == "Diocletian"][0]
    max_idx = [i for i, a in enumerate(class_names) if a == "Maximian"][0]

    # Compute pairwise cosine similarities in projected space
    dio_mask_train = train_labels == dio_idx
    max_mask_train = train_labels == max_idx
    dio_mask_test = test_labels == dio_idx
    max_mask_test = test_labels == max_idx

    dio_train = train_proj[dio_mask_train]
    max_train = train_proj[max_mask_train]
    dio_test = test_proj[dio_mask_test]
    max_test = test_proj[max_mask_test]

    # Intra-class similarity
    def mean_cosine_sim(a, b):
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        sims = a_norm @ b_norm.T
        # Exclude self-similarities on diagonal if same set
        if a is b:
            np.fill_diagonal(sims, 0)
            return sims.sum() / (sims.size - len(sims))
        return sims.mean()

    dio_intra = mean_cosine_sim(dio_train, dio_train)
    max_intra = mean_cosine_sim(max_train, max_train)
    cross = mean_cosine_sim(dio_train, max_train)

    print(f"  Diocletian intra-class cosine sim: {dio_intra:.4f}")
    print(f"  Maximian intra-class cosine sim:   {max_intra:.4f}")
    print(f"  Cross-class cosine sim:            {cross:.4f}")
    print(f"  Separation (intra_avg - cross):    {((dio_intra + max_intra)/2 - cross):.4f}")

    # Compare with a well-separated pair
    # Find the least-confused pair
    had_idx = [i for i, a in enumerate(class_names) if a == "Hadrian"][0]
    gor_idx = [i for i, a in enumerate(class_names) if a == "Gordian III"][0]

    had_train = train_proj[train_labels == had_idx]
    gor_train = train_proj[train_labels == gor_idx]

    had_intra = mean_cosine_sim(had_train, had_train)
    gor_intra = mean_cosine_sim(gor_train, gor_train)
    had_gor_cross = mean_cosine_sim(had_train, gor_train)

    print(f"\n  Comparison: Hadrian vs Gordian III (well-separated)")
    print(f"  Hadrian intra-class cosine sim:    {had_intra:.4f}")
    print(f"  Gordian III intra-class cosine sim: {gor_intra:.4f}")
    print(f"  Cross-class cosine sim:            {had_gor_cross:.4f}")
    print(f"  Separation (intra_avg - cross):    {((had_intra + gor_intra)/2 - had_gor_cross):.4f}")

    # Also compare raw 768-d space
    _, train_raw, _, _, _ = train_data
    dio_raw = train_raw[dio_mask_train]
    max_raw = train_raw[max_mask_train]
    raw_cross = mean_cosine_sim(dio_raw, max_raw)
    raw_dio_intra = mean_cosine_sim(dio_raw, dio_raw)
    raw_max_intra = mean_cosine_sim(max_raw, max_raw)

    print(f"\n  Raw 768-d space (Diocletian vs Maximian):")
    print(f"  Diocletian intra: {raw_dio_intra:.4f}  Maximian intra: {raw_max_intra:.4f}")
    print(f"  Cross: {raw_cross:.4f}  Separation: {((raw_dio_intra + raw_max_intra)/2 - raw_cross):.4f}")

    # Misclassified Diocletian coins — what do they look like?
    print(f"\n  Misclassified Diocletian test coins (predicted as Maximian):")
    for i, (cid, true, pred) in enumerate(zip(test_cids, test_labels, preds)):
        if true == dio_idx and pred == max_idx:
            # Find distance to nearest Diocletian vs nearest Maximian train point
            test_emb = test_proj[test_cids.index(cid) if isinstance(test_cids, list) else np.where(np.array(test_cids) == cid)[0][0]]
            dio_dists = 1 - (dio_train @ test_emb / (np.linalg.norm(dio_train, axis=1) * np.linalg.norm(test_emb)))
            max_dists = 1 - (max_train @ test_emb / (np.linalg.norm(max_train, axis=1) * np.linalg.norm(test_emb)))
            print(f"    coin {cid}: nearest Dio dist={dio_dists.min():.4f}  nearest Max dist={max_dists.min():.4f}")

    return cm


# ── 3. Top-K Calibration ────────────────────────────────────────────


def topk_calibration(model, test_data, class_map):
    """Calibration analysis: are the model's confidence scores reliable?"""
    _, test_proj, test_labels, _, test_cids = test_data
    test_raw, _, _, _, _ = test_data

    print("\n" + "=" * 70)
    print("3. TOP-K CONFIDENCE CALIBRATION (CosFace)")
    print("=" * 70)

    # Get classification logits (without margin) from the head
    with torch.no_grad():
        raw_tensors = torch.from_numpy(test_raw).float()
        logits = model(raw_tensors)
        probs = F.softmax(logits, dim=1).numpy()

    # Also compute cosine similarity scores to class centroids in projected space
    _, train_proj, train_labels, _, _ = collect_splits.__wrapped_train if hasattr(collect_splits, '__wrapped_train') else (None, None, None, None, None)

    # Calibration bins: group predictions by confidence, check actual accuracy
    top1_confs = probs.max(axis=1)
    top1_preds = probs.argmax(axis=1)
    top1_correct = (top1_preds == test_labels)

    bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    print("\nClassification head calibration (softmax confidence):")
    print(f"{'Conf Bin':<12} {'Count':>6} {'Accuracy':>9} {'Avg Conf':>9} {'Gap':>7}")
    print("-" * 47)

    for i in range(len(bins) - 1):
        mask = (top1_confs >= bins[i]) & (top1_confs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = top1_correct[mask].mean()
        avg_conf = top1_confs[mask].mean()
        count = mask.sum()
        gap = avg_conf - acc

        bin_centers.append(avg_conf)
        bin_accs.append(acc)
        bin_counts.append(count)

        print(f"[{bins[i]:.1f}-{bins[i+1]:.1f})  {count:>6}  {acc:>9.1%}  {avg_conf:>9.3f}  {gap:>+7.1%}")

    # Expected Calibration Error
    total = len(test_labels)
    ece = sum(c * abs(conf - acc) for c, conf, acc in zip(bin_counts, bin_centers, bin_accs)) / total
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

    # Top-K analysis
    print("\nTop-K candidate analysis:")
    print(f"{'K':>3} {'Accuracy':>9} {'Mean Score of Correct':>22} {'Mean Score of Top-1':>20}")
    print("-" * 57)

    for k in [1, 2, 3, 5, 10]:
        topk_idx = np.argsort(-probs, axis=1)[:, :k]
        topk_hit = np.array([test_labels[i] in topk_idx[i] for i in range(len(test_labels))])
        topk_acc = topk_hit.mean()

        # Mean softmax score of the correct class
        correct_scores = probs[np.arange(len(test_labels)), test_labels]
        # Mean softmax score of the top-1 prediction
        top1_scores = probs.max(axis=1)

        print(f"{k:>3} {topk_acc:>9.1%} {correct_scores.mean():>22.4f} {top1_scores.mean():>20.4f}")

    # Score distribution: correct vs wrong
    correct_conf = top1_confs[top1_correct]
    wrong_conf = top1_confs[~top1_correct]

    print(f"\nConfidence distribution:")
    print(f"  Correct predictions:  mean={correct_conf.mean():.3f}  "
          f"median={np.median(correct_conf):.3f}  p25={np.percentile(correct_conf, 25):.3f}")
    print(f"  Wrong predictions:    mean={wrong_conf.mean():.3f}  "
          f"median={np.median(wrong_conf):.3f}  p25={np.percentile(wrong_conf, 25):.3f}")
    print(f"  Separation:           {correct_conf.mean() - wrong_conf.mean():.3f}")

    # Reliability at confidence thresholds
    print(f"\nReliability at confidence thresholds:")
    print(f"{'Threshold':>10} {'N above':>8} {'Accuracy':>9} {'Coverage':>9}")
    print("-" * 38)
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = top1_confs >= thresh
        if mask.sum() == 0:
            continue
        acc = top1_correct[mask].mean()
        coverage = mask.mean()
        print(f"{thresh:>10.1f} {mask.sum():>8} {acc:>9.1%} {coverage:>9.1%}")

    # Plot calibration curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, bin_accs, width=0.08, alpha=0.6, label="Observed")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction Correct")
    ax1.set_title(f"Calibration (ECE={ece:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(correct_conf, bins=30, alpha=0.6, label=f"Correct (n={len(correct_conf)})", density=True)
    ax2.hist(wrong_conf, bins=30, alpha=0.6, label=f"Wrong (n={len(wrong_conf)})", density=True)
    ax2.set_xlabel("Top-1 Confidence")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_cosface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCalibration plot: {OUTPUT_DIR / 'calibration_cosface.png'}")


def main():
    model, class_map = load_cosface()
    train_data, test_data = collect_splits(model)

    per_authority_knn_recall(train_data, test_data, class_map)
    hard_pair_analysis(train_data, test_data, class_map)
    topk_calibration(model, test_data, class_map)


if __name__ == "__main__":
    main()
