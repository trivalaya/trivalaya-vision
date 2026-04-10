#!/usr/bin/env python3
"""Evaluate a trained embedding head on the test set.

    python -m embedding_heads.evaluate --head linear
    python -m embedding_heads.evaluate --head arcface
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

from embedding_heads.config import CHECKPOINT_DIR, OUTPUT_DIR, SPLITS_PATH, CONFUSION_SUBSET_PATH
from embedding_heads.dataset import build_dataloaders, extract_subset_embeddings
from embedding_heads.heads import build_head


def load_model(head_name):
    ck = torch.load(CHECKPOINT_DIR / f"{head_name}_best.pt", map_location="cpu", weights_only=False)
    model = build_head(head_name, ck["num_classes"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def get_predictions(model, loader, head_name):
    """Get all predictions, probabilities, and labels from a loader."""
    all_probs = []
    all_labels = []

    for embeddings, labels in loader:
        logits = model(embeddings)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
        all_labels.append(labels)

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds = probs.argmax(dim=1)

    return preds.numpy(), probs.numpy(), labels.numpy()


def topk_accuracy(probs, labels, k):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def plot_confusion(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel="True",
           xlabel="Predicted")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def hardest_pairs(cm, class_names, n=5):
    """Find the top-N off-diagonal confusion pairs."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], cm[i, j]))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:n]


@torch.no_grad()
def arcface_embedding_eval(model, head_name):
    """Evaluate cosine KNN accuracy on ArcFace/CosFace 128-d embeddings."""
    embed_dict, class_map = extract_subset_embeddings()

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    import pandas as pd
    subset = pd.read_csv(CONFUSION_SUBSET_PATH)
    coin_to_label = dict(zip(subset["coin_id"], subset["authority_label_int"]))

    def collect(ids):
        embs, labels = [], []
        for cid in ids:
            if cid in embed_dict and cid in coin_to_label:
                raw = torch.from_numpy(embed_dict[cid]).float().unsqueeze(0)
                projected = model.embed(raw).squeeze(0).numpy()
                embs.append(projected)
                labels.append(coin_to_label[cid])
        return np.array(embs), np.array(labels)

    train_embs, train_labels = collect(splits["train"])
    test_embs, test_labels = collect(splits["test"])

    # Also get raw 768-d for baseline comparison
    def collect_raw(ids):
        embs, labels = [], []
        for cid in ids:
            if cid in embed_dict and cid in coin_to_label:
                embs.append(embed_dict[cid])
                labels.append(coin_to_label[cid])
        return np.array(embs), np.array(labels)

    train_raw, _ = collect_raw(splits["train"])
    test_raw, _ = collect_raw(splits["test"])

    results = {}
    for k in [1, 10]:
        # Projected 128-d
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_embs, train_labels)
        proj_acc = float(knn.score(test_embs, test_labels))
        results[f"projected_knn{k}"] = proj_acc

        # Raw 768-d baseline
        knn_raw = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_raw.fit(train_raw, train_labels)
        raw_acc = float(knn_raw.score(test_raw, test_labels))
        results[f"raw_knn{k}"] = raw_acc

    return results


def evaluate(head_name):
    print(f"\n=== Evaluating: {head_name} ===\n")

    model, ck = load_model(head_name)
    class_map = ck["class_map"]
    class_names = [class_map[i] for i in sorted(int(k) for k in class_map.keys())]

    _, _, test_loader, num_classes, _ = build_dataloaders()
    preds, probs, labels = get_predictions(model, test_loader, head_name)

    # Accuracy
    top1 = topk_accuracy(probs, labels, 1)
    top3 = topk_accuracy(probs, labels, 3)
    top5 = topk_accuracy(probs, labels, 5)

    print(f"Top-1: {top1:.4f}  Top-3: {top3:.4f}  Top-5: {top5:.4f}")

    # Per-class report
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, class_names, OUTPUT_DIR / f"confusion_{head_name}.png",
                   f"Confusion Matrix — {head_name} (top-1={top1:.1%})")

    # Hardest pairs
    pairs = hardest_pairs(cm, class_names)
    print("Hardest pairs:")
    for true_cls, pred_cls, count in pairs:
        print(f"  {true_cls} -> {pred_cls}: {count}")

    results = {
        "head": head_name,
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "top5": round(top5, 4),
        "worst_pair": f"{pairs[0][0]} -> {pairs[0][1]}" if pairs else "",
        "worst_pair_count": int(pairs[0][2]) if pairs else 0,
    }

    # ArcFace/CosFace embedding eval
    if head_name in ("arcface", "cosface"):
        print("\nEmbedding space evaluation (cosine KNN):")
        emb_results = arcface_embedding_eval(model, head_name)
        for key, val in emb_results.items():
            print(f"  {key}: {val:.4f}")
            results[key] = round(val, 4)

    # Save results
    with open(OUTPUT_DIR / f"eval_{head_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding head")
    parser.add_argument("--head", required=True, choices=["linear", "mlp", "arcface", "cosface"])
    args = parser.parse_args()
    evaluate(args.head)


if __name__ == "__main__":
    main()
