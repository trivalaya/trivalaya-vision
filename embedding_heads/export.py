#!/usr/bin/env python3
"""Export CosFace checkpoint to production assets for visual_search.

Supports two modes:
  Legacy (v2): exports cosface_head.pt, cosface_authority_centroids.npy, etc.
  Scoped: exports {scope}_head.pt, {scope}_centroids.npy, {scope}_meta.json

Usage:
    # Legacy v2 export
    python -m embedding_heads.export [--checkpoint cosface_v2_best.pt]

    # Scoped export (spec-compliant naming)
    python -m embedding_heads.export --scope ri_authority
    python -m embedding_heads.export --scope ri_denomination
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from embedding_heads.config import CHECKPOINT_DIR, DATA_DIR, OBV_FEATURES_PATH, PROD_META_PATH, SCOPE_DATA_DIR
from embedding_heads.dataset import build_dataloaders_from_scope
from embedding_heads.heads import build_head

# Scope -> spec-compliant export prefix
EXPORT_PREFIXES = {
    "period": "period",
    "ri_authority": "roman_imperial_authority",
    "ri_denomination": "roman_imperial_denomination",
}


def fit_temperature(logits, labels):
    """Fit temperature T by minimizing NLL on validation set."""
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()

    def nll(T):
        scaled = logits_t / T
        log_probs = F.log_softmax(scaled, dim=1)
        return -log_probs[torch.arange(len(labels_t)), labels_t].mean().item()

    result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    return result.x


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    top1_confs = probs.max(axis=1)
    top1_preds = probs.argmax(axis=1)
    top1_correct = (top1_preds == labels)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (top1_confs >= bins[i]) & (top1_confs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = top1_correct[mask].mean()
        avg_conf = top1_confs[mask].mean()
        ece += mask.sum() * abs(avg_conf - acc)
    return ece / len(labels)


def calibrate_and_evaluate(model, scope, ck):
    """Fit calibration on val set, evaluate on test set. Returns (calibration_dict, performance_dict, split_sizes)."""
    _, val_loader, test_loader, num_classes, _ = build_dataloaders_from_scope(scope)

    # Collect val logits and labels
    val_logits, val_labels = [], []
    with torch.no_grad():
        for embs, labs in val_loader:
            val_logits.append(model(embs).numpy())
            val_labels.append(labs.numpy())
    val_logits = np.concatenate(val_logits)
    val_labels = np.concatenate(val_labels)

    # Collect test logits and labels
    test_logits, test_labels = [], []
    with torch.no_grad():
        for embs, labs in test_loader:
            test_logits.append(model(embs).numpy())
            test_labels.append(labs.numpy())
    test_logits = np.concatenate(test_logits)
    test_labels = np.concatenate(test_labels)

    # Fit temperature on val
    temperature = fit_temperature(val_logits, val_labels)

    # Evaluate on test with temperature scaling
    scaled_logits = torch.from_numpy(test_logits).float() / temperature
    probs = F.softmax(scaled_logits, dim=1).numpy()
    ece = compute_ece(probs, test_labels)

    calibration = {
        "method": "temperature_scaling",
        "temperature": round(temperature, 4),
        "ece": round(float(ece), 4),
    }

    # Performance metrics
    top1_preds = probs.argmax(axis=1)
    top1 = float((top1_preds == test_labels).mean())

    topk_idx = np.argsort(-probs, axis=1)
    top2 = float(np.mean([test_labels[i] in topk_idx[i, :2] for i in range(len(test_labels))]))
    top5 = float(np.mean([test_labels[i] in topk_idx[i, :5] for i in range(len(test_labels))]))

    # Load splits for sizes
    with open(SCOPE_DATA_DIR / f"{scope}_splits.json") as f:
        splits = json.load(f)

    performance = {
        "test_accuracy": round(top1, 4),
        "top2_accuracy": round(top2, 4),
        "top5_accuracy": round(top5, 4),
    }
    split_sizes = {
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "split_seed": splits.get("seed", 42),
    }

    return calibration, performance, split_sizes


def export_scoped(scope, checkpoint_path=None, out_dir=None):
    """Export a scoped CosFace head with spec-compliant naming and calibration."""
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / f"{scope}_cosface_best.pt"
    if out_dir is None:
        out_dir = Path("/root/trivalaya-pipeline/cluster_output_prod/head_models")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legacy scopes use mapped prefixes; cluster-driven scopes use scope name directly
    prefix = EXPORT_PREFIXES.get(scope, scope)

    # Load checkpoint
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    class_map = {int(k): v for k, v in ck["class_map"].items()}
    num_classes = ck["num_classes"]

    print(f"Exporting: {scope} -> {prefix}_*")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Classes: {num_classes}")

    # Rebuild model for calibration evaluation
    model = build_head("cosface", num_classes)
    model.load_state_dict(state)
    model.eval()

    # 1. {prefix}_head.pt — nn.Linear(768, 128) projection weights
    head_state = {
        "weight": state["proj.weight"],
        "bias": state["proj.bias"],
    }
    head_path = out_dir / f"{prefix}_head.pt"
    torch.save(head_state, head_path)
    print(f"\n1. Projection head: {head_path}")
    print(f"   Shape: weight={head_state['weight'].shape}, bias={head_state['bias'].shape}")

    # 2. {prefix}_centroids.npy — L2-normalized margin prototypes
    margin_weight = state["margin.weight"]
    centroids = F.normalize(margin_weight, dim=1).numpy()
    centroids_path = out_dir / f"{prefix}_centroids.npy"
    np.save(centroids_path, centroids.astype(np.float32))
    print(f"\n2. Centroids: {centroids_path}")
    print(f"   Shape: {centroids.shape}")

    # 3. Calibration + performance evaluation
    print("\n3. Fitting calibration...")
    calibration, performance, split_sizes = calibrate_and_evaluate(model, scope, ck)
    print(f"   Temperature: {calibration['temperature']}")
    print(f"   ECE: {calibration['ece']}")
    print(f"   Test accuracy: {performance['test_accuracy']:.1%}")

    # 4. {prefix}_meta.json — spec-compliant schema
    label_names = [class_map[i] for i in range(num_classes)]
    embed_dim_out = state["proj.weight"].shape[0]
    meta = {
        "scope": scope,
        "axis": "authority",
        "num_classes": num_classes,
        "embedding_dim": embed_dim_out,
        "embedding_input": "obverse",
        "class_names": label_names,
        "performance": performance,
        "calibration": calibration,
        "provenance": {
            "checkpoint": str(checkpoint_path),
            "split_seed": split_sizes["split_seed"],
            "train_size": split_sizes["train_size"],
            "val_size": split_sizes["val_size"],
            "test_size": split_sizes["test_size"],
            "training_config": {
                "head": "cosface",
                "lr": ck.get("lr", 0.001),
                "s": ck.get("s", 30.0),
                "m": ck.get("m", 0.35),
                "proj_dim": embed_dim_out,
                "epochs_trained": ck.get("epoch", None),
                "best_val_acc": round(ck.get("val_acc", 0), 4),
            },
            "feature_file": "obv_features_768.npy",
        },
    }
    meta_path = out_dir / f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n4. Metadata: {meta_path}")
    print(f"   Labels: {label_names[:5]} ... ({num_classes} total)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Export complete. Files in: {out_dir}")
    for fp in [head_path, centroids_path, meta_path]:
        size = fp.stat().st_size
        unit = "KB" if size < 1e6 else "MB"
        val = size / 1024 if size < 1e6 else size / 1e6
        print(f"  {fp.name:<50} {val:.1f} {unit}")

    return head_path, centroids_path, meta_path


def export_legacy(checkpoint_path, out_dir, temperature=1.0):
    """Legacy v2 export (cosface_head.pt, cosface_authority_centroids.npy, etc.)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ck["model_state_dict"]
    class_map = {int(k): v for k, v in ck["class_map"].items()}
    num_classes = ck["num_classes"]
    embed_dim_out = ck.get("embed_dim_out", 128)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  Classes: {num_classes}")
    print(f"  Embed dim: {embed_dim_out}")

    head_state = {
        "weight": state["proj.weight"],
        "bias": state["proj.bias"],
    }
    head_path = out_dir / "cosface_head.pt"
    torch.save(head_state, head_path)
    print(f"\n1. Projection head: {head_path}")

    margin_weight = state["margin.weight"]
    centroids = F.normalize(margin_weight, dim=1).numpy()
    centroids_path = out_dir / "cosface_authority_centroids.npy"
    np.save(centroids_path, centroids.astype(np.float32))
    print(f"\n2. Authority centroids: {centroids_path}")

    authority_names = [class_map[i] for i in range(num_classes)]
    meta = {
        "num_classes": num_classes,
        "embed_dim": embed_dim_out,
        "temperature": temperature,
        "authority_names": authority_names,
        "class_map": class_map,
        "source_checkpoint": str(checkpoint_path),
    }
    meta_path = out_dir / "cosface_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n3. Metadata: {meta_path}")

    # Precompute projections
    print(f"\n4. Precomputing projections for prod embeddings...")
    obv = np.load(OBV_FEATURES_PATH)
    print(f"   Loaded: {obv.shape}")

    proj_layer = torch.nn.Linear(768, embed_dim_out)
    proj_layer.load_state_dict(head_state)
    proj_layer.eval()

    batch_size = 4096
    projections = []
    with torch.no_grad():
        for i in range(0, len(obv), batch_size):
            batch = torch.from_numpy(obv[i:i + batch_size]).float()
            proj = F.normalize(proj_layer(batch), dim=1)
            projections.append(proj.numpy())

    projections = np.concatenate(projections, axis=0).astype(np.float32)
    proj_path = out_dir / "cosface_projections.npy"
    np.save(proj_path, projections)
    print(f"   Projections: {proj_path}  Shape: {projections.shape}")

    norms = np.linalg.norm(projections[:100], axis=1)
    print(f"   Norm check: min={norms.min():.4f} max={norms.max():.4f}")

    print(f"\n{'='*60}")
    print(f"Export complete. Files in: {out_dir}")
    for fp in [head_path, centroids_path, meta_path, proj_path]:
        size = fp.stat().st_size
        unit = "KB" if size < 1e6 else "MB"
        val = size / 1024 if size < 1e6 else size / 1e6
        print(f"  {fp.name:<40} {val:.1f} {unit}")


def main():
    parser = argparse.ArgumentParser(description="Export CosFace for production")
    parser.add_argument("--scope", default=None,
                        help="Scoped export (legacy or cluster-driven scope name)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if args.scope:
        export_scoped(args.scope, checkpoint_path=args.checkpoint, out_dir=args.output_dir)
    else:
        ck_path = args.checkpoint or str(CHECKPOINT_DIR / "cosface_v2_best.pt")
        out = args.output_dir or str(Path("/root/trivalaya-pipeline/visual_search/data"))
        export_legacy(ck_path, out, args.temperature)


if __name__ == "__main__":
    main()
