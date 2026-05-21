"""Three classification heads for frozen DINOv2 embeddings."""

import torch.nn as nn
import torch.nn.functional as F

from embedding_heads.arcface import ArcFaceMargin, CosFaceMargin
from embedding_heads.config import EMBED_DIM


class LinearHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x, labels=None):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, labels=None):
        return self.net(x)


class ArcFaceHead(nn.Module):
    """Projection to 128-d + ArcFace margin layer."""

    def __init__(self, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.proj = nn.Linear(EMBED_DIM, 128)
        self.margin = ArcFaceMargin(128, num_classes, s=s, m=m)

    def forward(self, x, labels=None):
        feat = F.normalize(self.proj(x), dim=1)
        if labels is not None:
            return self.margin(feat, labels)
        # Inference: return scaled cosine logits without margin
        return self.s * F.linear(feat, F.normalize(self.margin.weight))

    def embed(self, x):
        """Extract L2-normalized 128-d embeddings."""
        return F.normalize(self.proj(x), dim=1)


class CosFaceHead(nn.Module):
    """Projection to embed_dim-d + CosFace margin layer."""

    def __init__(self, num_classes, s=30.0, m=0.35, embed_dim_out=128):
        super().__init__()
        self.s = s
        self.proj = nn.Linear(EMBED_DIM, embed_dim_out)
        self.margin = CosFaceMargin(embed_dim_out, num_classes, s=s, m=m)

    def forward(self, x, labels=None):
        feat = F.normalize(self.proj(x), dim=1)
        if labels is not None:
            return self.margin(feat, labels)
        return self.s * F.linear(feat, F.normalize(self.margin.weight))

    def embed(self, x):
        """Extract L2-normalized 128-d embeddings."""
        return F.normalize(self.proj(x), dim=1)


def build_head(name, num_classes, **kwargs):
    heads = {
        "linear": LinearHead,
        "mlp": MLPHead,
        "arcface": ArcFaceHead,
        "cosface": CosFaceHead,
    }
    if name not in heads:
        raise ValueError(f"Unknown head: {name!r}. Choose from {list(heads.keys())}")
    return heads[name](num_classes, **kwargs)
