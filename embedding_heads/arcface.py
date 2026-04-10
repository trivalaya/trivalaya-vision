"""ArcFace and CosFace margin layers.

Implements additive angular margin (ArcFace) and multiplicative cosine
margin (CosFace) for learning discriminative embedding spaces.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceMargin(nn.Module):
    """Additive Angular Margin (ArcFace).

    cos(theta + m) applied to the ground-truth class logit.
    """

    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Monotonicity threshold: cos(pi - m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels):
        """
        Args:
            features: L2-normalized embeddings [B, D]
            labels: ground-truth class indices [B]
        Returns:
            scaled logits [B, num_classes]
        """
        cosine = F.linear(features, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical safety: when cos(theta) < cos(pi - m), use fallback
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to the target class
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class CosFaceMargin(nn.Module):
    """Multiplicative Cosine Margin (CosFace).

    cos(theta) - m applied to the ground-truth class logit.
    """

    def __init__(self, in_features, num_classes, s=30.0, m=0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        cosine = F.linear(features, F.normalize(self.weight))

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = cosine - one_hot * self.m
        output *= self.s

        return output
