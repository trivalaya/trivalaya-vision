"""Step 3C: LegendCNN — lightweight grayscale CNN for authority classification.

Input:  [B, 1, 64, 512]
Output: [B, num_classes]
"""

import torch
import torch.nn as nn


class LegendCNN(nn.Module):
    """4-block CNN for legend ribbon classification.

    Designed for horizontal texture and repeated local legend patterns
    on 64x512 grayscale ribbons.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1x64x512 -> 32x32x256
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x32x256 -> 64x16x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64x16x128 -> 128x8x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128x8x64 -> 256x1x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 256*1*4 = 1024
            nn.Linear(256 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
