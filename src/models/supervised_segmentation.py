from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov3_backbone import Dinov3Backbone


class SimpleSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, backbone: Dinov3Backbone, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = SimpleSegHead(backbone.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
