from __future__ import annotations

import torch.nn as nn


class ClipAdapter(nn.Module):
    def __init__(self, c_in: int, bottleneck: int = 768):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class CLIPInplanted(nn.Module):
    def __init__(self, c_in: int, _device=None):
        super().__init__()
        self.cls_token_adapter = nn.ModuleList([ClipAdapter(c_in=c_in) for _ in range(4)])
        self.prompt_adapter = nn.ModuleList([ClipAdapter(c_in=768) for _ in range(2)])
        self.patch_token_adapter = nn.ModuleList([ClipAdapter(c_in=c_in) for _ in range(4)])
